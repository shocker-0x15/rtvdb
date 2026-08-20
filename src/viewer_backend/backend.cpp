#include "viewer_backend/backend_internal.h"
#include "viewer_backend/helper_overlay.h"
#include "viewer_backend/rt_diagnostics.h"
#include "viewer_backend/rt_scene_builder.h"
#include "viewer_capture/png.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>

namespace rtvdb::viewer_backend {
namespace {

constexpr float kPi = 3.14159265f;
constexpr float kMinVectorLengthSq = 1.0e-8f;
constexpr auto kStaleSceneCoalesceDelay = std::chrono::milliseconds(100);
constexpr auto kLargeSceneCoalesceDelay = std::chrono::milliseconds(1500);
constexpr std::size_t kLargeScenePrimitiveThreshold = 65536;

struct backend_state {
    backend_config config{};
    bool initialized = false;
    const backend_ops* ops = nullptr;
    std::mutex mutex;
    std::condition_variable condition;
    std::chrono::steady_clock::time_point pending_updated_at{};
    std::thread worker;
    bool stop_worker = false;
    frame_scene pending_client_scene;
    bool pending_has_frame = false;
    bool pending_allow_auto_frame = true;
    bool pending_helper_overlay_update = false;
    std::uint64_t pending_revision = 0;
    std::uint64_t next_revision = 0;
    frame_scene present_client_scene;
    bool present_client_has_frame = false;
    std::uint64_t present_client_revision = 0;
    rt_scene_build present_client_rt_build{};
    std::shared_ptr<const frame_scene> present_render_scene;
    bool present_render_has_frame = false;
    std::uint64_t present_render_revision = 0;
    rt_scene_build present_render_rt_build{};
    bool build_in_progress = false;
    bool auto_frame = true;
    bool helper_overlay = true;
    helper_plane helper_overlay_plane = helper_plane::xy;
    display_mode display = display_mode::client_color;
    hover_highlight highlight{};
    selection_highlight selection{};
    bool recovery_in_progress = false;
    std::uint64_t recovery_count = 0;
} g_backend;

rt_render_request make_rt_render_request(
    int width,
    int height,
    const frame_scene &scene,
    bool has_frame)
{
    rt_render_request request{width, height, &scene, has_frame};
    std::scoped_lock lock(g_backend.mutex);
    request.render_scale_x = g_backend.config.render_scale_x;
    request.render_scale_y = g_backend.config.render_scale_y;
    return request;
}

struct scene_build_cancel_context {
    std::uint64_t revision = 0;
};

bool should_cancel_scene_build(void* user_data) {
    const auto* context = static_cast<const scene_build_cancel_context*>(user_data);
    if (context == nullptr) {
        return false;
    }

    std::scoped_lock lock(g_backend.mutex);
    return g_backend.stop_worker || context->revision != g_backend.pending_revision;
}

std::chrono::milliseconds scene_coalesce_delay_locked() {
    const std::size_t primitive_count =
        g_backend.pending_client_scene.triangles.size() +
        g_backend.pending_client_scene.points.size() +
        g_backend.pending_client_scene.lines.size();
    return primitive_count >= kLargeScenePrimitiveThreshold
        ? kLargeSceneCoalesceDelay
        : kStaleSceneCoalesceDelay;
}

backend_info unsupported_backend_info() {
    return {
        backend_kind::unsupported,
        "unsupported",
        {false},
        ""
    };
}

const backend_ops* select_backend_ops(backend_preference preference) {
    switch (preference) {
    case backend_preference::d3d12_dxr:
    case backend_preference::vulkan_rt:
        return select_rt_rhi(preference) ? rt_backend_ops() : nullptr;
    case backend_preference::metal_rt:
#if defined(__APPLE__)
        return select_rt_rhi(preference) ? rt_backend_ops() : nullptr;
#else
        return nullptr;
#endif
    case backend_preference::automatic:
    default:
        return select_rt_rhi(backend_preference::automatic) ? rt_backend_ops() : nullptr;
    }
}

bool backend_uses_shell_coordinated_recovery(backend_kind kind) {
    return kind == backend_kind::d3d12_dxr || kind == backend_kind::vulkan_rt;
}

const backend_ops* select_backend_ops() {
    return select_backend_ops(backend_preference::automatic);
}

rtvdb::vec3 operator+(const rtvdb::vec3 &a, const rtvdb::vec3 &b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

rtvdb::vec3 operator-(const rtvdb::vec3 &a, const rtvdb::vec3 &b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

rtvdb::vec3 operator*(const rtvdb::vec3 &v, float s) {
    return {v.x * s, v.y * s, v.z * s};
}

float dot(const rtvdb::vec3 &a, const rtvdb::vec3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

float length_sq(const rtvdb::vec3 &v) {
    return dot(v, v);
}

rtvdb::vec3 normalize_or(const rtvdb::vec3 &v, const rtvdb::vec3 &fallback) {
    const float len2 = length_sq(v);
    if (len2 <= kMinVectorLengthSq) {
        return fallback;
    }
    const float inv_len = 1.0f / std::sqrt(len2);
    return v * inv_len;
}

float clamped_projection_half_angle_radians(const rtvdb::camera &camera, float aspect) {
    switch (camera.projection) {
    case rtvdb::camera_projection::fisheye: {
        const float theta = (std::clamp)(camera.fisheye_theta_degrees, 1.0f, 180.0f) * (kPi / 180.0f);
        const float phi = (std::clamp)(camera.fisheye_phi_degrees, 1.0f, 360.0f) * (kPi / 180.0f);
        return (std::max)(0.15f, 0.5f * (std::min)(theta, phi));
    }
    case rtvdb::camera_projection::orthographic:
        return 0.0f;
    case rtvdb::camera_projection::perspective:
    default: {
        const float fov_y = (std::clamp)(camera.vertical_fov_degrees, 20.0f, 100.0f) * (kPi / 180.0f);
        const float fov_x = 2.0f * std::atan(std::tan(fov_y * 0.5f) * aspect);
        return (std::max)(0.15f, 0.5f * (std::min)(fov_x, fov_y));
    }
    }
}

void apply_auto_frame(frame_scene* scene, const backend_config &config, const scene_bounds &bounds) {
    if (scene == nullptr || !bounds.valid) {
        return;
    }

    const rtvdb::vec3 center = (bounds.min + bounds.max) * 0.5f;
    const rtvdb::vec3 diagonal = bounds.max - bounds.min;
    float radius = 0.5f * std::sqrt((std::max)(length_sq(diagonal), 1.0e-6f));
    radius = (std::max)(radius, 0.5f);

    const rtvdb::vec3 forward = normalize_or(scene->camera.target - scene->camera.origin, {0.0f, 1.0f, 0.0f});
    rtvdb::vec3 up = normalize_or(scene->camera.up, {0.0f, 0.0f, 1.0f});
    if (std::fabs(dot(forward, up)) > 0.98f) {
        up = {0.0f, 1.0f, 0.0f};
        if (std::fabs(dot(forward, up)) > 0.98f) {
            up = {1.0f, 0.0f, 0.0f};
        }
    }

    const float aspect = config.capture_height > 0
        ? static_cast<float>(config.capture_width) / static_cast<float>(config.capture_height)
        : (16.0f / 9.0f);
    scene->camera.target = center;
    scene->camera.up = up;
    if (scene->camera.projection == rtvdb::camera_projection::orthographic) {
        const float padded_diameter = (std::max)(radius * 2.7f, 0.01f);
        scene->camera.orthographic_height = aspect >= 1.0f
            ? padded_diameter
            : padded_diameter / (std::max)(aspect, 0.01f);
        const float distance = (std::max)(radius * 2.0f, kMinVectorLengthSq);
        scene->camera.origin = center - forward * distance;
        return;
    }

    const float half_angle = clamped_projection_half_angle_radians(scene->camera, aspect);
    const float distance = radius / std::tan(half_angle) + radius * 0.35f;
    scene->camera.origin = center - forward * distance;
}

void schedule_helper_overlay_rebuild_locked() {
    g_backend.pending_has_frame = g_backend.present_client_has_frame;
    g_backend.pending_allow_auto_frame = false;
    g_backend.pending_helper_overlay_update = true;
    g_backend.pending_revision = ++g_backend.next_revision;
    g_backend.pending_updated_at = std::chrono::steady_clock::now();
    g_backend.build_in_progress = g_backend.pending_has_frame;
    g_backend.condition.notify_all();
}

void backend_worker() {
    std::unique_lock lock(g_backend.mutex);
    for (;;) {
        g_backend.condition.wait(lock, [] {
            return g_backend.stop_worker || g_backend.pending_revision != 0;
        });
        if (g_backend.stop_worker) {
            break;
        }

        for (;;) {
            if (g_backend.pending_helper_overlay_update) {
                break;
            }
            const auto observed_pending_updated_at = g_backend.pending_updated_at;
            const auto coalesce_deadline = observed_pending_updated_at + scene_coalesce_delay_locked();
            const bool pending_changed = g_backend.condition.wait_until(
                lock,
                coalesce_deadline,
                [&] {
                    return g_backend.stop_worker
                        || g_backend.pending_revision == 0
                        || g_backend.pending_updated_at != observed_pending_updated_at;
                });
            if (g_backend.stop_worker) {
                break;
            }
            if (g_backend.pending_revision == 0) {
                break;
            }
            if (pending_changed) {
                continue;
            }
            break;
        }
        if (g_backend.stop_worker) {
            break;
        }
        if (g_backend.pending_revision == 0) {
            continue;
        }

        const bool pending_helper_overlay_update = g_backend.pending_helper_overlay_update;
        frame_scene pending_scene = pending_helper_overlay_update
            ? g_backend.present_client_scene
            : g_backend.pending_client_scene;
        const bool pending_has_frame = g_backend.pending_has_frame;
        const bool pending_allow_auto_frame = g_backend.pending_allow_auto_frame;
        const std::uint64_t pending_revision = g_backend.pending_revision;
        const backend_config config = g_backend.config;
        const bool auto_frame = g_backend.auto_frame;
        const bool helper_overlay = g_backend.helper_overlay;
        const helper_plane helper_overlay_plane = g_backend.helper_overlay_plane;
        const rt_scene_build previous_client_build = g_backend.present_client_rt_build;
        lock.unlock();

        scene_build_cancel_context cancel_context{pending_revision};

        const auto pending_scene_is_stale = [&]() {
            lock.lock();
            const bool stale = g_backend.stop_worker || pending_revision != g_backend.pending_revision;
            if (!stale) {
                lock.unlock();
            }
            return stale;
        };

        const bool can_reuse_client_build = pending_helper_overlay_update &&
            previous_client_build.revision != 0 &&
            previous_client_build.connection_serial == pending_scene.connection_serial &&
            previous_client_build.triangle_count == pending_scene.triangles.size() &&
            previous_client_build.point_count == pending_scene.points.size() &&
            previous_client_build.line_count == pending_scene.lines.size();
        rt_scene_build client_rt_build{};
        if (can_reuse_client_build) {
            client_rt_build = previous_client_build;
            client_rt_build.revision = pending_revision;
        } else if (!build_rt_scene_input(
                       pending_scene,
                       previous_client_build.revision != 0 ? &previous_client_build : nullptr,
                       pending_revision,
                       kDefaultRtSceneTriangleChunkPrimitives,
                       &client_rt_build,
                       should_cancel_scene_build,
                       &cancel_context)) {
            lock.lock();
            if (g_backend.pending_revision == pending_revision) {
                g_backend.build_in_progress = false;
            }
            continue;
        }
        if (pending_scene_is_stale()) {
            continue;
        }
        if (pending_has_frame && auto_frame && pending_allow_auto_frame) {
            apply_auto_frame(&pending_scene, config, client_rt_build.bounds);
        }

        frame_scene render_scene = pending_scene;
        if (pending_has_frame && helper_overlay) {
            scene_bounds helper_overlay_bounds = client_rt_build.bounds;
            if (pending_scene.helper_overlay_bounds_valid) {
                helper_overlay_bounds.min = pending_scene.helper_overlay_bounds_min;
                helper_overlay_bounds.max = pending_scene.helper_overlay_bounds_max;
                helper_overlay_bounds.valid = true;
            }
            append_default_helper_lines(helper_overlay_bounds, helper_overlay_plane, &render_scene);
        }

        if (pending_scene_is_stale()) {
            continue;
        }

        rt_scene_build render_rt_build{};
        if (!build_rt_scene_overlay_input(
                render_scene,
                client_rt_build,
                pending_revision,
                &render_rt_build,
                should_cancel_scene_build,
                &cancel_context)) {
            lock.lock();
            if (g_backend.pending_revision == pending_revision) {
                g_backend.build_in_progress = false;
            }
            continue;
        }

        frame_scene ready_scene{};
        bool ready_has_frame = false;
        scene_ready_callback ready_callback = nullptr;
        void* ready_user_data = nullptr;

        lock.lock();
        if (g_backend.stop_worker) {
            break;
        }
        const bool superseded_by_newer_scene = pending_revision != g_backend.pending_revision;

        if (superseded_by_newer_scene) {
            continue;
        }

        g_backend.present_client_scene = pending_scene;
        g_backend.present_client_has_frame = pending_has_frame;
        g_backend.present_client_revision = pending_revision;
        g_backend.present_client_rt_build = std::move(client_rt_build);
        g_backend.present_render_scene = std::make_shared<frame_scene>(std::move(render_scene));
        g_backend.present_render_has_frame = pending_has_frame;
        g_backend.present_render_revision = pending_revision;
        g_backend.present_render_rt_build = std::move(render_rt_build);
        g_backend.pending_revision = 0;
        g_backend.pending_helper_overlay_update = false;
        g_backend.build_in_progress = false;

        ready_scene = g_backend.present_client_scene;
        ready_has_frame = g_backend.present_client_has_frame;
        ready_callback = g_backend.config.scene_ready;
        ready_user_data = g_backend.config.scene_ready_user_data;
        lock.unlock();

        if (ready_callback != nullptr) {
            ready_callback(&ready_scene, ready_has_frame, ready_user_data);
        }

        lock.lock();
    }
}

bool try_recover_backend(const d3d12_interop_config* replacement_d3d12 = nullptr) {
    backend_config config{};
    frame_scene recovery_scene{};
    bool recovery_has_frame = false;
    bool recovery_allow_auto_frame = false;
    bool auto_frame = true;
    bool helper_overlay = true;
    helper_plane helper_overlay_plane = helper_plane::xy;
    display_mode display = display_mode::client_color;
    hover_highlight highlight{};
    selection_highlight selection{};
    {
        std::scoped_lock lock(g_backend.mutex);
        if (!g_backend.initialized || g_backend.ops == nullptr || g_backend.recovery_in_progress ||
            (replacement_d3d12 != nullptr &&
             g_backend.ops->info().kind != backend_kind::d3d12_dxr)) {
            return false;
        }
        g_backend.recovery_in_progress = true;
        config = g_backend.config;
        if (replacement_d3d12 != nullptr) {
            config.d3d12 = *replacement_d3d12;
        }
        auto_frame = g_backend.auto_frame;
        helper_overlay = g_backend.helper_overlay;
        helper_overlay_plane = g_backend.helper_overlay_plane;
        display = g_backend.display;
        highlight = g_backend.highlight;
        selection = g_backend.selection;
        if (g_backend.pending_revision != 0) {
            recovery_scene = g_backend.pending_helper_overlay_update
                ? g_backend.present_client_scene
                : g_backend.pending_client_scene;
            recovery_has_frame = g_backend.pending_has_frame;
            recovery_allow_auto_frame = g_backend.pending_allow_auto_frame;
        } else {
            recovery_scene = g_backend.present_client_scene;
            recovery_has_frame = g_backend.present_client_has_frame;
        }
    }

    append_rt_diagnostics_log_line("backend_recovery.log", "RT backend recovery begin");
    shutdown_backend();
    const bool initialized = initialize_backend(config);
    bool scene_queued = initialized;
    if (initialized) {
        set_auto_frame_enabled(auto_frame);
        set_helper_overlay_enabled(helper_overlay);
        set_helper_overlay_plane(helper_overlay_plane);
        set_display_mode(display);
        set_hover_highlight(highlight);
        set_selection_highlight(selection);
        if (recovery_has_frame) {
            scene_queued = submit_scene_build(
                recovery_scene,
                true,
                recovery_allow_auto_frame);
        }
        if (!scene_queued) {
            shutdown_backend();
        }
    }

    const bool recovered = initialized && scene_queued;

    {
        std::scoped_lock lock(g_backend.mutex);
        if (recovered) {
            ++g_backend.recovery_count;
        }
        g_backend.recovery_in_progress = false;
    }
    append_rt_diagnostics_log_line(
        "backend_recovery.log",
        recovered ? "RT backend recovery succeeded" : "RT backend recovery failed");
    return recovered;
}

} // namespace

backend_info select_planned_backend() {
    const backend_ops* ops = select_backend_ops();
    if (ops == nullptr || ops->info == nullptr) {
        return unsupported_backend_info();
    }
    return ops->info();
}

const char* backend_preference_name(backend_preference preference) {
    switch (preference) {
    case backend_preference::automatic:
        return "automatic";
    case backend_preference::d3d12_dxr:
        return "dxr";
    case backend_preference::vulkan_rt:
        return "vulkan";
    case backend_preference::metal_rt:
        return "metal";
    default:
        return "unknown";
    }
}

bool try_parse_backend_preference_name(const char* name, backend_preference* out_preference) {
    if (name == nullptr || out_preference == nullptr) {
        return false;
    }
    if (std::strcmp(name, "automatic") == 0 || std::strcmp(name, "auto") == 0) {
        *out_preference = backend_preference::automatic;
        return true;
    }
    if (std::strcmp(name, "dxr") == 0 || std::strcmp(name, "d3d12_dxr") == 0) {
        *out_preference = backend_preference::d3d12_dxr;
        return true;
    }
    if (std::strcmp(name, "vulkan") == 0 || std::strcmp(name, "vulkan_rt") == 0) {
        *out_preference = backend_preference::vulkan_rt;
        return true;
    }
    if (std::strcmp(name, "metal") == 0 || std::strcmp(name, "metal_rt") == 0) {
        *out_preference = backend_preference::metal_rt;
        return true;
    }
    return false;
}

backend_info current_backend() {
    const backend_ops* ops = g_backend.ops != nullptr ? g_backend.ops : select_backend_ops();
    if (ops == nullptr || ops->info == nullptr) {
        return unsupported_backend_info();
    }
    return ops->info();
}

const char* display_mode_name(display_mode mode) {
    switch (mode) {
    case display_mode::triangle_normal:
        return "triangle_normal";
    case display_mode::client_color:
        return "client_color";
    case display_mode::simple_shaded:
        return "simple_shaded";
    case display_mode::primitive_id:
        return "primitive_id";
    case display_mode::geometry_index:
        return "geometry_index";
    case display_mode::instance_index:
        return "instance_index";
    default:
        return "unknown";
    }
}

bool initialize_backend(const backend_config &config) {
    if (g_backend.initialized) {
        return true;
    }

    if (!g_backend.recovery_in_progress) {
        g_backend.recovery_count = 0;
    }
    g_backend.ops = select_backend_ops(config.preferred_backend);
    g_backend.config = config;
    if (g_backend.ops == nullptr || !g_backend.ops->initialize(config)) {
        if (g_backend.ops != nullptr) {
            g_backend.ops->shutdown();
        }
        g_backend.ops = nullptr;
        return false;
    }

    g_backend.stop_worker = false;
    g_backend.worker = std::thread(backend_worker);
    g_backend.initialized = true;
    return true;
}

void shutdown_backend() {
    if (!g_backend.initialized || g_backend.ops == nullptr) {
        return;
    }

    {
        std::scoped_lock lock(g_backend.mutex);
        g_backend.stop_worker = true;
        g_backend.condition.notify_all();
    }
    if (g_backend.worker.joinable()) {
        g_backend.worker.join();
    }

    g_backend.ops->shutdown();

    std::scoped_lock lock(g_backend.mutex);
    g_backend.config = {};
    g_backend.initialized = false;
    g_backend.ops = nullptr;
    g_backend.stop_worker = false;
    g_backend.pending_client_scene = {};
    g_backend.pending_has_frame = false;
    g_backend.pending_allow_auto_frame = true;
    g_backend.pending_helper_overlay_update = false;
    g_backend.pending_revision = 0;
    g_backend.pending_updated_at = {};
    g_backend.next_revision = 0;
    g_backend.present_client_scene = {};
    g_backend.present_client_has_frame = false;
    g_backend.present_client_revision = 0;
    g_backend.present_client_rt_build = {};
    g_backend.present_render_scene.reset();
    g_backend.present_render_has_frame = false;
    g_backend.present_render_revision = 0;
    g_backend.present_render_rt_build = {};
    g_backend.build_in_progress = false;
    g_backend.auto_frame = true;
    g_backend.helper_overlay = true;
    g_backend.helper_overlay_plane = helper_plane::xy;
    g_backend.display = display_mode::triangle_normal;
    g_backend.highlight = {};
    g_backend.selection = {};
}

bool recover_backend() {
    {
        std::scoped_lock lock(g_backend.mutex);
        if (!g_backend.initialized || g_backend.ops == nullptr) {
            return false;
        }
    }
    return try_recover_backend();
}

bool recover_backend_with_d3d12_interop(const d3d12_interop_config &d3d12) {
    if (d3d12.device == nullptr || d3d12.command_queue == nullptr) {
        return false;
    }
    return try_recover_backend(&d3d12);
}

bool submit_scene_build(const frame_scene &scene, bool has_frame, bool allow_auto_frame) {
    if (!g_backend.initialized || g_backend.ops == nullptr) {
        return false;
    }

    std::scoped_lock lock(g_backend.mutex);
    g_backend.pending_client_scene = scene;
    g_backend.pending_has_frame = has_frame;
    g_backend.pending_allow_auto_frame = allow_auto_frame;
    g_backend.pending_helper_overlay_update = false;
    g_backend.pending_revision = ++g_backend.next_revision;
    g_backend.pending_updated_at = std::chrono::steady_clock::now();
    g_backend.build_in_progress = true;
    g_backend.condition.notify_all();
    return true;
}

void copy_present_scene(frame_scene* out_scene, bool* out_has_frame) {
    std::scoped_lock lock(g_backend.mutex);
    if (out_scene != nullptr) {
        *out_scene = g_backend.present_client_scene;
    }
    if (out_has_frame != nullptr) {
        *out_has_frame = g_backend.present_client_has_frame;
    }
}

void copy_present_camera(
    rtvdb::camera* out_camera,
    rtvdb::camera_projection* out_projection_blend_from,
    rtvdb::camera_projection* out_projection_blend_to,
    float* out_projection_blend_t,
    bool* out_has_frame)
{
    std::scoped_lock lock(g_backend.mutex);
    if (out_camera != nullptr) {
        *out_camera = g_backend.present_render_scene != nullptr
            ? g_backend.present_render_scene->camera
            : rtvdb::camera{};
    }
    if (out_projection_blend_from != nullptr) {
        *out_projection_blend_from = g_backend.present_render_scene != nullptr
            ? g_backend.present_render_scene->projection_blend_from
            : rtvdb::camera_projection::perspective;
    }
    if (out_projection_blend_to != nullptr) {
        *out_projection_blend_to = g_backend.present_render_scene != nullptr
            ? g_backend.present_render_scene->projection_blend_to
            : rtvdb::camera_projection::perspective;
    }
    if (out_projection_blend_t != nullptr) {
        *out_projection_blend_t = g_backend.present_render_scene != nullptr
            ? g_backend.present_render_scene->projection_blend_t
            : 1.0f;
    }
    if (out_has_frame != nullptr) {
        *out_has_frame = g_backend.present_render_has_frame;
    }
}

void copy_present_render_scene(frame_scene* out_scene, bool* out_has_frame) {
    std::scoped_lock lock(g_backend.mutex);
    if (out_scene != nullptr) {
        *out_scene = g_backend.present_render_scene != nullptr ? *g_backend.present_render_scene : frame_scene{};
    }
    if (out_has_frame != nullptr) {
        *out_has_frame = g_backend.present_render_has_frame;
    }
}

bool acquire_present_render_scene(
    std::shared_ptr<const frame_scene>* out_scene,
    bool* out_has_frame)
{
    std::scoped_lock lock(g_backend.mutex);
    if (out_scene != nullptr) {
        *out_scene = g_backend.present_render_scene;
    }
    if (out_has_frame != nullptr) {
        *out_has_frame = g_backend.present_render_has_frame;
    }
    return g_backend.present_render_scene != nullptr;
}

void copy_present_client_rt_scene_build(rt_scene_build* out_build) {
    std::scoped_lock lock(g_backend.mutex);
    if (out_build != nullptr) {
        *out_build = g_backend.present_client_rt_build;
    }
}

void copy_present_render_rt_scene_build(rt_scene_build* out_build) {
    std::scoped_lock lock(g_backend.mutex);
    if (out_build != nullptr) {
        *out_build = g_backend.present_render_rt_build;
    }
}

void copy_present_build_info(scene_build_info* out_info) {
    if (out_info == nullptr) {
        return;
    }

    void (*fill_build_info)(scene_build_info*) = nullptr;
    std::uint64_t recovery_count = 0;
    {
        std::scoped_lock lock(g_backend.mutex);
        recovery_count = g_backend.recovery_count;
        out_info->revision = g_backend.present_client_rt_build.revision;
        out_info->triangle_count = g_backend.present_client_rt_build.triangle_count;
        out_info->point_count = g_backend.present_client_rt_build.point_count;
        out_info->line_count = g_backend.present_client_rt_build.line_count;
        out_info->triangle_chunk_count = g_backend.present_client_rt_build.triangle_chunks.size();
        out_info->point_chunk_count = g_backend.present_client_rt_build.point_chunks.size();
        out_info->line_chunk_count = g_backend.present_client_rt_build.line_chunks.size();
        out_info->triangle_blas_chunk_set_count = g_backend.present_client_rt_build.triangle_blas_chunk_sets.size();
        out_info->point_blas_chunk_set_count = g_backend.present_client_rt_build.point_blas_chunk_sets.size();
        out_info->line_blas_chunk_set_count = g_backend.present_client_rt_build.line_blas_chunk_sets.size();
        out_info->reused_triangle_chunk_count = g_backend.present_client_rt_build.reused_triangle_chunk_count;
        out_info->rebuilt_triangle_chunk_count = g_backend.present_client_rt_build.rebuilt_triangle_chunk_count;
        out_info->vertex_count = g_backend.present_client_rt_build.vertex_count;
        out_info->index_count = g_backend.present_client_rt_build.index_count;
        if (g_backend.ops != nullptr) {
            fill_build_info = g_backend.ops->fill_build_info;
        }
    }

    if (fill_build_info != nullptr) {
        fill_build_info(out_info);
    }
    out_info->backend_recovery_count = recovery_count;
}

bool copy_present_client_scene_bounds(rtvdb::vec3* out_min, rtvdb::vec3* out_max) {
    std::scoped_lock lock(g_backend.mutex);
    const scene_bounds &bounds = g_backend.present_client_rt_build.bounds;
    if (!bounds.valid) {
        return false;
    }
    if (out_min != nullptr) {
        *out_min = bounds.min;
    }
    if (out_max != nullptr) {
        *out_max = bounds.max;
    }
    return true;
}

bool build_in_progress() {
    std::scoped_lock lock(g_backend.mutex);
    return g_backend.build_in_progress;
}

void set_auto_frame_enabled(bool enabled) {
    std::scoped_lock lock(g_backend.mutex);
    g_backend.auto_frame = enabled;
}

bool auto_frame_enabled() {
    std::scoped_lock lock(g_backend.mutex);
    return g_backend.auto_frame;
}

void set_helper_overlay_enabled(bool enabled) {
    if (!g_backend.initialized || g_backend.ops == nullptr) {
        return;
    }

    std::scoped_lock lock(g_backend.mutex);
    if (g_backend.helper_overlay == enabled) {
        return;
    }

    g_backend.helper_overlay = enabled;
    schedule_helper_overlay_rebuild_locked();
}

bool helper_overlay_enabled() {
    std::scoped_lock lock(g_backend.mutex);
    return g_backend.helper_overlay;
}

void set_helper_overlay_plane(helper_plane plane) {
    if (!g_backend.initialized || g_backend.ops == nullptr) {
        return;
    }

    std::scoped_lock lock(g_backend.mutex);
    if (g_backend.helper_overlay_plane == plane) {
        return;
    }

    g_backend.helper_overlay_plane = plane;
    if (g_backend.helper_overlay) {
        schedule_helper_overlay_rebuild_locked();
    }
}

helper_plane current_helper_overlay_plane() {
    std::scoped_lock lock(g_backend.mutex);
    return g_backend.helper_overlay_plane;
}

void apply_reference_grid_request(rtvdb::reference_grid value) {
    if (!g_backend.initialized || g_backend.ops == nullptr || value == rtvdb::reference_grid::viewer_default) {
        return;
    }

    bool enabled = true;
    helper_plane plane = helper_plane::xy;
    switch (value) {
    case rtvdb::reference_grid::off:
        enabled = false;
        break;
    case rtvdb::reference_grid::xz_grid:
        plane = helper_plane::xz;
        break;
    case rtvdb::reference_grid::yz_grid:
        plane = helper_plane::yz;
        break;
    case rtvdb::reference_grid::xy_grid:
    case rtvdb::reference_grid::viewer_default:
    default:
        break;
    }

    std::scoped_lock lock(g_backend.mutex);
    if (g_backend.helper_overlay == enabled && (!enabled || g_backend.helper_overlay_plane == plane)) {
        return;
    }
    g_backend.helper_overlay = enabled;
    if (enabled) {
        g_backend.helper_overlay_plane = plane;
    }
    schedule_helper_overlay_rebuild_locked();
}

void set_capture_size(int width, int height) {
    if (width <= 0 || height <= 0) {
        return;
    }
    std::scoped_lock lock(g_backend.mutex);
    g_backend.config.capture_width = width;
    g_backend.config.capture_height = height;
}

void set_render_scale(float scale_x, float scale_y) {
    if (!std::isfinite(scale_x) || scale_x <= 0.0f ||
        !std::isfinite(scale_y) || scale_y <= 0.0f) {
        return;
    }
    std::scoped_lock lock(g_backend.mutex);
    g_backend.config.render_scale_x = scale_x;
    g_backend.config.render_scale_y = scale_y;
}

void set_display_mode(display_mode mode) {
    std::scoped_lock lock(g_backend.mutex);
    g_backend.display = mode;
}

bool get_display_mode(display_mode* out_mode) {
    std::scoped_lock lock(g_backend.mutex);
    if (out_mode != nullptr) {
        *out_mode = g_backend.display;
        return true;
    }
    return false;
}

void set_hover_highlight(const hover_highlight &highlight) {
    std::scoped_lock lock(g_backend.mutex);
    g_backend.highlight = highlight;
}

bool get_hover_highlight(hover_highlight* out_highlight) {
    std::scoped_lock lock(g_backend.mutex);
    if (out_highlight != nullptr) {
        *out_highlight = g_backend.highlight;
        return true;
    }
    return false;
}

void set_selection_highlight(const selection_highlight &highlight) {
    std::scoped_lock lock(g_backend.mutex);
    g_backend.selection = highlight;
}

bool get_selection_highlight(selection_highlight* out_highlight) {
    std::scoped_lock lock(g_backend.mutex);
    if (out_highlight != nullptr) {
        *out_highlight = g_backend.selection;
        return true;
    }
    return false;
}

bool pick(
    int width,
    int height,
    int pixel_x,
    int pixel_y,
    const frame_scene &scene,
    bool has_frame,
    pick_result* out_result)
{
    rt_pick_request request{};
    request.render = make_rt_render_request(width, height, scene, has_frame);
    request.pixel_x = pixel_x;
    request.pixel_y = pixel_y;
    if (!g_backend.initialized || g_backend.ops == nullptr || g_backend.ops->pick == nullptr) {
        return false;
    }
    return g_backend.ops->pick(request, out_result);
}

bool pick_query_pending() {
    return g_backend.initialized &&
        g_backend.ops != nullptr &&
        g_backend.ops->pick_query_pending != nullptr &&
        g_backend.ops->pick_query_pending();
}

bool accumulation_in_progress() {
    if (!g_backend.initialized || g_backend.ops == nullptr || g_backend.ops->accumulation_in_progress == nullptr) {
        return false;
    }
    return g_backend.ops->accumulation_in_progress();
}

bool native_d3d12_texture_present_supported() {
    if (!g_backend.initialized ||
        g_backend.ops == nullptr ||
        g_backend.ops->native_d3d12_texture_present_supported == nullptr) {
        return false;
    }
    return g_backend.ops->native_d3d12_texture_present_supported();
}

bool get_vulkan_renderer_interop(vulkan_renderer_interop* out_interop) {
    if (out_interop == nullptr) {
        return false;
    }
    *out_interop = {};
    if (!g_backend.initialized ||
        g_backend.ops == nullptr ||
        g_backend.ops->get_vulkan_renderer_interop == nullptr) {
        return false;
    }
    return g_backend.ops->get_vulkan_renderer_interop(out_interop);
}

bool track_latest_native_delivery() {
    return g_backend.initialized &&
        g_backend.ops != nullptr &&
        g_backend.ops->track_latest_native_delivery != nullptr &&
        g_backend.ops->track_latest_native_delivery();
}

bool notify_shell_post_present(bool* out_tracked_delivery_complete) {
    if (out_tracked_delivery_complete != nullptr) {
        *out_tracked_delivery_complete = false;
    }
    if (!g_backend.initialized || g_backend.ops == nullptr ||
        g_backend.ops->notify_shell_post_present == nullptr || out_tracked_delivery_complete == nullptr) {
        return false;
    }
    if (g_backend.ops->notify_shell_post_present(out_tracked_delivery_complete)) {
        return true;
    }
    if (!backend_uses_shell_coordinated_recovery(g_backend.ops->info().kind)) {
        (void)try_recover_backend();
    }
    return false;
}

bool render_frame_to_native_d3d12_texture(
    int width,
    int height,
    const frame_scene &scene,
    bool has_frame,
    void* texture_resource)
{
    const rt_render_request request = make_rt_render_request(width, height, scene, has_frame);
    if (!g_backend.initialized ||
        g_backend.ops == nullptr ||
        g_backend.ops->render_to_native_d3d12_texture == nullptr) {
        return false;
    }
    if (g_backend.ops->render_to_native_d3d12_texture(request, texture_resource)) {
        return true;
    }
    if (!backend_uses_shell_coordinated_recovery(g_backend.ops->info().kind)) {
        (void)try_recover_backend();
    }
    return false;
}

bool render_frame_to_native_metal_texture(
    int width,
    int height,
    const frame_scene &scene,
    bool has_frame,
    void* pixel_buffer)
{
    const rt_render_request request = make_rt_render_request(width, height, scene, has_frame);
    if (!g_backend.initialized ||
        g_backend.ops == nullptr ||
        g_backend.ops->render_to_native_metal_texture == nullptr) {
        return false;
    }
    if (g_backend.ops->render_to_native_metal_texture(request, pixel_buffer)) {
        return true;
    }
    if (!backend_uses_shell_coordinated_recovery(g_backend.ops->info().kind)) {
        (void)try_recover_backend();
    }
    return false;
}

bool render_frame_to_native_vulkan_texture(
    int width,
    int height,
    const frame_scene &scene,
    bool has_frame,
    void** out_image)
{
    const rt_render_request request = make_rt_render_request(width, height, scene, has_frame);
    if (out_image == nullptr ||
        !g_backend.initialized ||
        g_backend.ops == nullptr ||
        g_backend.ops->render_to_native_vulkan_texture == nullptr) {
        return false;
    }
    *out_image = nullptr;
    return g_backend.ops->render_to_native_vulkan_texture(request, out_image);
}

bool capture_frame_to_bgra(
    int width, int height,
    const frame_scene &scene,
    bool has_frame,
    std::vector<std::uint8_t>* out_pixels,
    bool update_build_info)
{
    const rt_render_request request = make_rt_render_request(width, height, scene, has_frame);
    if (!g_backend.initialized || g_backend.ops == nullptr || g_backend.ops->capture_to_bgra == nullptr) {
        return false;
    }
    if (g_backend.ops->capture_to_bgra(request, out_pixels, update_build_info)) {
        return true;
    }
    if (!backend_uses_shell_coordinated_recovery(g_backend.ops->info().kind)) {
        (void)try_recover_backend();
    }
    return false;
}

bool readback_current_frame_to_bgra(
    int width,
    int height,
    std::vector<std::uint8_t>* out_pixels)
{
    if (!g_backend.initialized || g_backend.ops == nullptr ||
        g_backend.ops->readback_current_frame_to_bgra == nullptr) {
        return false;
    }
    if (g_backend.ops->readback_current_frame_to_bgra(width, height, out_pixels)) {
        return true;
    }
    if (!backend_uses_shell_coordinated_recovery(g_backend.ops->info().kind)) {
        (void)try_recover_backend();
    }
    return false;
}

bool capture_frame_to_png(const wchar_t* path, int width, int height, const frame_scene &scene, bool has_frame) {
    if (!g_backend.initialized || g_backend.ops == nullptr ||
        g_backend.ops->capture_to_png == nullptr || path == nullptr) {
        return false;
    }
    const rt_render_request request = make_rt_render_request(width, height, scene, has_frame);
    if (g_backend.ops->capture_to_png(path, request)) {
        return true;
    }
    if (!backend_uses_shell_coordinated_recovery(g_backend.ops->info().kind)) {
        (void)try_recover_backend();
    }
    return false;
}

} // namespace rtvdb::viewer_backend
