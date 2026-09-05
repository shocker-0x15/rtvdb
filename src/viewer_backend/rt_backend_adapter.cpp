#include "viewer_backend/backend_internal.h"

#include "viewer_capture/png.h"
#include "viewer_diagnostics/output.h"

#include <chrono>
#include <fstream>

namespace rtvdb::viewer_backend {
namespace {

std::unique_ptr<rt_rhi_device> g_rhi;
rt_renderer g_renderer{};
bool g_pick_query_pending = false;
rt_submission_token g_latest_native_delivery_submission{};
rt_submission_token g_tracked_native_delivery_submission{};

void append_rt_rhi_error_log(const char* stage, const rt_rhi_error &error) {
    if (!rtvdb::viewer_diagnostics::output_enabled()) {
        return;
    }
    try {
        const std::filesystem::path path =
            rtvdb::viewer_diagnostics::output_directory() / "rt_backend_error.log";
        std::ofstream file(path, std::ios::binary | std::ios::app);
        file << (stage != nullptr ? stage : "unknown")
             << " operation=" << static_cast<std::uint32_t>(error.operation)
             << " native_code=" << error.native_code
             << " detail=" << error.detail << '\n';
    } catch (...) {
    }
}

class rt_renderer_access_scope {
public:
    explicit rt_renderer_access_scope(rt_renderer* renderer) : renderer_(renderer) {
        begin_rt_renderer_access(renderer_);
    }

    ~rt_renderer_access_scope() {
        end_rt_renderer_access(renderer_);
    }

private:
    rt_renderer* renderer_ = nullptr;
};

backend_info selected_info() {
    if (g_rhi == nullptr) {
        return {backend_kind::unsupported, "unsupported", {false}, ""};
    }
    const rt_rhi_device_info info = g_rhi->info();
    backend_kind kind = backend_kind::unsupported;
    switch (info.kind) {
    case rt_rhi_backend_kind::d3d12_dxr:
        kind = backend_kind::d3d12_dxr;
        break;
    case rt_rhi_backend_kind::vulkan_rt:
        kind = backend_kind::vulkan_rt;
        break;
    case rt_rhi_backend_kind::metal_rt:
        kind = backend_kind::metal_rt;
        break;
    }
    return {
        kind,
        info.name,
        {
            info.hardware_ray_tracing,
        },
        info.gpu_name,
    };
}

bool initialize_selected(const backend_config &config) {
    g_pick_query_pending = false;
    g_latest_native_delivery_submission = {};
    g_tracked_native_delivery_submission = {};
    if (g_rhi == nullptr) {
        return false;
    }
    rt_rhi_error error{};
    g_renderer.rhi = g_rhi.get();
    return initialize_rt_renderer(&g_renderer, config, &error);
}

void shutdown_selected() {
    g_pick_query_pending = false;
    g_latest_native_delivery_submission = {};
    g_tracked_native_delivery_submission = {};
    if (g_rhi == nullptr || g_renderer.rhi != g_rhi.get()) {
        return;
    }
    rt_rhi_error error{};
    if (!shutdown_rt_renderer(&g_renderer, &error)) {
        append_rt_rhi_error_log("shutdown_rt_renderer", error);
    }
    g_renderer.rhi = nullptr;
}

bool execute_present(const rt_present_request &request, rt_present_result* out_result = nullptr) {
    if (g_rhi == nullptr || g_renderer.rhi != g_rhi.get()) {
        return false;
    }
    rt_renderer* const renderer = &g_renderer;
    rt_renderer_access_scope access(renderer);
    rt_present_result result{};
    rt_present_result* const destination = out_result != nullptr ? out_result : &result;
    rt_rhi_error error{};
    if (!collect_rt_renderer_render_submissions(renderer, &error)) {
        append_rt_rhi_error_log("collect_rt_renderer_render_submissions", error);
        return false;
    }

    rt_native_frame_request native_request{};
    native_request.operation = request.operation;
    native_request.width = request.width;
    native_request.height = request.height;
    native_request.render_scale_x = request.render_scale_x;
    native_request.render_scale_y = request.render_scale_y;
    native_request.update_build_info = request.update_build_info;
    native_request.native_target = request.native_target;
    native_request.out_native_target = request.out_native_target;
    native_request.out_pixels = request.out_pixels;

    if (request.operation == rt_present_operation::readback_bgra ||
        request.operation == rt_present_operation::post_present) {
        const bool succeeded = execute_rt_renderer_native_frame(
            renderer,
            native_request,
            destination,
            &error);
        if (destination->readback_submission && !track_rt_renderer_delivery_submission(
                renderer,
                destination->readback_submission,
                &error)) {
            append_rt_rhi_error_log("track_rt_renderer_readback_submission", error);
            return false;
        }
        if (!succeeded) {
            append_rt_rhi_error_log("execute_rt_renderer_readback_or_post_present", error);
        }
        return succeeded;
    }
    if (request.scene == nullptr || request.width <= 0 || request.height <= 0) {
        return false;
    }

    const auto scene_snapshot_start = std::chrono::steady_clock::now();
    if (request.build_snapshot == nullptr) {
        return false;
    }
    const rt_scene_build &build = *request.build_snapshot;
    const double scene_snapshot_cpu_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - scene_snapshot_start).count();
    rt_renderer_frame_result frame_result{};
    rt_deferred_acceleration_submission deferred_acceleration{};
    rt_renderer_frame_request frame_request{&build, request.width, request.height, true, false};
    frame_request.scene_revision = request.scene_revision;
    frame_request.layer_visibility = request.layer_visibility.get();
    if (!prepare_rt_renderer_frame(
            renderer,
            frame_request,
            &frame_result,
            &error,
            &deferred_acceleration)) {
        append_rt_rhi_error_log("prepare_rt_renderer_frame", error);
        return false;
    }
    const auto rt_output_prepare_start = std::chrono::steady_clock::now();
    display_mode mode = display_mode::triangle_normal;
    get_display_mode(&mode);
    rt_accumulation_key next_key = make_rt_accumulation_key(
        *request.scene,
        request.has_frame,
        build,
        request.width,
        request.height,
        static_cast<std::uint32_t>(mode),
        request.render_scale_x,
        request.render_scale_y);
    next_key.build_revision = request.scene_revision != 0 ? request.scene_revision : build.revision;
    begin_rt_renderer_accumulation(
        renderer,
        next_key,
        renderer->continuous_render);
    std::uint64_t accumulation_generation = 0;
    std::uint32_t accumulation_sample_index = 0;
    const bool accumulation_sample_available = get_rt_accumulation_submission(
        renderer->accumulation_state,
        &accumulation_generation,
        &accumulation_sample_index);

    hover_highlight highlight{};
    get_hover_highlight(&highlight);
    selection_highlight selection{};
    get_selection_highlight(&selection);
    native_request.build = &build;
    native_request.constants = make_rt_viewer_constants(
        *request.scene,
        request.has_frame,
        build,
        request.width,
        request.height,
        static_cast<std::uint32_t>(mode),
        accumulation_sample_index,
        static_cast<std::uint32_t>(highlight.kind),
        highlight.primitive_index,
        static_cast<std::uint32_t>(selection.kind),
        selection.primitive_index,
        false,
        0,
        0,
        request.render_scale_x,
        request.render_scale_y);
    native_request.dispatch = make_rt_dispatch_plan(build, false).kind;
    native_request.reuse_output =
        native_request.dispatch == rt_dispatch_kind::render &&
        !accumulation_sample_available;
    const double rt_output_prepare_cpu_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - rt_output_prepare_start).count();

    const bool frame_succeeded = execute_rt_renderer_native_frame(
            renderer,
            native_request,
            destination,
            &error,
            &deferred_acceleration);
    if (destination->render_submission && !track_rt_renderer_render_submission(
            renderer,
            destination->render_submission,
            accumulation_generation,
            accumulation_sample_index,
            native_request.dispatch == rt_dispatch_kind::render && !destination->reused_output,
            native_request.dispatch == rt_dispatch_kind::render && !destination->reused_output &&
                renderer->frame_state.scene_valid,
            renderer->submitted_acceleration_revision,
            renderer->submitted_acceleration_summary,
            &error)) {
        append_rt_rhi_error_log("track_rt_renderer_render_submission", error);
        return false;
    }
    if (destination->native_publish_submission && !track_rt_renderer_delivery_submission(
            renderer,
            destination->native_publish_submission,
            &error)) {
        append_rt_rhi_error_log("track_rt_renderer_native_publish_submission", error);
        return false;
    }
    if (destination->readback_submission && !track_rt_renderer_delivery_submission(
            renderer,
            destination->readback_submission,
            &error)) {
        append_rt_rhi_error_log("track_rt_renderer_readback_submission", error);
        return false;
    }
    if (!frame_succeeded) {
        append_rt_rhi_error_log("execute_rt_renderer_native_frame", error);
        return false;
    }
    destination->scene_snapshot_cpu_ms = scene_snapshot_cpu_ms;
    destination->frame_pre_acceleration_prepare_cpu_ms =
        frame_result.pre_acceleration_prepare_cpu_ms;
    destination->frame_post_acceleration_prepare_cpu_ms =
        frame_result.post_acceleration_prepare_cpu_ms;
    destination->rt_output_prepare_cpu_ms = rt_output_prepare_cpu_ms;
    destination->acceleration_timing = frame_result.acceleration_timing;
    const auto accumulation_finalize_start = std::chrono::steady_clock::now();
    if (native_request.dispatch == rt_dispatch_kind::clear) {
        renderer->accumulation_state.active = false;
    }
    destination->accumulation_finalize_cpu_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - accumulation_finalize_start).count();
    renderer->last_present_result = *destination;
    return true;
}

rt_submission_token native_delivery_submission(
    const rt_present_result &result,
    rt_rhi_backend_kind backend)
{
    if (backend == rt_rhi_backend_kind::vulkan_rt) {
        return result.render_submission;
    }
    return result.native_publish_submission;
}

bool render_native_d3d12(const rt_render_request &request, void* texture_resource) {
    g_latest_native_delivery_submission = {};
    if (g_rhi == nullptr || g_rhi->info().kind != rt_rhi_backend_kind::d3d12_dxr) {
        return false;
    }
    rt_present_request present{
        rt_present_operation::native_texture,
        request.width,
        request.height,
        request.scene,
        request.has_frame,
        texture_resource,
    };
    present.render_scale_x = request.render_scale_x;
    present.render_scale_y = request.render_scale_y;
    present.scene_revision = request.scene_revision;
    present.layer_visibility = request.layer_visibility;
    present.build_snapshot = request.build_snapshot;
    rt_present_result result{};
    const bool succeeded = execute_present(present, &result);
    g_latest_native_delivery_submission = native_delivery_submission(
        result,
        rt_rhi_backend_kind::d3d12_dxr);
    return succeeded && g_latest_native_delivery_submission;
}

bool render_native_metal(const rt_render_request &request, void* pixel_buffer) {
    g_latest_native_delivery_submission = {};
    if (g_rhi == nullptr || g_rhi->info().kind != rt_rhi_backend_kind::metal_rt) {
        return false;
    }
    rt_present_request present{
        rt_present_operation::native_texture,
        request.width,
        request.height,
        request.scene,
        request.has_frame,
        pixel_buffer,
    };
    present.render_scale_x = request.render_scale_x;
    present.render_scale_y = request.render_scale_y;
    present.scene_revision = request.scene_revision;
    present.layer_visibility = request.layer_visibility;
    present.build_snapshot = request.build_snapshot;
    rt_present_result result{};
    const bool succeeded = execute_present(present, &result);
    g_latest_native_delivery_submission = native_delivery_submission(
        result,
        rt_rhi_backend_kind::metal_rt);
    return succeeded && g_latest_native_delivery_submission;
}

bool render_native_vulkan(const rt_render_request &request, void** out_image) {
    g_latest_native_delivery_submission = {};
    if (g_rhi == nullptr || g_rhi->info().kind != rt_rhi_backend_kind::vulkan_rt) {
        return false;
    }
    rt_present_request present{
        rt_present_operation::native_texture,
        request.width,
        request.height,
        request.scene,
        request.has_frame,
    };
    present.render_scale_x = request.render_scale_x;
    present.render_scale_y = request.render_scale_y;
    present.scene_revision = request.scene_revision;
    present.layer_visibility = request.layer_visibility;
    present.build_snapshot = request.build_snapshot;
    present.out_native_target = out_image;
    rt_present_result result{};
    const bool succeeded = execute_present(present, &result);
    g_latest_native_delivery_submission = native_delivery_submission(
        result,
        rt_rhi_backend_kind::vulkan_rt);
    return succeeded && g_latest_native_delivery_submission;
}

bool capture_bgra(
    const rt_render_request &request,
    std::vector<std::uint8_t>* out_pixels,
    bool update_build_info)
{
    rt_present_request present{
        rt_present_operation::capture_bgra,
        request.width,
        request.height,
        request.scene,
        request.has_frame,
    };
    present.render_scale_x = request.render_scale_x;
    present.render_scale_y = request.render_scale_y;
    present.scene_revision = request.scene_revision;
    present.layer_visibility = request.layer_visibility;
    present.build_snapshot = request.build_snapshot;
    present.out_pixels = out_pixels;
    present.update_build_info = update_build_info;
    return execute_present(present);
}

bool readback_bgra(int width, int height, std::vector<std::uint8_t>* out_pixels) {
    rt_present_request request{rt_present_operation::readback_bgra, width, height};
    request.out_pixels = out_pixels;
    return execute_present(request);
}

bool capture_png(const wchar_t* path, const rt_render_request &request) {
    if (path == nullptr || request.width <= 0 || request.height <= 0) {
        return false;
    }
    std::vector<std::uint8_t> pixels;
    rt_present_request present{
        rt_present_operation::capture_bgra,
        request.width,
        request.height,
        request.scene,
        request.has_frame};
    present.render_scale_x = request.render_scale_x;
    present.render_scale_y = request.render_scale_y;
    present.scene_revision = request.scene_revision;
    present.layer_visibility = request.layer_visibility;
    present.build_snapshot = request.build_snapshot;
    present.out_pixels = &pixels;
    rt_present_result result{};
    if (!execute_present(present, &result)) {
        return false;
    }
    if (pixels.empty() && result.readback_submission) {
        rt_rhi_timing timing{};
        rt_rhi_error error{};
        {
            rt_renderer_access_scope access(&g_renderer);
            if (!wait_for_rt_submission(
                    &g_renderer,
                    result.readback_submission,
                    &timing,
                    &error)) {
                append_rt_rhi_error_log("wait_for_rt_png_readback", error);
                return false;
            }
        }
        if (!readback_bgra(request.width, request.height, &pixels)) {
            return false;
        }
    }
    const std::size_t expected_size =
        static_cast<std::size_t>(request.width) *
        static_cast<std::size_t>(request.height) * 4u;
    if (pixels.size() != expected_size) {
        return false;
    }
    std::vector<std::uint8_t> save_pixels;
    if (!viewer_capture::composite_bgra8_over_color(
            pixels.data(),
            request.width,
            request.height,
            request.width * 4,
            0,
            0,
            0,
            255,
            &save_pixels)) {
        return false;
    }
    return viewer_capture::write_png_bgra8(
        path,
        save_pixels.data(),
        request.width,
        request.height,
        request.width * 4);
}

void fill_build_info(scene_build_info* out_info) {
    if (g_rhi == nullptr || g_renderer.rhi != g_rhi.get() || out_info == nullptr) {
        return;
    }
    rt_renderer* const renderer = &g_renderer;
    rt_renderer_access_scope access(renderer);
    rt_rhi_diagnostics diagnostics{};
    g_rhi->get_diagnostics(&diagnostics);
    copy_rt_rhi_diagnostics(out_info, diagnostics);
    copy_rt_renderer_diagnostics(out_info, *renderer);
    out_info->dispatch_gpu_ms = diagnostics.dispatch_gpu_ms;
}

bool pick(const rt_pick_request &request, pick_result* out_result) {
    g_pick_query_pending = false;
    if (g_rhi == nullptr || g_renderer.rhi != g_rhi.get() || request.render.scene == nullptr ||
        out_result == nullptr || !request.render.has_frame || request.render.width <= 0 ||
        request.render.height <= 0 || request.pixel_x < 0 || request.pixel_y < 0) {
        return false;
    }

    rt_renderer* const renderer = &g_renderer;
    rt_renderer_access_scope access(renderer);
    rt_rhi_error error{};
    if (!collect_rt_renderer_render_submissions(renderer, &error)) {
        append_rt_rhi_error_log("collect_rt_renderer_render_submissions", error);
        return false;
    }
    if (request.render.build_snapshot == nullptr) {
        return false;
    }
    const rt_scene_build &build = *request.render.build_snapshot;
    if (make_rt_dispatch_plan(build, true).kind != rt_dispatch_kind::pick) {
        return false;
    }

    rt_renderer_frame_request frame_request{
        &build,
        request.render.width,
        request.render.height,
        true,
        true,
    };
    frame_request.scene_revision = request.render.scene_revision;
    frame_request.layer_visibility = request.render.layer_visibility.get();
    rt_renderer_frame_result frame_result{};
    if (!prepare_rt_renderer_frame(renderer, frame_request, &frame_result, &error)) {
        append_rt_rhi_error_log("prepare_rt_renderer_frame", error);
        return false;
    }

    display_mode mode = display_mode::triangle_normal;
    get_display_mode(&mode);
    hover_highlight highlight{};
    get_hover_highlight(&highlight);
    selection_highlight selection{};
    get_selection_highlight(&selection);
    rt_pick_dispatch_request dispatch_request{
        request.render.scene_revision != 0 ? request.render.scene_revision : build.revision,
        request.render.width,
        request.render.height,
        request.pixel_x,
        request.pixel_y,
    };
    dispatch_request.constants = make_rt_viewer_constants(
        *request.render.scene,
        request.render.has_frame,
        build,
        request.render.width,
        request.render.height,
        static_cast<std::uint32_t>(mode),
        renderer->accumulation_state.sample_count,
        static_cast<std::uint32_t>(highlight.kind),
        highlight.primitive_index,
        static_cast<std::uint32_t>(selection.kind),
        selection.primitive_index,
        true,
        request.pixel_x,
        request.pixel_y,
        request.render.render_scale_x,
        request.render.render_scale_y);
    bool native_pending = false;
    rt_pick_dispatch_request completed_request{};
    const bool succeeded = dispatch_rt_renderer_pick(
        renderer,
        dispatch_request,
        out_result,
        &completed_request,
        &error,
        &native_pending);
    if (succeeded) {
        const bool stale = !rt_pick_dispatch_request_matches(completed_request, dispatch_request);
        out_result->pixel_x = completed_request.pixel_x;
        out_result->pixel_y = completed_request.pixel_y;
        out_result->completed = rt_pick_dispatch_request_matches_scene_and_view(
            completed_request,
            dispatch_request);
        g_pick_query_pending = native_pending || stale;
        return !stale;
    }
    if (native_pending) {
        g_pick_query_pending = true;
    } else {
        append_rt_rhi_error_log("dispatch_rt_renderer_pick", error);
        g_pick_query_pending = false;
    }
    return false;
}

bool selected_pick_query_pending() {
    return g_pick_query_pending;
}

bool accumulation_in_progress() {
    if (g_rhi == nullptr || g_renderer.rhi != g_rhi.get()) {
        return false;
    }
    rt_renderer_access_scope access(&g_renderer);
    return g_renderer.accumulation_state.active;
}

bool native_d3d12_target_supported() {
    return g_rhi != nullptr &&
        g_renderer.rhi == g_rhi.get() &&
        g_renderer.capabilities.native_d3d12_target;
}

bool get_vulkan_interop(vulkan_renderer_interop* out_interop) {
    if (g_rhi == nullptr || g_renderer.rhi != g_rhi.get()) {
        return false;
    }
    rt_renderer_access_scope access(&g_renderer);
    rt_vulkan_interop_extension* const extension = g_rhi->vulkan_interop_extension();
    return extension != nullptr && extension->get_interop(out_interop);
}

bool track_latest_native_delivery() {
    if (g_rhi == nullptr || g_renderer.rhi != g_rhi.get()) {
        return false;
    }
    g_tracked_native_delivery_submission = g_latest_native_delivery_submission;
    return true;
}

bool notify_post_present(bool* out_tracked_delivery_complete) {
    if (out_tracked_delivery_complete == nullptr) {
        return false;
    }
    *out_tracked_delivery_complete = false;
    if (!execute_present({rt_present_operation::post_present})) {
        return false;
    }
    if (!g_tracked_native_delivery_submission) {
        *out_tracked_delivery_complete = true;
        return true;
    }

    rt_renderer_access_scope access(&g_renderer);
    bool complete = false;
    rt_rhi_error error{};
    if (!g_rhi->is_complete(g_tracked_native_delivery_submission, &complete, &error)) {
        append_rt_rhi_error_log("query_tracked_native_delivery", error);
        return false;
    }
    *out_tracked_delivery_complete = complete;
    if (complete) {
        g_tracked_native_delivery_submission = {};
    }
    return true;
}

bool wait_for_idle() {
    if (g_rhi == nullptr || g_renderer.rhi != g_rhi.get()) {
        return false;
    }
    rt_rhi_timing timing{};
    rt_rhi_error error{};
    rt_renderer_access_scope access(&g_renderer);
    if (!wait_for_rt_renderer_idle(&g_renderer, &timing, &error)) {
        append_rt_rhi_error_log("wait_for_rt_renderer_idle", error);
        return false;
    }
    return true;
}

const backend_ops kRtBackendOps{
    selected_info,
    initialize_selected,
    shutdown_selected,
    render_native_d3d12,
    render_native_metal,
    render_native_vulkan,
    capture_bgra,
    readback_bgra,
    capture_png,
    fill_build_info,
    pick,
    selected_pick_query_pending,
    accumulation_in_progress,
    native_d3d12_target_supported,
    get_vulkan_interop,
    track_latest_native_delivery,
    notify_post_present,
    wait_for_idle,
};

} // namespace

bool select_rt_rhi(backend_preference preference) {
    g_latest_native_delivery_submission = {};
    g_tracked_native_delivery_submission = {};
    rt_rhi_factory_result selected{};
    switch (preference) {
    case backend_preference::d3d12_dxr:
#if defined(_WIN32) && defined(RTVDB_ENABLE_D3D12_DXR)
        selected = create_d3d12_dxr_rhi();
#endif
        break;
    case backend_preference::vulkan_rt:
#if defined(RTVDB_ENABLE_VULKAN_RT) && \
    (defined(_WIN32) || defined(__linux__) || defined(__FreeBSD__))
        selected = create_vulkan_rhi();
#endif
        break;
    case backend_preference::metal_rt:
#if defined(__APPLE__) && !defined(RTVDB_ENABLE_VULKAN_RT)
        selected = create_metal_rhi();
#endif
        break;
    case backend_preference::automatic:
    default:
#if defined(__APPLE__) && !defined(RTVDB_ENABLE_VULKAN_RT)
        selected = create_metal_rhi();
#elif defined(_WIN32) && defined(RTVDB_ENABLE_D3D12_DXR)
        selected = create_d3d12_dxr_rhi();
#elif defined(RTVDB_ENABLE_VULKAN_RT) && (defined(_WIN32) || defined(__linux__) || defined(__FreeBSD__))
        selected = create_vulkan_rhi();
#endif
        break;
    }
    if (selected.device == nullptr ||
        !validate_rt_shader_package_desc(selected.shader_package)) {
        selected = {};
    }
    g_renderer.rhi = nullptr;
    g_renderer.shader_package = selected.shader_package;
    g_rhi = std::move(selected.device);
    g_renderer.rhi = g_rhi.get();
    return g_rhi != nullptr;
}
const backend_ops* rt_backend_ops() {
    return &kRtBackendOps;
}

} // namespace rtvdb::viewer_backend
