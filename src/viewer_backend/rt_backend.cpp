#include "viewer_backend/backend_internal.h"

#include "viewer_capture/png.h"
#include "viewer_diagnostics/output.h"

#include <chrono>
#include <fstream>

namespace rtvdb::viewer_backend {
namespace {

std::unique_ptr<rt_rhi_device> g_rhi;
bool g_pick_query_pending = false;

void append_rt_device_error_log(const char* stage, const rt_device_error &error) {
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

class rt_device_access_scope {
public:
    explicit rt_device_access_scope(rt_device* device) : device_(device) {
        begin_rt_device_access(device_);
    }

    ~rt_device_access_scope() {
        end_rt_device_access(device_);
    }

private:
    rt_device* device_ = nullptr;
};
backend_info selected_info() {
    if (g_rhi == nullptr) {
        return {backend_kind::unsupported, "unsupported", {false}};
    }
    const rt_rhi_device_info info = g_rhi->info();
    const backend_kind kind = info.kind == rt_rhi_backend_kind::d3d12_dxr
        ? backend_kind::d3d12_dxr
        : backend_kind::vulkan_rt;
    return {
        kind,
        info.name,
        {
            info.hardware_ray_tracing,
        },
    };
}

bool initialize_selected(const backend_config &config) {
    g_pick_query_pending = false;
    if (g_rhi == nullptr || g_rhi->device() == nullptr) {
        return false;
    }
    rt_device_error error{};
    return initialize_rt_device(g_rhi->device(), config, &error);
}

void shutdown_selected() {
    g_pick_query_pending = false;
    if (g_rhi == nullptr || g_rhi->device() == nullptr) {
        return;
    }
    rt_device_error error{};
    (void)shutdown_rt_device(g_rhi->device(), &error);
}

bool execute_present(const rt_present_request &request, rt_present_result* out_result = nullptr) {
    if (g_rhi == nullptr || g_rhi->device() == nullptr) {
        return false;
    }
    rt_device* const device = g_rhi->device();
    rt_device_access_scope access(device);
    rt_present_result result{};
    rt_present_result* const destination = out_result != nullptr ? out_result : &result;

    rt_native_frame_request native_request{};
    native_request.operation = request.operation;
    native_request.width = request.width;
    native_request.height = request.height;
    native_request.update_build_info = request.update_build_info;
    native_request.native_target = request.native_target;
    native_request.out_native_target = request.out_native_target;
    native_request.out_pixels = request.out_pixels;

    if (request.operation == rt_present_operation::readback_bgra ||
        request.operation == rt_present_operation::post_present) {
        return execute_rt_device_native_frame(device, native_request, destination, nullptr);
    }
    if (request.scene == nullptr || request.width <= 0 || request.height <= 0) {
        return false;
    }

    const auto scene_snapshot_start = std::chrono::steady_clock::now();
    rt_scene_build build{};
    copy_present_render_rt_scene_build(&build);
    const double scene_snapshot_cpu_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - scene_snapshot_start).count();
    rt_device_frame_result frame_result{};
    rt_deferred_acceleration_submission deferred_acceleration{};
    rt_device_error error{};
    const rt_device_frame_request frame_request{&build, request.width, request.height, true, false};
    if (!prepare_rt_device_frame(
            device,
            frame_request,
            &frame_result,
            &error,
            &deferred_acceleration)) {
        append_rt_device_error_log("prepare_rt_device_frame", error);
        return false;
    }
    const auto rt_output_prepare_start = std::chrono::steady_clock::now();
    display_mode mode = display_mode::triangle_normal;
    get_display_mode(&mode);
    const rt_accumulation_key next_key = make_rt_accumulation_key(
        *request.scene,
        request.has_frame,
        build,
        request.width,
        request.height,
        static_cast<std::uint32_t>(mode));
    const bool accumulation_changed = begin_rt_device_accumulation(
        device,
        next_key,
        device->continuous_render);

    hover_highlight highlight{};
    get_hover_highlight(&highlight);
    native_request.build = &build;
    native_request.constants = make_rt_viewer_constants(
        *request.scene,
        request.has_frame,
        build,
        request.width,
        request.height,
        static_cast<std::uint32_t>(mode),
        device->accumulation_state.sample_count,
        static_cast<std::uint32_t>(highlight.kind),
        highlight.primitive_index,
        false,
        0,
        0);
    native_request.dispatch = make_rt_dispatch_plan(build, false).kind;
    native_request.reuse_output =
        (request.operation == rt_present_operation::native_d3d12 ||
            request.operation == rt_present_operation::native_vulkan) &&
        native_request.dispatch == rt_dispatch_kind::render &&
        !accumulation_changed &&
        !device->continuous_render &&
        device->accumulation_state.sample_count >= kRtMaxAccumulationSamples;
    const double rt_output_prepare_cpu_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - rt_output_prepare_start).count();

    if (!execute_rt_device_native_frame(
            device,
            native_request,
            destination,
            &error,
            &deferred_acceleration)) {
        append_rt_device_error_log("execute_rt_device_native_frame", error);
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
    if (native_request.dispatch == rt_dispatch_kind::render && !destination->reused_output) {
        complete_rt_device_accumulation(device, device->continuous_render);
    } else if (native_request.dispatch == rt_dispatch_kind::clear) {
        device->accumulation_state.active = false;
    }
    destination->accumulation_finalize_cpu_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - accumulation_finalize_start).count();
    device->last_present_result = *destination;
    return true;
}

bool render_native_d3d12(const rt_render_request &request, void* texture_resource) {
    rt_present_request present{
        rt_present_operation::native_d3d12,
        request.width,
        request.height,
        request.scene,
        request.has_frame,
        texture_resource,
    };
    return execute_present(present);
}

bool render_native_vulkan(const rt_render_request &request, void** out_image) {
    rt_present_request present{
        rt_present_operation::native_vulkan,
        request.width,
        request.height,
        request.scene,
        request.has_frame,
    };
    present.out_native_target = out_image;
    return execute_present(present);
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
    if (path == nullptr) {
        return false;
    }
    std::vector<std::uint8_t> pixels;
    if (!capture_bgra(request, &pixels, false)) {
        return false;
    }
    return viewer_capture::write_png_bgra8(
        path,
        pixels.data(),
        request.width,
        request.height,
        request.width * 4);
}

void fill_build_info(scene_build_info* out_info) {
    if (g_rhi == nullptr || g_rhi->device() == nullptr || out_info == nullptr) {
        return;
    }
    rt_device* const device = g_rhi->device();
    rt_device_access_scope access(device);
    rt_rhi_diagnostics diagnostics{};
    g_rhi->get_diagnostics(&diagnostics);
    copy_rt_rhi_diagnostics(out_info, diagnostics);
    copy_rt_device_diagnostics(out_info, *device);
}

bool pick(const rt_pick_request &request, pick_result* out_result) {
    g_pick_query_pending = false;
    if (g_rhi == nullptr || g_rhi->device() == nullptr || request.render.scene == nullptr ||
        out_result == nullptr || !request.render.has_frame || request.render.width <= 0 ||
        request.render.height <= 0 || request.pixel_x < 0 || request.pixel_y < 0) {
        return false;
    }

    rt_device* const device = g_rhi->device();
    rt_device_access_scope access(device);
    rt_scene_build build{};
    copy_present_render_rt_scene_build(&build);
    if (make_rt_dispatch_plan(build, true).kind != rt_dispatch_kind::pick) {
        return false;
    }

    const rt_device_frame_request frame_request{
        &build,
        request.render.width,
        request.render.height,
        true,
        true,
    };
    rt_device_frame_result frame_result{};
    rt_device_error error{};
    if (!prepare_rt_device_frame(device, frame_request, &frame_result, &error)) {
        append_rt_device_error_log("prepare_rt_device_frame", error);
        return false;
    }

    display_mode mode = display_mode::triangle_normal;
    get_display_mode(&mode);
    hover_highlight highlight{};
    get_hover_highlight(&highlight);
    rt_pick_dispatch_request dispatch_request{
        build.revision,
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
        device->accumulation_state.sample_count,
        static_cast<std::uint32_t>(highlight.kind),
        highlight.primitive_index,
        true,
        request.pixel_x,
        request.pixel_y);
    bool native_pending = false;
    rt_pick_dispatch_request completed_request{};
    const bool succeeded = dispatch_rt_device_pick(
        device,
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
        append_rt_device_error_log("dispatch_rt_device_pick", error);
        g_pick_query_pending = false;
    }
    return false;
}

bool selected_pick_query_pending() {
    return g_pick_query_pending;
}

bool accumulation_in_progress() {
    if (g_rhi == nullptr || g_rhi->device() == nullptr) {
        return false;
    }
    rt_device_access_scope access(g_rhi->device());
    return g_rhi->device()->accumulation_state.active;
}

bool native_d3d12_target_supported() {
    return g_rhi != nullptr &&
        g_rhi->device() != nullptr &&
        g_rhi->device()->capabilities.native_d3d12_target;
}

bool get_vulkan_interop(vulkan_renderer_interop* out_interop) {
    if (g_rhi == nullptr || g_rhi->device() == nullptr) {
        return false;
    }
    rt_device_access_scope access(g_rhi->device());
    rt_vulkan_interop_extension* const extension = g_rhi->vulkan_interop_extension();
    return extension != nullptr && extension->get_interop(out_interop);
}

void notify_post_present() {
    (void)execute_present({rt_present_operation::post_present});
}

const backend_ops kRtBackendOps{
    selected_info,
    initialize_selected,
    shutdown_selected,
    render_native_d3d12,
    nullptr,
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
    notify_post_present,
};

} // namespace

bool select_rt_rhi(backend_preference preference) {
    std::unique_ptr<rt_rhi_device> selected;
    switch (preference) {
    case backend_preference::d3d12_dxr:
#if defined(_WIN32) && defined(RTVDB_ENABLE_D3D12_DXR)
        selected = create_d3d12_dxr_rhi_device();
#endif
        break;
    case backend_preference::vulkan_rt:
#if defined(RTVDB_ENABLE_VULKAN_RT) && \
    (defined(_WIN32) || defined(__linux__) || defined(__FreeBSD__))
        selected = create_vulkan_rhi_device();
#endif
        break;
    case backend_preference::automatic:
    default:
#if defined(_WIN32) && defined(RTVDB_ENABLE_D3D12_DXR)
        selected = create_d3d12_dxr_rhi_device();
#elif defined(RTVDB_ENABLE_VULKAN_RT) && (defined(_WIN32) || defined(__linux__) || defined(__FreeBSD__))
        selected = create_vulkan_rhi_device();
#endif
        break;
    }
    g_rhi = std::move(selected);
    return g_rhi != nullptr && g_rhi->device() != nullptr;
}
const backend_ops* rt_backend_ops() {
    return &kRtBackendOps;
}

} // namespace rtvdb::viewer_backend
