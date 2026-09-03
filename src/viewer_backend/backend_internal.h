#pragma once

#include "viewer_backend/backend.h"
#include "viewer_backend/rt_renderer.h"
#include "viewer_backend/rt_scene_builder.h"

namespace rtvdb::viewer_backend {

struct rt_render_request {
    int width = 0;
    int height = 0;
    const frame_scene* scene = nullptr;
    bool has_frame = false;
    float render_scale_x = 1.0f;
    float render_scale_y = 1.0f;
    std::uint64_t scene_revision = 0;
    std::shared_ptr<const layer_visibility_map> layer_visibility;
    std::shared_ptr<const rt_scene_build> build_snapshot;
};

struct rt_pick_request {
    rt_render_request render{};
    int pixel_x = 0;
    int pixel_y = 0;
};

struct backend_ops {
    backend_info (*info)();
    bool (*initialize)(const backend_config &config);
    void (*shutdown)();
    bool (*render_to_native_d3d12_texture)(
        const rt_render_request &request,
        void* texture_resource);
    bool (*render_to_native_metal_texture)(
        const rt_render_request &request,
        void* pixel_buffer);
    bool (*render_to_native_vulkan_texture)(
        const rt_render_request &request,
        void** out_image);
    bool (*capture_to_bgra)(
        const rt_render_request &request,
        std::vector<std::uint8_t>* out_pixels,
        bool update_build_info);
    bool (*readback_current_frame_to_bgra)(
        int width,
        int height,
        std::vector<std::uint8_t>* out_pixels);
    bool (*capture_to_png)(const wchar_t* path, const rt_render_request &request);
    void (*fill_build_info)(scene_build_info* out_info);
    bool (*pick)(const rt_pick_request &request, pick_result* out_result);
    bool (*pick_query_pending)();
    bool (*accumulation_in_progress)();
    bool (*native_d3d12_texture_present_supported)();
    bool (*get_vulkan_renderer_interop)(vulkan_renderer_interop* out_interop);
    bool (*track_latest_native_delivery)();
    bool (*notify_shell_post_present)(bool* out_tracked_delivery_complete);
};

bool select_rt_rhi(backend_preference preference);
const backend_ops* rt_backend_ops();

void copy_present_render_scene(frame_scene* out_scene, bool* out_has_frame);
void copy_present_client_rt_scene_build(rt_scene_build* out_build);

} // namespace rtvdb::viewer_backend
