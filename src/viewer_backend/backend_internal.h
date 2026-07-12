#pragma once

#include "viewer_backend/backend.h"
#include "viewer_backend/rt_scene_builder.h"

namespace rtvdb::viewer_backend {

struct backend_ops {
    backend_info (*info)();
    bool (*initialize)(const backend_config &config);
    void (*shutdown)();
    bool (*render_to_native_d3d12_texture)(
        int width,
        int height,
        const frame_scene &scene,
        bool has_frame,
        void* texture_resource);
    bool (*render_to_native_metal_texture)(
        int width,
        int height,
        const frame_scene &scene,
        bool has_frame,
        void* pixel_buffer);
    bool (*render_to_native_vulkan_texture)(
        int width,
        int height,
        const frame_scene &scene,
        bool has_frame,
        void** out_image);
    bool (*capture_to_bgra)(
        int width, int height,
        const frame_scene &scene,
        bool has_frame,
        std::vector<std::uint8_t>* out_pixels,
        bool update_build_info);
    bool (*capture_to_png)(const wchar_t* path, int width, int height, const frame_scene &scene, bool has_frame);
    void (*fill_build_info)(scene_build_info* out_info);
    bool (*pick)(
        int width,
        int height,
        int pixel_x,
        int pixel_y,
        const frame_scene &scene,
        bool has_frame,
        pick_result* out_result);
    bool (*accumulation_in_progress)();
    bool (*native_d3d12_texture_present_supported)();
    bool (*get_vulkan_renderer_interop)(vulkan_renderer_interop* out_interop);
    void (*notify_shell_post_present)();
};

#if defined(RTVDB_ENABLE_D3D12_DXR)
const backend_ops* d3d12_dxr_backend_ops();
#endif
#if defined(RTVDB_ENABLE_VULKAN_RT)
const backend_ops* vulkan_rt_backend_ops();
#endif
#if defined(__APPLE__)
const backend_ops* metal_rt_backend_ops();
#endif
void copy_present_render_scene(frame_scene* out_scene, bool* out_has_frame);
void copy_present_client_rt_scene_build(rt_scene_build* out_build);
void copy_present_render_rt_scene_build(rt_scene_build* out_build);

} // namespace rtvdb::viewer_backend
