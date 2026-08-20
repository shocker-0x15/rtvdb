#pragma once

#include "rtvdb/rtvdb.h"
#include "viewer_shell/shell.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rtvdb::viewer_backend {

enum class backend_kind {
    unsupported,
    d3d12_dxr,
    vulkan_rt,
    metal_rt,
};

enum class backend_preference {
    automatic,
    d3d12_dxr,
    vulkan_rt,
    metal_rt,
};

struct backend_caps {
    bool hardware_ray_tracing;
};

enum class display_mode {
    triangle_normal,
    client_color,
    simple_shaded,
    primitive_id,
    geometry_index,
    instance_index,
};

enum class helper_plane {
    xy,
    xz,
    yz,
};

enum class hover_highlight_kind : std::uint32_t {
    none = 0,
    triangle = 1,
    point = 2,
    line = 3,
};

enum class line_flags : std::uint32_t {
    none = 0,
    fixed_color = 1u << 0,
    non_pickable = 1u << 1,
};

constexpr line_flags operator|(line_flags a, line_flags b) {
    return static_cast<line_flags>(static_cast<std::uint32_t>(a) | static_cast<std::uint32_t>(b));
}

constexpr line_flags operator&(line_flags a, line_flags b) {
    return static_cast<line_flags>(static_cast<std::uint32_t>(a) & static_cast<std::uint32_t>(b));
}

constexpr bool has_line_flags(line_flags value, line_flags mask) {
    return static_cast<std::uint32_t>(value & mask) != 0u;
}

struct backend_info {
    backend_kind kind;
    const char* name;
    backend_caps caps;
    const char* gpu_name;
};

struct triangle {
    rtvdb::vec3 a;
    rtvdb::vec3 b;
    rtvdb::vec3 c;
    rtvdb::rgba color;
    std::uint32_t user_data = 0;
    std::string layer = "Default";
    bool visible = true;
};

struct point {
    rtvdb::vec3 position;
    float radius;
    rtvdb::rgba color;
    std::uint32_t user_data = 0;
    std::string layer = "Default";
    bool visible = true;
};

struct line {
    rtvdb::vec3 a;
    float radius;
    rtvdb::vec3 b;
    rtvdb::rgba color;
    std::uint32_t user_data = 0;
    line_flags flags = line_flags::none;
    std::string layer = "Default";
    bool visible = true;
};

struct scene_bounds {
    rtvdb::vec3 min{};
    rtvdb::vec3 max{};
    bool valid = false;
};

struct frame_scene {
    std::uint64_t frame_serial = 0;
    std::uint64_t connection_serial = 0;
    std::uint64_t view_revision = 0;
    std::string app_name = "waiting";
    rtvdb::camera camera{};
    bool camera_set_by_client = false;
    rtvdb::camera_projection projection_blend_from = rtvdb::camera_projection::perspective;
    rtvdb::camera_projection projection_blend_to = rtvdb::camera_projection::perspective;
    float projection_blend_t = 1.0f;
    rtvdb::vec3 helper_overlay_bounds_min{};
    rtvdb::vec3 helper_overlay_bounds_max{};
    bool helper_overlay_bounds_valid = false;
    std::vector<triangle> triangles;
    std::vector<point> points;
    std::vector<line> lines;
};

struct scene_build_info {
    std::uint64_t backend_recovery_count = 0;
    std::uint64_t revision = 0;
    std::size_t triangle_count = 0;
    std::size_t point_count = 0;
    std::size_t line_count = 0;
    std::size_t triangle_chunk_count = 0;
    std::size_t point_chunk_count = 0;
    std::size_t line_chunk_count = 0;
    std::size_t triangle_blas_chunk_set_count = 0;
    std::size_t point_blas_chunk_set_count = 0;
    std::size_t line_blas_chunk_set_count = 0;
    std::size_t reused_triangle_chunk_count = 0;
    std::size_t rebuilt_triangle_chunk_count = 0;
    std::size_t blas_reused_count = 0;
    std::size_t blas_rebuilt_count = 0;
    std::size_t blas_reused_triangle_chunk_count = 0;
    std::size_t blas_rebuilt_triangle_chunk_count = 0;
    std::size_t tlas_rebuild_count = 0;
    double accel_build_ms = 0.0;
    double accel_host_prep_ms = 0.0;
    double accel_instance_build_ms = 0.0;
    double accel_procedural_aabb_ms = 0.0;
    double accel_command_record_ms = 0.0;
    double accel_resource_alloc_ms = 0.0;
    double accel_build_call_record_ms = 0.0;
    double accel_prebuild_info_ms = 0.0;
    double accel_chunk_blas_prebuild_info_ms = 0.0;
    std::uint32_t accel_chunk_blas_prebuild_info_count = 0;
    double accel_group_blas_prebuild_info_ms = 0.0;
    std::uint32_t accel_group_blas_prebuild_info_count = 0;
    double accel_point_blas_prebuild_info_ms = 0.0;
    std::uint32_t accel_point_blas_prebuild_info_count = 0;
    double accel_line_blas_prebuild_info_ms = 0.0;
    std::uint32_t accel_line_blas_prebuild_info_count = 0;
    double accel_tlas_prebuild_info_ms = 0.0;
    std::uint32_t accel_tlas_prebuild_info_count = 0;
    double accel_startup_prebuild_warmup_ms = 0.0;
    double accel_tlas_instance_upload_ms = 0.0;
    double accel_submit_cpu_ms = 0.0;
    double accel_gpu_wait_ms = 0.0;
    double accel_gpu_ms = 0.0;
    double paint_rt_scene_snapshot_ms = 0.0;
    double paint_rt_pre_acceleration_prepare_ms = 0.0;
    double paint_as_command_slot_wait_ms = 0.0;
    double paint_accel_command_record_ms = 0.0;
    double paint_rt_post_acceleration_prepare_ms = 0.0;
    double paint_rt_output_prepare_ms = 0.0;
    double paint_rt_output_command_slot_wait_ms = 0.0;
    double paint_rt_command_record_ms = 0.0;
    double paint_rt_submit_ms = 0.0;
    double paint_as_finalize_ms = 0.0;
    double paint_native_target_publish_ms = 0.0;
    double paint_rt_accumulation_finalize_ms = 0.0;
    double dispatch_ms = 0.0;
    double dispatch_submit_cpu_ms = 0.0;
    double dispatch_gpu_wait_ms = 0.0;
    double dispatch_gpu_ms = 0.0;
    double command_slot_reuse_wait_ms = 0.0;
    double readback_ms = 0.0;
    std::uint32_t accumulation_sample_count = 0;
    std::uint32_t accumulation_target_sample_count = 0;
    bool accumulation_in_progress = false;
    std::size_t vertex_count = 0;
    std::size_t index_count = 0;
    std::uint64_t scratch_growth_count = 0;
    std::size_t scratch_capacity_bytes = 0;
    std::size_t scratch_peak_capacity_bytes = 0;
    std::uint64_t acceleration_resource_allocation_count = 0;
    std::uint64_t acceleration_resource_reallocation_count = 0;
    std::size_t acceleration_capacity_bytes = 0;
    std::size_t acceleration_peak_capacity_bytes = 0;
    std::uint64_t scene_buffer_allocation_count = 0;
    std::uint64_t scene_buffer_growth_count = 0;
    std::size_t scene_buffer_capacity_bytes = 0;
    std::size_t scene_buffer_peak_capacity_bytes = 0;
    std::uint64_t blas_storage_pool_hit_count = 0;
    std::uint64_t blas_storage_pool_miss_count = 0;
    std::uint64_t scene_buffer_pool_hit_count = 0;
    std::uint64_t scene_buffer_pool_miss_count = 0;
    std::size_t blas_storage_pool_bytes = 0;
    std::size_t scene_buffer_pool_bytes = 0;
    std::size_t retired_resource_bytes = 0;
    std::uint64_t resource_pool_eviction_count = 0;
};

struct hover_highlight {
    hover_highlight_kind kind = hover_highlight_kind::none;
    std::uint32_t primitive_index = 0;
};

struct selection_highlight {
    hover_highlight_kind kind = hover_highlight_kind::none;
    std::uint32_t primitive_index = 0;
};

struct pick_result {
    hover_highlight_kind kind = hover_highlight_kind::none;
    std::uint32_t primitive_index = 0;
    float distance = 0.0f;
    int pixel_x = -1;
    int pixel_y = -1;
    bool completed = false;
};

struct d3d12_interop_config {
    void* device = nullptr;
    void* command_queue = nullptr;
};

struct vulkan_renderer_interop {
    void* instance = nullptr;
    void* physical_device = nullptr;
    void* device = nullptr;
    std::uint32_t graphics_queue_family_index = 0;
    std::uint32_t present_queue_family_index = 0;
};

using scene_ready_callback = void (*)(const frame_scene* scene, bool has_frame, void* user_data);

struct backend_config {
    int capture_width = 1280;
    int capture_height = 720;
    scene_ready_callback scene_ready = nullptr;
    void* scene_ready_user_data = nullptr;
    backend_preference preferred_backend = backend_preference::automatic;
    bool continuous_render = false;
    d3d12_interop_config d3d12{};
    float render_scale_x = 1.0f;
    float render_scale_y = 1.0f;
};

backend_info select_planned_backend();
const char* backend_preference_name(backend_preference preference);
bool try_parse_backend_preference_name(const char* name, backend_preference* out_preference);
backend_info current_backend();
const char* display_mode_name(display_mode mode);
bool initialize_backend(const backend_config &config);
void shutdown_backend();
bool recover_backend();
bool recover_backend_with_d3d12_interop(const d3d12_interop_config &d3d12);
bool submit_scene_build(const frame_scene &scene, bool has_frame, bool allow_auto_frame = true);
void copy_present_scene(frame_scene* out_scene, bool* out_has_frame);
void copy_present_render_scene(frame_scene* out_scene, bool* out_has_frame);
bool acquire_present_render_scene(
    std::shared_ptr<const frame_scene>* out_scene,
    bool* out_has_frame);
void copy_present_camera(
    rtvdb::camera* out_camera,
    rtvdb::camera_projection* out_projection_blend_from,
    rtvdb::camera_projection* out_projection_blend_to,
    float* out_projection_blend_t,
    bool* out_has_frame);
void copy_present_build_info(scene_build_info* out_info);
bool copy_present_client_scene_bounds(rtvdb::vec3* out_min, rtvdb::vec3* out_max);
bool build_in_progress();
void set_auto_frame_enabled(bool enabled);
bool auto_frame_enabled();
void set_helper_overlay_enabled(bool enabled);
bool helper_overlay_enabled();
void set_helper_overlay_plane(helper_plane plane);
helper_plane current_helper_overlay_plane();
void apply_reference_grid_request(rtvdb::reference_grid value);
void set_capture_size(int width, int height);
void set_render_scale(float scale_x, float scale_y);
void set_display_mode(display_mode mode);
bool get_display_mode(display_mode* out_mode);
void set_hover_highlight(const hover_highlight &highlight);
bool get_hover_highlight(hover_highlight* out_highlight);
void set_selection_highlight(const selection_highlight &highlight);
bool get_selection_highlight(selection_highlight* out_highlight);
bool pick(
    int width,
    int height,
    int pixel_x,
    int pixel_y,
    const frame_scene &scene,
    bool has_frame,
    pick_result* out_result);
bool pick_query_pending();
bool accumulation_in_progress();
bool native_d3d12_texture_present_supported();
bool get_vulkan_renderer_interop(vulkan_renderer_interop* out_interop);
bool track_latest_native_delivery();
bool notify_shell_post_present(bool* out_tracked_delivery_complete);
bool render_frame_to_native_d3d12_texture(
    int width,
    int height,
    const frame_scene &scene,
    bool has_frame,
    void* texture_resource);
bool render_frame_to_native_metal_texture(
    int width,
    int height,
    const frame_scene &scene,
    bool has_frame,
    void* pixel_buffer);
bool render_frame_to_native_vulkan_texture(
    int width,
    int height,
    const frame_scene &scene,
    bool has_frame,
    void** out_image);
bool capture_frame_to_bgra(
    int width, int height,
    const frame_scene &scene,
    bool has_frame,
    std::vector<std::uint8_t>* out_pixels,
    bool update_build_info = true);
bool readback_current_frame_to_bgra(
    int width,
    int height,
    std::vector<std::uint8_t>* out_pixels);
bool capture_frame_to_png(const wchar_t* path, int width, int height, const frame_scene &scene, bool has_frame);

} // namespace rtvdb::viewer_backend
