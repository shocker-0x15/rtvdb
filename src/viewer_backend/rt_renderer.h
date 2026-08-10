#pragma once

#include "viewer_backend/backend.h"
#include "viewer_backend/rt_acceleration_plan.h"
#include "viewer_backend/rt_diagnostics.h"
#include "viewer_backend/rt_render_plan.h"
#include "viewer_backend/rt_rhi_device.h"
#include "viewer_backend/rt_scene_builder.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace rtvdb::viewer_backend {

struct rt_scene_buffer_upload {
    rt_buffer_handle staging{};
    rt_buffer_handle destination{};
    std::size_t size = 0;
    rt_resource_usage destination_usage = rt_resource_usage::undefined;
};

struct rt_scene_buffer_resources {
    rt_buffer_handle positions{};
    rt_buffer_handle indices{};
    rt_buffer_handle triangle_colors{};
    rt_buffer_handle instance_metadata{};
    rt_buffer_handle points{};
    rt_buffer_handle lines{};
    rt_buffer_handle point_aabbs{};
    rt_buffer_handle line_aabbs{};
    std::size_t position_count = 0;
    std::size_t index_count = 0;
    std::size_t triangle_color_count = 0;
    std::size_t point_count = 0;
    std::size_t line_count = 0;
    std::size_t point_aabb_count = 0;
    std::size_t line_aabb_count = 0;
    std::size_t instance_metadata_count = 0;
    std::size_t positions_capacity_bytes = 0;
    std::size_t indices_capacity_bytes = 0;
    std::size_t triangle_colors_capacity_bytes = 0;
    std::size_t instance_metadata_capacity_bytes = 0;
    std::size_t points_capacity_bytes = 0;
    std::size_t lines_capacity_bytes = 0;
    std::size_t point_aabbs_capacity_bytes = 0;
    std::size_t line_aabbs_capacity_bytes = 0;
    std::uint64_t point_geometry_fingerprint = 0;
    std::uint64_t line_geometry_fingerprint = 0;
    std::uint64_t revision = 0;
    std::uint64_t connection_serial = 0;
    double procedural_aabb_upload_ms = 0.0;
    std::vector<rt_scene_buffer_upload> uploads;
};

enum class rt_present_operation : std::uint32_t {
    native_texture,
    capture_bgra,
    readback_bgra,
    post_present,
};

struct rt_present_request {
    rt_present_operation operation = rt_present_operation::capture_bgra;
    int width = 0;
    int height = 0;
    const frame_scene* scene = nullptr;
    bool has_frame = false;
    void* native_target = nullptr;
    void** out_native_target = nullptr;
    std::vector<std::uint8_t>* out_pixels = nullptr;
    bool update_build_info = false;
};

struct rt_present_result {
    bool rendered = false;
    bool captured = false;
    bool reused_output = false;
    rt_submission_token render_submission{};
    rt_submission_token native_publish_submission{};
    rt_submission_token readback_submission{};
    double scene_snapshot_cpu_ms = 0.0;
    double frame_pre_acceleration_prepare_cpu_ms = 0.0;
    double frame_post_acceleration_prepare_cpu_ms = 0.0;
    double rt_output_prepare_cpu_ms = 0.0;
    double acceleration_finalize_cpu_ms = 0.0;
    double accumulation_finalize_cpu_ms = 0.0;
    rt_rhi_timing acceleration_timing{};
    rt_rhi_timing output_timing{};
    rt_rhi_timing timing{};
    rt_rhi_timing native_publish_timing{};
    double native_publish_cpu_ms = 0.0;
    double readback_ms = 0.0;
};

struct rt_native_frame_request {
    rt_present_operation operation = rt_present_operation::capture_bgra;
    rt_dispatch_kind dispatch = rt_dispatch_kind::clear;
    int width = 0;
    int height = 0;
    const rt_scene_build* build = nullptr;
    rt_viewer_constants constants{};
    float clear_color[4]{0.0f, 0.0f, 0.0f, 0.0f};
    bool reuse_output = false;
    bool update_build_info = false;
    void* native_target = nullptr;
    void** out_native_target = nullptr;
    std::vector<std::uint8_t>* out_pixels = nullptr;
};

struct rt_pick_dispatch_request {
    std::uint64_t scene_revision = 0;
    int width = 0;
    int height = 0;
    int pixel_x = 0;
    int pixel_y = 0;
    rt_viewer_constants constants{};
};

bool rt_pick_dispatch_request_matches(
    const rt_pick_dispatch_request &left,
    const rt_pick_dispatch_request &right);

bool rt_pick_dispatch_request_matches_scene_and_view(
    const rt_pick_dispatch_request &left,
    const rt_pick_dispatch_request &right);

struct rt_pick_slot {
    rt_buffer_handle readback_buffer{};
    rt_pick_dispatch_request request{};
    rt_submission_token submission{};
    bool pending = false;
};

struct rt_pick_gpu_result {
    std::uint32_t primitive_kind = 0;
    std::uint32_t primitive_index = 0;
    float distance = 0.0f;
    std::uint32_t hit = 0;
};

static_assert(sizeof(rt_pick_gpu_result) == 16);

struct rt_renderer_frame_request {
    const rt_scene_build* build = nullptr;
    int width = 0;
    int height = 0;
    bool require_output = true;
    bool require_pick = false;
    const rt_acceleration_build_plan* acceleration_plan = nullptr;
    rt_blas_cache_update_plan* blas_cache_plan = nullptr;
    const rt_scene_resource_data* resources = nullptr;
    const rt_acceleration_command_plan* acceleration_commands = nullptr;
};

struct rt_blas_storage_pool_entry {
    rt_blas_storage_key key{};
    rt_blas_handle acceleration{};
    std::size_t capacity_bytes = 0;
    rt_submission_token retirement_submission{};
    std::uint64_t sequence = 0;
};

enum class rt_scene_buffer_role : std::uint8_t {
    positions,
    indices,
    triangle_colors,
    instance_metadata,
    points,
    lines,
    point_aabbs,
    line_aabbs,
};

struct rt_scene_buffer_pool_entry {
    rt_buffer_handle buffer{};
    std::size_t capacity_bytes = 0;
    rt_scene_buffer_role role = rt_scene_buffer_role::positions;
    std::size_t format_stride = 0;
    std::uint32_t usage = 0;
    rt_memory_domain memory_domain = rt_memory_domain::device;
    rt_submission_token retirement_submission{};
    std::uint64_t sequence = 0;
};

struct rt_acceleration_build_summary {
    std::size_t blas_reused_count = 0;
    std::size_t blas_rebuilt_count = 0;
    std::size_t blas_reused_triangle_chunk_count = 0;
    std::size_t blas_rebuilt_triangle_chunk_count = 0;
    std::size_t tlas_rebuild_count = 0;
};
struct rt_renderer_frame_result {
    bool viewer_resources_changed = false;
    bool output_changed = false;
    bool pick_resources_changed = false;
    bool scene_changed = false;
    bool acceleration_changed = false;
    bool pipeline_changed = false;
    std::size_t blas_reused_count = 0;
    std::size_t blas_rebuilt_count = 0;
    std::size_t blas_reused_triangle_chunk_count = 0;
    std::size_t blas_rebuilt_triangle_chunk_count = 0;
    std::size_t tlas_rebuild_count = 0;
    double pre_acceleration_prepare_cpu_ms = 0.0;
    double post_acceleration_prepare_cpu_ms = 0.0;
    rt_rhi_timing acceleration_timing{};
};

struct rt_renderer_frame_state {
    std::uint64_t scene_revision = 0;
    int output_width = 0;
    int output_height = 0;
    std::uint64_t serial = 0;
    bool scene_valid = false;
    bool output_valid = false;
    bool active = false;
};

struct rt_deferred_acceleration_submission {
    rt_command_encoder encoder{};
    rt_blas_cache_state next_blas_cache_state{};
    std::vector<rt_blas_handle> created_accelerations;
    std::vector<rt_blas_storage_pool_entry> acquired_accelerations;
    std::vector<rt_blas_storage_pool_entry> retired_accelerations;
    std::uint64_t scene_revision = 0;
    std::uint64_t connection_serial = 0;
};

struct rt_pending_render_submission {
    rt_submission_token submission{};
    std::uint64_t accumulation_generation = 0;
    std::uint32_t sample_index = 0;
    bool contributes_sample = false;
    bool confirms_scene = false;
    std::uint64_t scene_revision = 0;
    rt_acceleration_build_summary acceleration_summary{};
};

struct rt_renderer {
    rt_rhi_capabilities capabilities{};
    rt_rhi_device* rhi = nullptr;
    rt_shader_package_desc shader_package{};
    mutable std::mutex access_mutex;
    rt_renderer_frame_state frame_state{};
    rt_scene_buffer_resources scene_buffers{};
    rt_buffer_handle viewer_constant_buffer{};
    rt_texture_handle output_texture{};
    rt_texture_handle accumulation_texture{};
    rt_buffer_handle output_readback_buffer{};
    rt_texture_copy_footprint output_readback_footprint{};
    rt_submission_token output_readback_submission{};
    int output_readback_width = 0;
    int output_readback_height = 0;
    rt_buffer_handle pick_output_buffer{};
    std::array<rt_pick_slot, kRtCommandSlotCount> pick_slots{};
    rt_blas_cache_state blas_cache_state{};
    std::uint64_t current_connection_serial = 0;
    std::vector<rt_blas_storage_pool_entry> blas_storage_pool;
    std::vector<rt_scene_buffer_pool_entry> scene_buffer_pool;
    std::uint64_t resource_pool_sequence = 1;
    rt_submission_token last_submission{};
    std::uint64_t blas_storage_pool_hit_count = 0;
    std::uint64_t blas_storage_pool_miss_count = 0;
    std::uint64_t scene_buffer_pool_hit_count = 0;
    std::uint64_t scene_buffer_pool_miss_count = 0;
    std::uint64_t scene_buffer_allocation_count = 0;
    std::uint64_t scene_buffer_growth_count = 0;
    std::uint64_t resource_pool_eviction_count = 0;
    std::size_t acceleration_peak_capacity_bytes = 0;
    std::size_t scene_buffer_peak_capacity_bytes = 0;
    rt_tlas_handle tlas{};
    std::uint64_t submitted_acceleration_revision = 0;
    rt_acceleration_build_summary submitted_acceleration_summary{};
    std::uint64_t last_acceleration_revision = 0;
    rt_acceleration_build_summary last_acceleration_summary{};
    double last_acceleration_cpu_ms = 0.0;
    double last_point_blas_prebuild_info_ms = 0.0;
    std::uint32_t last_point_blas_prebuild_info_count = 0;
    double last_line_blas_prebuild_info_ms = 0.0;
    std::uint32_t last_line_blas_prebuild_info_count = 0;
    std::vector<rt_shader_module_handle> shader_modules;
    std::array<rt_shader_module_handle, kViewerRtShaderEntryCount> shader_entry_modules{};
    rt_pipeline_handle pipeline{};
    rt_accumulation_state accumulation_state{};
    std::vector<rt_pending_render_submission> pending_render_submissions;
    std::vector<rt_submission_token> pending_delivery_submissions;
    rt_rhi_error async_error{};
    bool async_failed = false;
    rt_present_result last_present_result{};
    bool output_readback_pending = false;
    bool blas_reuse_enabled = true;
    bool continuous_render = false;
};

bool initialize_rt_renderer(
    rt_renderer* renderer,
    const backend_config &config,
    rt_rhi_error* out_error);
bool shutdown_rt_renderer(rt_renderer* renderer, rt_rhi_error* out_error);
bool wait_for_rt_renderer_idle(
    rt_renderer* renderer,
    rt_rhi_timing* out_timing,
    rt_rhi_error* out_error);
bool begin_rt_commands(
    rt_renderer* renderer,
    rt_queue_class queue,
    rt_command_encoder* out_encoder,
    rt_rhi_error* out_error,
    rt_rhi_timing* out_timing = nullptr);
bool submit_rt_commands(
    rt_renderer* renderer,
    rt_command_encoder encoder,
    rt_submission_token* out_submission,
    rt_rhi_timing* out_timing,
    rt_rhi_error* out_error);
void discard_rt_commands(
    rt_renderer* renderer,
    rt_command_encoder encoder);
bool is_rt_submission_complete(
    rt_renderer* renderer,
    rt_submission_token submission,
    bool* out_complete,
    rt_rhi_error* out_error);
bool wait_for_rt_submission(
    rt_renderer* renderer,
    rt_submission_token submission,
    rt_rhi_timing* out_timing,
    rt_rhi_error* out_error);
bool record_rt_barriers(
    rt_renderer* renderer,
    rt_command_encoder encoder,
    const rt_resource_barrier* barriers,
    std::size_t barrier_count,
    rt_rhi_error* out_error);
bool record_rt_buffer_copy(
    rt_renderer* renderer,
    rt_command_encoder encoder,
    rt_buffer_handle source,
    rt_buffer_handle destination,
    const rt_buffer_copy_region &region,
    rt_rhi_error* out_error);
bool record_rt_texture_to_buffer_copy(
    rt_renderer* renderer,
    rt_command_encoder encoder,
    rt_texture_handle source,
    rt_buffer_handle destination,
    const rt_texture_buffer_copy_region &region,
    rt_rhi_error* out_error);
bool record_rt_texture_clear(
    rt_renderer* renderer,
    rt_command_encoder encoder,
    rt_texture_handle texture,
    const float color[4],
    rt_rhi_error* out_error);
bool record_rt_trace_rays(
    rt_renderer* renderer,
    rt_command_encoder encoder,
    const rt_trace_rays_desc &desc,
    rt_rhi_error* out_error);
bool write_rt_trace_constants(
    rt_renderer* renderer,
    rt_command_encoder encoder,
    const rt_viewer_constants &constants,
    rt_rhi_error* out_error);
bool read_rt_buffer(
    rt_renderer* renderer,
    rt_buffer_handle buffer,
    std::size_t offset,
    void* data,
    std::size_t size,
    rt_rhi_error* out_error);
bool transition_rt_texture(
    rt_renderer* renderer,
    rt_command_encoder encoder,
    rt_texture_handle texture,
    rt_resource_usage usage,
    rt_rhi_error* out_error);
bool transition_rt_buffer(
    rt_renderer* renderer,
    rt_command_encoder encoder,
    rt_buffer_handle buffer,
    rt_resource_usage usage,
    rt_rhi_error* out_error);
bool prepare_rt_renderer_pipeline(
    rt_renderer* renderer,
    const rt_pipeline_desc &pipeline_desc,
    rt_renderer_frame_result* in_out_result,
    rt_rhi_error* out_error);
void begin_rt_renderer_access(rt_renderer* renderer);
void end_rt_renderer_access(rt_renderer* renderer);
bool execute_rt_renderer_native_frame(
    rt_renderer* renderer,
    const rt_native_frame_request &request,
    rt_present_result* out_result,
    rt_rhi_error* out_error,
    rt_deferred_acceleration_submission* deferred_acceleration = nullptr);
bool dispatch_rt_renderer_pick(
    rt_renderer* renderer,
    const rt_pick_dispatch_request &request,
    pick_result* out_result,
    rt_pick_dispatch_request* out_completed_request,
    rt_rhi_error* out_error,
    bool* out_pending = nullptr);
void copy_rt_renderer_diagnostics(scene_build_info* out_info, const rt_renderer &renderer);
void copy_rt_rhi_diagnostics(
    scene_build_info* out_info,
    const rt_rhi_diagnostics &diagnostics);
void reset_rt_renderer_accumulation(rt_renderer* renderer);
bool begin_rt_renderer_accumulation(
    rt_renderer* renderer,
    const rt_accumulation_key &next_key,
    bool continuous_render);
bool track_rt_renderer_render_submission(
    rt_renderer* renderer,
    rt_submission_token submission,
    std::uint64_t accumulation_generation,
    std::uint32_t sample_index,
    bool contributes_sample,
    bool confirms_scene,
    std::uint64_t scene_revision,
    const rt_acceleration_build_summary &acceleration_summary,
    rt_rhi_error* out_error);
bool collect_rt_renderer_render_submissions(
    rt_renderer* renderer,
    rt_rhi_error* out_error);
bool track_rt_renderer_delivery_submission(
    rt_renderer* renderer,
    rt_submission_token submission,
    rt_rhi_error* out_error);
bool prepare_rt_renderer_frame(
    rt_renderer* renderer,
    const rt_renderer_frame_request &request,
    rt_renderer_frame_result* out_result,
    rt_rhi_error* out_error,
    rt_deferred_acceleration_submission* deferred_acceleration = nullptr);

} // namespace rtvdb::viewer_backend
