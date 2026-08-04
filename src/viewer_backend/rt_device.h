#pragma once

#include "viewer_backend/backend.h"
#include "viewer_backend/rt_backend_common.h"
#include "viewer_backend/rt_scene_builder.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace rtvdb::viewer_backend {

constexpr std::uint32_t kRtCommandSlotCount = 3;
constexpr std::size_t kRtViewerConstantSlotStride = 256;
constexpr std::size_t kRtViewerConstantBufferBytes =
    static_cast<std::size_t>(kRtCommandSlotCount) * kRtViewerConstantSlotStride;

enum class rt_device_kind : std::uint32_t {
    d3d12_dxr,
    vulkan_rt,
};

enum class rt_device_operation : std::uint32_t {
    initialize,
    shutdown,
    wait_idle,
    begin_commands,
    submit_commands,
    query_submission,
    wait_submission,
    begin_frame,
    create_resource,
    upload_scene_buffers,
    build_blas,
    build_tlas,
    update_bindings,
    create_shader_module,
    transition_resource,
    copy_resource,
    clear_texture,
    trace_rays,
    prepare_pipeline,
    native_texture,
    dispatch_pick,
    present,
    capture,
    readback,
    end_frame,
};

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
    std::uint64_t point_geometry_fingerprint = 0;
    std::uint64_t line_geometry_fingerprint = 0;
    std::uint64_t revision = 0;
    double procedural_aabb_upload_ms = 0.0;
    std::vector<rt_scene_buffer_upload> uploads;
};

struct rt_binding_write {
    rt_binding_location location{};
    rt_descriptor_type type = rt_descriptor_type::storage_buffer;
    rt_buffer_handle resource{};
    std::size_t element_count = 0;
    std::size_t element_stride = 0;
    rt_texture_handle texture{};
};

struct rt_binding_update_request {
    const rt_binding_write* writes = nullptr;
    std::size_t write_count = 0;
};

struct rt_device_capabilities {
    bool hardware_ray_tracing = false;
    bool timestamp_queries = false;
    bool native_d3d12_target = false;
    bool native_vulkan_target = false;
    bool bgra_capture = false;
    bool bgra_readback = false;
    rt_shader_binary_format shader_binary_format = rt_shader_binary_format::spirv;
    rt_texture_format output_format = rt_texture_format::rgba8_unorm;
    rt_texture_format accumulation_format = rt_texture_format::rgba16_float;
};

enum class rt_present_operation : std::uint32_t {
    native_d3d12,
    native_vulkan,
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

struct rt_device_error {
    rt_device_operation operation = rt_device_operation::initialize;
    std::int64_t native_code = 0;
    std::string detail;
};

struct rt_device_timing {
    double command_slot_wait_ms = 0.0;
    double command_record_ms = 0.0;
    double submit_cpu_ms = 0.0;
    double gpu_wait_ms = 0.0;
    double gpu_ms = 0.0;
};

struct rt_present_result {
    bool rendered = false;
    bool captured = false;
    bool reused_output = false;
    double scene_snapshot_cpu_ms = 0.0;
    double frame_pre_acceleration_prepare_cpu_ms = 0.0;
    double frame_post_acceleration_prepare_cpu_ms = 0.0;
    double rt_output_prepare_cpu_ms = 0.0;
    double acceleration_finalize_cpu_ms = 0.0;
    double accumulation_finalize_cpu_ms = 0.0;
    rt_device_timing acceleration_timing{};
    rt_device_timing output_timing{};
    rt_device_timing timing{};
    rt_device_timing native_publish_timing{};
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
    float clear_color[4]{0.0f, 0.0f, 0.0f, 1.0f};
    bool reuse_output = false;
    bool update_build_info = false;
    void* native_target = nullptr;
    void** out_native_target = nullptr;
    std::vector<std::uint8_t>* out_pixels = nullptr;
};

struct rt_native_texture_publish_desc {
    void* target = nullptr;
    void** out_target = nullptr;
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

struct rt_device_frame_request {
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

struct rt_blas_build_result {
    rt_blas_handle acceleration{};
    double prebuild_info_ms = 0.0;
    bool reused = false;
};

struct rt_acceleration_build_summary {
    std::size_t blas_reused_count = 0;
    std::size_t blas_rebuilt_count = 0;
    std::size_t blas_reused_triangle_chunk_count = 0;
    std::size_t blas_rebuilt_triangle_chunk_count = 0;
    std::size_t tlas_rebuild_count = 0;
};
struct rt_device_frame_result {
    bool viewer_resources_changed = false;
    bool output_changed = false;
    bool pick_resources_changed = false;
    bool scene_changed = false;
    bool acceleration_changed = false;
    bool pipeline_changed = false;
    bool shader_table_changed = false;
    std::size_t blas_reused_count = 0;
    std::size_t blas_rebuilt_count = 0;
    std::size_t blas_reused_triangle_chunk_count = 0;
    std::size_t blas_rebuilt_triangle_chunk_count = 0;
    std::size_t tlas_rebuild_count = 0;
    double pre_acceleration_prepare_cpu_ms = 0.0;
    double post_acceleration_prepare_cpu_ms = 0.0;
    rt_device_timing acceleration_timing{};
};

struct rt_device;

class rt_native_texture_extension {
public:
    virtual ~rt_native_texture_extension() = default;

    virtual bool publish_texture(
        rt_texture_handle texture,
        const rt_native_texture_publish_desc &desc,
        rt_device_timing* out_timing,
        rt_device_error* out_error) = 0;
};

class rt_vulkan_interop_extension {
public:
    virtual ~rt_vulkan_interop_extension() = default;

    virtual bool get_interop(vulkan_renderer_interop* out_interop) = 0;
};

class rt_rhi_device {
public:
    virtual ~rt_rhi_device() = default;

    virtual rt_rhi_device_info info() const = 0;
    virtual rt_device* device() = 0;
    virtual rt_native_texture_extension* native_texture_extension() = 0;
    virtual rt_vulkan_interop_extension* vulkan_interop_extension() = 0;
    virtual bool initialize(
        const rt_rhi_device_desc &desc,
        rt_device_error* out_error) = 0;
    virtual bool shutdown(rt_device_error* out_error) = 0;
    virtual bool wait_idle(
        rt_device_timing* out_timing,
        rt_device_error* out_error) = 0;
    virtual bool begin_commands(
        rt_queue_class queue,
        rt_command_encoder* out_encoder,
        rt_device_error* out_error) = 0;
    virtual bool submit_commands(
        rt_command_encoder encoder,
        rt_submission_token* out_submission,
        rt_device_timing* out_timing,
        rt_device_error* out_error) = 0;
    virtual void discard_commands(
        rt_command_encoder encoder) = 0;
    virtual bool is_complete(
        rt_submission_token submission,
        bool* out_complete,
        rt_device_error* out_error) = 0;
    virtual bool wait(
        rt_submission_token submission,
        rt_device_timing* out_timing,
        rt_device_error* out_error) = 0;
    virtual bool barrier(
        rt_command_encoder encoder,
        const rt_resource_barrier* barriers,
        std::size_t barrier_count,
        rt_device_error* out_error) = 0;
    virtual bool copy_buffer(
        rt_command_encoder encoder,
        rt_buffer_handle source,
        rt_buffer_handle destination,
        const rt_buffer_copy_region &region,
        rt_device_error* out_error) = 0;
    virtual bool copy_texture_to_buffer(
        rt_command_encoder encoder,
        rt_texture_handle source,
        rt_buffer_handle destination,
        const rt_texture_buffer_copy_region &region,
        rt_device_error* out_error) = 0;
    virtual bool clear_texture(
        rt_command_encoder encoder,
        rt_texture_handle texture,
        const float color[4],
        rt_device_error* out_error) = 0;
    virtual bool trace_rays(
        rt_command_encoder encoder,
        const rt_trace_rays_desc &desc,
        rt_device_error* out_error) = 0;
    virtual bool create_buffer(
        const rt_buffer_desc &desc,
        rt_buffer_handle* out_buffer,
        rt_device_error* out_error) = 0;
    virtual bool upload_buffer(
        rt_buffer_handle buffer,
        std::size_t offset,
        const void* data,
        std::size_t size,
        rt_device_error* out_error) = 0;
    virtual bool read_buffer(
        rt_buffer_handle buffer,
        std::size_t offset,
        void* data,
        std::size_t size,
        rt_device_error* out_error) = 0;
    virtual void destroy_buffer(rt_buffer_handle buffer) = 0;
    virtual bool create_texture(
        const rt_texture_desc &desc,
        rt_texture_handle* out_texture,
        rt_device_error* out_error) = 0;
    virtual void destroy_texture(rt_texture_handle texture) = 0;
    virtual bool get_texture_copy_footprint(
        rt_texture_handle texture,
        rt_texture_copy_footprint* out_footprint,
        rt_device_error* out_error) = 0;
    virtual bool create_blas(
        rt_blas_handle* out_blas,
        rt_device_error* out_error) = 0;
    virtual void destroy_blas(rt_blas_handle blas) = 0;
    virtual bool create_tlas(
        rt_tlas_handle* out_tlas,
        rt_device_error* out_error) = 0;
    virtual void destroy_tlas(rt_tlas_handle tlas) = 0;

    virtual bool build_blas(
        rt_command_encoder encoder,
        const rt_blas_build_desc &desc,
        rt_blas_build_result* out_result,
        rt_device_error* out_error) = 0;
    virtual bool build_tlas(
        rt_command_encoder encoder,
        const rt_tlas_build_desc &desc,
        rt_device_error* out_error) = 0;
    virtual bool update_bindings(
        const rt_binding_update_request &request,
        rt_device_error* out_error) = 0;
    virtual bool create_shader_module(
        const rt_shader_module_desc &desc,
        rt_shader_module_handle* out_module,
        rt_device_error* out_error) = 0;
    virtual void destroy_shader_module(
        rt_shader_module_handle module) = 0;
    virtual bool create_pipeline(
        const rt_pipeline_desc &desc,
        rt_pipeline_handle* out_pipeline,
        rt_device_error* out_error) = 0;
    virtual bool create_shader_table(
        const rt_shader_table_desc &desc,
        rt_shader_table_handle* out_shader_table,
        rt_device_error* out_error) = 0;
    virtual void get_diagnostics(rt_rhi_diagnostics* out_diagnostics) const = 0;
};
struct rt_device_frame_state {
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
    std::vector<rt_blas_handle> retired_accelerations;
    std::uint64_t scene_revision = 0;
};

struct rt_device {
    rt_device_kind kind = rt_device_kind::d3d12_dxr;
    rt_device_capabilities capabilities{};
    rt_rhi_device* api = nullptr;
    mutable std::mutex access_mutex;

    void* native_state = nullptr;
    rt_device_frame_state frame_state{};
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
    rt_tlas_handle tlas{};
    std::uint64_t last_acceleration_revision = 0;
    rt_acceleration_build_summary last_acceleration_summary{};
    double last_acceleration_cpu_ms = 0.0;
    double last_point_blas_prebuild_info_ms = 0.0;
    std::uint32_t last_point_blas_prebuild_info_count = 0;
    double last_line_blas_prebuild_info_ms = 0.0;
    std::uint32_t last_line_blas_prebuild_info_count = 0;
    std::vector<rt_shader_module_handle> shader_modules;
    rt_pipeline_handle pipeline{};
    rt_shader_table_handle shader_table{};
    rt_accumulation_state accumulation_state{};
    rt_present_result last_present_result{};
    bool output_readback_pending = false;
    bool blas_reuse_enabled = true;
    bool continuous_render = false;
};

bool initialize_rt_device(
    rt_device* device,
    const backend_config &config,
    rt_device_error* out_error);
bool shutdown_rt_device(rt_device* device, rt_device_error* out_error);
bool wait_for_rt_device_idle(
    rt_device* device,
    rt_device_timing* out_timing,
    rt_device_error* out_error);
bool begin_rt_commands(
    rt_device* device,
    rt_queue_class queue,
    rt_command_encoder* out_encoder,
    rt_device_error* out_error,
    rt_device_timing* out_timing = nullptr);
bool submit_rt_commands(
    rt_device* device,
    rt_command_encoder encoder,
    rt_submission_token* out_submission,
    rt_device_timing* out_timing,
    rt_device_error* out_error);
void discard_rt_commands(
    rt_device* device,
    rt_command_encoder encoder);
bool is_rt_submission_complete(
    rt_device* device,
    rt_submission_token submission,
    bool* out_complete,
    rt_device_error* out_error);
bool wait_for_rt_submission(
    rt_device* device,
    rt_submission_token submission,
    rt_device_timing* out_timing,
    rt_device_error* out_error);
bool record_rt_barriers(
    rt_device* device,
    rt_command_encoder encoder,
    const rt_resource_barrier* barriers,
    std::size_t barrier_count,
    rt_device_error* out_error);
bool record_rt_buffer_copy(
    rt_device* device,
    rt_command_encoder encoder,
    rt_buffer_handle source,
    rt_buffer_handle destination,
    const rt_buffer_copy_region &region,
    rt_device_error* out_error);
bool record_rt_texture_to_buffer_copy(
    rt_device* device,
    rt_command_encoder encoder,
    rt_texture_handle source,
    rt_buffer_handle destination,
    const rt_texture_buffer_copy_region &region,
    rt_device_error* out_error);
bool record_rt_texture_clear(
    rt_device* device,
    rt_command_encoder encoder,
    rt_texture_handle texture,
    const float color[4],
    rt_device_error* out_error);
bool record_rt_trace_rays(
    rt_device* device,
    rt_command_encoder encoder,
    const rt_trace_rays_desc &desc,
    rt_device_error* out_error);
bool write_rt_trace_constants(
    rt_device* device,
    rt_command_encoder encoder,
    const rt_viewer_constants &constants,
    rt_device_error* out_error);
bool read_rt_buffer(
    rt_device* device,
    rt_buffer_handle buffer,
    std::size_t offset,
    void* data,
    std::size_t size,
    rt_device_error* out_error);
bool transition_rt_texture(
    rt_device* device,
    rt_command_encoder encoder,
    rt_texture_handle texture,
    rt_resource_usage usage,
    rt_device_error* out_error);
bool transition_rt_buffer(
    rt_device* device,
    rt_command_encoder encoder,
    rt_buffer_handle buffer,
    rt_resource_usage usage,
    rt_device_error* out_error);
bool prepare_rt_device_pipeline(
    rt_device* device,
    const rt_pipeline_desc &pipeline_desc,
    const rt_shader_table_desc &shader_table_desc,
    rt_device_frame_result* in_out_result,
    rt_device_error* out_error);
void begin_rt_device_access(rt_device* device);
void end_rt_device_access(rt_device* device);
bool execute_rt_device_native_frame(
    rt_device* device,
    const rt_native_frame_request &request,
    rt_present_result* out_result,
    rt_device_error* out_error,
    rt_deferred_acceleration_submission* deferred_acceleration = nullptr);
bool dispatch_rt_device_pick(
    rt_device* device,
    const rt_pick_dispatch_request &request,
    pick_result* out_result,
    rt_pick_dispatch_request* out_completed_request,
    rt_device_error* out_error,
    bool* out_pending = nullptr);
void copy_rt_device_diagnostics(scene_build_info* out_info, const rt_device &device);
void copy_rt_rhi_diagnostics(
    scene_build_info* out_info,
    const rt_rhi_diagnostics &diagnostics);
void reset_rt_device_accumulation(rt_device* device);
bool begin_rt_device_accumulation(
    rt_device* device,
    const rt_accumulation_key &next_key,
    bool continuous_render);
void complete_rt_device_accumulation(rt_device* device, bool continuous_render);
bool prepare_rt_device_frame(
    rt_device* device,
    const rt_device_frame_request &request,
    rt_device_frame_result* out_result,
    rt_device_error* out_error,
    rt_deferred_acceleration_submission* deferred_acceleration = nullptr);

} // namespace rtvdb::viewer_backend
