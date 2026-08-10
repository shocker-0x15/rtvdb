#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace rtvdb::viewer_backend {

constexpr std::size_t kRtMaxBlasGeometryCount = 4;
constexpr std::uint32_t kRtCommandSlotCount = 3;
constexpr std::uint32_t kRtTimestampQueryCountPerRegion = 2;
constexpr std::uint32_t kRtTimestampQueryRegionCount = 2;
constexpr std::uint32_t kRtTimestampQueryCountPerCommandSlot =
    kRtTimestampQueryCountPerRegion * kRtTimestampQueryRegionCount;

enum class rt_timestamp_query_region : std::uint32_t {
    acceleration = 0,
    dispatch = kRtTimestampQueryCountPerRegion,
};

enum class rt_rhi_backend_kind : std::uint8_t {
    d3d12_dxr,
    vulkan_rt,
    metal_rt,
};

struct rt_rhi_device_info {
    rt_rhi_backend_kind kind = rt_rhi_backend_kind::d3d12_dxr;
    const char* name = "";
    bool hardware_ray_tracing = false;
};

struct rt_rhi_device_desc {
    std::uint32_t initial_width = 0;
    std::uint32_t initial_height = 0;
    void* native_device = nullptr;
    void* native_graphics_queue = nullptr;
};

template <typename Tag>
struct rt_typed_handle {
    std::uint64_t value = 0;

    explicit operator bool() const {
        return value != 0;
    }

    friend bool operator==(rt_typed_handle a, rt_typed_handle b) {
        return a.value == b.value;
    }
};

using rt_buffer_handle = rt_typed_handle<struct rt_buffer_handle_tag>;
using rt_texture_handle = rt_typed_handle<struct rt_texture_handle_tag>;
using rt_blas_handle = rt_typed_handle<struct rt_blas_handle_tag>;
using rt_tlas_handle = rt_typed_handle<struct rt_tlas_handle_tag>;
using rt_shader_module_handle = rt_typed_handle<struct rt_shader_module_handle_tag>;
using rt_pipeline_handle = rt_typed_handle<struct rt_pipeline_handle_tag>;
using rt_timestamp_query_handle = rt_typed_handle<struct rt_timestamp_query_handle_tag>;

struct rt_submission_token {
    std::uint64_t serial = 0;

    explicit operator bool() const {
        return serial != 0;
    }

    friend bool operator==(rt_submission_token a, rt_submission_token b) {
        return a.serial == b.serial;
    }

    friend bool operator!=(rt_submission_token a, rt_submission_token b) {
        return !(a == b);
    }
};

enum class rt_queue_class : std::uint8_t {
    graphics
};

struct rt_command_encoder {
    std::uint64_t id = 0;
    std::uint32_t slot = 0;
    rt_queue_class queue = rt_queue_class::graphics;

    explicit operator bool() const {
        return id != 0;
    }
};

struct rt_rhi_diagnostics {
    double acceleration_cpu_ms = 0.0;
    double acceleration_host_prepare_ms = 0.0;
    double acceleration_instance_build_ms = 0.0;
    double acceleration_procedural_aabb_ms = 0.0;
    double acceleration_command_record_ms = 0.0;
    double acceleration_resource_allocate_ms = 0.0;
    double acceleration_build_call_record_ms = 0.0;
    double acceleration_prebuild_query_ms = 0.0;
    double chunk_blas_prebuild_query_ms = 0.0;
    std::uint32_t chunk_blas_prebuild_query_count = 0;
    double grouped_blas_prebuild_query_ms = 0.0;
    std::uint32_t grouped_blas_prebuild_query_count = 0;
    double point_blas_prebuild_query_ms = 0.0;
    std::uint32_t point_blas_prebuild_query_count = 0;
    double line_blas_prebuild_query_ms = 0.0;
    std::uint32_t line_blas_prebuild_query_count = 0;
    double tlas_prebuild_query_ms = 0.0;
    std::uint32_t tlas_prebuild_query_count = 0;
    double startup_prebuild_warmup_ms = 0.0;
    double tlas_instance_upload_ms = 0.0;
    double acceleration_submit_cpu_ms = 0.0;
    double acceleration_gpu_wait_ms = 0.0;
    double acceleration_gpu_ms = 0.0;
    double dispatch_cpu_ms = 0.0;
    double dispatch_submit_cpu_ms = 0.0;
    double dispatch_gpu_wait_ms = 0.0;
    double dispatch_gpu_ms = 0.0;
    double command_slot_reuse_wait_ms = 0.0;
    double readback_cpu_ms = 0.0;
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
};

enum class rt_logical_dispatch_entry : std::uint8_t {
    render,
    pick,
    count,
};

struct rt_trace_rays_desc {
    rt_pipeline_handle pipeline{};
    rt_logical_dispatch_entry entry = rt_logical_dispatch_entry::render;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t depth = 1;
    bool measure_gpu_time = false;
};

enum rt_buffer_usage_flag : std::uint32_t {
    rt_buffer_usage_shader_read = 1u << 0u,
    rt_buffer_usage_acceleration_build_input = 1u << 1u,
    rt_buffer_usage_device_address = 1u << 2u,
    rt_buffer_usage_shader_write = 1u << 3u,
    rt_buffer_usage_uniform = 1u << 4u,
    rt_buffer_usage_copy_source = 1u << 5u,
    rt_buffer_usage_copy_destination = 1u << 6u,
};

enum class rt_memory_domain : std::uint8_t {
    device,
    upload,
    readback,
};

struct rt_buffer_desc {
    std::size_t size = 0;
    std::uint32_t usage = 0;
    rt_memory_domain memory_domain = rt_memory_domain::upload;
};

enum class rt_texture_format : std::uint8_t {
    rgba8_unorm,
    bgra8_unorm,
    rgba16_float,
    rgba32_float,
};

constexpr std::size_t rt_texture_format_bytes_per_pixel(rt_texture_format format) {
    switch (format) {
    case rt_texture_format::rgba8_unorm:
    case rt_texture_format::bgra8_unorm:
        return 4;
    case rt_texture_format::rgba16_float:
        return 8;
    case rt_texture_format::rgba32_float:
        return 16;
    }
    return 0;
}

enum rt_texture_usage_flag : std::uint32_t {
    rt_texture_usage_shader_read = 1u << 0u,
    rt_texture_usage_shader_write = 1u << 1u,
    rt_texture_usage_copy_source = 1u << 2u,
    rt_texture_usage_copy_destination = 1u << 3u,
};

struct rt_texture_desc {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    rt_texture_format format = rt_texture_format::rgba8_unorm;
    std::uint32_t usage = 0;
};

struct rt_texture_copy_footprint {
    std::size_t row_pitch = 0;
    std::size_t total_size = 0;
};

struct rt_buffer_copy_region {
    std::size_t source_offset = 0;
    std::size_t destination_offset = 0;
    std::size_t size = 0;
};

struct rt_texture_buffer_copy_region {
    std::size_t buffer_offset = 0;
    std::size_t buffer_row_pitch = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

enum class rt_descriptor_type : std::uint8_t {
    acceleration_structure,
    storage_texture,
    structured_buffer,
    storage_buffer,
    uniform_buffer,
};

struct rt_binding_location {
    std::uint32_t group = 0;
    std::uint32_t binding = 0;

    friend bool operator==(rt_binding_location a, rt_binding_location b) {
        return a.group == b.group && a.binding == b.binding;
    }
};

struct rt_binding_layout_desc {
    rt_binding_location location{};
    rt_descriptor_type type = rt_descriptor_type::storage_buffer;
    std::uint32_t count = 1;
};

enum class rt_shader_stage : std::uint8_t {
    compute,
    ray_generation,
    miss,
    closest_hit,
    any_hit,
    intersection,
    callable,
};

enum class rt_shader_binary_format : std::uint8_t {
    dxil_library,
    spirv,
    metallib,
};

enum class rt_pipeline_model : std::uint8_t {
    native_ray_tracing,
    compute_intersector,
};

enum class rt_logical_shader_entry : std::uint8_t {
    render,
    pick,
    miss,
    triangle_closest_hit,
    point_closest_hit,
    point_intersection,
    line_closest_hit,
    line_intersection,
    count,
};

struct rt_shader_module_desc {
    rt_shader_binary_format format = rt_shader_binary_format::spirv;
    const void* data = nullptr;
    std::size_t size = 0;
};

struct rt_shader_package_desc {
    rt_pipeline_model pipeline_model = rt_pipeline_model::native_ray_tracing;
    const rt_shader_module_desc* modules = nullptr;
    std::size_t module_count = 0;
    const std::uint32_t* entry_module_indices = nullptr;
    const rt_logical_shader_entry* logical_entries = nullptr;
    std::size_t entry_count = 0;
};

struct rt_shader_entry_desc {
    rt_shader_module_handle module{};
    rt_shader_stage stage = rt_shader_stage::ray_generation;
    const char* entry_point = nullptr;
};

enum class rt_shader_group_type : std::uint8_t {
    general,
    triangles_hit_group,
    procedural_hit_group,
};

constexpr std::uint32_t kRtUnusedShaderIndex = ~std::uint32_t{0};

struct rt_shader_group_desc {
    rt_shader_group_type type = rt_shader_group_type::general;
    const char* export_name = nullptr;
    std::uint32_t general_shader = kRtUnusedShaderIndex;
    std::uint32_t closest_hit_shader = kRtUnusedShaderIndex;
    std::uint32_t any_hit_shader = kRtUnusedShaderIndex;
    std::uint32_t intersection_shader = kRtUnusedShaderIndex;
};

struct rt_pipeline_dispatch_entry_desc {
    rt_logical_dispatch_entry logical_entry = rt_logical_dispatch_entry::render;
    std::uint32_t shader_or_group_index = kRtUnusedShaderIndex;
};

struct rt_pipeline_desc {
    rt_pipeline_model model = rt_pipeline_model::native_ray_tracing;
    const rt_binding_layout_desc* bindings = nullptr;
    std::size_t binding_count = 0;
    const rt_shader_entry_desc* shaders = nullptr;
    std::size_t shader_count = 0;
    const rt_shader_group_desc* groups = nullptr;
    std::size_t group_count = 0;
    const rt_pipeline_dispatch_entry_desc* dispatch_entries = nullptr;
    std::size_t dispatch_entry_count = 0;
    std::uint32_t max_recursion_depth = 1;
    std::uint32_t max_payload_size = 0;
    std::uint32_t max_attribute_size = 0;
};

enum class rt_resource_usage : std::uint8_t {
    undefined,
    shader_write,
    shader_read,
    acceleration_build_input,
    acceleration_storage,
    copy_source,
    copy_destination,
    clear_destination,
    host_read,
};

enum class rt_resource_kind : std::uint8_t {
    buffer,
    texture,
};

struct rt_resource_barrier {
    rt_resource_kind kind = rt_resource_kind::buffer;
    rt_buffer_handle buffer{};
    rt_texture_handle texture{};
    rt_resource_usage before = rt_resource_usage::undefined;
    rt_resource_usage after = rt_resource_usage::undefined;
};

enum class rt_acceleration_geometry_type : std::uint32_t {
    triangles,
    aabbs,
};

enum class rt_vertex_format : std::uint32_t {
    float3,
};

enum class rt_index_format : std::uint32_t {
    uint32,
};

enum rt_acceleration_geometry_flag : std::uint32_t {
    rt_acceleration_geometry_opaque = 1u << 0u,
};

enum rt_acceleration_instance_flag : std::uint32_t {
    rt_acceleration_instance_triangle_cull_disable = 1u << 0u,
    rt_acceleration_instance_triangle_front_counterclockwise = 1u << 1u,
    rt_acceleration_instance_force_opaque = 1u << 2u,
    rt_acceleration_instance_force_non_opaque = 1u << 3u,
};

struct rt_triangle_geometry_desc {
    rt_buffer_handle vertex_buffer{};
    std::size_t vertex_offset = 0;
    std::size_t vertex_count = 0;
    std::size_t vertex_stride = 0;
    rt_vertex_format vertex_format = rt_vertex_format::float3;
    rt_buffer_handle index_buffer{};
    std::size_t index_offset = 0;
    std::size_t index_count = 0;
    rt_index_format index_format = rt_index_format::uint32;
};

struct rt_aabb_geometry_desc {
    rt_buffer_handle buffer{};
    std::size_t offset = 0;
    std::size_t count = 0;
    std::size_t stride = 0;
};

struct rt_acceleration_geometry_desc {
    rt_acceleration_geometry_type type = rt_acceleration_geometry_type::triangles;
    std::uint32_t flags = 0;
    rt_triangle_geometry_desc triangles{};
    rt_aabb_geometry_desc aabbs{};
};

enum rt_acceleration_build_flag : std::uint32_t {
    rt_acceleration_build_prefer_fast_trace = 1u << 0u,
    rt_acceleration_build_allow_update = 1u << 1u,
};

struct rt_blas_build_desc {
    rt_blas_handle destination{};
    const rt_acceleration_geometry_desc* geometries = nullptr;
    std::size_t geometry_count = 0;
    std::uint32_t flags = rt_acceleration_build_prefer_fast_trace;
    const std::size_t* allocation_primitive_counts = nullptr;
};

struct rt_tlas_instance_desc {
    rt_blas_handle acceleration{};
    std::array<float, 12> transform{
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f};
    std::uint32_t instance_id = 0;
    std::uint8_t mask = 0xff;
    std::uint32_t hit_group_contribution = 0;
    std::uint32_t flags = rt_acceleration_instance_triangle_cull_disable;
};

struct rt_tlas_build_desc {
    rt_tlas_handle destination{};
    const rt_tlas_instance_desc* instances = nullptr;
    std::size_t instance_count = 0;
    std::uint32_t flags = rt_acceleration_build_prefer_fast_trace;
};

struct rt_blas_geometry_counts {
    std::size_t actual = 0;
    std::size_t allocation = 0;
};

bool get_rt_blas_geometry_counts(
    const rt_blas_build_desc &desc,
    std::size_t geometry_index,
    rt_blas_geometry_counts* out_counts);
bool validate_rt_blas_build_desc(const rt_blas_build_desc &desc);
bool validate_rt_tlas_build_desc(const rt_tlas_build_desc &desc);
bool validate_rt_shader_package_desc(const rt_shader_package_desc &desc);
bool validate_rt_pipeline_desc(const rt_pipeline_desc &desc);
bool get_rt_pipeline_dispatch_entry_index(
    const rt_pipeline_desc &desc,
    rt_logical_dispatch_entry logical_entry,
    std::uint32_t* out_index);

} // namespace rtvdb::viewer_backend
