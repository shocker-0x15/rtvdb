#pragma once

#include "viewer_backend/backend.h"
#include "viewer_backend/rt_rhi.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace rtvdb::viewer_backend {

enum class rt_rhi_operation : std::uint32_t {
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

struct rt_binding_write {
    rt_binding_location location{};
    rt_descriptor_type type = rt_descriptor_type::storage_buffer;
    rt_buffer_handle resource{};
    std::size_t element_count = 0;
    std::size_t element_stride = 0;
    rt_texture_handle texture{};
    rt_tlas_handle acceleration{};
};

struct rt_binding_update_request {
    const rt_binding_write* writes = nullptr;
    std::size_t write_count = 0;
};

struct rt_rhi_capabilities {
    bool hardware_ray_tracing = false;
    bool timestamp_queries = false;
    bool native_d3d12_target = false;
    bool native_vulkan_target = false;
    bool native_texture_publish = false;
    bool bgra_capture = false;
    bool bgra_readback = false;
    rt_shader_binary_format shader_binary_format = rt_shader_binary_format::spirv;
    rt_texture_format output_format = rt_texture_format::rgba8_unorm;
    rt_texture_format accumulation_format = rt_texture_format::rgba16_float;
};

struct rt_rhi_error {
    rt_rhi_operation operation = rt_rhi_operation::initialize;
    std::int64_t native_code = 0;
    std::string detail;
};

struct rt_rhi_timing {
    double command_slot_wait_ms = 0.0;
    double command_record_ms = 0.0;
    double submit_cpu_ms = 0.0;
    double gpu_wait_ms = 0.0;
    double gpu_ms = 0.0;
};

struct rt_native_texture_publish_desc {
    void* target = nullptr;
    void** out_target = nullptr;
    rt_submission_token* out_submission = nullptr;
};

struct rt_blas_build_result {
    rt_blas_handle acceleration{};
    double prebuild_info_ms = 0.0;
    std::size_t allocation_size_bytes = 0;
    bool reused = false;
};

class rt_native_texture_extension {
public:
    virtual ~rt_native_texture_extension() = default;

    virtual bool publish_texture(
        rt_texture_handle texture,
        const rt_native_texture_publish_desc &desc,
        rt_rhi_timing* out_timing,
        rt_rhi_error* out_error) = 0;
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
    virtual rt_rhi_capabilities capabilities() const = 0;
    virtual rt_native_texture_extension* native_texture_extension() = 0;
    virtual rt_vulkan_interop_extension* vulkan_interop_extension() = 0;
    virtual bool initialize(
        const rt_rhi_device_desc &desc,
        rt_rhi_error* out_error) = 0;
    virtual bool shutdown(rt_rhi_error* out_error) = 0;
    virtual bool wait_idle(
        rt_rhi_timing* out_timing,
        rt_rhi_error* out_error) = 0;
    virtual bool begin_commands(
        rt_queue_class queue,
        rt_command_encoder* out_encoder,
        rt_rhi_error* out_error) = 0;
    virtual bool submit_commands(
        rt_command_encoder encoder,
        rt_submission_token* out_submission,
        rt_rhi_timing* out_timing,
        rt_rhi_error* out_error) = 0;
    virtual void discard_commands(rt_command_encoder encoder) = 0;
    virtual bool is_complete(
        rt_submission_token submission,
        bool* out_complete,
        rt_rhi_error* out_error) = 0;
    virtual bool wait(
        rt_submission_token submission,
        rt_rhi_timing* out_timing,
        rt_rhi_error* out_error) = 0;
    virtual bool barrier(
        rt_command_encoder encoder,
        const rt_resource_barrier* barriers,
        std::size_t barrier_count,
        rt_rhi_error* out_error) = 0;
    virtual bool copy_buffer(
        rt_command_encoder encoder,
        rt_buffer_handle source,
        rt_buffer_handle destination,
        const rt_buffer_copy_region &region,
        rt_rhi_error* out_error) = 0;
    virtual bool copy_texture_to_buffer(
        rt_command_encoder encoder,
        rt_texture_handle source,
        rt_buffer_handle destination,
        const rt_texture_buffer_copy_region &region,
        rt_rhi_error* out_error) = 0;
    virtual bool clear_texture(
        rt_command_encoder encoder,
        rt_texture_handle texture,
        const float color[4],
        rt_rhi_error* out_error) = 0;
    virtual bool trace_rays(
        rt_command_encoder encoder,
        const rt_trace_rays_desc &desc,
        rt_rhi_error* out_error) = 0;
    virtual bool create_buffer(
        const rt_buffer_desc &desc,
        rt_buffer_handle* out_buffer,
        rt_rhi_error* out_error) = 0;
    virtual bool upload_buffer(
        rt_buffer_handle buffer,
        std::size_t offset,
        const void* data,
        std::size_t size,
        rt_rhi_error* out_error) = 0;
    virtual bool read_buffer(
        rt_buffer_handle buffer,
        std::size_t offset,
        void* data,
        std::size_t size,
        rt_rhi_error* out_error) = 0;
    virtual void destroy_buffer(rt_buffer_handle buffer) = 0;
    virtual bool create_texture(
        const rt_texture_desc &desc,
        rt_texture_handle* out_texture,
        rt_rhi_error* out_error) = 0;
    virtual void destroy_texture(rt_texture_handle texture) = 0;
    virtual bool get_texture_copy_footprint(
        rt_texture_handle texture,
        rt_texture_copy_footprint* out_footprint,
        rt_rhi_error* out_error) = 0;
    virtual bool create_blas(
        rt_blas_handle* out_blas,
        rt_rhi_error* out_error) = 0;
    virtual void destroy_blas(rt_blas_handle blas) = 0;
    virtual bool create_tlas(
        rt_tlas_handle* out_tlas,
        rt_rhi_error* out_error) = 0;
    virtual void destroy_tlas(rt_tlas_handle tlas) = 0;
    virtual bool build_blas(
        rt_command_encoder encoder,
        const rt_blas_build_desc &desc,
        rt_blas_build_result* out_result,
        rt_rhi_error* out_error) = 0;
    virtual bool build_tlas(
        rt_command_encoder encoder,
        const rt_tlas_build_desc &desc,
        rt_rhi_error* out_error) = 0;
    virtual bool update_bindings(
        const rt_binding_update_request &request,
        rt_rhi_error* out_error) = 0;
    virtual bool create_shader_module(
        const rt_shader_module_desc &desc,
        rt_shader_module_handle* out_module,
        rt_rhi_error* out_error) = 0;
    virtual void destroy_shader_module(rt_shader_module_handle module) = 0;
    virtual bool create_pipeline(
        const rt_pipeline_desc &desc,
        rt_pipeline_handle* out_pipeline,
        rt_rhi_error* out_error) = 0;
    virtual void get_diagnostics(rt_rhi_diagnostics* out_diagnostics) const = 0;
};

struct rt_rhi_factory_result {
    std::unique_ptr<rt_rhi_device> device;
    rt_shader_package_desc shader_package{};
};

rt_rhi_factory_result create_d3d12_dxr_rhi();
rt_rhi_factory_result create_vulkan_rhi();
rt_rhi_factory_result create_metal_rhi();

} // namespace rtvdb::viewer_backend
