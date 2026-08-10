#import <Metal/Metal.h>
#import <CoreVideo/CoreVideo.h>
#import <CoreVideo/CVPixelBufferIOSurface.h>

#include "viewer_backend/metal/metal_rt_shaders.h"
#include "viewer_backend/rt_object_registry.h"
#include "viewer_backend/rt_rhi_device.h"

#include <dispatch/dispatch.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

namespace rtvdb::viewer_backend {
namespace {

constexpr std::size_t kMetalTextureRowPitchAlignment = 256;
constexpr std::uint32_t kMetalBufferUsageMask =
    rt_buffer_usage_shader_read |
    rt_buffer_usage_acceleration_build_input |
    rt_buffer_usage_device_address |
    rt_buffer_usage_shader_write |
    rt_buffer_usage_uniform |
    rt_buffer_usage_copy_source |
    rt_buffer_usage_copy_destination;
constexpr std::uint32_t kMetalTextureUsageMask =
    rt_texture_usage_shader_read |
    rt_texture_usage_shader_write |
    rt_texture_usage_copy_source |
    rt_texture_usage_copy_destination;
constexpr std::size_t kMetalDispatchEntryCount =
    static_cast<std::size_t>(rt_logical_dispatch_entry::count);
constexpr std::uint32_t kMetalTimestampSampleCount = 1024;

bool consume_metal_test_completion_failure() {
    static bool consumed = false;
    const char* const value = std::getenv("RTVDB_TEST_FAIL_METAL_SUBMISSION_COMPLETION_ONCE");
    if (consumed || value == nullptr ||
        (value[0] != '1' && value[0] != 't' && value[0] != 'T' &&
         value[0] != 'y' && value[0] != 'Y')) {
        return false;
    }
    consumed = true;
    return true;
}

enum class metal_timestamp_mode : std::uint8_t {
    unsupported,
    dispatch_boundary,
    stage_boundary,
};

struct metal_buffer {
    id<MTLBuffer> resource = nil;
    rt_buffer_desc desc{};
};

struct metal_texture {
    id<MTLTexture> resource = nil;
    rt_texture_desc desc{};
};

struct metal_blas {
    id<MTLAccelerationStructure> resource = nil;
    id<MTLBuffer> scratch = nil;
    std::size_t allocation_size = 0;
    std::size_t scratch_size = 0;
};

struct metal_tlas {
    id<MTLAccelerationStructure> resource = nil;
    id<MTLBuffer> scratch = nil;
    id<MTLBuffer> instance_buffer = nil;
    id<MTLBuffer> instance_kind_buffer = nil;
    std::vector<id<MTLAccelerationStructure>> referenced_accelerations;
    std::size_t allocation_size = 0;
    std::size_t scratch_size = 0;
    std::size_t instance_buffer_size = 0;
};

struct metal_binding {
    rt_binding_location location{};
    rt_descriptor_type type = rt_descriptor_type::storage_buffer;
    id<MTLBuffer> buffer = nil;
    id<MTLTexture> texture = nil;
    id<MTLAccelerationStructure> acceleration = nil;
    id<MTLBuffer> instance_kind_buffer = nil;
    std::vector<id<MTLAccelerationStructure>> referenced_accelerations;
    std::size_t element_count = 0;
    std::size_t element_stride = 0;
};

struct metal_shader_module {
    id<MTLLibrary> library = nil;
};

struct metal_pipeline {
    std::array<id<MTLComputePipelineState>,
        static_cast<std::size_t>(rt_logical_dispatch_entry::count)> dispatch_states{};
};

void release_metal_pipeline(metal_pipeline &pipeline) {
    for (id<MTLComputePipelineState> state : pipeline.dispatch_states) {
        if (state != nil) {
            [state release];
        }
    }
    pipeline = {};
}

void release_acceleration_list(std::vector<id<MTLAccelerationStructure>> &accelerations) {
    for (id<MTLAccelerationStructure> acceleration : accelerations) {
        [acceleration release];
    }
    accelerations.clear();
}

bool append_unique_acceleration(
    std::vector<id<MTLAccelerationStructure>> &accelerations,
    id<MTLAccelerationStructure> acceleration)
{
    if (std::find(accelerations.begin(), accelerations.end(), acceleration) != accelerations.end()) {
        return true;
    }
    try {
        accelerations.push_back(acceleration);
    } catch (const std::bad_alloc &) {
        return false;
    }
    return true;
}

bool retain_acceleration_list(
    const std::vector<id<MTLAccelerationStructure>> &source,
    std::vector<id<MTLAccelerationStructure>>* destination)
{
    if (destination == nullptr) {
        return false;
    }
    std::vector<id<MTLAccelerationStructure>> retained;
    try {
        retained = source;
    } catch (const std::bad_alloc &) {
        return false;
    }
    for (id<MTLAccelerationStructure> acceleration : retained) {
        [acceleration retain];
    }
    *destination = std::move(retained);
    return true;
}

void release_metal_binding(metal_binding &binding) {
    if (binding.buffer != nil) {
        [binding.buffer release];
    }
    if (binding.texture != nil) {
        [binding.texture release];
    }
    if (binding.acceleration != nil) {
        [binding.acceleration release];
    }
    if (binding.instance_kind_buffer != nil) {
        [binding.instance_kind_buffer release];
    }
    release_acceleration_list(binding.referenced_accelerations);
    binding = {};
}

void release_metal_bindings(std::vector<metal_binding> &bindings) {
    for (metal_binding &binding : bindings) {
        release_metal_binding(binding);
    }
    bindings.clear();
}

struct metal_command_slot {
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLBlitCommandEncoder> blit_encoder = nil;
    id<MTLAccelerationStructureCommandEncoder> acceleration_encoder = nil;
    id<MTLComputeCommandEncoder> compute_encoder = nil;
    std::vector<id<MTLResource>> retained_resources;
    std::vector<id<MTLAccelerationStructure>> retained_acceleration_structures;
    std::vector<CVMetalTextureRef> retained_cv_textures;
    std::array<id<MTLCounterSampleBuffer>, kMetalDispatchEntryCount> dispatch_timestamp_buffers{};
    std::array<std::uint32_t, kMetalDispatchEntryCount> dispatch_timestamp_next_sample_indices{};
    std::array<std::uint32_t, kMetalDispatchEntryCount> dispatch_timestamp_sample_indices{};
    std::chrono::steady_clock::time_point record_begin{};
    std::uint64_t encoder_id = 0;
    std::uint64_t submission_serial = 0;
    std::uint32_t dispatch_timestamp_entries = 0;
    rt_logical_dispatch_entry last_dispatch_timestamp_entry = rt_logical_dispatch_entry::render;
    bool active = false;
    bool pending = false;
};

bool metal_device_supports_raytracing(id<MTLDevice> device) {
    if (device == nil || ![device respondsToSelector:@selector(supportsRaytracing)]) {
        return false;
    }
    return device.supportsRaytracing;
}

bool metal_device_supports_hardware_raytracing(id<MTLDevice> device) {
    if (device == nil || ![device respondsToSelector:@selector(supportsFamily:)]) {
        return false;
    }
    return [device supportsFamily:MTLGPUFamilyApple9] ||
        [device supportsFamily:MTLGPUFamilyApple10];
}

MTLPixelFormat metal_texture_format(rt_texture_format format) {
    switch (format) {
    case rt_texture_format::rgba8_unorm:
        return MTLPixelFormatRGBA8Unorm;
    case rt_texture_format::bgra8_unorm:
        return MTLPixelFormatBGRA8Unorm;
    case rt_texture_format::rgba16_float:
        return MTLPixelFormatRGBA16Float;
    case rt_texture_format::rgba32_float:
        return MTLPixelFormatRGBA32Float;
    }
    return MTLPixelFormatInvalid;
}

MTLTextureUsage metal_texture_usage(std::uint32_t usage) {
    MTLTextureUsage result = MTLTextureUsageUnknown;
    if ((usage & rt_texture_usage_shader_read) != 0u) {
        result |= MTLTextureUsageShaderRead;
    }
    if ((usage & rt_texture_usage_shader_write) != 0u) {
        result |= MTLTextureUsageShaderRead;
        result |= MTLTextureUsageShaderWrite;
    }
    if ((usage & (rt_texture_usage_shader_write | rt_texture_usage_copy_destination)) != 0u) {
        result |= MTLTextureUsageRenderTarget;
    }
    return result;
}

MTLResourceOptions metal_buffer_options(rt_memory_domain domain) {
    return domain == rt_memory_domain::device
        ? MTLResourceStorageModePrivate
        : MTLResourceStorageModeShared;
}

MTLAccelerationStructureUsage metal_acceleration_usage(std::uint32_t flags) {
    MTLAccelerationStructureUsage usage = MTLAccelerationStructureUsageNone;
    if ((flags & rt_acceleration_build_allow_update) != 0u) {
        usage |= MTLAccelerationStructureUsageRefit;
    }
    if ((flags & rt_acceleration_build_prefer_fast_trace) == 0u) {
        usage |= MTLAccelerationStructureUsagePreferFastBuild;
    } else if (@available(macOS 26.0, *)) {
        usage |= MTLAccelerationStructureUsagePreferFastIntersection;
    }
    return usage;
}

MTLAccelerationStructureInstanceOptions metal_instance_options(std::uint32_t flags) {
    MTLAccelerationStructureInstanceOptions options = MTLAccelerationStructureInstanceOptionNone;
    if ((flags & rt_acceleration_instance_triangle_cull_disable) != 0u) {
        options |= MTLAccelerationStructureInstanceOptionDisableTriangleCulling;
    }
    if ((flags & rt_acceleration_instance_triangle_front_counterclockwise) != 0u) {
        options |= MTLAccelerationStructureInstanceOptionTriangleFrontFacingWindingCounterClockwise;
    }
    if ((flags & rt_acceleration_instance_force_opaque) != 0u) {
        options |= MTLAccelerationStructureInstanceOptionOpaque;
    }
    if ((flags & rt_acceleration_instance_force_non_opaque) != 0u) {
        options |= MTLAccelerationStructureInstanceOptionNonOpaque;
    }
    return options;
}

MTLPackedFloat4x3 metal_instance_transform(const std::array<float, 12> &transform) {
    return MTLPackedFloat4x3(
        MTLPackedFloat3(transform[0], transform[4], transform[8]),
        MTLPackedFloat3(transform[1], transform[5], transform[9]),
        MTLPackedFloat3(transform[2], transform[6], transform[10]),
        MTLPackedFloat3(transform[3], transform[7], transform[11]));
}

bool range_is_valid(std::size_t capacity, std::size_t offset, std::size_t size) {
    return size > 0 && offset <= capacity && size <= capacity - offset;
}

bool align_up(std::size_t value, std::size_t alignment, std::size_t* out_value) {
    if (out_value == nullptr || alignment == 0 ||
        value > (std::numeric_limits<std::size_t>::max)() - (alignment - 1u)) {
        return false;
    }
    *out_value = ((value + alignment - 1u) / alignment) * alignment;
    return true;
}

bool size_product(std::size_t left, std::size_t right, std::size_t* out_value) {
    if (out_value == nullptr || left == 0 || right == 0 ||
        left > (std::numeric_limits<std::size_t>::max)() / right) {
        return false;
    }
    *out_value = left * right;
    return true;
}

double elapsed_ms(
    std::chrono::steady_clock::time_point begin,
    std::chrono::steady_clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

void set_metal_native_error(
    rt_rhi_error* out_error,
    rt_rhi_operation operation,
    const char* context,
    NSError* native_error)
{
    if (out_error == nullptr) {
        return;
    }
    *out_error = {
        operation,
        native_error != nil ? static_cast<std::int64_t>(native_error.code) : 0,
        context != nullptr ? context : "Metal operation failed"};
    const char* const native_detail = native_error != nil
        ? native_error.localizedDescription.UTF8String
        : nullptr;
    if (native_detail != nullptr && native_detail[0] != '\0') {
        out_error->detail += ": ";
        out_error->detail += native_detail;
    }
}

class metal_rhi_device final : public rt_rhi_device, public rt_native_texture_extension {
public:
    ~metal_rhi_device() override;

    rt_rhi_device_info info() const override;
    rt_rhi_capabilities capabilities() const override;
    rt_native_texture_extension* native_texture_extension() override;
    rt_vulkan_interop_extension* vulkan_interop_extension() override;
    bool publish_texture(
        rt_texture_handle texture,
        const rt_native_texture_publish_desc &desc,
        rt_rhi_timing* out_timing,
        rt_rhi_error* out_error) override;
    bool initialize(
        const rt_rhi_device_desc &desc,
        rt_rhi_error* out_error) override;
    bool shutdown(rt_rhi_error* out_error) override;
    bool wait_idle(
        rt_rhi_timing* out_timing,
        rt_rhi_error* out_error) override;
    bool begin_commands(
        rt_queue_class queue,
        rt_command_encoder* out_encoder,
        rt_rhi_error* out_error) override;
    bool submit_commands(
        rt_command_encoder encoder,
        rt_submission_token* out_submission,
        rt_rhi_timing* out_timing,
        rt_rhi_error* out_error) override;
    void discard_commands(rt_command_encoder encoder) override;
    bool is_complete(
        rt_submission_token submission,
        bool* out_complete,
        rt_rhi_error* out_error) override;
    bool wait(
        rt_submission_token submission,
        rt_rhi_timing* out_timing,
        rt_rhi_error* out_error) override;
    bool barrier(
        rt_command_encoder encoder,
        const rt_resource_barrier* barriers,
        std::size_t barrier_count,
        rt_rhi_error* out_error) override;
    bool copy_buffer(
        rt_command_encoder encoder,
        rt_buffer_handle source,
        rt_buffer_handle destination,
        const rt_buffer_copy_region &region,
        rt_rhi_error* out_error) override;
    bool copy_texture_to_buffer(
        rt_command_encoder encoder,
        rt_texture_handle source,
        rt_buffer_handle destination,
        const rt_texture_buffer_copy_region &region,
        rt_rhi_error* out_error) override;
    bool clear_texture(
        rt_command_encoder encoder,
        rt_texture_handle texture,
        const float color[4],
        rt_rhi_error* out_error) override;
    bool trace_rays(
        rt_command_encoder encoder,
        const rt_trace_rays_desc &desc,
        rt_rhi_error* out_error) override;
    bool create_buffer(
        const rt_buffer_desc &desc,
        rt_buffer_handle* out_buffer,
        rt_rhi_error* out_error) override;
    bool upload_buffer(
        rt_buffer_handle buffer,
        std::size_t offset,
        const void* data,
        std::size_t size,
        rt_rhi_error* out_error) override;
    bool read_buffer(
        rt_buffer_handle buffer,
        std::size_t offset,
        void* data,
        std::size_t size,
        rt_rhi_error* out_error) override;
    void destroy_buffer(rt_buffer_handle buffer) override;
    bool create_texture(
        const rt_texture_desc &desc,
        rt_texture_handle* out_texture,
        rt_rhi_error* out_error) override;
    void destroy_texture(rt_texture_handle texture) override;
    bool get_texture_copy_footprint(
        rt_texture_handle texture,
        rt_texture_copy_footprint* out_footprint,
        rt_rhi_error* out_error) override;
    bool create_blas(
        rt_blas_handle* out_blas,
        rt_rhi_error* out_error) override;
    void destroy_blas(rt_blas_handle blas) override;
    bool create_tlas(
        rt_tlas_handle* out_tlas,
        rt_rhi_error* out_error) override;
    void destroy_tlas(rt_tlas_handle tlas) override;
    bool build_blas(
        rt_command_encoder encoder,
        const rt_blas_build_desc &desc,
        rt_blas_build_result* out_result,
        rt_rhi_error* out_error) override;
    bool build_tlas(
        rt_command_encoder encoder,
        const rt_tlas_build_desc &desc,
        rt_rhi_error* out_error) override;
    bool update_bindings(
        const rt_binding_update_request &request,
        rt_rhi_error* out_error) override;
    bool create_shader_module(
        const rt_shader_module_desc &desc,
        rt_shader_module_handle* out_module,
        rt_rhi_error* out_error) override;
    void destroy_shader_module(rt_shader_module_handle module) override;
    bool create_pipeline(
        const rt_pipeline_desc &desc,
        rt_pipeline_handle* out_pipeline,
        rt_rhi_error* out_error) override;
    void get_diagnostics(rt_rhi_diagnostics* out_diagnostics) const override;

private:
    metal_command_slot* command_slot(
        rt_command_encoder encoder,
        rt_rhi_operation operation,
        rt_rhi_error* out_error);
    bool acquire_command_slot(
        std::uint32_t* out_slot,
        double* out_wait_ms,
        rt_rhi_error* out_error);
    bool begin_blit_encoder(metal_command_slot &slot, rt_rhi_error* out_error);
    bool begin_acceleration_encoder(
        metal_command_slot &slot,
        rt_rhi_operation operation,
        rt_rhi_error* out_error);
    bool begin_compute_encoder(metal_command_slot &slot, rt_rhi_error* out_error);
    void end_active_encoder(metal_command_slot &slot);
    bool retain_resource(metal_command_slot &slot, id<MTLResource> resource);
    bool adopt_cv_texture(metal_command_slot &slot, CVMetalTextureRef texture);
    bool retain_acceleration_structure(
        metal_command_slot &slot,
        id<MTLAccelerationStructure> acceleration);
    bool ensure_acceleration_storage(
        std::size_t required_size,
        std::size_t required_scratch_size,
        id<MTLAccelerationStructure>* acceleration,
        std::size_t* acceleration_size,
        id<MTLBuffer>* scratch,
        std::size_t* scratch_size,
        bool* out_reused);
    bool initialize_timestamp_resources();
    void release_timestamp_resources();
    bool create_dispatch_timestamp_buffer(
        metal_command_slot &slot,
        std::size_t dispatch_index,
        rt_rhi_error* out_error);
    bool read_dispatch_timestamp_ms(
        const metal_command_slot &slot,
        double* out_ms) const;
    void release_command_slot(metal_command_slot &slot);
    void poll_completed_slots();
    bool write_failed_submission_error(
        rt_rhi_operation operation,
        std::uint64_t submission_serial,
        rt_rhi_error* out_error) const;
    bool finish_pending_slot(
        metal_command_slot &slot,
        bool wait_for_completion,
        rt_rhi_timing* out_timing,
        rt_rhi_error* out_error);
    bool buffer_supports_usage(const metal_buffer &buffer, rt_resource_usage usage) const;
    bool texture_supports_usage(const metal_texture &texture, rt_resource_usage usage) const;
    void release_binding_state();
    void release_shader_state();
    void release_resources();
    void release_native_objects();

    bool initialized_ = false;
    bool hardware_raytracing_ = false;
    metal_timestamp_mode timestamp_mode_ = metal_timestamp_mode::unsupported;
    bool owns_device_ = false;
    bool owns_command_queue_ = false;
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> command_queue_ = nil;
    CVMetalTextureCacheRef texture_cache_ = nullptr;
    id<MTLCounterSet> timestamp_counter_set_ = nil;
    std::uint64_t timestamp_frequency_ = 0;
    rt_object_registry<metal_buffer, rt_buffer_handle> buffer_registry_;
    rt_object_registry<metal_texture, rt_texture_handle> texture_registry_;
    rt_object_registry<metal_blas, rt_blas_handle> blas_registry_;
    rt_object_registry<metal_tlas, rt_tlas_handle> tlas_registry_;
    rt_object_registry<metal_shader_module, rt_shader_module_handle> shader_module_registry_;
    rt_object_registry<metal_pipeline, rt_pipeline_handle> pipeline_registry_;
    std::vector<metal_binding> bindings_;
    std::array<metal_command_slot, kRtCommandSlotCount> command_slots_{};
    std::uint64_t next_encoder_id_ = 1;
    std::uint64_t next_submission_serial_ = 1;
    std::uint64_t completed_submission_serial_ = 0;
    std::uint64_t failed_submission_serial_ = 0;
    std::int64_t failed_submission_code_ = 0;
    std::string failed_submission_detail_;
    rt_rhi_diagnostics diagnostics_{};
};

metal_rhi_device::~metal_rhi_device() {
    rt_rhi_error error{};
    (void)shutdown(&error);
}

rt_rhi_device_info metal_rhi_device::info() const {
    return {
        rt_rhi_backend_kind::metal_rt,
        "metal_rt",
        hardware_raytracing_,
    };
}

rt_rhi_capabilities metal_rhi_device::capabilities() const {
    return {
        hardware_raytracing_,
        timestamp_mode_ != metal_timestamp_mode::unsupported,
        false,
        false,
        true,
        true,
        true,
        rt_shader_binary_format::metallib,
        rt_texture_format::bgra8_unorm,
        rt_texture_format::rgba16_float,
    };
}

rt_native_texture_extension* metal_rhi_device::native_texture_extension() {
    return this;
}

rt_vulkan_interop_extension* metal_rhi_device::vulkan_interop_extension() {
    return nullptr;
}

void metal_rhi_device::release_resources() {
    release_shader_state();
    release_binding_state();
    tlas_registry_.clear([](metal_tlas &tlas) {
        if (tlas.resource != nil) {
            [tlas.resource release];
        }
        if (tlas.scratch != nil) {
            [tlas.scratch release];
        }
        if (tlas.instance_buffer != nil) {
            [tlas.instance_buffer release];
        }
        if (tlas.instance_kind_buffer != nil) {
            [tlas.instance_kind_buffer release];
        }
        release_acceleration_list(tlas.referenced_accelerations);
    });
    blas_registry_.clear([](metal_blas &blas) {
        if (blas.resource != nil) {
            [blas.resource release];
        }
        if (blas.scratch != nil) {
            [blas.scratch release];
        }
    });
    buffer_registry_.clear([](metal_buffer &buffer) {
        if (buffer.resource != nil) {
            [buffer.resource release];
        }
    });
    texture_registry_.clear([](metal_texture &texture) {
        if (texture.resource != nil) {
            [texture.resource release];
        }
    });
}

void metal_rhi_device::release_shader_state() {
    pipeline_registry_.clear([](metal_pipeline &pipeline) {
        release_metal_pipeline(pipeline);
    });
    shader_module_registry_.clear([](metal_shader_module &module) {
        if (module.library != nil) {
            [module.library release];
        }
    });
}

void metal_rhi_device::release_binding_state() {
    release_metal_bindings(bindings_);
}

bool metal_rhi_device::initialize_timestamp_resources() {
    release_timestamp_resources();
    if (device_ == nil) {
        return false;
    }
    if (@available(macOS 26.0, *)) {
        const bool dispatch_boundary =
            [device_ supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary];
        const bool stage_boundary =
            [device_ supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary];
        if (!dispatch_boundary && !stage_boundary) {
            return false;
        }
        std::uint64_t timestamp_frequency = [device_ queryTimestampFrequency];
        if (timestamp_frequency == 0) {
            return false;
        }
        MTLTimestamp cpu_start_timestamp = 0;
        MTLTimestamp gpu_start_timestamp = 0;
        MTLTimestamp cpu_end_timestamp = 0;
        MTLTimestamp gpu_end_timestamp = 0;
        [device_ sampleTimestamps:&cpu_start_timestamp gpuTimestamp:&gpu_start_timestamp];
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        [device_ sampleTimestamps:&cpu_end_timestamp gpuTimestamp:&gpu_end_timestamp];
        if (cpu_end_timestamp > cpu_start_timestamp &&
            gpu_end_timestamp > gpu_start_timestamp) {
            const double cpu_span = static_cast<double>(cpu_end_timestamp - cpu_start_timestamp);
            const double gpu_span = static_cast<double>(gpu_end_timestamp - gpu_start_timestamp);
            const double calibrated_frequency = gpu_span * 1.0e9 / cpu_span;
            if (std::isfinite(calibrated_frequency) && calibrated_frequency > 0.0) {
                timestamp_frequency = static_cast<std::uint64_t>(calibrated_frequency + 0.5);
            }
        }

        id<MTLCounterSet> timestamp_set = nil;
        for (id<MTLCounterSet> counter_set in device_.counterSets) {
            if ([counter_set.name isEqualToString:MTLCommonCounterSetTimestamp]) {
                timestamp_set = counter_set;
                break;
            }
        }
        if (timestamp_set == nil) {
            return false;
        }

        timestamp_counter_set_ = [timestamp_set retain];
        MTLCounterSampleBufferDescriptor* const descriptor =
            [[MTLCounterSampleBufferDescriptor alloc] init];
        descriptor.counterSet = timestamp_counter_set_;
        descriptor.label = @"rtvdb dispatch timestamp";
        descriptor.storageMode = MTLStorageModeShared;
        descriptor.sampleCount = kMetalTimestampSampleCount;
        bool created = true;
        for (metal_command_slot &slot : command_slots_) {
            for (id<MTLCounterSampleBuffer> &sample_buffer : slot.dispatch_timestamp_buffers) {
                NSError* native_error = nil;
                sample_buffer = [device_ newCounterSampleBufferWithDescriptor:descriptor
                                                                            error:&native_error];
                if (sample_buffer == nil) {
                    created = false;
                    break;
                }
            }
            if (!created) {
                break;
            }
        }
        [descriptor release];
        if (!created) {
            release_timestamp_resources();
            return false;
        }
        timestamp_frequency_ = timestamp_frequency;
        timestamp_mode_ = dispatch_boundary
            ? metal_timestamp_mode::dispatch_boundary
            : metal_timestamp_mode::stage_boundary;
        return true;
    }
    return false;
}

void metal_rhi_device::release_timestamp_resources() {
    for (metal_command_slot &slot : command_slots_) {
        for (id<MTLCounterSampleBuffer> &sample_buffer : slot.dispatch_timestamp_buffers) {
            if (sample_buffer != nil) {
                [sample_buffer release];
                sample_buffer = nil;
            }
        }
        slot.dispatch_timestamp_entries = 0;
        slot.last_dispatch_timestamp_entry = rt_logical_dispatch_entry::render;
        slot.dispatch_timestamp_next_sample_indices = {};
        slot.dispatch_timestamp_sample_indices = {};
    }
    if (timestamp_counter_set_ != nil) {
        [timestamp_counter_set_ release];
        timestamp_counter_set_ = nil;
    }
    timestamp_frequency_ = 0;
    timestamp_mode_ = metal_timestamp_mode::unsupported;
}

bool metal_rhi_device::create_dispatch_timestamp_buffer(
    metal_command_slot &slot,
    std::size_t dispatch_index,
    rt_rhi_error* out_error)
{
    if (dispatch_index >= slot.dispatch_timestamp_buffers.size() ||
        timestamp_counter_set_ == nil || device_ == nil) {
        if (out_error != nullptr) {
            *out_error = {
                rt_rhi_operation::trace_rays,
                0,
                "Metal dispatch timestamp resources are unavailable"};
        }
        return false;
    }
    if (slot.dispatch_timestamp_buffers[dispatch_index] != nil) {
        [slot.dispatch_timestamp_buffers[dispatch_index] release];
        slot.dispatch_timestamp_buffers[dispatch_index] = nil;
    }

    MTLCounterSampleBufferDescriptor* const descriptor =
        [[MTLCounterSampleBufferDescriptor alloc] init];
    descriptor.counterSet = timestamp_counter_set_;
    descriptor.label = @"rtvdb dispatch timestamp";
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.sampleCount = kMetalTimestampSampleCount;
    NSError* native_error = nil;
    id<MTLCounterSampleBuffer> const sample_buffer =
        [device_ newCounterSampleBufferWithDescriptor:descriptor error:&native_error];
    [descriptor release];
    if (sample_buffer == nil) {
        set_metal_native_error(
            out_error,
            rt_rhi_operation::trace_rays,
            "Metal dispatch timestamp sample buffer creation failed",
            native_error);
        return false;
    }
    slot.dispatch_timestamp_buffers[dispatch_index] = sample_buffer;
    return true;
}

bool metal_rhi_device::read_dispatch_timestamp_ms(
    const metal_command_slot &slot,
    double* out_ms) const
{
    if (out_ms == nullptr || timestamp_mode_ == metal_timestamp_mode::unsupported ||
        timestamp_frequency_ == 0 ||
        slot.dispatch_timestamp_entries == 0) {
        return false;
    }
    const std::size_t entry_index = static_cast<std::size_t>(slot.last_dispatch_timestamp_entry);
    if (entry_index >= slot.dispatch_timestamp_buffers.size()) {
        return false;
    }
    id<MTLCounterSampleBuffer> const sample_buffer = slot.dispatch_timestamp_buffers[entry_index];
    if (sample_buffer == nil) {
        return false;
    }
    const NSUInteger sample_index = slot.dispatch_timestamp_sample_indices[entry_index];
    NSData* const resolved = [sample_buffer resolveCounterRange:NSMakeRange(
        sample_index,
        kRtTimestampQueryCountPerRegion)];
    if (resolved == nil || resolved.length < sizeof(MTLCounterResultTimestamp) * 2u) {
        return false;
    }
    MTLCounterResultTimestamp timestamps[2]{};
    std::memcpy(timestamps, resolved.bytes, sizeof(timestamps));
    if (timestamps[0].timestamp == MTLCounterErrorValue ||
        timestamps[1].timestamp == MTLCounterErrorValue ||
        timestamps[1].timestamp <= timestamps[0].timestamp) {
        return false;
    }
    *out_ms = static_cast<double>(timestamps[1].timestamp - timestamps[0].timestamp) * 1000.0 /
        static_cast<double>(timestamp_frequency_);
    return true;
}

void metal_rhi_device::release_native_objects() {
    for (metal_command_slot &slot : command_slots_) {
        release_command_slot(slot);
    }
    release_timestamp_resources();
    if (texture_cache_ != nullptr) {
        CVMetalTextureCacheFlush(texture_cache_, 0);
        CFRelease(texture_cache_);
        texture_cache_ = nullptr;
    }
    if (owns_command_queue_ && command_queue_ != nil) {
        [command_queue_ release];
    }
    if (owns_device_ && device_ != nil) {
        [device_ release];
    }
    command_queue_ = nil;
    device_ = nil;
    owns_command_queue_ = false;
    owns_device_ = false;
    initialized_ = false;
    hardware_raytracing_ = false;
    timestamp_mode_ = metal_timestamp_mode::unsupported;
    next_encoder_id_ = 1;
    next_submission_serial_ = 1;
    completed_submission_serial_ = 0;
    failed_submission_serial_ = 0;
    failed_submission_code_ = 0;
    failed_submission_detail_.clear();
    diagnostics_ = {};
}

bool metal_rhi_device::initialize(
    const rt_rhi_device_desc &desc,
    rt_rhi_error* out_error)
{
    if (out_error != nullptr) {
        *out_error = {rt_rhi_operation::initialize, 0, {}};
    }
    if (initialized_) {
        if (out_error != nullptr) {
            out_error->detail = "Metal RHI is already initialized";
        }
        return false;
    }

    id<MTLDevice> requested_device = desc.native_device != nullptr
        ? static_cast<id<MTLDevice>>(desc.native_device)
        : nil;
    id<MTLCommandQueue> requested_queue = desc.native_graphics_queue != nullptr
        ? static_cast<id<MTLCommandQueue>>(desc.native_graphics_queue)
        : nil;
    if (requested_device == nil && requested_queue != nil) {
        requested_device = requested_queue.device;
    }
    if (requested_device == nil) {
        requested_device = [MTLCreateSystemDefaultDevice() retain];
        owns_device_ = requested_device != nil;
    }
    if (requested_device == nil) {
        if (out_error != nullptr) {
            out_error->detail = "Metal RHI could not create a system default device";
        }
        release_native_objects();
        return false;
    }
    device_ = requested_device;
    if (!metal_device_supports_raytracing(device_)) {
        if (out_error != nullptr) {
            out_error->detail = "Metal RHI device does not support ray tracing";
        }
        release_native_objects();
        return false;
    }
    if (requested_queue != nil && requested_queue.device != device_) {
        if (out_error != nullptr) {
            out_error->detail = "Metal RHI native device and command queue do not match";
        }
        release_native_objects();
        return false;
    }
    if (requested_queue == nil) {
        requested_queue = [device_ newCommandQueue];
        owns_command_queue_ = requested_queue != nil;
    }
    if (requested_queue == nil) {
        if (out_error != nullptr) {
            out_error->detail = "Metal RHI could not create a command queue";
        }
        release_native_objects();
        return false;
    }
    command_queue_ = requested_queue;
    const CVReturn cache_result = CVMetalTextureCacheCreate(
        kCFAllocatorDefault,
        nullptr,
        device_,
        nullptr,
        &texture_cache_);
    if (cache_result != kCVReturnSuccess || texture_cache_ == nullptr) {
        if (out_error != nullptr) {
            out_error->native_code = cache_result;
            out_error->detail = "Metal RHI could not create a CoreVideo texture cache";
        }
        release_native_objects();
        return false;
    }
    hardware_raytracing_ = metal_device_supports_hardware_raytracing(device_);
    (void)initialize_timestamp_resources();
    initialized_ = true;
    return true;
}

bool metal_rhi_device::shutdown(rt_rhi_error* out_error) {
    if (out_error != nullptr) {
        *out_error = {rt_rhi_operation::shutdown, 0, {}};
    }
    rt_rhi_timing timing{};
    rt_rhi_error wait_error{};
    const bool wait_succeeded = wait_idle(&timing, &wait_error);
    release_resources();
    release_native_objects();
    if (!wait_succeeded && out_error != nullptr) {
        *out_error = wait_error;
        out_error->operation = rt_rhi_operation::shutdown;
    }
    return wait_succeeded;
}

bool metal_rhi_device::wait_idle(
    rt_rhi_timing* out_timing,
    rt_rhi_error* out_error)
{
    if (out_timing != nullptr) {
        *out_timing = {};
    }
    if (out_error != nullptr) {
        *out_error = {rt_rhi_operation::wait_idle, 0, {}};
    }
    if (!initialized_) {
        return true;
    }
    for (metal_command_slot &slot : command_slots_) {
        if (slot.active) {
            release_command_slot(slot);
        }
    }
    for (metal_command_slot &slot : command_slots_) {
        if (!slot.pending) {
            continue;
        }
        (void)finish_pending_slot(slot, true, out_timing, nullptr);
    }
    if (write_failed_submission_error(
            rt_rhi_operation::wait_idle,
            failed_submission_serial_,
            out_error)) {
        return false;
    }
    return true;
}

void metal_rhi_device::end_active_encoder(metal_command_slot &slot) {
    if (slot.compute_encoder != nil) {
        [slot.compute_encoder endEncoding];
        [slot.compute_encoder release];
        slot.compute_encoder = nil;
    }
    if (slot.acceleration_encoder != nil) {
        [slot.acceleration_encoder endEncoding];
        [slot.acceleration_encoder release];
        slot.acceleration_encoder = nil;
    }
    if (slot.blit_encoder != nil) {
        [slot.blit_encoder endEncoding];
        [slot.blit_encoder release];
        slot.blit_encoder = nil;
    }
}

bool metal_rhi_device::retain_resource(
    metal_command_slot &slot,
    id<MTLResource> resource)
{
    if (resource == nil) {
        return false;
    }
    for (id<MTLResource> retained : slot.retained_resources) {
        if (retained == resource) {
            return true;
        }
    }
    [resource retain];
    try {
        slot.retained_resources.push_back(resource);
    } catch (const std::bad_alloc &) {
        [resource release];
        return false;
    }
    return true;
}

bool metal_rhi_device::adopt_cv_texture(
    metal_command_slot &slot,
    CVMetalTextureRef texture)
{
    if (texture == nullptr) {
        return false;
    }
    try {
        slot.retained_cv_textures.push_back(texture);
    } catch (const std::bad_alloc &) {
        return false;
    }
    return true;
}

bool metal_rhi_device::retain_acceleration_structure(
    metal_command_slot &slot,
    id<MTLAccelerationStructure> acceleration)
{
    if (acceleration == nil) {
        return false;
    }
    for (id<MTLAccelerationStructure> retained : slot.retained_acceleration_structures) {
        if (retained == acceleration) {
            return true;
        }
    }
    [acceleration retain];
    try {
        slot.retained_acceleration_structures.push_back(acceleration);
    } catch (const std::bad_alloc &) {
        [acceleration release];
        return false;
    }
    return true;
}

bool metal_rhi_device::ensure_acceleration_storage(
    std::size_t required_size,
    std::size_t required_scratch_size,
    id<MTLAccelerationStructure>* acceleration,
    std::size_t* acceleration_size,
    id<MTLBuffer>* scratch,
    std::size_t* scratch_size,
    bool* out_reused)
{
    if (out_reused != nullptr) {
        *out_reused = false;
    }
    if (device_ == nil || required_size == 0 || required_scratch_size == 0 ||
        acceleration == nullptr || acceleration_size == nullptr || scratch == nullptr ||
        scratch_size == nullptr || out_reused == nullptr ||
        required_size > (std::numeric_limits<NSUInteger>::max)() ||
        required_scratch_size > (std::numeric_limits<NSUInteger>::max)()) {
        return false;
    }

    const bool acceleration_reused = *acceleration != nil && *acceleration_size >= required_size;
    const bool scratch_reused = *scratch != nil && *scratch_size >= required_scratch_size;
    id<MTLAccelerationStructure> replacement_acceleration = nil;
    id<MTLBuffer> replacement_scratch = nil;
    if (!acceleration_reused) {
        replacement_acceleration = [device_ newAccelerationStructureWithSize:required_size];
        if (replacement_acceleration == nil) {
            return false;
        }
    }
    if (!scratch_reused) {
        replacement_scratch = [device_ newBufferWithLength:required_scratch_size
                                                    options:MTLResourceStorageModePrivate];
        if (replacement_scratch == nil) {
            [replacement_acceleration release];
            return false;
        }
    }

    if (!acceleration_reused) {
        const bool reallocated = *acceleration != nil;
        if (*acceleration != nil) {
            [*acceleration release];
        }
        if (diagnostics_.acceleration_capacity_bytes >= *acceleration_size) {
            diagnostics_.acceleration_capacity_bytes -= *acceleration_size;
        }
        *acceleration = replacement_acceleration;
        *acceleration_size = required_size;
        diagnostics_.acceleration_capacity_bytes += required_size;
        diagnostics_.acceleration_peak_capacity_bytes = (std::max)(
            diagnostics_.acceleration_peak_capacity_bytes,
            diagnostics_.acceleration_capacity_bytes);
        ++diagnostics_.acceleration_resource_allocation_count;
        if (reallocated) {
            ++diagnostics_.acceleration_resource_reallocation_count;
        }
    }
    if (!scratch_reused) {
        const bool grew = *scratch != nil;
        if (*scratch != nil) {
            [*scratch release];
        }
        if (diagnostics_.scratch_capacity_bytes >= *scratch_size) {
            diagnostics_.scratch_capacity_bytes -= *scratch_size;
        }
        *scratch = replacement_scratch;
        *scratch_size = required_scratch_size;
        diagnostics_.scratch_capacity_bytes += required_scratch_size;
        diagnostics_.scratch_peak_capacity_bytes = (std::max)(
            diagnostics_.scratch_peak_capacity_bytes,
            diagnostics_.scratch_capacity_bytes);
        if (grew) {
            ++diagnostics_.scratch_growth_count;
        }
    }
    *out_reused = acceleration_reused;
    return true;
}

void metal_rhi_device::release_command_slot(metal_command_slot &slot) {
    end_active_encoder(slot);
    for (id<MTLResource> resource : slot.retained_resources) {
        [resource release];
    }
    slot.retained_resources.clear();
    for (id<MTLAccelerationStructure> acceleration : slot.retained_acceleration_structures) {
        [acceleration release];
    }
    slot.retained_acceleration_structures.clear();
    for (CVMetalTextureRef texture : slot.retained_cv_textures) {
        CFRelease(texture);
    }
    slot.retained_cv_textures.clear();
    if (slot.command_buffer != nil) {
        [slot.command_buffer release];
    }
    slot.command_buffer = nil;
    slot.record_begin = {};
    slot.encoder_id = 0;
    slot.submission_serial = 0;
    slot.dispatch_timestamp_entries = 0;
    slot.last_dispatch_timestamp_entry = rt_logical_dispatch_entry::render;
    slot.active = false;
    slot.pending = false;
}

bool metal_rhi_device::finish_pending_slot(
    metal_command_slot &slot,
    bool wait_for_completion,
    rt_rhi_timing* out_timing,
    rt_rhi_error* out_error)
{
    if (!slot.pending || slot.command_buffer == nil) {
        return true;
    }
    if (wait_for_completion) {
        const auto wait_begin = std::chrono::steady_clock::now();
        [slot.command_buffer waitUntilCompleted];
        if (out_timing != nullptr) {
            out_timing->gpu_wait_ms += elapsed_ms(wait_begin, std::chrono::steady_clock::now());
        }
    }
    const MTLCommandBufferStatus status = slot.command_buffer.status;
    if (status != MTLCommandBufferStatusCompleted && status != MTLCommandBufferStatusError) {
        return true;
    }
    const std::uint64_t completed_serial = slot.submission_serial;
    if (completed_serial > completed_submission_serial_) {
        completed_submission_serial_ = completed_serial;
    }
    const bool injected_failure =
        status == MTLCommandBufferStatusCompleted && consume_metal_test_completion_failure();
    if (status == MTLCommandBufferStatusError || injected_failure) {
        NSError* const native_error = status == MTLCommandBufferStatusError
            ? slot.command_buffer.error
            : nil;
        if (failed_submission_serial_ == 0 || completed_serial < failed_submission_serial_) {
            const char* const native_detail = native_error != nil
                ? native_error.localizedDescription.UTF8String
                : nullptr;
            failed_submission_serial_ = completed_serial;
            failed_submission_code_ = injected_failure
                ? -1
                : (native_error != nil ? native_error.code : 0);
            failed_submission_detail_ = "Metal submission ";
            failed_submission_detail_ += std::to_string(completed_serial);
            failed_submission_detail_ += " failed";
            if (injected_failure) {
                failed_submission_detail_ += ": Injected completion failure";
            } else if (native_detail != nullptr && native_detail[0] != '\0') {
                failed_submission_detail_ += ": ";
                failed_submission_detail_ += native_detail;
            }
        }
        (void)write_failed_submission_error(
            rt_rhi_operation::wait_submission,
            completed_serial,
            out_error);
        release_command_slot(slot);
        return false;
    }
    if (slot.dispatch_timestamp_entries != 0) {
        double dispatch_gpu_ms = 0.0;
        diagnostics_.dispatch_gpu_ms = 0.0;
        if (read_dispatch_timestamp_ms(slot, &dispatch_gpu_ms)) {
            diagnostics_.dispatch_gpu_ms = dispatch_gpu_ms;
            if (out_timing != nullptr) {
                out_timing->gpu_ms += dispatch_gpu_ms;
            }
        }
    }
    release_command_slot(slot);
    return true;
}

void metal_rhi_device::poll_completed_slots() {
    for (metal_command_slot &slot : command_slots_) {
        if (slot.pending) {
            (void)finish_pending_slot(slot, false, nullptr, nullptr);
        }
    }
}

bool metal_rhi_device::write_failed_submission_error(
    rt_rhi_operation operation,
    std::uint64_t submission_serial,
    rt_rhi_error* out_error) const
{
    if (failed_submission_serial_ == 0 || submission_serial < failed_submission_serial_) {
        return false;
    }
    if (out_error != nullptr) {
        *out_error = {
            operation,
            failed_submission_code_,
            failed_submission_detail_};
    }
    return true;
}

bool metal_rhi_device::acquire_command_slot(
    std::uint32_t* out_slot,
    double* out_wait_ms,
    rt_rhi_error* out_error)
{
    if (out_slot == nullptr || out_wait_ms == nullptr) {
        return false;
    }
    *out_slot = 0;
    *out_wait_ms = 0.0;
    poll_completed_slots();
    if (write_failed_submission_error(
            rt_rhi_operation::begin_commands,
            failed_submission_serial_,
            out_error)) {
        return false;
    }
    for (std::uint32_t index = 0; index < command_slots_.size(); ++index) {
        const metal_command_slot &slot = command_slots_[index];
        if (!slot.active && !slot.pending) {
            *out_slot = index;
            return true;
        }
    }

    metal_command_slot* oldest = nullptr;
    std::uint32_t oldest_index = 0;
    for (std::uint32_t index = 0; index < command_slots_.size(); ++index) {
        metal_command_slot &slot = command_slots_[index];
        if (slot.pending && (oldest == nullptr || slot.submission_serial < oldest->submission_serial)) {
            oldest = &slot;
            oldest_index = index;
        }
    }
    if (oldest == nullptr) {
        if (out_error != nullptr) {
            *out_error = {
                rt_rhi_operation::begin_commands,
                0,
                "All Metal command slots are recording and cannot be reused"};
        }
        return false;
    }
    rt_rhi_timing timing{};
    if (!finish_pending_slot(*oldest, true, &timing, out_error)) {
        return false;
    }
    *out_wait_ms = timing.gpu_wait_ms;
    diagnostics_.command_slot_reuse_wait_ms += timing.gpu_wait_ms;
    *out_slot = oldest_index;
    return true;
}

metal_command_slot* metal_rhi_device::command_slot(
    rt_command_encoder encoder,
    rt_rhi_operation operation,
    rt_rhi_error* out_error)
{
    if (!initialized_ || !encoder || encoder.queue != rt_queue_class::graphics ||
        encoder.slot >= command_slots_.size()) {
        if (out_error != nullptr) {
            *out_error = {operation, 0, "Metal command encoder is invalid"};
        }
        return nullptr;
    }
    metal_command_slot &slot = command_slots_[encoder.slot];
    if (!slot.active || slot.encoder_id != encoder.id || slot.command_buffer == nil) {
        if (out_error != nullptr) {
            *out_error = {operation, 0, "Metal command encoder is stale or not recording"};
        }
        return nullptr;
    }
    return &slot;
}

bool metal_rhi_device::begin_blit_encoder(
    metal_command_slot &slot,
    rt_rhi_error* out_error)
{
    if (slot.blit_encoder != nil) {
        return true;
    }
    end_active_encoder(slot);
    slot.blit_encoder = [[slot.command_buffer blitCommandEncoder] retain];
    if (slot.blit_encoder == nil) {
        if (out_error != nullptr) {
            *out_error = {
                rt_rhi_operation::copy_resource,
                0,
                "Metal blit command encoder creation failed"};
        }
        return false;
    }
    return true;
}

bool metal_rhi_device::begin_acceleration_encoder(
    metal_command_slot &slot,
    rt_rhi_operation operation,
    rt_rhi_error* out_error)
{
    if (slot.acceleration_encoder != nil) {
        return true;
    }
    end_active_encoder(slot);
    slot.acceleration_encoder =
        [[slot.command_buffer accelerationStructureCommandEncoder] retain];
    if (slot.acceleration_encoder == nil) {
        if (out_error != nullptr) {
            *out_error = {
                operation,
                0,
                "Metal acceleration structure command encoder creation failed"};
        }
        return false;
    }
    return true;
}

bool metal_rhi_device::begin_compute_encoder(
    metal_command_slot &slot,
    rt_rhi_error* out_error)
{
    if (slot.compute_encoder != nil) {
        return true;
    }
    end_active_encoder(slot);
    slot.compute_encoder = [[slot.command_buffer computeCommandEncoder] retain];
    if (slot.compute_encoder == nil) {
        if (out_error != nullptr) {
            *out_error = {
                rt_rhi_operation::trace_rays,
                0,
                "Metal compute command encoder creation failed"};
        }
        return false;
    }
    return true;
}

bool metal_rhi_device::begin_commands(
    rt_queue_class queue,
    rt_command_encoder* out_encoder,
    rt_rhi_error* out_error)
{
    if (out_encoder != nullptr) {
        *out_encoder = {};
    }
    if (out_error != nullptr) {
        *out_error = {rt_rhi_operation::begin_commands, 0, {}};
    }
    if (!initialized_ || command_queue_ == nil || out_encoder == nullptr ||
        queue != rt_queue_class::graphics) {
        if (out_error != nullptr) {
            out_error->detail = "Metal begin_commands request is invalid";
        }
        return false;
    }
    std::uint32_t slot_index = 0;
    double slot_wait_ms = 0.0;
    if (!acquire_command_slot(&slot_index, &slot_wait_ms, out_error)) {
        return false;
    }
    (void)slot_wait_ms;
    metal_command_slot &slot = command_slots_[slot_index];
    slot.command_buffer = [[command_queue_ commandBuffer] retain];
    if (slot.command_buffer == nil) {
        if (out_error != nullptr) {
            out_error->detail = "Metal command buffer creation failed";
        }
        return false;
    }
    slot.record_begin = std::chrono::steady_clock::now();
    slot.encoder_id = next_encoder_id_++;
    if (next_encoder_id_ == 0) {
        next_encoder_id_ = 1;
    }
    slot.active = true;
    *out_encoder = {slot.encoder_id, slot_index, queue};
    return true;
}

bool metal_rhi_device::submit_commands(
    rt_command_encoder encoder,
    rt_submission_token* out_submission,
    rt_rhi_timing* out_timing,
    rt_rhi_error* out_error)
{
    if (out_submission != nullptr) {
        *out_submission = {};
    }
    if (out_timing != nullptr) {
        *out_timing = {};
    }
    if (out_error != nullptr) {
        *out_error = {rt_rhi_operation::submit_commands, 0, {}};
    }
    metal_command_slot* const slot = command_slot(
        encoder,
        rt_rhi_operation::submit_commands,
        out_error);
    if (slot == nullptr || out_submission == nullptr) {
        if (slot != nullptr && out_error != nullptr) {
            out_error->detail = "Metal submit_commands requires an output submission token";
        }
        return false;
    }
    end_active_encoder(*slot);
    const auto submit_begin = std::chrono::steady_clock::now();
    if (out_timing != nullptr) {
        out_timing->command_record_ms = elapsed_ms(slot->record_begin, submit_begin);
    }
    [slot->command_buffer commit];
    const auto submit_end = std::chrono::steady_clock::now();
    const double submit_cpu_ms = elapsed_ms(submit_begin, submit_end);
    diagnostics_.dispatch_submit_cpu_ms = submit_cpu_ms;
    diagnostics_.dispatch_gpu_wait_ms = 0.0;
    if (out_timing != nullptr) {
        out_timing->submit_cpu_ms = submit_cpu_ms;
        out_timing->gpu_ms = diagnostics_.dispatch_gpu_ms;
    }
    slot->active = false;
    slot->pending = true;
    slot->submission_serial = next_submission_serial_++;
    if (next_submission_serial_ == 0) {
        next_submission_serial_ = 1;
    }
    *out_submission = {slot->submission_serial};
    return true;
}

void metal_rhi_device::discard_commands(rt_command_encoder encoder) {
    if (!encoder || encoder.slot >= command_slots_.size()) {
        return;
    }
    metal_command_slot &slot = command_slots_[encoder.slot];
    if (slot.active && slot.encoder_id == encoder.id) {
        release_command_slot(slot);
    }
}

bool metal_rhi_device::is_complete(
    rt_submission_token submission,
    bool* out_complete,
    rt_rhi_error* out_error)
{
    if (out_complete != nullptr) {
        *out_complete = false;
    }
    if (out_error != nullptr) {
        *out_error = {rt_rhi_operation::query_submission, 0, {}};
    }
    if (!submission || out_complete == nullptr || submission.serial >= next_submission_serial_) {
        if (out_error != nullptr) {
            out_error->detail = "Metal submission token is invalid";
        }
        return false;
    }
    poll_completed_slots();
    if (write_failed_submission_error(
            rt_rhi_operation::query_submission,
            submission.serial,
            out_error)) {
        return false;
    }
    *out_complete = submission.serial <= completed_submission_serial_;
    return true;
}

bool metal_rhi_device::wait(
    rt_submission_token submission,
    rt_rhi_timing* out_timing,
    rt_rhi_error* out_error)
{
    if (out_timing != nullptr) {
        *out_timing = {};
    }
    if (out_error != nullptr) {
        *out_error = {rt_rhi_operation::wait_submission, 0, {}};
    }
    if (!submission || submission.serial >= next_submission_serial_) {
        if (out_error != nullptr) {
            out_error->detail = "Metal submission token is invalid";
        }
        return false;
    }
    poll_completed_slots();
    if (write_failed_submission_error(
            rt_rhi_operation::wait_submission,
            submission.serial,
            out_error)) {
        return false;
    }
    if (submission.serial <= completed_submission_serial_) {
        return true;
    }
    for (metal_command_slot &slot : command_slots_) {
        if (slot.pending && slot.submission_serial == submission.serial) {
            return finish_pending_slot(slot, true, out_timing, out_error);
        }
    }
    if (out_error != nullptr) {
        out_error->detail = "Metal submission is not associated with a command slot";
    }
    return false;
}

bool metal_rhi_device::buffer_supports_usage(
    const metal_buffer &buffer,
    rt_resource_usage usage) const
{
    switch (usage) {
    case rt_resource_usage::undefined:
        return true;
    case rt_resource_usage::shader_write:
        return (buffer.desc.usage & rt_buffer_usage_shader_write) != 0u;
    case rt_resource_usage::shader_read:
        return (buffer.desc.usage & (rt_buffer_usage_shader_read | rt_buffer_usage_uniform)) != 0u;
    case rt_resource_usage::acceleration_build_input:
        return (buffer.desc.usage & rt_buffer_usage_acceleration_build_input) != 0u;
    case rt_resource_usage::acceleration_storage:
        return false;
    case rt_resource_usage::copy_source:
        return (buffer.desc.usage & rt_buffer_usage_copy_source) != 0u;
    case rt_resource_usage::copy_destination:
        return (buffer.desc.usage & rt_buffer_usage_copy_destination) != 0u;
    case rt_resource_usage::clear_destination:
        return false;
    case rt_resource_usage::host_read:
        return buffer.desc.memory_domain != rt_memory_domain::device;
    }
    return false;
}

bool metal_rhi_device::texture_supports_usage(
    const metal_texture &texture,
    rt_resource_usage usage) const
{
    switch (usage) {
    case rt_resource_usage::undefined:
        return true;
    case rt_resource_usage::shader_write:
        return (texture.desc.usage & rt_texture_usage_shader_write) != 0u;
    case rt_resource_usage::shader_read:
        return (texture.desc.usage & rt_texture_usage_shader_read) != 0u;
    case rt_resource_usage::copy_source:
        return (texture.desc.usage & rt_texture_usage_copy_source) != 0u;
    case rt_resource_usage::copy_destination:
    case rt_resource_usage::clear_destination:
        return (texture.desc.usage & rt_texture_usage_copy_destination) != 0u;
    case rt_resource_usage::acceleration_build_input:
    case rt_resource_usage::acceleration_storage:
    case rt_resource_usage::host_read:
        return false;
    }
    return false;
}

bool metal_rhi_device::barrier(
    rt_command_encoder encoder,
    const rt_resource_barrier* barriers,
    std::size_t barrier_count,
    rt_rhi_error* out_error)
{
    if (out_error != nullptr) {
        *out_error = {rt_rhi_operation::transition_resource, 0, {}};
    }
    metal_command_slot* const slot = command_slot(
        encoder,
        rt_rhi_operation::transition_resource,
        out_error);
    if (slot == nullptr || (barrier_count != 0 && barriers == nullptr)) {
        if (slot != nullptr && out_error != nullptr) {
            out_error->detail = "Metal barrier list is invalid";
        }
        return false;
    }
    for (std::size_t index = 0; index < barrier_count; ++index) {
        const rt_resource_barrier &barrier = barriers[index];
        bool valid = false;
        if (barrier.kind == rt_resource_kind::buffer) {
            const metal_buffer* const buffer = buffer_registry_.get(barrier.buffer);
            valid = buffer != nullptr &&
                buffer_supports_usage(*buffer, barrier.before) &&
                buffer_supports_usage(*buffer, barrier.after);
        } else if (barrier.kind == rt_resource_kind::texture) {
            const metal_texture* const texture = texture_registry_.get(barrier.texture);
            valid = texture != nullptr &&
                texture_supports_usage(*texture, barrier.before) &&
                texture_supports_usage(*texture, barrier.after);
        }
        if (!valid) {
            if (out_error != nullptr) {
                out_error->detail = "Metal barrier resource or semantic usage is invalid";
            }
            return false;
        }
    }
    end_active_encoder(*slot);
    return true;
}

bool metal_rhi_device::copy_buffer(
    rt_command_encoder encoder,
    rt_buffer_handle source,
    rt_buffer_handle destination,
    const rt_buffer_copy_region &region,
    rt_rhi_error* out_error)
{
    if (out_error != nullptr) {
        *out_error = {rt_rhi_operation::copy_resource, 0, {}};
    }
    metal_command_slot* const slot = command_slot(
        encoder,
        rt_rhi_operation::copy_resource,
        out_error);
    metal_buffer* const source_buffer = buffer_registry_.get(source);
    metal_buffer* const destination_buffer = buffer_registry_.get(destination);
    const bool same_buffer_overlap = source == destination &&
        (region.source_offset < region.destination_offset
            ? region.destination_offset - region.source_offset < region.size
            : region.source_offset - region.destination_offset < region.size);
    if (slot == nullptr || source_buffer == nullptr || destination_buffer == nullptr ||
        (source_buffer->desc.usage & rt_buffer_usage_copy_source) == 0u ||
        (destination_buffer->desc.usage & rt_buffer_usage_copy_destination) == 0u ||
        !range_is_valid(source_buffer->desc.size, region.source_offset, region.size) ||
        !range_is_valid(destination_buffer->desc.size, region.destination_offset, region.size) ||
        same_buffer_overlap) {
        if (slot != nullptr && out_error != nullptr) {
            out_error->detail = "Metal buffer copy request is invalid";
        }
        return false;
    }
    if (!retain_resource(*slot, source_buffer->resource) ||
        !retain_resource(*slot, destination_buffer->resource) ||
        !begin_blit_encoder(*slot, out_error)) {
        if (out_error != nullptr && out_error->detail.empty()) {
            out_error->detail = "Metal buffer copy resource retention failed";
        }
        return false;
    }
    [slot->blit_encoder copyFromBuffer:source_buffer->resource
                          sourceOffset:region.source_offset
                              toBuffer:destination_buffer->resource
                     destinationOffset:region.destination_offset
                                  size:region.size];
    return true;
}

bool metal_rhi_device::copy_texture_to_buffer(
    rt_command_encoder encoder,
    rt_texture_handle source,
    rt_buffer_handle destination,
    const rt_texture_buffer_copy_region &region,
    rt_rhi_error* out_error)
{
    if (out_error != nullptr) {
        *out_error = {rt_rhi_operation::copy_resource, 0, {}};
    }
    metal_command_slot* const slot = command_slot(
        encoder,
        rt_rhi_operation::copy_resource,
        out_error);
    metal_texture* const source_texture = texture_registry_.get(source);
    metal_buffer* const destination_buffer = buffer_registry_.get(destination);
    const std::size_t bytes_per_pixel = source_texture != nullptr
        ? rt_texture_format_bytes_per_pixel(source_texture->desc.format)
        : 0;
    const bool row_size_valid = bytes_per_pixel != 0 &&
        region.width <= (std::numeric_limits<std::size_t>::max)() / bytes_per_pixel;
    const std::size_t row_size = row_size_valid
        ? static_cast<std::size_t>(region.width) * bytes_per_pixel
        : 0;
    const bool copy_size_valid = region.height != 0 &&
        region.buffer_row_pitch <= (std::numeric_limits<std::size_t>::max)() / region.height;
    const std::size_t copy_size = copy_size_valid
        ? region.buffer_row_pitch * static_cast<std::size_t>(region.height)
        : 0;
    if (slot == nullptr || source_texture == nullptr || destination_buffer == nullptr ||
        (source_texture->desc.usage & rt_texture_usage_copy_source) == 0u ||
        (destination_buffer->desc.usage & rt_buffer_usage_copy_destination) == 0u ||
        region.width == 0 || region.height == 0 ||
        region.width > source_texture->desc.width ||
        region.height > source_texture->desc.height ||
        !row_size_valid || region.buffer_row_pitch < row_size ||
        region.buffer_row_pitch % kMetalTextureRowPitchAlignment != 0 ||
        !copy_size_valid ||
        !range_is_valid(destination_buffer->desc.size, region.buffer_offset, copy_size)) {
        if (slot != nullptr && out_error != nullptr) {
            out_error->detail = "Metal texture-to-buffer copy request is invalid";
        }
        return false;
    }
    if (!retain_resource(*slot, source_texture->resource) ||
        !retain_resource(*slot, destination_buffer->resource) ||
        !begin_blit_encoder(*slot, out_error)) {
        if (out_error != nullptr && out_error->detail.empty()) {
            out_error->detail = "Metal texture copy resource retention failed";
        }
        return false;
    }
    [slot->blit_encoder copyFromTexture:source_texture->resource
                            sourceSlice:0
                            sourceLevel:0
                           sourceOrigin:MTLOriginMake(0, 0, 0)
                             sourceSize:MTLSizeMake(region.width, region.height, 1)
                               toBuffer:destination_buffer->resource
                      destinationOffset:region.buffer_offset
                 destinationBytesPerRow:region.buffer_row_pitch
               destinationBytesPerImage:copy_size];
    return true;
}

bool metal_rhi_device::clear_texture(
    rt_command_encoder encoder,
    rt_texture_handle texture,
    const float color[4],
    rt_rhi_error* out_error)
{
    if (out_error != nullptr) {
        *out_error = {rt_rhi_operation::clear_texture, 0, {}};
    }
    metal_command_slot* const slot = command_slot(
        encoder,
        rt_rhi_operation::clear_texture,
        out_error);
    metal_texture* const target = texture_registry_.get(texture);
    if (slot == nullptr || target == nullptr || color == nullptr ||
        (target->desc.usage & rt_texture_usage_copy_destination) == 0u) {
        if (slot != nullptr && out_error != nullptr) {
            out_error->detail = "Metal texture clear request is invalid";
        }
        return false;
    }
    if (!retain_resource(*slot, target->resource)) {
        if (out_error != nullptr) {
            out_error->detail = "Metal texture clear resource retention failed";
        }
        return false;
    }
    end_active_encoder(*slot);
    MTLRenderPassDescriptor* const pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = target->resource;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(
        color[0],
        color[1],
        color[2],
        color[3]);
    id<MTLRenderCommandEncoder> clear_encoder =
        [[slot->command_buffer renderCommandEncoderWithDescriptor:pass] retain];
    if (clear_encoder == nil) {
        if (out_error != nullptr) {
            out_error->detail = "Metal texture clear encoder creation failed";
        }
        return false;
    }
    [clear_encoder endEncoding];
    [clear_encoder release];
    return true;
}

bool metal_rhi_device::trace_rays(
    rt_command_encoder encoder,
    const rt_trace_rays_desc &desc,
    rt_rhi_error* out_error)
{
    if (out_error != nullptr) {
        *out_error = {rt_rhi_operation::trace_rays, 0, {}};
    }
    const auto record_begin = std::chrono::steady_clock::now();
    metal_command_slot* const slot = command_slot(
        encoder,
        rt_rhi_operation::trace_rays,
        out_error);
    metal_pipeline* const pipeline = pipeline_registry_.get(desc.pipeline);
    const std::size_t dispatch_index = static_cast<std::size_t>(desc.entry);
    if (slot == nullptr || pipeline == nullptr ||
        dispatch_index >= pipeline->dispatch_states.size() ||
        pipeline->dispatch_states[dispatch_index] == nil ||
        desc.width == 0 || desc.height == 0 || desc.depth != 1) {
        if (slot != nullptr && out_error != nullptr) {
            out_error->detail = "Metal trace dispatch descriptor is invalid";
        }
        return false;
    }

    constexpr std::array<rt_descriptor_type, 11> kExpectedTypes{{
        rt_descriptor_type::acceleration_structure,
        rt_descriptor_type::storage_texture,
        rt_descriptor_type::structured_buffer,
        rt_descriptor_type::structured_buffer,
        rt_descriptor_type::structured_buffer,
        rt_descriptor_type::structured_buffer,
        rt_descriptor_type::structured_buffer,
        rt_descriptor_type::structured_buffer,
        rt_descriptor_type::uniform_buffer,
        rt_descriptor_type::storage_buffer,
        rt_descriptor_type::storage_texture,
    }};
    std::array<const metal_binding*, kExpectedTypes.size()> resolved_bindings{};
    for (std::size_t binding_index = 0; binding_index < resolved_bindings.size(); ++binding_index) {
        for (const metal_binding &binding : bindings_) {
            if (binding.location.group == 0 && binding.location.binding == binding_index &&
                binding.type == kExpectedTypes[binding_index]) {
                resolved_bindings[binding_index] = &binding;
                break;
            }
        }
        if (resolved_bindings[binding_index] == nullptr) {
            if (out_error != nullptr) {
                out_error->detail =
                    "Metal trace binding " + std::to_string(binding_index) + " is unavailable";
            }
            return false;
        }
    }

    const metal_binding &scene_binding = *resolved_bindings[0];
    const metal_binding &output_binding = *resolved_bindings[1];
    const metal_binding &uniform_binding = *resolved_bindings[8];
    const metal_binding &pick_binding = *resolved_bindings[9];
    const metal_binding &accumulation_binding = *resolved_bindings[10];
    const bool render_dispatch = desc.entry == rt_logical_dispatch_entry::render;
    const std::uint32_t timestamp_entry_bit = 1u << dispatch_index;
    const bool stage_timestamp =
        desc.measure_gpu_time && timestamp_mode_ == metal_timestamp_mode::stage_boundary;
    id<MTLCounterSampleBuffer> timestamp_buffer = nil;
    std::uint32_t timestamp_sample_index = 0;
    if (desc.measure_gpu_time) {
        if (timestamp_mode_ == metal_timestamp_mode::unsupported) {
            if (out_error != nullptr) {
                out_error->detail = "Metal dispatch timestamp queries are unsupported";
            }
            return false;
        }
        if ((slot->dispatch_timestamp_entries & timestamp_entry_bit) != 0u) {
            if (out_error != nullptr) {
                out_error->detail = "Metal logical dispatch entry was already timestamped in this command buffer";
            }
            return false;
        }
        timestamp_sample_index = slot->dispatch_timestamp_next_sample_indices[dispatch_index];
        if (timestamp_sample_index + kRtTimestampQueryCountPerRegion >
            kMetalTimestampSampleCount) {
            if (!create_dispatch_timestamp_buffer(*slot, dispatch_index, out_error)) {
                return false;
            }
            timestamp_sample_index = 0;
            slot->dispatch_timestamp_next_sample_indices[dispatch_index] =
                kRtTimestampQueryCountPerRegion;
        } else {
            slot->dispatch_timestamp_next_sample_indices[dispatch_index] =
                timestamp_sample_index + kRtTimestampQueryCountPerRegion;
        }
        timestamp_buffer = slot->dispatch_timestamp_buffers[dispatch_index];
        if (timestamp_buffer == nil) {
            if (out_error != nullptr) {
                out_error->detail = "Metal dispatch timestamp sample buffer is unavailable";
            }
            return false;
        }
    }
    if (scene_binding.acceleration == nil || uniform_binding.buffer == nil ||
        uniform_binding.element_stride == 0 || encoder.slot >= uniform_binding.element_count ||
        pick_binding.buffer == nil || scene_binding.instance_kind_buffer == nil ||
        (render_dispatch &&
            (output_binding.texture == nil || accumulation_binding.texture == nil ||
                desc.width > output_binding.texture.width ||
                desc.height > output_binding.texture.height ||
                desc.width > accumulation_binding.texture.width ||
                desc.height > accumulation_binding.texture.height))) {
        if (out_error != nullptr) {
            out_error->detail = "Metal trace native binding state is incomplete";
        }
        return false;
    }

    if (!retain_acceleration_structure(*slot, scene_binding.acceleration) ||
        !retain_resource(*slot, scene_binding.instance_kind_buffer)) {
        if (out_error != nullptr) {
            out_error->detail = "Metal trace acceleration binding retention failed";
        }
        return false;
    }
    for (id<MTLAccelerationStructure> acceleration : scene_binding.referenced_accelerations) {
        if (!retain_acceleration_structure(*slot, acceleration)) {
            if (out_error != nullptr) {
                out_error->detail = "Metal trace referenced BLAS retention failed";
            }
            return false;
        }
    }
    for (const metal_binding* binding : resolved_bindings) {
        if ((binding->buffer != nil && !retain_resource(*slot, binding->buffer)) ||
            (binding->texture != nil && !retain_resource(*slot, binding->texture))) {
            if (out_error != nullptr) {
                out_error->detail = "Metal trace resource binding retention failed";
            }
            return false;
        }
    }
    if (stage_timestamp) {
        end_active_encoder(*slot);
        MTLComputePassDescriptor* const pass = [MTLComputePassDescriptor computePassDescriptor];
        MTLComputePassSampleBufferAttachmentDescriptor* const attachment =
            pass.sampleBufferAttachments[0];
        attachment.sampleBuffer = timestamp_buffer;
        attachment.startOfEncoderSampleIndex = timestamp_sample_index;
        attachment.endOfEncoderSampleIndex = timestamp_sample_index + 1u;
        slot->compute_encoder =
            [[slot->command_buffer computeCommandEncoderWithDescriptor:pass] retain];
        if (slot->compute_encoder == nil) {
            if (out_error != nullptr) {
                out_error->detail = "Metal timestamped compute encoder creation failed";
            }
            return false;
        }
    } else if (!begin_compute_encoder(*slot, out_error)) {
        return false;
    }

    for (id<MTLAccelerationStructure> acceleration : scene_binding.referenced_accelerations) {
        [slot->compute_encoder useResource:acceleration usage:MTLResourceUsageRead];
    }
    id<MTLComputePipelineState> const state = pipeline->dispatch_states[dispatch_index];
    [slot->compute_encoder setComputePipelineState:state];
    [slot->compute_encoder setAccelerationStructure:scene_binding.acceleration atBufferIndex:0];
    for (std::size_t binding_index = 2; binding_index <= 9; ++binding_index) {
        const metal_binding &binding = *resolved_bindings[binding_index];
        const NSUInteger offset = binding_index == 8
            ? encoder.slot * binding.element_stride
            : 0;
        [slot->compute_encoder setBuffer:binding.buffer offset:offset atIndex:binding_index];
    }
    [slot->compute_encoder setBuffer:scene_binding.instance_kind_buffer offset:0 atIndex:11];
    if (render_dispatch) {
        [slot->compute_encoder setTexture:output_binding.texture atIndex:1];
        [slot->compute_encoder setTexture:accumulation_binding.texture atIndex:10];
    }

    const NSUInteger thread_width = state.threadExecutionWidth;
    const NSUInteger thread_height = (std::max)(
        static_cast<NSUInteger>(1),
        state.maxTotalThreadsPerThreadgroup / thread_width);
    if (desc.measure_gpu_time && timestamp_mode_ == metal_timestamp_mode::dispatch_boundary) {
        [slot->compute_encoder sampleCountersInBuffer:timestamp_buffer
                                       atSampleIndex:timestamp_sample_index
                                         withBarrier:YES];
    }
    [slot->compute_encoder dispatchThreads:MTLSizeMake(desc.width, desc.height, 1)
                         threadsPerThreadgroup:MTLSizeMake(thread_width, thread_height, 1)];
    if (desc.measure_gpu_time && timestamp_mode_ == metal_timestamp_mode::dispatch_boundary) {
        [slot->compute_encoder sampleCountersInBuffer:timestamp_buffer
                                       atSampleIndex:timestamp_sample_index + 1u
                                         withBarrier:YES];
    }
    if (stage_timestamp) {
        end_active_encoder(*slot);
    }
    if (desc.measure_gpu_time) {
        slot->dispatch_timestamp_entries |= timestamp_entry_bit;
        slot->last_dispatch_timestamp_entry = desc.entry;
        slot->dispatch_timestamp_sample_indices[dispatch_index] = timestamp_sample_index;
    }
    diagnostics_.dispatch_cpu_ms += elapsed_ms(record_begin, std::chrono::steady_clock::now());
    return true;
}

bool metal_rhi_device::create_buffer(
    const rt_buffer_desc &desc,
    rt_buffer_handle* out_buffer,
    rt_rhi_error* out_error)
{
    if (out_buffer != nullptr) {
        *out_buffer = {};
    }
    if (!initialized_ || device_ == nil || out_buffer == nullptr || desc.size == 0 ||
        desc.size > (std::numeric_limits<NSUInteger>::max)() ||
        (desc.usage & ~kMetalBufferUsageMask) != 0u) {
        if (out_error != nullptr) {
            *out_error = {
                rt_rhi_operation::create_resource,
                0,
                "Metal RHI buffer descriptor is invalid"};
        }
        return false;
    }
    id<MTLBuffer> resource = [device_ newBufferWithLength:desc.size
                                                  options:metal_buffer_options(desc.memory_domain)];
    if (resource == nil) {
        if (out_error != nullptr) {
            *out_error = {
                rt_rhi_operation::create_resource,
                0,
                "Metal RHI buffer allocation failed"};
        }
        return false;
    }
    if (!buffer_registry_.insert({resource, desc}, out_buffer)) {
        [resource release];
        if (out_error != nullptr) {
            *out_error = {
                rt_rhi_operation::create_resource,
                0,
                "Metal RHI buffer handle allocation failed"};
        }
        return false;
    }
    return true;
}

bool metal_rhi_device::upload_buffer(
    rt_buffer_handle buffer,
    std::size_t offset,
    const void* data,
    std::size_t size,
    rt_rhi_error* out_error)
{
    metal_buffer* const target = buffer_registry_.get(buffer);
    if (target == nullptr || target->resource == nil ||
        target->desc.memory_domain == rt_memory_domain::device || data == nullptr ||
        !range_is_valid(target->desc.size, offset, size)) {
        if (out_error != nullptr) {
            *out_error = {
                rt_rhi_operation::upload_scene_buffers,
                0,
                "Metal RHI buffer upload request is invalid or buffer is not host visible"};
        }
        return false;
    }
    void* const contents = target->resource.contents;
    if (contents == nullptr) {
        if (out_error != nullptr) {
            *out_error = {
                rt_rhi_operation::upload_scene_buffers,
                0,
                "Metal RHI host-visible buffer has no mapped contents"};
        }
        return false;
    }
    std::memcpy(static_cast<std::uint8_t*>(contents) + offset, data, size);
    return true;
}

bool metal_rhi_device::read_buffer(
    rt_buffer_handle buffer,
    std::size_t offset,
    void* data,
    std::size_t size,
    rt_rhi_error* out_error)
{
    const metal_buffer* const source = buffer_registry_.get(buffer);
    if (source == nullptr || source->resource == nil ||
        source->desc.memory_domain == rt_memory_domain::device || data == nullptr ||
        !range_is_valid(source->desc.size, offset, size)) {
        if (out_error != nullptr) {
            *out_error = {
                rt_rhi_operation::readback,
                0,
                "Metal RHI buffer read request is invalid or buffer is not host visible"};
        }
        return false;
    }
    const void* const contents = source->resource.contents;
    if (contents == nullptr) {
        if (out_error != nullptr) {
            *out_error = {
                rt_rhi_operation::readback,
                0,
                "Metal RHI host-visible buffer has no mapped contents"};
        }
        return false;
    }
    std::memcpy(data, static_cast<const std::uint8_t*>(contents) + offset, size);
    return true;
}

void metal_rhi_device::destroy_buffer(rt_buffer_handle buffer) {
    metal_buffer removed{};
    if (buffer_registry_.erase(buffer, &removed) && removed.resource != nil) {
        [removed.resource release];
    }
}

bool metal_rhi_device::create_texture(
    const rt_texture_desc &desc,
    rt_texture_handle* out_texture,
    rt_rhi_error* out_error)
{
    if (out_texture != nullptr) {
        *out_texture = {};
    }
    const MTLPixelFormat format = metal_texture_format(desc.format);
    if (!initialized_ || device_ == nil || out_texture == nullptr ||
        desc.width == 0 || desc.height == 0 || format == MTLPixelFormatInvalid ||
        (desc.usage & ~kMetalTextureUsageMask) != 0u) {
        if (out_error != nullptr) {
            *out_error = {
                rt_rhi_operation::create_resource,
                0,
                "Metal RHI texture descriptor is invalid"};
        }
        return false;
    }
    MTLTextureDescriptor* native_desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                          width:desc.width
                                                         height:desc.height
                                                      mipmapped:NO];
    native_desc.storageMode = MTLStorageModePrivate;
    native_desc.usage = metal_texture_usage(desc.usage);
    id<MTLTexture> resource = [device_ newTextureWithDescriptor:native_desc];
    if (resource == nil) {
        if (out_error != nullptr) {
            *out_error = {
                rt_rhi_operation::create_resource,
                0,
                "Metal RHI texture allocation failed"};
        }
        return false;
    }
    if (!texture_registry_.insert({resource, desc}, out_texture)) {
        [resource release];
        if (out_error != nullptr) {
            *out_error = {
                rt_rhi_operation::create_resource,
                0,
                "Metal RHI texture handle allocation failed"};
        }
        return false;
    }
    return true;
}

void metal_rhi_device::destroy_texture(rt_texture_handle texture) {
    metal_texture removed{};
    if (texture_registry_.erase(texture, &removed) && removed.resource != nil) {
        [removed.resource release];
    }
}

bool metal_rhi_device::get_texture_copy_footprint(
    rt_texture_handle texture,
    rt_texture_copy_footprint* out_footprint,
    rt_rhi_error* out_error)
{
    if (out_footprint != nullptr) {
        *out_footprint = {};
    }
    const metal_texture* const source = texture_registry_.get(texture);
    const std::size_t bytes_per_pixel = source != nullptr
        ? rt_texture_format_bytes_per_pixel(source->desc.format)
        : 0;
    if (source == nullptr || source->resource == nil || out_footprint == nullptr ||
        bytes_per_pixel == 0 ||
        source->desc.width > (std::numeric_limits<std::size_t>::max)() / bytes_per_pixel) {
        if (out_error != nullptr) {
            *out_error = {
                rt_rhi_operation::create_resource,
                0,
                "Metal RHI texture copy footprint request is invalid"};
        }
        return false;
    }
    const std::size_t unaligned_row_pitch =
        static_cast<std::size_t>(source->desc.width) * bytes_per_pixel;
    std::size_t row_pitch = 0;
    if (!align_up(unaligned_row_pitch, kMetalTextureRowPitchAlignment, &row_pitch) ||
        source->desc.height > (std::numeric_limits<std::size_t>::max)() / row_pitch) {
        if (out_error != nullptr) {
            *out_error = {
                rt_rhi_operation::create_resource,
                0,
                "Metal RHI texture copy footprint exceeds native size limits"};
        }
        return false;
    }
    *out_footprint = {
        row_pitch,
        row_pitch * static_cast<std::size_t>(source->desc.height),
    };
    return true;
}

bool metal_rhi_device::create_blas(
    rt_blas_handle* out_blas,
    rt_rhi_error* out_error)
{
    if (out_blas != nullptr) {
        *out_blas = {};
    }
    if (!initialized_ || out_blas == nullptr || !blas_registry_.insert({}, out_blas)) {
        if (out_error != nullptr) {
            *out_error = {
                rt_rhi_operation::build_blas,
                0,
                "Metal BLAS object creation failed"};
        }
        return false;
    }
    return true;
}

void metal_rhi_device::destroy_blas(rt_blas_handle handle) {
    metal_blas blas{};
    if (!blas_registry_.erase(handle, &blas)) {
        return;
    }
    if (diagnostics_.acceleration_capacity_bytes >= blas.allocation_size) {
        diagnostics_.acceleration_capacity_bytes -= blas.allocation_size;
    }
    if (diagnostics_.scratch_capacity_bytes >= blas.scratch_size) {
        diagnostics_.scratch_capacity_bytes -= blas.scratch_size;
    }
    if (blas.resource != nil) {
        [blas.resource release];
    }
    if (blas.scratch != nil) {
        [blas.scratch release];
    }
}

bool metal_rhi_device::create_tlas(
    rt_tlas_handle* out_tlas,
    rt_rhi_error* out_error)
{
    if (out_tlas != nullptr) {
        *out_tlas = {};
    }
    if (!initialized_ || out_tlas == nullptr || !tlas_registry_.insert({}, out_tlas)) {
        if (out_error != nullptr) {
            *out_error = {
                rt_rhi_operation::build_tlas,
                0,
                "Metal TLAS object creation failed"};
        }
        return false;
    }
    return true;
}

void metal_rhi_device::destroy_tlas(rt_tlas_handle handle) {
    metal_tlas tlas{};
    if (!tlas_registry_.erase(handle, &tlas)) {
        return;
    }
    if (diagnostics_.acceleration_capacity_bytes >= tlas.allocation_size) {
        diagnostics_.acceleration_capacity_bytes -= tlas.allocation_size;
    }
    if (diagnostics_.scratch_capacity_bytes >= tlas.scratch_size) {
        diagnostics_.scratch_capacity_bytes -= tlas.scratch_size;
    }
    if (tlas.resource != nil) {
        [tlas.resource release];
    }
    if (tlas.scratch != nil) {
        [tlas.scratch release];
    }
    if (tlas.instance_buffer != nil) {
        [tlas.instance_buffer release];
    }
    if (tlas.instance_kind_buffer != nil) {
        [tlas.instance_kind_buffer release];
    }
    release_acceleration_list(tlas.referenced_accelerations);
}

bool metal_rhi_device::build_blas(
    rt_command_encoder encoder,
    const rt_blas_build_desc &desc,
    rt_blas_build_result* out_result,
    rt_rhi_error* out_error)
{
    if (out_result != nullptr) {
        *out_result = {};
    }
    if (out_error != nullptr) {
        *out_error = {rt_rhi_operation::build_blas, 0, {}};
    }
    const auto build_begin = std::chrono::steady_clock::now();
    metal_command_slot* const slot = command_slot(
        encoder,
        rt_rhi_operation::build_blas,
        out_error);
    metal_blas* const destination = blas_registry_.get(desc.destination);
    if (slot == nullptr || out_result == nullptr || destination == nullptr ||
        !validate_rt_blas_build_desc(desc)) {
        if (slot != nullptr && out_error != nullptr) {
            out_error->detail = "Metal BLAS build request is invalid";
        }
        return false;
    }

    NSMutableArray<MTLAccelerationStructureGeometryDescriptor*>* build_geometries =
        [NSMutableArray arrayWithCapacity:desc.geometry_count];
    NSMutableArray<MTLAccelerationStructureGeometryDescriptor*>* sizing_geometries =
        [NSMutableArray arrayWithCapacity:desc.geometry_count];
    for (std::size_t geometry_index = 0; geometry_index < desc.geometry_count; ++geometry_index) {
        const rt_acceleration_geometry_desc &source = desc.geometries[geometry_index];
        rt_blas_geometry_counts counts{};
        if (!get_rt_blas_geometry_counts(desc, geometry_index, &counts)) {
            if (out_error != nullptr) {
                out_error->detail = "Metal BLAS geometry counts are invalid";
            }
            return false;
        }

        if (source.type == rt_acceleration_geometry_type::triangles) {
            const rt_triangle_geometry_desc &triangles = source.triangles;
            metal_buffer* const vertex_buffer = buffer_registry_.get(triangles.vertex_buffer);
            metal_buffer* const index_buffer = buffer_registry_.get(triangles.index_buffer);
            std::size_t vertex_size = 0;
            std::size_t index_size = 0;
            const bool valid = vertex_buffer != nullptr && index_buffer != nullptr &&
                (vertex_buffer->desc.usage & rt_buffer_usage_acceleration_build_input) != 0u &&
                (index_buffer->desc.usage & rt_buffer_usage_acceleration_build_input) != 0u &&
                triangles.vertex_offset % triangles.vertex_stride == 0 &&
                triangles.index_offset % sizeof(std::uint32_t) == 0 &&
                size_product(triangles.vertex_count, triangles.vertex_stride, &vertex_size) &&
                size_product(triangles.index_count, sizeof(std::uint32_t), &index_size) &&
                range_is_valid(vertex_buffer->desc.size, triangles.vertex_offset, vertex_size) &&
                range_is_valid(index_buffer->desc.size, triangles.index_offset, index_size);
            if (!valid || !retain_resource(*slot, vertex_buffer->resource) ||
                !retain_resource(*slot, index_buffer->resource)) {
                if (out_error != nullptr) {
                    out_error->detail = "Metal BLAS triangle geometry is invalid";
                }
                return false;
            }

            MTLAccelerationStructureTriangleGeometryDescriptor* build_geometry =
                [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
            build_geometry.vertexBuffer = vertex_buffer->resource;
            build_geometry.vertexBufferOffset = triangles.vertex_offset;
            build_geometry.vertexStride = triangles.vertex_stride;
            build_geometry.indexBuffer = index_buffer->resource;
            build_geometry.indexBufferOffset = triangles.index_offset;
            build_geometry.indexType = MTLIndexTypeUInt32;
            build_geometry.triangleCount = counts.actual;
            build_geometry.opaque =
                (source.flags & rt_acceleration_geometry_opaque) != 0u ? YES : NO;
            if (@available(macOS 13.0, *)) {
                build_geometry.vertexFormat = MTLAttributeFormatFloat3;
            }

            MTLAccelerationStructureTriangleGeometryDescriptor* sizing_geometry =
                [build_geometry copy];
            sizing_geometry.triangleCount = counts.allocation;
            [build_geometries addObject:build_geometry];
            [sizing_geometries addObject:sizing_geometry];
            [sizing_geometry release];
        } else if (source.type == rt_acceleration_geometry_type::aabbs) {
            const rt_aabb_geometry_desc &aabbs = source.aabbs;
            metal_buffer* const buffer = buffer_registry_.get(aabbs.buffer);
            std::size_t buffer_size = 0;
            const bool valid = buffer != nullptr &&
                (buffer->desc.usage & rt_buffer_usage_acceleration_build_input) != 0u &&
                aabbs.offset % sizeof(float) == 0 &&
                size_product(aabbs.count, aabbs.stride, &buffer_size) &&
                range_is_valid(buffer->desc.size, aabbs.offset, buffer_size);
            if (!valid || !retain_resource(*slot, buffer->resource)) {
                if (out_error != nullptr) {
                    out_error->detail = "Metal BLAS AABB geometry is invalid";
                }
                return false;
            }

            MTLAccelerationStructureBoundingBoxGeometryDescriptor* build_geometry =
                [MTLAccelerationStructureBoundingBoxGeometryDescriptor descriptor];
            build_geometry.boundingBoxBuffer = buffer->resource;
            build_geometry.boundingBoxBufferOffset = aabbs.offset;
            build_geometry.boundingBoxStride = aabbs.stride;
            build_geometry.boundingBoxCount = counts.actual;
            build_geometry.opaque =
                (source.flags & rt_acceleration_geometry_opaque) != 0u ? YES : NO;

            MTLAccelerationStructureBoundingBoxGeometryDescriptor* sizing_geometry =
                [build_geometry copy];
            sizing_geometry.boundingBoxCount = counts.allocation;
            [build_geometries addObject:build_geometry];
            [sizing_geometries addObject:sizing_geometry];
            [sizing_geometry release];
        } else {
            if (out_error != nullptr) {
                out_error->detail = "Metal BLAS geometry type is unsupported";
            }
            return false;
        }
    }

    MTLPrimitiveAccelerationStructureDescriptor* build_descriptor =
        [MTLPrimitiveAccelerationStructureDescriptor descriptor];
    build_descriptor.geometryDescriptors = build_geometries;
    build_descriptor.usage = metal_acceleration_usage(desc.flags);
    MTLPrimitiveAccelerationStructureDescriptor* sizing_descriptor =
        [MTLPrimitiveAccelerationStructureDescriptor descriptor];
    sizing_descriptor.geometryDescriptors = sizing_geometries;
    sizing_descriptor.usage = build_descriptor.usage;

    const auto prebuild_begin = std::chrono::steady_clock::now();
    const MTLAccelerationStructureSizes sizes =
        [device_ accelerationStructureSizesWithDescriptor:sizing_descriptor];
    const double prebuild_ms = elapsed_ms(prebuild_begin, std::chrono::steady_clock::now());
    diagnostics_.acceleration_prebuild_query_ms += prebuild_ms;
    if (desc.geometries[0].type == rt_acceleration_geometry_type::triangles) {
        if (desc.geometry_count == 1) {
            diagnostics_.chunk_blas_prebuild_query_ms += prebuild_ms;
            ++diagnostics_.chunk_blas_prebuild_query_count;
        } else {
            diagnostics_.grouped_blas_prebuild_query_ms += prebuild_ms;
            ++diagnostics_.grouped_blas_prebuild_query_count;
        }
    }

    bool reused = false;
    const auto allocation_begin = std::chrono::steady_clock::now();
    const bool storage_ready = ensure_acceleration_storage(
        sizes.accelerationStructureSize,
        sizes.buildScratchBufferSize,
        &destination->resource,
        &destination->allocation_size,
        &destination->scratch,
        &destination->scratch_size,
        &reused);
    diagnostics_.acceleration_resource_allocate_ms +=
        elapsed_ms(allocation_begin, std::chrono::steady_clock::now());
    if (!storage_ready ||
        !retain_acceleration_structure(*slot, destination->resource) ||
        !retain_resource(*slot, destination->scratch) ||
        !begin_acceleration_encoder(*slot, rt_rhi_operation::build_blas, out_error)) {
        if (out_error != nullptr && out_error->detail.empty()) {
            out_error->detail = "Metal BLAS storage or command encoder preparation failed";
        }
        return false;
    }

    const auto record_begin = std::chrono::steady_clock::now();
    [slot->acceleration_encoder buildAccelerationStructure:destination->resource
                                                descriptor:build_descriptor
                                             scratchBuffer:destination->scratch
                                       scratchBufferOffset:0];
    const double record_ms = elapsed_ms(record_begin, std::chrono::steady_clock::now());
    diagnostics_.acceleration_build_call_record_ms += record_ms;
    diagnostics_.acceleration_command_record_ms += record_ms;
    diagnostics_.acceleration_cpu_ms += elapsed_ms(build_begin, std::chrono::steady_clock::now());
    *out_result = {
        desc.destination,
        prebuild_ms,
        destination->allocation_size,
        reused,
    };
    return true;
}

bool metal_rhi_device::build_tlas(
    rt_command_encoder encoder,
    const rt_tlas_build_desc &desc,
    rt_rhi_error* out_error)
{
    if (out_error != nullptr) {
        *out_error = {rt_rhi_operation::build_tlas, 0, {}};
    }
    const auto build_begin = std::chrono::steady_clock::now();
    metal_command_slot* const slot = command_slot(
        encoder,
        rt_rhi_operation::build_tlas,
        out_error);
    metal_tlas* const destination = tlas_registry_.get(desc.destination);
    if (slot == nullptr || destination == nullptr || !validate_rt_tlas_build_desc(desc)) {
        if (slot != nullptr && out_error != nullptr) {
            out_error->detail = "Metal TLAS build request is invalid";
        }
        return false;
    }
    if (desc.instance_count == 0) {
        if (diagnostics_.acceleration_capacity_bytes >= destination->allocation_size) {
            diagnostics_.acceleration_capacity_bytes -= destination->allocation_size;
        }
        if (diagnostics_.scratch_capacity_bytes >= destination->scratch_size) {
            diagnostics_.scratch_capacity_bytes -= destination->scratch_size;
        }
        if (destination->resource != nil) {
            [destination->resource release];
        }
        if (destination->scratch != nil) {
            [destination->scratch release];
        }
        if (destination->instance_buffer != nil) {
            [destination->instance_buffer release];
        }
        if (destination->instance_kind_buffer != nil) {
            [destination->instance_kind_buffer release];
        }
        release_acceleration_list(destination->referenced_accelerations);
        *destination = {};
        return true;
    }

    std::vector<MTLAccelerationStructureUserIDInstanceDescriptor> instances(desc.instance_count);
    std::vector<std::uint32_t> instance_kinds(desc.instance_count);
    std::vector<id<MTLAccelerationStructure>> owned_referenced_accelerations;
    try {
        owned_referenced_accelerations.reserve(desc.instance_count);
    } catch (const std::bad_alloc &) {
        if (out_error != nullptr) {
            out_error->detail = "Metal TLAS referenced BLAS ownership allocation failed";
        }
        return false;
    }
    NSMutableArray<id<MTLAccelerationStructure>>* referenced_accelerations =
        [NSMutableArray arrayWithCapacity:desc.instance_count];
    for (std::size_t instance_index = 0; instance_index < desc.instance_count; ++instance_index) {
        const rt_tlas_instance_desc &source = desc.instances[instance_index];
        const metal_blas* const acceleration = blas_registry_.get(source.acceleration);
        if (acceleration == nullptr || acceleration->resource == nil ||
            !retain_acceleration_structure(*slot, acceleration->resource) ||
            !append_unique_acceleration(
                owned_referenced_accelerations,
                acceleration->resource)) {
            if (out_error != nullptr) {
                out_error->detail = "Metal TLAS references an invalid BLAS";
            }
            return false;
        }
        MTLAccelerationStructureUserIDInstanceDescriptor &instance = instances[instance_index];
        instance.transformationMatrix = metal_instance_transform(source.transform);
        instance.options = metal_instance_options(source.flags);
        instance.mask = source.mask;
        instance.intersectionFunctionTableOffset = source.hit_group_contribution;
        instance.accelerationStructureIndex = static_cast<std::uint32_t>(instance_index);
        instance.userID = source.instance_id;
        instance_kinds[instance_index] = source.hit_group_contribution;
        [referenced_accelerations addObject:acceleration->resource];
    }

    const std::size_t instance_buffer_size =
        instances.size() * sizeof(MTLAccelerationStructureUserIDInstanceDescriptor);
    const auto instance_upload_begin = std::chrono::steady_clock::now();
    id<MTLBuffer> instance_buffer =
        [device_ newBufferWithBytes:instances.data()
                            length:instance_buffer_size
                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> instance_kind_buffer =
        [device_ newBufferWithBytes:instance_kinds.data()
                            length:instance_kinds.size() * sizeof(std::uint32_t)
                           options:MTLResourceStorageModeShared];
    diagnostics_.tlas_instance_upload_ms +=
        elapsed_ms(instance_upload_begin, std::chrono::steady_clock::now());
    if (instance_buffer == nil || instance_kind_buffer == nil) {
        [instance_buffer release];
        [instance_kind_buffer release];
        if (out_error != nullptr) {
            out_error->detail = "Metal TLAS instance or kind buffer allocation failed";
        }
        return false;
    }

    MTLInstanceAccelerationStructureDescriptor* descriptor =
        [MTLInstanceAccelerationStructureDescriptor descriptor];
    descriptor.instanceDescriptorBuffer = instance_buffer;
    descriptor.instanceDescriptorBufferOffset = 0;
    descriptor.instanceDescriptorStride = sizeof(MTLAccelerationStructureUserIDInstanceDescriptor);
    descriptor.instanceCount = desc.instance_count;
    descriptor.instancedAccelerationStructures = referenced_accelerations;
    descriptor.instanceDescriptorType = MTLAccelerationStructureInstanceDescriptorTypeUserID;
    descriptor.usage = metal_acceleration_usage(desc.flags);

    const auto prebuild_begin = std::chrono::steady_clock::now();
    const MTLAccelerationStructureSizes sizes =
        [device_ accelerationStructureSizesWithDescriptor:descriptor];
    const double prebuild_ms = elapsed_ms(prebuild_begin, std::chrono::steady_clock::now());
    diagnostics_.acceleration_prebuild_query_ms += prebuild_ms;
    diagnostics_.tlas_prebuild_query_ms += prebuild_ms;
    ++diagnostics_.tlas_prebuild_query_count;

    bool reused = false;
    const auto allocation_begin = std::chrono::steady_clock::now();
    const bool storage_ready = ensure_acceleration_storage(
        sizes.accelerationStructureSize,
        sizes.buildScratchBufferSize,
        &destination->resource,
        &destination->allocation_size,
        &destination->scratch,
        &destination->scratch_size,
        &reused);
    diagnostics_.acceleration_resource_allocate_ms +=
        elapsed_ms(allocation_begin, std::chrono::steady_clock::now());
    (void)reused;
    if (!storage_ready ||
        !retain_acceleration_structure(*slot, destination->resource) ||
        !retain_resource(*slot, destination->scratch) ||
        !retain_resource(*slot, instance_buffer) ||
        !retain_resource(*slot, instance_kind_buffer) ||
        !begin_acceleration_encoder(*slot, rt_rhi_operation::build_tlas, out_error)) {
        [instance_buffer release];
        [instance_kind_buffer release];
        if (out_error != nullptr) {
            out_error->operation = rt_rhi_operation::build_tlas;
            if (out_error->detail.empty()) {
                out_error->detail = "Metal TLAS storage or command encoder preparation failed";
            }
        }
        return false;
    }

    if (destination->instance_buffer != nil) {
        [destination->instance_buffer release];
    }
    if (destination->instance_kind_buffer != nil) {
        [destination->instance_kind_buffer release];
    }
    for (id<MTLAccelerationStructure> acceleration : owned_referenced_accelerations) {
        [acceleration retain];
    }
    release_acceleration_list(destination->referenced_accelerations);
    destination->instance_buffer = instance_buffer;
    destination->instance_kind_buffer = instance_kind_buffer;
    destination->referenced_accelerations = std::move(owned_referenced_accelerations);
    destination->instance_buffer_size = instance_buffer_size;
    const auto record_begin = std::chrono::steady_clock::now();
    [slot->acceleration_encoder buildAccelerationStructure:destination->resource
                                                descriptor:descriptor
                                             scratchBuffer:destination->scratch
                                       scratchBufferOffset:0];
    const double record_ms = elapsed_ms(record_begin, std::chrono::steady_clock::now());
    diagnostics_.acceleration_build_call_record_ms += record_ms;
    diagnostics_.acceleration_command_record_ms += record_ms;
    diagnostics_.acceleration_cpu_ms += elapsed_ms(build_begin, std::chrono::steady_clock::now());
    return true;
}

bool metal_rhi_device::update_bindings(
    const rt_binding_update_request &request,
    rt_rhi_error* out_error)
{
    if (out_error != nullptr) {
        *out_error = {rt_rhi_operation::update_bindings, 0, {}};
    }
    std::vector<metal_binding> next_bindings;
    const auto fail = [&next_bindings, out_error](
                          const rt_binding_write* write,
                          const char* reason) {
        release_metal_bindings(next_bindings);
        if (out_error != nullptr) {
            *out_error = {
                rt_rhi_operation::update_bindings,
                0,
                "Metal binding update failed"};
            if (write != nullptr) {
                out_error->detail +=
                    " at group " + std::to_string(write->location.group) +
                    " binding " + std::to_string(write->location.binding);
            }
            out_error->detail += ": ";
            out_error->detail += reason != nullptr ? reason : "unknown error";
        }
        return false;
    };
    if (!initialized_ || request.writes == nullptr || request.write_count == 0) {
        return fail(nullptr, "binding writes are unavailable");
    }
    try {
        next_bindings.reserve(request.write_count);
    } catch (const std::bad_alloc &) {
        return fail(nullptr, "binding state allocation failed");
    }

    for (std::size_t write_index = 0; write_index < request.write_count; ++write_index) {
        const rt_binding_write &write = request.writes[write_index];
        const auto duplicate = std::find_if(
            next_bindings.begin(),
            next_bindings.end(),
            [&write](const metal_binding &binding) {
                return binding.location == write.location;
            });
        if (duplicate != next_bindings.end()) {
            return fail(&write, "binding location is duplicated");
        }

        metal_binding binding{};
        binding.location = write.location;
        binding.type = write.type;
        binding.element_count = write.element_count;
        binding.element_stride = write.element_stride;
        if (write.type == rt_descriptor_type::acceleration_structure) {
            const metal_tlas* const tlas = tlas_registry_.get(write.acceleration);
            if (write.element_count != 1 || write.element_stride != 0 ||
                tlas == nullptr || tlas->resource == nil || tlas->instance_kind_buffer == nil) {
                return fail(&write, "specified TLAS is unavailable or the element range is invalid");
            }
            binding.acceleration = [tlas->resource retain];
            binding.instance_kind_buffer = [tlas->instance_kind_buffer retain];
            if (!retain_acceleration_list(
                    tlas->referenced_accelerations,
                    &binding.referenced_accelerations)) {
                release_metal_binding(binding);
                return fail(&write, "specified TLAS binding bundle allocation failed");
            }
        } else if (write.type == rt_descriptor_type::storage_texture) {
            const metal_texture* const texture = texture_registry_.get(write.texture);
            if (write.element_count != 1 || write.element_stride != 0 ||
                texture == nullptr || texture->resource == nil ||
                (texture->desc.usage & rt_texture_usage_shader_write) == 0u) {
                return fail(&write, "storage texture is unavailable or lacks shader-write usage");
            }
            binding.texture = [texture->resource retain];
        } else if (write.type == rt_descriptor_type::structured_buffer) {
            if (write.element_count == 0) {
                next_bindings.push_back(std::move(binding));
                continue;
            }
            const metal_buffer* const buffer = buffer_registry_.get(write.resource);
            std::size_t range_size = 0;
            if (write.element_stride == 0 ||
                !size_product(write.element_count, write.element_stride, &range_size) ||
                buffer == nullptr || buffer->resource == nil || range_size > buffer->desc.size ||
                (buffer->desc.usage & rt_buffer_usage_shader_read) == 0u) {
                return fail(&write, "structured buffer range or shader-read usage is invalid");
            }
            binding.buffer = [buffer->resource retain];
        } else if (write.type == rt_descriptor_type::storage_buffer ||
            write.type == rt_descriptor_type::uniform_buffer) {
            const metal_buffer* const buffer = buffer_registry_.get(write.resource);
            std::size_t range_size = 0;
            const std::uint32_t required_usage = write.type == rt_descriptor_type::storage_buffer
                ? rt_buffer_usage_shader_write
                : rt_buffer_usage_uniform;
            if (write.element_count == 0 || write.element_stride == 0 ||
                !size_product(write.element_count, write.element_stride, &range_size) ||
                buffer == nullptr || buffer->resource == nil || range_size > buffer->desc.size ||
                (buffer->desc.usage & required_usage) == 0u) {
                return fail(&write, "buffer range or descriptor usage is invalid");
            }
            binding.buffer = [buffer->resource retain];
        } else {
            return fail(&write, "descriptor type is unsupported");
        }
        next_bindings.push_back(std::move(binding));
    }

    release_binding_state();
    bindings_ = std::move(next_bindings);
    return true;
}

bool metal_rhi_device::create_shader_module(
    const rt_shader_module_desc &desc,
    rt_shader_module_handle* out_module,
    rt_rhi_error* out_error)
{
    if (out_module != nullptr) {
        *out_module = {};
    }
    if (out_error != nullptr) {
        *out_error = {rt_rhi_operation::create_shader_module, 0, {}};
    }
    if (!initialized_ || device_ == nil || out_module == nullptr ||
        desc.format != rt_shader_binary_format::metallib ||
        desc.data == nullptr || desc.size == 0) {
        if (out_error != nullptr) {
            out_error->detail = "Metal metallib shader module descriptor is invalid";
        }
        return false;
    }

    dispatch_data_t shader_data = dispatch_data_create(
        desc.data,
        desc.size,
        dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0),
        DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    if (shader_data == nullptr) {
        if (out_error != nullptr) {
            out_error->detail = "Metal metallib dispatch data creation failed";
        }
        return false;
    }
    NSError* native_error = nil;
    id<MTLLibrary> library = [device_ newLibraryWithData:shader_data error:&native_error];
    dispatch_release(shader_data);
    if (library == nil) {
        set_metal_native_error(
            out_error,
            rt_rhi_operation::create_shader_module,
            "Metal newLibraryWithData failed",
            native_error);
        return false;
    }
    if (!shader_module_registry_.insert({library}, out_module)) {
        [library release];
        if (out_error != nullptr) {
            out_error->detail = "Metal shader module handle allocation failed";
        }
        return false;
    }
    return true;
}

void metal_rhi_device::destroy_shader_module(rt_shader_module_handle handle) {
    metal_shader_module module{};
    if (shader_module_registry_.erase(handle, &module) && module.library != nil) {
        [module.library release];
    }
}

bool metal_rhi_device::create_pipeline(
    const rt_pipeline_desc &desc,
    rt_pipeline_handle* out_pipeline,
    rt_rhi_error* out_error)
{
    if (out_pipeline != nullptr) {
        *out_pipeline = {};
    }
    if (out_error != nullptr) {
        *out_error = {rt_rhi_operation::prepare_pipeline, 0, {}};
    }
    if (desc.model == rt_pipeline_model::native_ray_tracing) {
        if (out_error != nullptr) {
            out_error->detail = "Metal RHI rejects the native-raytracing pipeline model";
        }
        return false;
    }
    if (!initialized_ || device_ == nil || out_pipeline == nullptr ||
        desc.model != rt_pipeline_model::compute_intersector ||
        !validate_rt_pipeline_desc(desc)) {
        if (out_error != nullptr) {
            out_error->detail = "Metal compute-intersector pipeline descriptor is invalid";
        }
        return false;
    }

    metal_pipeline pipeline{};
    constexpr std::array<rt_logical_dispatch_entry, 2> kDispatchEntries{{
        rt_logical_dispatch_entry::render,
        rt_logical_dispatch_entry::pick,
    }};
    for (rt_logical_dispatch_entry logical_entry : kDispatchEntries) {
        std::uint32_t shader_index = kRtUnusedShaderIndex;
        if (!get_rt_pipeline_dispatch_entry_index(desc, logical_entry, &shader_index) ||
            shader_index >= desc.shader_count) {
            release_metal_pipeline(pipeline);
            if (out_error != nullptr) {
                out_error->detail = "Metal pipeline dispatch entry could not resolve a shader";
            }
            return false;
        }
        const rt_shader_entry_desc &shader = desc.shaders[shader_index];
        const metal_shader_module* const module = shader_module_registry_.get(shader.module);
        if (module == nullptr || module->library == nil) {
            release_metal_pipeline(pipeline);
            if (out_error != nullptr) {
                out_error->detail = "Metal pipeline references an unavailable shader module";
            }
            return false;
        }
        NSString* const function_name = [NSString stringWithUTF8String:shader.entry_point];
        id<MTLFunction> function = function_name != nil
            ? [module->library newFunctionWithName:function_name]
            : nil;
        if (function == nil) {
            release_metal_pipeline(pipeline);
            if (out_error != nullptr) {
                out_error->detail = "Metal shader function '";
                out_error->detail += shader.entry_point;
                out_error->detail += "' was not found in the metallib";
            }
            return false;
        }

        NSError* native_error = nil;
        id<MTLComputePipelineState> state =
            [device_ newComputePipelineStateWithFunction:function error:&native_error];
        [function release];
        if (state == nil) {
            release_metal_pipeline(pipeline);
            std::string context = "Metal compute pipeline compilation failed for function '";
            context += shader.entry_point;
            context += "'";
            set_metal_native_error(
                out_error,
                rt_rhi_operation::prepare_pipeline,
                context.c_str(),
                native_error);
            return false;
        }
        pipeline.dispatch_states[static_cast<std::size_t>(logical_entry)] = state;
    }
    if (!pipeline_registry_.insert(std::move(pipeline), out_pipeline)) {
        release_metal_pipeline(pipeline);
        if (out_error != nullptr) {
            out_error->detail = "Metal pipeline handle allocation failed";
        }
        return false;
    }
    return true;
}

void metal_rhi_device::get_diagnostics(rt_rhi_diagnostics* out_diagnostics) const {
    if (out_diagnostics != nullptr) {
        *out_diagnostics = diagnostics_;
    }
}

bool metal_rhi_device::publish_texture(
    rt_texture_handle texture,
    const rt_native_texture_publish_desc &desc,
    rt_rhi_timing* out_timing,
    rt_rhi_error* out_error)
{
    if (out_timing != nullptr) {
        *out_timing = {};
    }
    if (out_error != nullptr) {
        *out_error = {rt_rhi_operation::native_texture, 0, {}};
    }
    if (desc.out_submission != nullptr) {
        *desc.out_submission = {};
    }
    if (desc.out_target != nullptr) {
        *desc.out_target = nullptr;
    }
    const metal_texture* const source = texture_registry_.get(texture);
    if (!initialized_ || texture_cache_ == nullptr || command_queue_ == nil ||
        desc.target == nullptr || source == nullptr || source->resource == nil ||
        source->desc.format != rt_texture_format::bgra8_unorm ||
        (source->desc.usage & rt_texture_usage_copy_source) == 0u ||
        source->resource.pixelFormat != MTLPixelFormatBGRA8Unorm ||
        source->resource.textureType != MTLTextureType2D) {
        if (out_error != nullptr) {
            out_error->detail = "Metal native output request is invalid";
        }
        return false;
    }

    CVPixelBufferRef const pixel_buffer = static_cast<CVPixelBufferRef>(desc.target);
    const std::size_t width = CVPixelBufferGetWidth(pixel_buffer);
    const std::size_t height = CVPixelBufferGetHeight(pixel_buffer);
    if (width != source->desc.width || height != source->desc.height ||
        CVPixelBufferGetPixelFormatType(pixel_buffer) != kCVPixelFormatType_32BGRA ||
        CVPixelBufferIsPlanar(pixel_buffer) || CVPixelBufferGetIOSurface(pixel_buffer) == nullptr) {
        if (out_error != nullptr) {
            out_error->detail = "Metal native output pixel buffer is not a compatible BGRA target";
        }
        return false;
    }

    CVMetalTextureRef cv_texture = nullptr;
    const CVReturn create_result = CVMetalTextureCacheCreateTextureFromImage(
        kCFAllocatorDefault,
        texture_cache_,
        pixel_buffer,
        nullptr,
        MTLPixelFormatBGRA8Unorm,
        width,
        height,
        0,
        &cv_texture);
    if (create_result != kCVReturnSuccess || cv_texture == nullptr) {
        if (out_error != nullptr) {
            out_error->native_code = create_result;
            out_error->detail = "Metal native output could not import the CoreVideo pixel buffer";
        }
        return false;
    }

    id<MTLTexture> const destination = CVMetalTextureGetTexture(cv_texture);
    if (destination == nil || destination.device != device_ ||
        destination.textureType != MTLTextureType2D ||
        destination.pixelFormat != MTLPixelFormatBGRA8Unorm ||
        destination.width != width || destination.height != height) {
        CFRelease(cv_texture);
        if (out_error != nullptr) {
            out_error->detail = "Metal native output imported an incompatible Metal texture";
        }
        return false;
    }

    rt_command_encoder encoder{};
    const auto begin_start = std::chrono::steady_clock::now();
    if (!begin_commands(rt_queue_class::graphics, &encoder, out_error)) {
        CFRelease(cv_texture);
        if (out_timing != nullptr) {
            out_timing->command_slot_wait_ms = elapsed_ms(begin_start, std::chrono::steady_clock::now());
        }
        if (out_error != nullptr) {
            out_error->operation = rt_rhi_operation::native_texture;
        }
        return false;
    }
    const double command_slot_wait_ms = elapsed_ms(begin_start, std::chrono::steady_clock::now());
    metal_command_slot* const slot = command_slot(
        encoder,
        rt_rhi_operation::native_texture,
        out_error);
    if (slot == nullptr || !retain_resource(*slot, source->resource) ||
        !adopt_cv_texture(*slot, cv_texture)) {
        discard_commands(encoder);
        CFRelease(cv_texture);
        if (out_timing != nullptr) {
            out_timing->command_slot_wait_ms = command_slot_wait_ms;
        }
        if (out_error != nullptr && out_error->detail.empty()) {
            out_error->detail = "Metal native output resource retention failed";
        }
        return false;
    }
    cv_texture = nullptr;

    if (!begin_blit_encoder(*slot, out_error)) {
        discard_commands(encoder);
        if (out_timing != nullptr) {
            out_timing->command_slot_wait_ms = command_slot_wait_ms;
        }
        if (out_error != nullptr) {
            out_error->operation = rt_rhi_operation::native_texture;
        }
        return false;
    }
    [slot->blit_encoder copyFromTexture:source->resource
                            sourceSlice:0
                            sourceLevel:0
                           sourceOrigin:MTLOriginMake(0, 0, 0)
                             sourceSize:MTLSizeMake(width, height, 1)
                              toTexture:destination
                       destinationSlice:0
                       destinationLevel:0
                      destinationOrigin:MTLOriginMake(0, 0, 0)];

    rt_submission_token submission{};
    rt_rhi_timing submit_timing{};
    if (!submit_commands(encoder, &submission, &submit_timing, out_error)) {
        discard_commands(encoder);
        if (out_timing != nullptr) {
            out_timing->command_slot_wait_ms = command_slot_wait_ms;
        }
        if (out_error != nullptr) {
            out_error->operation = rt_rhi_operation::native_texture;
        }
        return false;
    }
    submit_timing.command_slot_wait_ms = command_slot_wait_ms;
    submit_timing.gpu_wait_ms = 0.0;
    submit_timing.gpu_ms = 0.0;
    if (out_timing != nullptr) {
        *out_timing = submit_timing;
    }
    if (desc.out_target != nullptr) {
        *desc.out_target = desc.target;
    }
    if (desc.out_submission != nullptr) {
        *desc.out_submission = submission;
    }
    return true;
}

} // namespace

rt_rhi_factory_result create_metal_rhi() {
    return {
        std::make_unique<metal_rhi_device>(),
        metal_rt_shader_package(),
    };
}

} // namespace rtvdb::viewer_backend
