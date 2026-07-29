#include "viewer_backend/backend_internal.h"
#include "viewer_backend/rt_backend_common.h"
#include "viewer_backend/rt_object_registry.h"
#include "viewer_shell/shell.h"

#if defined(_WIN32)
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#include <vulkan/vulkan.h>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi1_6.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
#include <vector>

namespace rtvdb::viewer_backend {
namespace {

constexpr VkFormat kOutputFormat = VK_FORMAT_R8G8B8A8_UNORM;
constexpr std::uint32_t kTimestampQueryCountPerRegion = 2;
constexpr std::uint32_t kTimestampQueryRegionCount = 2;
constexpr std::uint32_t kTimestampQueryCountPerCommandSlot =
    kTimestampQueryCountPerRegion * kTimestampQueryRegionCount;
#if defined(_WIN32)
constexpr std::array<const char*, 2> kRequiredInstanceExtensions = {
    "VK_KHR_surface",
    "VK_KHR_win32_surface",
};
#else
constexpr std::array<const char*, 1> kRequiredInstanceExtensions = {
    "VK_KHR_surface",
};
#endif

#if defined(_WIN32)
bool query_dxgi_adapter_desc_from_luid(
    const std::uint8_t luid_bytes[VK_LUID_SIZE],
    DXGI_ADAPTER_DESC1* out_desc)
{
    if (luid_bytes == nullptr || out_desc == nullptr) {
        return false;
    }

    LUID luid{};
    std::memcpy(&luid, luid_bytes, VK_LUID_SIZE);

    IDXGIFactory4* factory = nullptr;
    IDXGIAdapter1* adapter = nullptr;
    bool found = false;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (SUCCEEDED(hr) && factory != nullptr) {
        hr = factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter));
        if (SUCCEEDED(hr) && adapter != nullptr && SUCCEEDED(adapter->GetDesc1(out_desc))) {
            found = true;
        }
    }

    if (adapter != nullptr) {
        adapter->Release();
    }
    if (factory != nullptr) {
        factory->Release();
    }
    return found;
}

bool query_preferred_dxgi_adapter(
    LUID* out_luid,
    DXGI_ADAPTER_DESC1* out_desc)
{
    if (out_luid == nullptr) {
        return false;
    }

    *out_luid = {};
    if (out_desc != nullptr) {
        *out_desc = {};
    }

    IDXGIFactory6* factory = nullptr;
    IDXGIAdapter1* adapter = nullptr;
    bool found = false;
    std::uint64_t best_score = 0;
    const HRESULT factory_hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (SUCCEEDED(factory_hr) && factory != nullptr) {
        for (UINT adapter_index = 0;
             factory->EnumAdapters1(adapter_index, &adapter) != DXGI_ERROR_NOT_FOUND;
             ++adapter_index) {
            DXGI_ADAPTER_DESC1 desc{};
            if (SUCCEEDED(adapter->GetDesc1(&desc)) && (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0) {
                std::uint64_t score = desc.DedicatedVideoMemory / (64ull * 1024ull * 1024ull);
                if (desc.DedicatedVideoMemory > 0) {
                    score += 100000ull;
                }
                if (!found || score > best_score) {
                    *out_luid = desc.AdapterLuid;
                    if (out_desc != nullptr) {
                        *out_desc = desc;
                    }
                    best_score = score;
                    found = true;
                }
            }
            adapter->Release();
            adapter = nullptr;
        }
    }

    if (adapter != nullptr) {
        adapter->Release();
    }
    if (factory != nullptr) {
        factory->Release();
    }
    return found;
}

#endif

static_assert(sizeof(rt_scene_gpu_aabb) == sizeof(VkAabbPositionsKHR));

struct vulkan_buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    rt_resource_usage usage = rt_resource_usage::undefined;
};

struct vulkan_acceleration_structure {
    vulkan_buffer storage{};
    VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
    VkDeviceAddress device_address = 0;
};

struct vulkan_texture {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    rt_texture_desc desc{};
};

struct vulkan_deferred_buffer_release {
    rt_submission_token submission{};
    vulkan_buffer buffer{};
};

struct vulkan_deferred_acceleration_release {
    rt_submission_token submission{};
    vulkan_acceleration_structure acceleration{};
};

struct vulkan_deferred_texture_release {
    rt_submission_token submission{};
    vulkan_texture texture{};
};

struct vulkan_acceleration_build_context {
    bool active = false;
    bool recorded = false;
    bool can_update_tlas = false;
    VkDeviceSize max_scratch_size = 0;
    std::chrono::steady_clock::time_point total_start{};
    std::chrono::steady_clock::time_point command_start{};
    std::vector<VkAccelerationStructureGeometryKHR> geometries;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> build_ranges;
    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> build_infos;
    std::vector<std::size_t> build_range_offsets;
    VkAccelerationStructureGeometryInstancesDataKHR top_instances{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR};
    VkAccelerationStructureGeometryKHR top_geometry{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    VkAccelerationStructureBuildRangeInfoKHR top_build_range{};
    VkAccelerationStructureBuildGeometryInfoKHR top_build_info{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
};

struct vulkan_command_slot {
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer buffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool submitted = false;
    rt_submission_token submission{};
    std::uint32_t timestamp_regions = 0;
};

struct vulkan_backend_state {
    bool initialized = false;
    bool hardware_ray_tracing = false;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphics_queue = VK_NULL_HANDLE;
    std::uint32_t graphics_queue_family = VK_QUEUE_FAMILY_IGNORED;
    std::uint32_t present_queue_family = VK_QUEUE_FAMILY_IGNORED;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkFence command_fence = VK_NULL_HANDLE;
    std::array<vulkan_command_slot, kRtCommandSlotCount> command_slots{};
    std::uint32_t command_slot_index = 0;
    std::uint32_t active_command_slot_index = 0;
    std::uint64_t next_submission_serial = 1;
    std::uint64_t completed_submission_serial = 0;
    std::uint64_t next_encoder_id = 1;
    std::uint64_t active_encoder_id = 0;
    VkQueryPool timestamp_query_pool = VK_NULL_HANDLE;
    float timestamp_period_ns = 0.0f;
    bool timestamp_queries_supported = false;
    rt_object_registry<vulkan_buffer, rt_buffer_handle> buffer_registry;
    rt_object_registry<vulkan_texture, rt_texture_handle> texture_registry;
    vulkan_buffer tlas_instance_buffer{};
    vulkan_buffer scratch_buffer{};
    rt_object_registry<vulkan_acceleration_structure, rt_blas_handle> blas_registry;
    rt_object_registry<vulkan_acceleration_structure, rt_tlas_handle> tlas_registry;
    rt_object_registry<VkShaderModule, rt_shader_module_handle> shader_module_registry;
    rt_object_registry<VkPipeline, rt_pipeline_handle> pipeline_registry;
    rt_object_registry<VkBuffer, rt_shader_table_handle> shader_table_registry;
    std::vector<vulkan_deferred_buffer_release> deferred_buffer_releases;
    std::vector<vulkan_deferred_acceleration_release> deferred_acceleration_releases;
    std::vector<vulkan_deferred_texture_release> deferred_texture_releases;
    rt_tlas_handle tlas{};
    rt_pipeline_handle pipeline_handle{};
    rt_shader_table_handle shader_table_handle{};
    std::size_t tlas_instance_count = 0;
    vulkan_acceleration_build_context acceleration_build{};
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    vulkan_buffer shader_binding_table{};
    std::vector<VkStridedDeviceAddressRegionKHR> raygen_regions;
    VkStridedDeviceAddressRegionKHR miss_region{};
    VkStridedDeviceAddressRegionKHR hit_region{};
    VkStridedDeviceAddressRegionKHR callable_region{};
    std::uint32_t pipeline_chunk_count = 0;
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rt_pipeline_properties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
    VkPhysicalDeviceAccelerationStructurePropertiesKHR accel_properties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
    PFN_vkCreateAccelerationStructureKHR create_acceleration_structure = nullptr;
    PFN_vkDestroyAccelerationStructureKHR destroy_acceleration_structure = nullptr;
    PFN_vkGetAccelerationStructureBuildSizesKHR get_acceleration_structure_build_sizes = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR cmd_build_acceleration_structures = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR get_acceleration_structure_device_address = nullptr;
    PFN_vkCreateRayTracingPipelinesKHR create_ray_tracing_pipelines = nullptr;
    PFN_vkGetRayTracingShaderGroupHandlesKHR get_ray_tracing_shader_group_handles = nullptr;
    PFN_vkCmdTraceRaysKHR cmd_trace_rays = nullptr;
    rt_rhi_diagnostics diagnostics{};
};

class vulkan_rhi_device final :
    public rt_rhi_device,
    public rt_native_texture_extension,
    public rt_vulkan_interop_extension
{
public:
    vulkan_rhi_device();
    ~vulkan_rhi_device() override;

    rt_rhi_device_info info() const override;
    rt_device* device() override;
    rt_native_texture_extension* native_texture_extension() override;
    rt_vulkan_interop_extension* vulkan_interop_extension() override;
    bool initialize(
        const rt_rhi_device_desc &desc,
        rt_device_error* out_error) override;
    bool shutdown(rt_device_error* out_error) override;
    bool wait_idle(
        rt_device_timing* out_timing,
        rt_device_error* out_error) override;
    bool begin_commands(
        rt_queue_class queue,
        rt_command_encoder* out_encoder,
        rt_device_error* out_error) override;
    bool submit_commands(
        rt_command_encoder encoder,
        rt_submission_token* out_submission,
        rt_device_timing* out_timing,
        rt_device_error* out_error) override;
    void discard_commands(
        rt_command_encoder encoder) override;
    bool is_complete(
        rt_submission_token submission,
        bool* out_complete,
        rt_device_error* out_error) override;
    bool wait(
        rt_submission_token submission,
        rt_device_timing* out_timing,
        rt_device_error* out_error) override;
    bool barrier(
        rt_command_encoder encoder,
        const rt_resource_barrier* barriers,
        std::size_t barrier_count,
        rt_device_error* out_error) override;
    bool copy_buffer(
        rt_command_encoder encoder,
        rt_buffer_handle source,
        rt_buffer_handle destination,
        const rt_buffer_copy_region &region,
        rt_device_error* out_error) override;
    bool copy_texture_to_buffer(
        rt_command_encoder encoder,
        rt_texture_handle source,
        rt_buffer_handle destination,
        const rt_texture_buffer_copy_region &region,
        rt_device_error* out_error) override;
    bool clear_texture(
        rt_command_encoder encoder,
        rt_texture_handle texture,
        const float color[4],
        rt_device_error* out_error) override;
    bool trace_rays(
        rt_command_encoder encoder,
        const rt_trace_rays_desc &desc,
        rt_device_error* out_error) override;
    bool create_buffer(
        const rt_buffer_desc &desc,
        rt_buffer_handle* out_buffer,
        rt_device_error* out_error) override;
    bool upload_buffer(
        rt_buffer_handle buffer,
        std::size_t offset,
        const void* data,
        std::size_t size,
        rt_device_error* out_error) override;
    bool read_buffer(
        rt_buffer_handle buffer,
        std::size_t offset,
        void* data,
        std::size_t size,
        rt_device_error* out_error) override;
    void destroy_buffer(rt_buffer_handle buffer) override;
    bool create_texture(
        const rt_texture_desc &desc,
        rt_texture_handle* out_texture,
        rt_device_error* out_error) override;
    void destroy_texture(rt_texture_handle texture) override;
    bool get_texture_copy_footprint(
        rt_texture_handle texture,
        rt_texture_copy_footprint* out_footprint,
        rt_device_error* out_error) override;
    bool create_blas(
        rt_blas_handle* out_blas,
        rt_device_error* out_error) override;
    void destroy_blas(rt_blas_handle blas) override;
    bool create_tlas(
        rt_tlas_handle* out_tlas,
        rt_device_error* out_error) override;
    void destroy_tlas(rt_tlas_handle tlas) override;
    bool build_blas(
        rt_command_encoder encoder,
        const rt_blas_build_desc &desc,
        rt_blas_build_result* out_result,
        rt_device_error* out_error) override;
    bool build_tlas(
        rt_command_encoder encoder,
        const rt_tlas_build_desc &desc,
        rt_device_error* out_error) override;
    bool update_bindings(
        const rt_binding_update_request &request,
        rt_device_error* out_error) override;
    bool create_shader_module(
        const rt_shader_module_desc &desc,
        rt_shader_module_handle* out_module,
        rt_device_error* out_error) override;
    void destroy_shader_module(
        rt_shader_module_handle module) override;
    bool create_pipeline(
        const rt_pipeline_desc &desc,
        rt_pipeline_handle* out_pipeline,
        rt_device_error* out_error) override;
    bool create_shader_table(
        const rt_shader_table_desc &desc,
        rt_shader_table_handle* out_shader_table,
        rt_device_error* out_error) override;
    void get_diagnostics(rt_rhi_diagnostics* out_diagnostics) const override;
    bool publish_texture(
        rt_texture_handle texture,
        const rt_native_texture_publish_desc &desc,
        rt_device_timing* out_timing,
        rt_device_error* out_error) override;
    bool get_interop(vulkan_renderer_interop* out_interop) override;

private:
    vulkan_backend_state native_state_{};
    rt_device device_{};
};

enum class timestamp_query_region : std::uint32_t {
    accel = 0,
    dispatch = 2,
};

bool timestamp_queries_enabled(const vulkan_backend_state &state);
void reset_timestamp_query_region(
    vulkan_backend_state &state,
    std::uint32_t slot_index,
    timestamp_query_region region);
void write_timestamp_query_begin(
    vulkan_backend_state &state,
    std::uint32_t slot_index,
    timestamp_query_region region);
void write_timestamp_query_end(
    vulkan_backend_state &state,
    std::uint32_t slot_index,
    timestamp_query_region region);
bool read_timestamp_query_ms(
    vulkan_backend_state &state,
    std::uint32_t slot_index,
    timestamp_query_region region,
    double* out_ms);
bool begin_command_buffer(vulkan_backend_state &state);
bool submit_command_buffer(
    vulkan_backend_state &state,
    double* out_submit_cpu_ms,
    rt_submission_token* out_submission = nullptr);
bool wait_for_all_command_slots(vulkan_backend_state &state);

#if defined(_WIN32)
constexpr std::array<const char*, 9> kRequiredDeviceExtensions = {
#else
constexpr std::array<const char*, 9> kRequiredDeviceExtensions = {
#endif
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
    VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
    VK_KHR_SPIRV_1_4_EXTENSION_NAME,
    VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME,
    VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
    VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

rt_rhi_device_info vulkan_backend_info_locked(const vulkan_backend_state &state) {
    return {
        rt_rhi_backend_kind::vulkan_rt,
        "vulkan_rt",
        state.hardware_ray_tracing,
    };
}

// =============================================================================
// Resource lifetime and allocation.
// =============================================================================

void destroy_native_buffer(vulkan_backend_state &state, vulkan_buffer* buffer) {
    if (buffer == nullptr || state.device == VK_NULL_HANDLE) {
        return;
    }
    if (buffer->buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(state.device, buffer->buffer, nullptr);
        buffer->buffer = VK_NULL_HANDLE;
    }
    if (buffer->memory != VK_NULL_HANDLE) {
        vkFreeMemory(state.device, buffer->memory, nullptr);
        buffer->memory = VK_NULL_HANDLE;
    }
    buffer->size = 0;
}

void destroy_native_texture(vulkan_backend_state &state, vulkan_texture* texture) {
    if (texture == nullptr || state.device == VK_NULL_HANDLE) {
        return;
    }
    if (texture->view != VK_NULL_HANDLE) {
        vkDestroyImageView(state.device, texture->view, nullptr);
    }
    if (texture->image != VK_NULL_HANDLE) {
        vkDestroyImage(state.device, texture->image, nullptr);
    }
    if (texture->memory != VK_NULL_HANDLE) {
        vkFreeMemory(state.device, texture->memory, nullptr);
    }
    *texture = {};
}

void destroy_acceleration_structure(vulkan_backend_state &state, vulkan_acceleration_structure* accel) {
    if (accel == nullptr || state.device == VK_NULL_HANDLE) {
        return;
    }
    if (accel->handle != VK_NULL_HANDLE && state.destroy_acceleration_structure != nullptr) {
        state.destroy_acceleration_structure(state.device, accel->handle, nullptr);
        accel->handle = VK_NULL_HANDLE;
    }
    destroy_native_buffer(state, &accel->storage);
    accel->device_address = 0;
}

rt_submission_token latest_submission(const vulkan_backend_state &state) {
    return state.next_submission_serial > 1
        ? rt_submission_token{state.next_submission_serial - 1u}
        : rt_submission_token{};
}

void defer_native_buffer_release(vulkan_backend_state &state, vulkan_buffer* buffer) {
    if (buffer == nullptr || buffer->buffer == VK_NULL_HANDLE) {
        return;
    }
    const rt_submission_token submission = latest_submission(state);
    if (!submission || submission.serial <= state.completed_submission_serial) {
        destroy_native_buffer(state, buffer);
        return;
    }
    state.deferred_buffer_releases.push_back({submission, *buffer});
    *buffer = {};
}

void defer_acceleration_release(
    vulkan_backend_state &state,
    vulkan_acceleration_structure* acceleration)
{
    if (acceleration == nullptr || acceleration->handle == VK_NULL_HANDLE) {
        return;
    }
    const rt_submission_token submission = latest_submission(state);
    if (!submission || submission.serial <= state.completed_submission_serial) {
        destroy_acceleration_structure(state, acceleration);
        return;
    }
    state.deferred_acceleration_releases.push_back({submission, *acceleration});
    *acceleration = {};
}

void defer_texture_release(vulkan_backend_state &state, vulkan_texture* texture) {
    if (texture == nullptr || texture->image == VK_NULL_HANDLE) {
        return;
    }
    const rt_submission_token submission = latest_submission(state);
    if (!submission || submission.serial <= state.completed_submission_serial) {
        destroy_native_texture(state, texture);
        return;
    }
    state.deferred_texture_releases.push_back({submission, *texture});
    *texture = {};
}

void collect_deferred_releases(vulkan_backend_state &state, bool release_all = false) {
    for (std::size_t index = 0; index < state.deferred_buffer_releases.size();) {
        vulkan_deferred_buffer_release &release = state.deferred_buffer_releases[index];
        if (!release_all && release.submission.serial > state.completed_submission_serial) {
            ++index;
            continue;
        }
        destroy_native_buffer(state, &release.buffer);
        release = state.deferred_buffer_releases.back();
        state.deferred_buffer_releases.pop_back();
    }
    for (std::size_t index = 0; index < state.deferred_acceleration_releases.size();) {
        vulkan_deferred_acceleration_release &release =
            state.deferred_acceleration_releases[index];
        if (!release_all && release.submission.serial > state.completed_submission_serial) {
            ++index;
            continue;
        }
        destroy_acceleration_structure(state, &release.acceleration);
        release = state.deferred_acceleration_releases.back();
        state.deferred_acceleration_releases.pop_back();
    }
    for (std::size_t index = 0; index < state.deferred_texture_releases.size();) {
        vulkan_deferred_texture_release &release = state.deferred_texture_releases[index];
        if (!release_all && release.submission.serial > state.completed_submission_serial) {
            ++index;
            continue;
        }
        destroy_native_texture(state, &release.texture);
        release = state.deferred_texture_releases.back();
        state.deferred_texture_releases.pop_back();
    }
}

void destroy_tlas_handle(vulkan_backend_state &state) {
    vulkan_acceleration_structure object{};
    if (state.tlas_registry.erase(state.tlas, &object)) {
        defer_acceleration_release(state, &object);
    }
    state.tlas = {};
}

void destroy_native_shader_module(vulkan_backend_state &state, VkShaderModule* shader_module) {
    if (shader_module == nullptr || *shader_module == VK_NULL_HANDLE || state.device == VK_NULL_HANDLE) {
        return;
    }
    vkDestroyShaderModule(state.device, *shader_module, nullptr);
    *shader_module = VK_NULL_HANDLE;
}

void reset_vulkan_state_locked(vulkan_backend_state &state) {
    state.command_pool = VK_NULL_HANDLE;
    state.command_buffer = VK_NULL_HANDLE;
    state.command_fence = VK_NULL_HANDLE;
    if (state.timestamp_query_pool != VK_NULL_HANDLE && state.device != VK_NULL_HANDLE) {
        vkDestroyQueryPool(state.device, state.timestamp_query_pool, nullptr);
        state.timestamp_query_pool = VK_NULL_HANDLE;
    }
    if (state.device != VK_NULL_HANDLE) {
        for (vulkan_command_slot &slot : state.command_slots) {
            if (slot.fence != VK_NULL_HANDLE) {
                vkDestroyFence(state.device, slot.fence, nullptr);
            }
            if (slot.pool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(state.device, slot.pool, nullptr);
            }
            slot = {};
        }
    }
    state.buffer_registry.clear([&state](vulkan_buffer &buffer) {
        destroy_native_buffer(state, &buffer);
    });
    state.texture_registry.clear([&state](vulkan_texture &texture) {
        destroy_native_texture(state, &texture);
    });
    state.shader_table_registry.clear([](VkBuffer &) {});
    state.pipeline_registry.clear([](VkPipeline &) {});
    state.shader_table_handle = {};
    state.pipeline_handle = {};
    destroy_native_buffer(state, &state.shader_binding_table);
    state.raygen_regions.clear();
    state.miss_region = {};
    state.hit_region = {};
    state.callable_region = {};
    state.pipeline_chunk_count = 0;
    if (state.pipeline != VK_NULL_HANDLE && state.device != VK_NULL_HANDLE) {
        vkDestroyPipeline(state.device, state.pipeline, nullptr);
        state.pipeline = VK_NULL_HANDLE;
    }
    if (state.pipeline_layout != VK_NULL_HANDLE && state.device != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(state.device, state.pipeline_layout, nullptr);
        state.pipeline_layout = VK_NULL_HANDLE;
    }
    if (state.descriptor_pool != VK_NULL_HANDLE && state.device != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(state.device, state.descriptor_pool, nullptr);
        state.descriptor_pool = VK_NULL_HANDLE;
    }
    if (state.descriptor_set_layout != VK_NULL_HANDLE && state.device != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(state.device, state.descriptor_set_layout, nullptr);
        state.descriptor_set_layout = VK_NULL_HANDLE;
    }
    destroy_tlas_handle(state);
    state.tlas_registry.clear([&state](vulkan_acceleration_structure &accel) {
        destroy_acceleration_structure(state, &accel);
    });
    state.tlas_instance_count = 0;
    state.blas_registry.clear([&state](vulkan_acceleration_structure &accel) {
        destroy_acceleration_structure(state, &accel);
    });
    collect_deferred_releases(state, true);
    destroy_native_buffer(state, &state.scratch_buffer);
    destroy_native_buffer(state, &state.tlas_instance_buffer);
    state.shader_module_registry.clear([&state](VkShaderModule &shader_module) {
        destroy_native_shader_module(state, &shader_module);
    });
    if (state.device != VK_NULL_HANDLE) {
        vkDestroyDevice(state.device, nullptr);
        state.device = VK_NULL_HANDLE;
    }
    if (state.instance != VK_NULL_HANDLE) {
        vkDestroyInstance(state.instance, nullptr);
        state.instance = VK_NULL_HANDLE;
    }
    state.initialized = false;
    state.hardware_ray_tracing = false;
    state.physical_device = VK_NULL_HANDLE;
    state.graphics_queue = VK_NULL_HANDLE;
    state.graphics_queue_family = VK_QUEUE_FAMILY_IGNORED;
    state.present_queue_family = VK_QUEUE_FAMILY_IGNORED;
    state.rt_pipeline_properties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
    state.accel_properties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
    state.diagnostics = {};
}

bool check_device_extension_support(VkPhysicalDevice physical_device) {
    std::uint32_t extension_count = 0;
    if (vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, nullptr) != VK_SUCCESS) {
        return false;
    }
    std::vector<VkExtensionProperties> extensions(extension_count);
    if (extension_count > 0 &&
        vkEnumerateDeviceExtensionProperties(
            physical_device,
            nullptr,
            &extension_count,
            extensions.data()) != VK_SUCCESS) {
        return false;
    }

    for (const char* required_extension : kRequiredDeviceExtensions) {
        const auto match = std::find_if(
            extensions.begin(),
            extensions.end(),
            [required_extension](const VkExtensionProperties &extension) {
                return std::strcmp(extension.extensionName, required_extension) == 0;
            });
        if (match == extensions.end()) {
            return false;
        }
    }
    return true;
}

bool choose_physical_device(vulkan_backend_state &state) {
    bool require_matching_adapter = false;
    const bool presentation_required = viewer_shell::native_window().value != nullptr;
#if defined(_WIN32)
    LUID target_adapter_luid{};
    DXGI_ADAPTER_DESC1 preferred_adapter_desc{};
    if (query_preferred_dxgi_adapter(&target_adapter_luid, &preferred_adapter_desc)) {
        require_matching_adapter = true;
        char line[512]{};
        std::snprintf(
            line,
            sizeof(line),
            "Vulkan startup preferred DXGI adapter: source=dxgi_preferred name=%ws "
            "vendor=0x%04x device=0x%04x luid=%08x:%08x",
            preferred_adapter_desc.Description,
            static_cast<unsigned>(preferred_adapter_desc.VendorId),
            static_cast<unsigned>(preferred_adapter_desc.DeviceId),
            static_cast<unsigned>(preferred_adapter_desc.AdapterLuid.HighPart),
            static_cast<unsigned>(preferred_adapter_desc.AdapterLuid.LowPart));
        append_rt_startup_log(line);
    }
#endif

    VkPhysicalDevice best_physical_device = VK_NULL_HANDLE;
    std::uint32_t best_queue_family_index = VK_QUEUE_FAMILY_IGNORED;
    std::uint32_t best_present_queue_family_index = VK_QUEUE_FAMILY_IGNORED;
    std::uint32_t best_timestamp_valid_bits = 0;
    int best_score = -1;
    bool found = false;
    bool found_matching_adapter = false;

    const auto enumerate_candidates = [&]() -> bool {
        best_physical_device = VK_NULL_HANDLE;
        best_queue_family_index = VK_QUEUE_FAMILY_IGNORED;
        best_present_queue_family_index = VK_QUEUE_FAMILY_IGNORED;
        best_score = -1;
        found = false;
        found_matching_adapter = false;

        std::uint32_t physical_device_count = 0;
        if (vkEnumeratePhysicalDevices(state.instance, &physical_device_count, nullptr) != VK_SUCCESS ||
            physical_device_count == 0) {
            return false;
        }

        std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
        if (vkEnumeratePhysicalDevices(state.instance, &physical_device_count, physical_devices.data()) != VK_SUCCESS) {
            return false;
        }

        for (VkPhysicalDevice physical_device : physical_devices) {
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(physical_device, &properties);

            VkPhysicalDeviceIDProperties id_properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
            VkPhysicalDeviceProperties2 properties2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            properties2.pNext = &id_properties;
            vkGetPhysicalDeviceProperties2(physical_device, &properties2);

            bool adapter_matches = false;
            bool adapter_visible_in_dxgi = true;
#if defined(_WIN32)
            if (require_matching_adapter && id_properties.deviceLUIDValid) {
                adapter_matches =
                    std::memcmp(id_properties.deviceLUID, &target_adapter_luid, VK_LUID_SIZE) == 0;
            }
            if (id_properties.deviceLUIDValid) {
                DXGI_ADAPTER_DESC1 adapter_desc{};
                adapter_visible_in_dxgi = query_dxgi_adapter_desc_from_luid(id_properties.deviceLUID, &adapter_desc);
                if (adapter_visible_in_dxgi && (adapter_desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0) {
                    adapter_visible_in_dxgi = false;
                }
            }
#endif
            if (!check_device_extension_support(physical_device)) {
                continue;
            }

            std::uint32_t queue_family_count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);
            if (queue_family_count == 0) {
                continue;
            }

            std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
            vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families.data());

            VkPhysicalDeviceAccelerationStructureFeaturesKHR accel_features{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
            VkPhysicalDeviceRayTracingPipelineFeaturesKHR rt_pipeline_features{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
            VkPhysicalDeviceBufferDeviceAddressFeatures bda_features{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
            VkPhysicalDeviceFeatures2 device_features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            device_features.pNext = &accel_features;
            accel_features.pNext = &rt_pipeline_features;
            rt_pipeline_features.pNext = &bda_features;
            vkGetPhysicalDeviceFeatures2(physical_device, &device_features);
            if (!accel_features.accelerationStructure ||
                !rt_pipeline_features.rayTracingPipeline ||
                !bda_features.bufferDeviceAddress) {
                continue;
            }

            if (require_matching_adapter && (!adapter_visible_in_dxgi || !adapter_matches)) {
                continue;
            }

            for (std::uint32_t queue_family_index = 0; queue_family_index < queue_family_count; ++queue_family_index) {
                if (queue_families[queue_family_index].queueCount == 0 ||
                    (queue_families[queue_family_index].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) {
                    continue;
                }

                const bool graphics_queue_presents =
                    presentation_required &&
                    viewer_shell::vulkan_presentation_supported(
                        state.instance,
                        physical_device,
                        queue_family_index);
                std::uint32_t present_queue_family_index = graphics_queue_presents
                    ? queue_family_index
                    : VK_QUEUE_FAMILY_IGNORED;
                if (presentation_required && !graphics_queue_presents) {
                    for (std::uint32_t present_index = 0; present_index < queue_family_count; ++present_index) {
                        if (queue_families[present_index].queueCount == 0) {
                            continue;
                        }
                        if (viewer_shell::vulkan_presentation_supported(
                                state.instance,
                                physical_device,
                                present_index)) {
                            present_queue_family_index = present_index;
                            break;
                        }
                    }
                }
                if (presentation_required && present_queue_family_index == VK_QUEUE_FAMILY_IGNORED) {
                    continue;
                }

                append_rt_startup_log(
                    std::string("Vulkan candidate device: name=") +
                    properties.deviceName +
                    " vendor=0x" + [&properties]() {
                        char buffer[16]{};
                        std::snprintf(buffer, sizeof(buffer), "%04x", properties.vendorID);
                        return std::string(buffer);
                    }() +
                    " device=0x" + [&properties]() {
                        char buffer[16]{};
                        std::snprintf(buffer, sizeof(buffer), "%04x", properties.deviceID);
                        return std::string(buffer);
                    }() +
                    " type=" + std::to_string(static_cast<int>(properties.deviceType)) +
                    " queue_family=" + std::to_string(queue_family_index) +
                    " present_queue_family=" + std::to_string(present_queue_family_index) +
                    " presentation_required=" + std::to_string(presentation_required ? 1 : 0) +
                    " graphics_queue_presents=" + std::to_string(graphics_queue_presents ? 1 : 0) +
                    " dxgi_visible=" + std::to_string(adapter_visible_in_dxgi ? 1 : 0) +
                    " adapter_match=" + std::to_string(adapter_matches ? 1 : 0));

                int score = 0;
                switch (properties.deviceType) {
                case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                    score += 1000;
                    break;
                case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
                    score += 500;
                    break;
                case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
                    score += 250;
                    break;
                case VK_PHYSICAL_DEVICE_TYPE_CPU:
                    score += 100;
                    break;
                default:
                    break;
                }
                if (graphics_queue_presents) {
                    score += 100;
                }
                if (adapter_matches) {
                    score += 10000;
                    found_matching_adapter = true;
                }
                if (!found || score > best_score) {
                    best_physical_device = physical_device;
                    best_queue_family_index = queue_family_index;
                    best_present_queue_family_index = present_queue_family_index;
                    best_timestamp_valid_bits = queue_families[queue_family_index].timestampValidBits;
                    best_score = score;
                    found = true;
                }
                break;
            }
        }
        return found;
    };

    const bool initial_enumeration_found = enumerate_candidates();
    if (!initial_enumeration_found && !require_matching_adapter) {
        return false;
    }
    if (!found) {
        if (require_matching_adapter) {
            append_rt_startup_log(
                "Vulkan startup failed: no RT-capable Vulkan device matched the preferred DXGI adapter");
        }
        return false;
    }

    VkPhysicalDeviceProperties best_properties{};
    vkGetPhysicalDeviceProperties(best_physical_device, &best_properties);
    state.physical_device = best_physical_device;
    state.graphics_queue_family = best_queue_family_index;
    state.present_queue_family = best_present_queue_family_index;
    state.hardware_ray_tracing = true;
    state.timestamp_queries_supported = best_timestamp_valid_bits != 0;
    append_rt_startup_log(
        std::string("Vulkan selected device: name=") +
        best_properties.deviceName +
        " vendor=0x" + [&best_properties]() {
            char buffer[16]{};
            std::snprintf(buffer, sizeof(buffer), "%04x", best_properties.vendorID);
            return std::string(buffer);
        }() +
        " device=0x" + [&best_properties]() {
            char buffer[16]{};
            std::snprintf(buffer, sizeof(buffer), "%04x", best_properties.deviceID);
            return std::string(buffer);
        }() +
        " type=" + std::to_string(static_cast<int>(best_properties.deviceType)) +
        " queue_family=" + std::to_string(best_queue_family_index) +
        " present_queue_family=" + std::to_string(best_present_queue_family_index) +
        " score=" + std::to_string(best_score));
    return true;
}

bool load_rt_functions(vulkan_backend_state &state) {
    state.create_acceleration_structure =
        reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(
            vkGetDeviceProcAddr(state.device, "vkCreateAccelerationStructureKHR"));
    state.destroy_acceleration_structure =
        reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(
            vkGetDeviceProcAddr(state.device, "vkDestroyAccelerationStructureKHR"));
    state.get_acceleration_structure_build_sizes =
        reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(
            vkGetDeviceProcAddr(state.device, "vkGetAccelerationStructureBuildSizesKHR"));
    state.cmd_build_acceleration_structures =
        reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(
            vkGetDeviceProcAddr(state.device, "vkCmdBuildAccelerationStructuresKHR"));
    state.get_acceleration_structure_device_address =
        reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(
            vkGetDeviceProcAddr(state.device, "vkGetAccelerationStructureDeviceAddressKHR"));
    state.create_ray_tracing_pipelines =
        reinterpret_cast<PFN_vkCreateRayTracingPipelinesKHR>(
            vkGetDeviceProcAddr(state.device, "vkCreateRayTracingPipelinesKHR"));
    state.get_ray_tracing_shader_group_handles =
        reinterpret_cast<PFN_vkGetRayTracingShaderGroupHandlesKHR>(
            vkGetDeviceProcAddr(state.device, "vkGetRayTracingShaderGroupHandlesKHR"));
    state.cmd_trace_rays =
        reinterpret_cast<PFN_vkCmdTraceRaysKHR>(
            vkGetDeviceProcAddr(state.device, "vkCmdTraceRaysKHR"));

    return state.create_acceleration_structure != nullptr &&
        state.destroy_acceleration_structure != nullptr &&
        state.get_acceleration_structure_build_sizes != nullptr &&
        state.cmd_build_acceleration_structures != nullptr &&
        state.get_acceleration_structure_device_address != nullptr &&
        state.create_ray_tracing_pipelines != nullptr &&
        state.get_ray_tracing_shader_group_handles != nullptr &&
        state.cmd_trace_rays != nullptr;
}

bool create_logical_device(vulkan_backend_state &state) {
    const float queue_priority = 1.0f;
    std::array<VkDeviceQueueCreateInfo, 2> queue_infos{};
    queue_infos[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_infos[0].queueFamilyIndex = state.graphics_queue_family;
    queue_infos[0].queueCount = 1;
    queue_infos[0].pQueuePriorities = &queue_priority;
    std::uint32_t queue_info_count = 1;
    if (state.present_queue_family != VK_QUEUE_FAMILY_IGNORED &&
        state.present_queue_family != state.graphics_queue_family) {
        queue_infos[1].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_infos[1].queueFamilyIndex = state.present_queue_family;
        queue_infos[1].queueCount = 1;
        queue_infos[1].pQueuePriorities = &queue_priority;
        queue_info_count = 2;
    }

    VkPhysicalDeviceAccelerationStructureFeaturesKHR accel_features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    accel_features.accelerationStructure = VK_TRUE;
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rt_pipeline_features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
    rt_pipeline_features.rayTracingPipeline = VK_TRUE;
    rt_pipeline_features.pNext = &accel_features;
    VkPhysicalDeviceBufferDeviceAddressFeatures bda_features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    bda_features.bufferDeviceAddress = VK_TRUE;
    bda_features.pNext = &rt_pipeline_features;
    VkPhysicalDeviceSynchronization2Features sync2_features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES};
    sync2_features.synchronization2 = VK_TRUE;
    sync2_features.pNext = &bda_features;

    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.queueCreateInfoCount = queue_info_count;
    device_info.pQueueCreateInfos = queue_infos.data();
    device_info.enabledExtensionCount = static_cast<std::uint32_t>(kRequiredDeviceExtensions.size());
    device_info.ppEnabledExtensionNames = kRequiredDeviceExtensions.data();
    device_info.pNext = &sync2_features;
    if (vkCreateDevice(state.physical_device, &device_info, nullptr, &state.device) != VK_SUCCESS) {
        return false;
    }

    vkGetDeviceQueue(state.device, state.graphics_queue_family, 0, &state.graphics_queue);
    if (state.graphics_queue == VK_NULL_HANDLE || !load_rt_functions(state)) {
        return false;
    }

    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    properties.pNext = &state.rt_pipeline_properties;
    state.rt_pipeline_properties.pNext = &state.accel_properties;
    vkGetPhysicalDeviceProperties2(state.physical_device, &properties);
    state.timestamp_period_ns = properties.properties.limits.timestampPeriod;
    return true;
}

std::uint32_t find_memory_type_index(
    vulkan_backend_state &state,
    std::uint32_t type_bits,
    VkMemoryPropertyFlags required_flags)
{
    VkPhysicalDeviceMemoryProperties memory_properties{};
    vkGetPhysicalDeviceMemoryProperties(state.physical_device, &memory_properties);
    for (std::uint32_t memory_index = 0; memory_index < memory_properties.memoryTypeCount; ++memory_index) {
        const bool supported = (type_bits & (1u << memory_index)) != 0u;
        const bool matches =
            (memory_properties.memoryTypes[memory_index].propertyFlags & required_flags) == required_flags;
        if (supported && matches) {
            return memory_index;
        }
    }
    return UINT32_MAX;
}

bool create_native_buffer(
    vulkan_backend_state &state,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags memory_flags,
    vulkan_buffer* out_buffer)
{
    if (out_buffer == nullptr) {
        return false;
    }
    defer_native_buffer_release(state, out_buffer);

    VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(state.device, &buffer_info, nullptr, &out_buffer->buffer) != VK_SUCCESS) {
        return false;
    }

    VkMemoryRequirements memory_requirements{};
    vkGetBufferMemoryRequirements(state.device, out_buffer->buffer, &memory_requirements);
    const std::uint32_t memory_index = find_memory_type_index(state, memory_requirements.memoryTypeBits, memory_flags);
    if (memory_index == UINT32_MAX) {
        return false;
    }

    VkMemoryAllocateFlagsInfo allocate_flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    if ((usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0u) {
        allocate_flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    }

    VkMemoryAllocateInfo allocate_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocate_info.allocationSize = memory_requirements.size;
    allocate_info.memoryTypeIndex = memory_index;
    if (allocate_flags.flags != 0u) {
        allocate_info.pNext = &allocate_flags;
    }
    if (vkAllocateMemory(state.device, &allocate_info, nullptr, &out_buffer->memory) != VK_SUCCESS) {
        return false;
    }
    if (vkBindBufferMemory(state.device, out_buffer->buffer, out_buffer->memory, 0) != VK_SUCCESS) {
        return false;
    }
    out_buffer->size = size;
    return true;
}

bool upload_buffer_data(
    vulkan_backend_state &state,
    vulkan_buffer* buffer,
    std::size_t offset,
    const void* data,
    std::size_t size)
{
    if (buffer == nullptr || buffer->memory == VK_NULL_HANDLE ||
        offset > buffer->size || size > buffer->size - offset) {
        return false;
    }
    if (size == 0) {
        return true;
    }
    void* mapped = nullptr;
    if (vkMapMemory(state.device, buffer->memory, offset, size, 0, &mapped) != VK_SUCCESS ||
        mapped == nullptr) {
        return false;
    }
    std::memcpy(mapped, data, size);
    vkUnmapMemory(state.device, buffer->memory);
    return true;
}

VkDeviceSize normalized_buffer_size(VkDeviceSize size) {
    return size == 0 ? VkDeviceSize{16} : size;
}

bool ensure_uploaded_buffer(
    vulkan_backend_state &state,
    vulkan_buffer* buffer,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags memory_flags,
    const void* data,
    std::size_t data_size)
{
    if (buffer == nullptr) {
        return false;
    }
    const VkDeviceSize allocation_size = normalized_buffer_size(size);
    if (buffer->buffer == VK_NULL_HANDLE || buffer->size < allocation_size) {
        if (!create_native_buffer(state, allocation_size, usage, memory_flags, buffer)) {
            return false;
        }
    }
    return upload_buffer_data(state, buffer, 0, data, data_size);
}

vulkan_buffer* vulkan_api_buffer(vulkan_backend_state &state, rt_buffer_handle handle) {
    return state.buffer_registry.get(handle);
}

bool vulkan_rhi_device::create_buffer(
    const rt_buffer_desc &desc,
    rt_buffer_handle* out_buffer,
    rt_device_error* out_error)
{
    if (out_buffer != nullptr) {
        *out_buffer = {};
    }
    if (device_.native_state != &native_state_ || out_buffer == nullptr) {
        if (out_error != nullptr) {
            out_error->detail = "Vulkan buffer request is invalid";
        }
        return false;
    }

    vulkan_buffer buffer{};
    VkBufferUsageFlags usage = 0;
    if ((desc.usage & (rt_buffer_usage_shader_read | rt_buffer_usage_shader_write)) != 0u) {
        usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    }
    if ((desc.usage & rt_buffer_usage_uniform) != 0u) {
        usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    }
    if ((desc.usage & rt_buffer_usage_copy_source) != 0u) {
        usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    }
    if ((desc.usage & rt_buffer_usage_copy_destination) != 0u) {
        usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    }
    if ((desc.usage & rt_buffer_usage_acceleration_build_input) != 0u) {
        usage |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    }
    if ((desc.usage & rt_buffer_usage_device_address) != 0u) {
        usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }
    const VkMemoryPropertyFlags memory_flags =
        desc.memory_domain == rt_memory_domain::device
            ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            : VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (usage == 0u || !create_native_buffer(
            native_state_,
            normalized_buffer_size(desc.size),
            usage,
            memory_flags,
            &buffer)) {
        destroy_native_buffer(native_state_, &buffer);
        if (out_error != nullptr) {
            out_error->detail = "Vulkan buffer creation failed";
        }
        return false;
    }
    buffer.usage = desc.memory_domain == rt_memory_domain::readback
        ? rt_resource_usage::copy_destination
        : rt_resource_usage::undefined;
    if (!native_state_.buffer_registry.insert(buffer, out_buffer)) {
        destroy_native_buffer(native_state_, &buffer);
        if (out_error != nullptr) {
            out_error->detail = "Vulkan buffer registry allocation failed";
        }
        return false;
    }
    return static_cast<bool>(*out_buffer);
}

bool vulkan_rhi_device::upload_buffer(
    rt_buffer_handle handle,
    std::size_t offset,
    const void* data,
    std::size_t size,
    rt_device_error* out_error)
{
    vulkan_buffer* buffer = vulkan_api_buffer(native_state_, handle);
    const bool uploaded = device_.native_state == &native_state_ && buffer != nullptr &&
        upload_buffer_data(native_state_, buffer, offset, data, size);
    if (!uploaded && out_error != nullptr) {
        out_error->detail = "Vulkan buffer upload failed";
    }
    return uploaded;
}

bool vulkan_rhi_device::read_buffer(
    rt_buffer_handle handle,
    std::size_t offset,
    void* data,
    std::size_t size,
    rt_device_error* out_error)
{
    const vulkan_buffer* const buffer = vulkan_api_buffer(native_state_, handle);
    const bool valid = device_.native_state == &native_state_ &&
        buffer != nullptr && buffer->memory != VK_NULL_HANDLE &&
        data != nullptr && size > 0 &&
        offset <= buffer->size && size <= buffer->size - offset;
    if (!valid) {
        if (out_error != nullptr) {
            *out_error = {rt_device_operation::readback, 0, "Vulkan buffer read request is invalid"};
        }
        return false;
    }
    void* mapped = nullptr;
    const VkResult result = vkMapMemory(
        native_state_.device,
        buffer->memory,
        offset,
        size,
        0,
        &mapped);
    if (result != VK_SUCCESS || mapped == nullptr) {
        if (out_error != nullptr) {
            *out_error = {
                rt_device_operation::readback,
                static_cast<std::int64_t>(result),
                "Vulkan buffer map failed"};
        }
        return false;
    }
    std::memcpy(data, mapped, size);
    vkUnmapMemory(native_state_.device, buffer->memory);
    return true;
}

void vulkan_rhi_device::destroy_buffer(rt_buffer_handle handle) {
    if (device_.native_state != &native_state_) {
        return;
    }
    vulkan_buffer buffer{};
    if (native_state_.buffer_registry.erase(handle, &buffer)) {
        defer_native_buffer_release(native_state_, &buffer);
    }
}

VkFormat vulkan_texture_format(rt_texture_format format) {
    switch (format) {
    case rt_texture_format::rgba8_unorm:
        return VK_FORMAT_R8G8B8A8_UNORM;
    case rt_texture_format::bgra8_unorm:
        return VK_FORMAT_B8G8R8A8_UNORM;
    case rt_texture_format::rgba16_float:
        return VK_FORMAT_R16G16B16A16_SFLOAT;
    case rt_texture_format::rgba32_float:
        return VK_FORMAT_R32G32B32A32_SFLOAT;
    }
    return VK_FORMAT_UNDEFINED;
}

bool vulkan_rhi_device::create_texture(
    const rt_texture_desc &desc,
    rt_texture_handle* out_texture,
    rt_device_error* out_error)
{
    if (out_texture != nullptr) {
        *out_texture = {};
    }
    const VkFormat format = vulkan_texture_format(desc.format);
    if (device_.native_state != &native_state_ ||
        out_texture == nullptr || desc.width == 0 || desc.height == 0 ||
        desc.usage == 0 || format == VK_FORMAT_UNDEFINED) {
        if (out_error != nullptr) {
            out_error->detail = "Vulkan texture request is invalid";
        }
        return false;
    }

    VkImageUsageFlags usage = 0;
    if ((desc.usage & rt_texture_usage_shader_read) != 0u) {
        usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
    }
    if ((desc.usage & rt_texture_usage_shader_write) != 0u) {
        usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    }
    if ((desc.usage & rt_texture_usage_copy_source) != 0u) {
        usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    }
    if ((desc.usage & rt_texture_usage_copy_destination) != 0u) {
        usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    }

    vulkan_texture texture{};
    texture.desc = desc;
    VkImageCreateInfo image_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = format;
    image_info.extent = {desc.width, desc.height, 1u};
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage = usage;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_info.initialLayout = texture.layout;
    if (vkCreateImage(native_state_.device, &image_info, nullptr, &texture.image) != VK_SUCCESS) {
        if (out_error != nullptr) {
            out_error->detail = "Vulkan texture image creation failed";
        }
        return false;
    }

    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(native_state_.device, texture.image, &requirements);
    const std::uint32_t memory_index = find_memory_type_index(
        native_state_,
        requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = memory_index;
    VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view_info.image = texture.image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = format;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.layerCount = 1;
    if (memory_index == UINT32_MAX ||
        vkAllocateMemory(native_state_.device, &allocation, nullptr, &texture.memory) != VK_SUCCESS ||
        vkBindImageMemory(native_state_.device, texture.image, texture.memory, 0) != VK_SUCCESS ||
        vkCreateImageView(native_state_.device, &view_info, nullptr, &texture.view) != VK_SUCCESS ||
        !native_state_.texture_registry.insert(texture, out_texture)) {
        destroy_native_texture(native_state_, &texture);
        if (out_error != nullptr) {
            out_error->detail = "Vulkan texture allocation failed";
        }
        return false;
    }
    return true;
}

void vulkan_rhi_device::destroy_texture(rt_texture_handle handle) {
    if (device_.native_state != &native_state_) {
        return;
    }
    vulkan_texture texture{};
    if (native_state_.texture_registry.erase(handle, &texture)) {
        defer_texture_release(native_state_, &texture);
    }
}

bool vulkan_rhi_device::get_texture_copy_footprint(
    rt_texture_handle handle,
    rt_texture_copy_footprint* out_footprint,
    rt_device_error* out_error)
{
    if (out_footprint != nullptr) {
        *out_footprint = {};
    }
    const vulkan_texture* const texture = native_state_.texture_registry.get(handle);
    if (device_.native_state != &native_state_ ||
        texture == nullptr || out_footprint == nullptr) {
        if (out_error != nullptr) {
            out_error->detail = "Vulkan texture copy footprint request is invalid";
        }
        return false;
    }
    std::size_t pixel_size = 0;
    switch (texture->desc.format) {
    case rt_texture_format::rgba8_unorm:
    case rt_texture_format::bgra8_unorm:
        pixel_size = 4;
        break;
    case rt_texture_format::rgba16_float:
        pixel_size = 8;
        break;
    case rt_texture_format::rgba32_float:
        pixel_size = 16;
        break;
    }
    out_footprint->row_pitch = static_cast<std::size_t>(texture->desc.width) * pixel_size;
    out_footprint->total_size =
        out_footprint->row_pitch * static_cast<std::size_t>(texture->desc.height);
    return out_footprint->total_size != 0;
}

// =============================================================================
// Acceleration structure creation and build recording.
// =============================================================================

bool ensure_acceleration_structure(
    vulkan_backend_state &state,
    vulkan_acceleration_structure* accel,
    VkDeviceSize size,
    VkAccelerationStructureTypeKHR type)
{
    if (accel == nullptr) {
        return false;
    }

    if (accel->handle != VK_NULL_HANDLE && accel->storage.size >= size) {
        return true;
    }

    defer_acceleration_release(state, accel);
    if (!create_native_buffer(
            state,
            size,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            &accel->storage)) {
        return false;
    }

    VkAccelerationStructureCreateInfoKHR accel_create_info{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    accel_create_info.type = type;
    accel_create_info.size = size;
    accel_create_info.buffer = accel->storage.buffer;
    if (state.create_acceleration_structure(state.device, &accel_create_info, nullptr, &accel->handle) !=
        VK_SUCCESS) {
        destroy_acceleration_structure(state, accel);
        return false;
    }

    return true;
}

VkDeviceAddress buffer_device_address(vulkan_backend_state &state, const vulkan_buffer &buffer) {
    VkBufferDeviceAddressInfo address_info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    address_info.buffer = buffer.buffer;
    return vkGetBufferDeviceAddress(state.device, &address_info);
}

bool create_native_shader_module(
    vulkan_backend_state &state,
    const std::uint8_t* bytes,
    std::size_t size,
    VkShaderModule* out_shader_module)
{
    if (bytes == nullptr || size == 0 || out_shader_module == nullptr || (size % 4) != 0) {
        return false;
    }
    VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shader_info.codeSize = size;
    shader_info.pCode = reinterpret_cast<const std::uint32_t*>(bytes);
    return vkCreateShaderModule(state.device, &shader_info, nullptr, out_shader_module) == VK_SUCCESS;
}

bool create_command_objects(vulkan_backend_state &state) {
    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = state.graphics_queue_family;
    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    for (vulkan_command_slot &slot : state.command_slots) {
        if (vkCreateCommandPool(state.device, &pool_info, nullptr, &slot.pool) != VK_SUCCESS) {
            return false;
        }
        VkCommandBufferAllocateInfo command_buffer_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        command_buffer_info.commandPool = slot.pool;
        command_buffer_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_buffer_info.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(state.device, &command_buffer_info, &slot.buffer) != VK_SUCCESS ||
            vkCreateFence(state.device, &fence_info, nullptr, &slot.fence) != VK_SUCCESS) {
            return false;
        }
    }
    if (!state.timestamp_queries_supported) {
        return true;
    }
    VkQueryPoolCreateInfo query_pool_info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    query_pool_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    query_pool_info.queryCount = kRtCommandSlotCount * kTimestampQueryCountPerCommandSlot;
    if (vkCreateQueryPool(state.device, &query_pool_info, nullptr, &state.timestamp_query_pool) != VK_SUCCESS) {
        state.timestamp_queries_supported = false;
    }
    return true;
}

bool ensure_scratch_buffer(vulkan_backend_state &state, VkDeviceSize size) {
    if (state.scratch_buffer.buffer != VK_NULL_HANDLE && state.scratch_buffer.size >= size) {
        return true;
    }
    return create_native_buffer(
        state,
        size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        &state.scratch_buffer);
}

bool begin_acceleration_recording(vulkan_backend_state &state) {
    vulkan_acceleration_build_context &context = state.acceleration_build;
    if (context.active) {
        return !context.recorded;
    }
    context = {};
    context.active = true;
    context.total_start = std::chrono::steady_clock::now();

    state.diagnostics.acceleration_prebuild_query_ms = 0.0;
    state.diagnostics.chunk_blas_prebuild_query_ms = 0.0;
    state.diagnostics.chunk_blas_prebuild_query_count = 0;
    state.diagnostics.grouped_blas_prebuild_query_ms = 0.0;
    state.diagnostics.grouped_blas_prebuild_query_count = 0;
    state.diagnostics.point_blas_prebuild_query_ms = 0.0;
    state.diagnostics.point_blas_prebuild_query_count = 0;
    state.diagnostics.line_blas_prebuild_query_ms = 0.0;
    state.diagnostics.line_blas_prebuild_query_count = 0;
    state.diagnostics.tlas_prebuild_query_ms = 0.0;
    state.diagnostics.tlas_prebuild_query_count = 0;
    state.diagnostics.startup_prebuild_warmup_ms = 0.0;
    state.diagnostics.acceleration_procedural_aabb_ms = 0.0;
    state.diagnostics.acceleration_command_record_ms = 0.0;
    state.diagnostics.acceleration_resource_allocate_ms = 0.0;
    state.diagnostics.acceleration_build_call_record_ms = 0.0;
    state.diagnostics.tlas_instance_upload_ms = 0.0;
    context.command_start = std::chrono::steady_clock::now();
    return true;
}

VkBuildAccelerationStructureFlagsKHR vulkan_acceleration_build_flags(
    std::uint32_t flags)
{
    VkBuildAccelerationStructureFlagsKHR native_flags = 0;
    if ((flags & rt_acceleration_build_prefer_fast_trace) != 0u) {
        native_flags |= VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    }
    if ((flags & rt_acceleration_build_allow_update) != 0u) {
        native_flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    }
    return native_flags;
}

bool build_triangle_blas(
    vulkan_backend_state &state,
    const rt_blas_build_desc &command,
    vulkan_acceleration_structure* entry,
    rt_blas_build_result* out_result)
{
    if (entry == nullptr || out_result == nullptr || command.geometry_count == 0 ||
        command.geometry_count > kRtBlasChunkSetChunkCount) {
        return false;
    }

    vulkan_acceleration_build_context &context = state.acceleration_build;
    const std::size_t geometry_offset = context.geometries.size();
    std::array<std::uint32_t, kRtBlasChunkSetChunkCount> primitive_counts{};
    for (std::size_t geometry_index = 0; geometry_index < command.geometry_count; ++geometry_index) {
        const rt_acceleration_geometry_desc &source = command.geometries[geometry_index];
        const rt_triangle_geometry_desc &triangle = source.triangles;
        const vulkan_buffer* const vertex_buffer =
            vulkan_api_buffer(state, triangle.vertex_buffer);
        const vulkan_buffer* const index_buffer =
            vulkan_api_buffer(state, triangle.index_buffer);
        if (source.type != rt_acceleration_geometry_type::triangles ||
            triangle.vertex_format != rt_vertex_format::float3 ||
            triangle.index_format != rt_index_format::uint32 ||
            vertex_buffer == nullptr || index_buffer == nullptr ||
            triangle.vertex_count == 0 || triangle.index_count == 0 ||
            triangle.index_count % 3u != 0 ||
            triangle.vertex_count > (std::numeric_limits<std::uint32_t>::max)() ||
            triangle.index_count / 3u > (std::numeric_limits<std::uint32_t>::max)()) {
            return false;
        }
        const VkDeviceAddress vertex_address =
            buffer_device_address(state, *vertex_buffer) + triangle.vertex_offset;
        const VkDeviceAddress index_address =
            buffer_device_address(state, *index_buffer) + triangle.index_offset;
        VkAccelerationStructureGeometryTrianglesDataKHR triangles{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
        triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        triangles.vertexData.deviceAddress = vertex_address;
        triangles.vertexStride = triangle.vertex_stride;
        triangles.maxVertex = static_cast<std::uint32_t>(triangle.vertex_count);
        triangles.indexType = VK_INDEX_TYPE_UINT32;
        triangles.indexData.deviceAddress = index_address;
        VkAccelerationStructureGeometryKHR geometry{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geometry.flags = (source.flags & rt_acceleration_geometry_opaque) != 0u
            ? VK_GEOMETRY_OPAQUE_BIT_KHR
            : 0;
        geometry.geometry.triangles = triangles;
        VkAccelerationStructureBuildRangeInfoKHR build_range{};
        build_range.primitiveCount = static_cast<std::uint32_t>(triangle.index_count / 3u);
        primitive_counts[geometry_index] = build_range.primitiveCount;
        context.geometries.push_back(geometry);
        context.build_ranges.push_back(build_range);
    }

    VkAccelerationStructureBuildGeometryInfoKHR build_info{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    build_info.flags = vulkan_acceleration_build_flags(command.flags);
    build_info.geometryCount = static_cast<std::uint32_t>(command.geometry_count);
    build_info.pGeometries = context.geometries.data() + geometry_offset;
    VkAccelerationStructureBuildSizesInfoKHR build_sizes{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    const auto prebuild_start = std::chrono::steady_clock::now();
    state.get_acceleration_structure_build_sizes(
        state.device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &build_info,
        primitive_counts.data(),
        &build_sizes);
    const double prebuild_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - prebuild_start).count();
    state.diagnostics.acceleration_prebuild_query_ms += prebuild_ms;
    if (command.geometry_count == 1) {
        state.diagnostics.chunk_blas_prebuild_query_ms += prebuild_ms;
        ++state.diagnostics.chunk_blas_prebuild_query_count;
    } else {
        state.diagnostics.grouped_blas_prebuild_query_ms += prebuild_ms;
        ++state.diagnostics.grouped_blas_prebuild_query_count;
    }

    const auto alloc_start = std::chrono::steady_clock::now();
    const bool allocated = ensure_acceleration_structure(
        state,
        entry,
        build_sizes.accelerationStructureSize,
        VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR);
    state.diagnostics.acceleration_resource_allocate_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - alloc_start).count();
    if (!allocated) {
        return false;
    }
    build_info.dstAccelerationStructure = entry->handle;
    context.build_infos.push_back(build_info);
    context.build_range_offsets.push_back(geometry_offset);
    context.max_scratch_size =
        (std::max)(context.max_scratch_size, build_sizes.buildScratchSize);

    VkAccelerationStructureDeviceAddressInfoKHR address_info{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    address_info.accelerationStructure = entry->handle;
    entry->device_address =
        state.get_acceleration_structure_device_address(state.device, &address_info);
    out_result->acceleration = command.destination;
    out_result->reused = false;
    return static_cast<bool>(out_result->acceleration);
}

bool build_procedural_blas(
    vulkan_backend_state &state,
    const rt_blas_build_desc &command,
    vulkan_acceleration_structure* entry,
    rt_blas_build_result* out_result)
{
    if (entry == nullptr || out_result == nullptr || command.geometry_count == 0 ||
        command.geometry_count > kRtBlasChunkSetChunkCount) {
        return false;
    }

    vulkan_acceleration_build_context &context = state.acceleration_build;
    const std::size_t geometry_offset = context.geometries.size();
    std::array<std::uint32_t, kRtBlasChunkSetChunkCount> primitive_counts{};
    for (std::size_t geometry_index = 0; geometry_index < command.geometry_count; ++geometry_index) {
        const rt_acceleration_geometry_desc &source = command.geometries[geometry_index];
        const rt_aabb_geometry_desc &aabbs = source.aabbs;
        const vulkan_buffer* const aabb_buffer = vulkan_api_buffer(state, aabbs.buffer);
        if (source.type != rt_acceleration_geometry_type::aabbs || aabb_buffer == nullptr ||
            aabbs.count == 0 || aabbs.count > (std::numeric_limits<std::uint32_t>::max)() ||
            aabbs.offset > (std::numeric_limits<std::uint32_t>::max)()) {
            return false;
        }
        primitive_counts[geometry_index] = static_cast<std::uint32_t>(aabbs.count);
        VkAccelerationStructureGeometryAabbsDataKHR aabb_data{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR};
        aabb_data.data.deviceAddress = buffer_device_address(state, *aabb_buffer);
        aabb_data.stride = aabbs.stride;
        VkAccelerationStructureGeometryKHR geometry{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        geometry.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
        geometry.flags = (source.flags & rt_acceleration_geometry_opaque) != 0u
            ? VK_GEOMETRY_OPAQUE_BIT_KHR
            : 0;
        geometry.geometry.aabbs = aabb_data;
        VkAccelerationStructureBuildRangeInfoKHR build_range{};
        build_range.primitiveCount = primitive_counts[geometry_index];
        build_range.primitiveOffset = static_cast<std::uint32_t>(aabbs.offset);
        context.geometries.push_back(geometry);
        context.build_ranges.push_back(build_range);
    }

    VkAccelerationStructureBuildGeometryInfoKHR build_info{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    build_info.flags = vulkan_acceleration_build_flags(command.flags);
    build_info.geometryCount = static_cast<std::uint32_t>(command.geometry_count);
    build_info.pGeometries = context.geometries.data() + geometry_offset;
    VkAccelerationStructureBuildSizesInfoKHR build_sizes{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    const auto prebuild_start = std::chrono::steady_clock::now();
    state.get_acceleration_structure_build_sizes(
        state.device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &build_info,
        primitive_counts.data(),
        &build_sizes);
    const double prebuild_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - prebuild_start).count();
    state.diagnostics.acceleration_prebuild_query_ms += prebuild_ms;
    out_result->prebuild_info_ms = prebuild_ms;

    const auto alloc_start = std::chrono::steady_clock::now();
    const bool allocated = ensure_acceleration_structure(
        state,
        entry,
        build_sizes.accelerationStructureSize,
        VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR);
    state.diagnostics.acceleration_resource_allocate_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - alloc_start).count();
    if (!allocated) {
        return false;
    }
    build_info.dstAccelerationStructure = entry->handle;
    context.build_infos.push_back(build_info);
    context.build_range_offsets.push_back(geometry_offset);
    context.max_scratch_size =
        (std::max)(context.max_scratch_size, build_sizes.buildScratchSize);

    VkAccelerationStructureDeviceAddressInfoKHR address_info{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    address_info.accelerationStructure = entry->handle;
    entry->device_address =
        state.get_acceleration_structure_device_address(state.device, &address_info);
    out_result->acceleration = command.destination;
    out_result->reused = false;
    return static_cast<bool>(out_result->acceleration);
}

bool vulkan_rhi_device::create_blas(
    rt_blas_handle* out_blas,
    rt_device_error* out_error)
{
    if (out_blas != nullptr) {
        *out_blas = {};
    }
    if (device_.native_state != &native_state_ ||
        out_blas == nullptr || !native_state_.blas_registry.insert({}, out_blas)) {
        if (out_error != nullptr) {
            out_error->detail = "Vulkan BLAS object creation failed";
        }
        return false;
    }
    return true;
}

void vulkan_rhi_device::destroy_blas(rt_blas_handle handle) {
    if (device_.native_state != &native_state_ || !handle) {
        return;
    }
    vulkan_acceleration_structure acceleration{};
    if (native_state_.blas_registry.erase(handle, &acceleration)) {
        defer_acceleration_release(native_state_, &acceleration);
    }
}

bool vulkan_rhi_device::create_tlas(
    rt_tlas_handle* out_tlas,
    rt_device_error* out_error)
{
    if (out_tlas != nullptr) {
        *out_tlas = {};
    }
    if (device_.native_state != &native_state_ ||
        out_tlas == nullptr || native_state_.tlas ||
        !native_state_.tlas_registry.insert({}, out_tlas)) {
        if (out_error != nullptr) {
            out_error->detail = "Vulkan TLAS object creation failed";
        }
        return false;
    }
    native_state_.tlas = *out_tlas;
    return true;
}

void vulkan_rhi_device::destroy_tlas(rt_tlas_handle handle) {
    if (device_.native_state != &native_state_ || !handle) {
        return;
    }
    vulkan_acceleration_structure acceleration{};
    if (native_state_.tlas_registry.erase(handle, &acceleration)) {
        defer_acceleration_release(native_state_, &acceleration);
    }
    if (native_state_.tlas == handle) {
        native_state_.tlas = {};
    }
}

bool vulkan_rhi_device::build_blas(
    rt_command_encoder encoder,
    const rt_blas_build_desc &command,
    rt_blas_build_result* out_result,
    rt_device_error* out_error)
{
    if (out_result != nullptr) {
        *out_result = {};
    }
    if (device_.native_state != &native_state_ || out_result == nullptr ||
        !encoder || encoder.id != native_state_.active_encoder_id ||
        !begin_acceleration_recording(native_state_) ||
        command.geometry_count == 0 || command.geometries == nullptr) {
        if (out_error != nullptr) {
            out_error->detail = "Vulkan BLAS build request is invalid";
        }
        return false;
    }
    vulkan_acceleration_structure* const acceleration =
        native_state_.blas_registry.get(command.destination);
    if (acceleration == nullptr) {
        if (out_error != nullptr) {
            out_error->detail = "Vulkan BLAS build failed";
        }
        return false;
    }
    const rt_acceleration_geometry_type geometry_type =
        command.geometries[0].type;
    const bool built = geometry_type == rt_acceleration_geometry_type::triangles
        ? build_triangle_blas(native_state_, command, acceleration, out_result)
        : geometry_type == rt_acceleration_geometry_type::aabbs
            ? build_procedural_blas(native_state_, command, acceleration, out_result)
            : false;
    if (!built && out_error != nullptr) {
        out_error->detail = "Vulkan BLAS build failed";
    }
    return built;
}

bool vulkan_rhi_device::build_tlas(
    rt_command_encoder encoder,
    const rt_tlas_build_desc &request,
    rt_device_error* out_error)
{
    if (device_.native_state != &native_state_ ||
        !encoder || encoder.id != native_state_.active_encoder_id ||
        !begin_acceleration_recording(native_state_) ||
        (request.instance_count != 0 && request.instances == nullptr)) {
        if (out_error != nullptr) {
            out_error->detail = "Vulkan TLAS build request is invalid";
        }
        return false;
    }
    vulkan_acceleration_build_context &context = native_state_.acceleration_build;
    vulkan_acceleration_structure* const tlas =
        native_state_.tlas_registry.get(request.destination);
    if (tlas == nullptr) {
        return false;
    }
    if (request.instance_count == 0) {
        defer_acceleration_release(native_state_, tlas);
        destroy_native_buffer(native_state_, &native_state_.tlas_instance_buffer);
        native_state_.tlas_instance_count = 0;
        return true;
    }

    std::vector<VkAccelerationStructureInstanceKHR> instances;
    instances.reserve(request.instance_count);
    for (std::size_t instance_index = 0; instance_index < request.instance_count; ++instance_index) {
        const rt_tlas_instance_desc &source = request.instances[instance_index];
        const vulkan_acceleration_structure* const acceleration =
            native_state_.blas_registry.get(source.acceleration);
        if (acceleration == nullptr || acceleration->device_address == 0) {
            return false;
        }
        VkAccelerationStructureInstanceKHR instance{};
        std::memcpy(
            instance.transform.matrix,
            source.transform.data(),
            sizeof(instance.transform.matrix));
        instance.instanceCustomIndex = source.instance_id;
        instance.mask = source.mask;
        instance.instanceShaderBindingTableRecordOffset = source.hit_group_contribution;
        instance.flags = 0;
        if ((source.flags & rt_acceleration_instance_triangle_cull_disable) != 0u) {
            instance.flags |= VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        }
        if ((source.flags & rt_acceleration_instance_triangle_front_counterclockwise) != 0u) {
            instance.flags |= VK_GEOMETRY_INSTANCE_TRIANGLE_FRONT_COUNTERCLOCKWISE_BIT_KHR;
        }
        if ((source.flags & rt_acceleration_instance_force_opaque) != 0u) {
            instance.flags |= VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;
        }
        if ((source.flags & rt_acceleration_instance_force_non_opaque) != 0u) {
            instance.flags |= VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR;
        }
        instance.accelerationStructureReference = acceleration->device_address;
        instances.push_back(instance);
    }

    const auto instance_upload_start = std::chrono::steady_clock::now();
    const bool instance_uploaded = ensure_uploaded_buffer(
        native_state_,
        &native_state_.tlas_instance_buffer,
        instances.size() * sizeof(VkAccelerationStructureInstanceKHR),
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        instances.data(),
        instances.size() * sizeof(VkAccelerationStructureInstanceKHR));
    native_state_.diagnostics.tlas_instance_upload_ms +=
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - instance_upload_start).count();
    if (!instance_uploaded) {
        if (out_error != nullptr) {
            out_error->detail = "Vulkan TLAS input upload failed";
        }
        return false;
    }

    context.top_instances.arrayOfPointers = VK_FALSE;
    context.top_instances.data.deviceAddress = buffer_device_address(native_state_, native_state_.tlas_instance_buffer);
    context.top_geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    context.top_geometry.geometry.instances = context.top_instances;
    context.top_build_range.primitiveCount = static_cast<std::uint32_t>(instances.size());
    context.top_build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    context.top_build_info.flags =
        vulkan_acceleration_build_flags(request.flags);
    context.top_build_info.geometryCount = 1;
    context.top_build_info.pGeometries = &context.top_geometry;

    VkAccelerationStructureBuildSizesInfoKHR top_build_sizes{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    const std::uint32_t primitive_count = static_cast<std::uint32_t>(instances.size());
    const auto prebuild_start = std::chrono::steady_clock::now();
    native_state_.get_acceleration_structure_build_sizes(
        native_state_.device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &context.top_build_info,
        &primitive_count,
        &top_build_sizes);
    const double prebuild_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - prebuild_start).count();
    native_state_.diagnostics.acceleration_prebuild_query_ms += prebuild_ms;
    native_state_.diagnostics.tlas_prebuild_query_ms += prebuild_ms;
    ++native_state_.diagnostics.tlas_prebuild_query_count;

    context.can_update_tlas =
        (request.flags & rt_acceleration_build_allow_update) != 0u &&
        tlas->handle != VK_NULL_HANDLE &&
        native_state_.tlas_instance_count == instances.size() &&
        tlas->storage.size >= top_build_sizes.accelerationStructureSize;
    const auto accel_alloc_start = std::chrono::steady_clock::now();
    const bool acceleration_allocated = ensure_acceleration_structure(
        native_state_,
        tlas,
        top_build_sizes.accelerationStructureSize,
        VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR);
    native_state_.diagnostics.acceleration_resource_allocate_ms +=
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - accel_alloc_start).count();
    if (!acceleration_allocated) {
        return false;
    }
    context.max_scratch_size = (std::max)(
        context.max_scratch_size,
        context.can_update_tlas
            ? top_build_sizes.updateScratchSize
            : top_build_sizes.buildScratchSize);
    const auto scratch_alloc_start = std::chrono::steady_clock::now();
    const bool scratch_allocated = ensure_scratch_buffer(native_state_, context.max_scratch_size);
    native_state_.diagnostics.acceleration_resource_allocate_ms +=
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - scratch_alloc_start).count();
    if (!scratch_allocated) {
        return false;
    }

    reset_timestamp_query_region(
        native_state_,
        native_state_.active_command_slot_index,
        timestamp_query_region::accel);
    write_timestamp_query_begin(
        native_state_,
        native_state_.active_command_slot_index,
        timestamp_query_region::accel);

    const auto build_record_start = std::chrono::steady_clock::now();
    const VkDeviceAddress scratch_address = buffer_device_address(native_state_, native_state_.scratch_buffer);
    for (std::size_t build_index = 0; build_index < context.build_infos.size(); ++build_index) {
        context.build_infos[build_index].pGeometries =
            context.geometries.data() + context.build_range_offsets[build_index];
        context.build_infos[build_index].scratchData.deviceAddress = scratch_address;
        const VkAccelerationStructureBuildRangeInfoKHR* build_ranges[] = {
            context.build_ranges.data() + context.build_range_offsets[build_index]};
        native_state_.cmd_build_acceleration_structures(
            native_state_.command_buffer,
            1,
            &context.build_infos[build_index],
            build_ranges);
        VkMemoryBarrier build_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        build_barrier.srcAccessMask =
            VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
            VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        build_barrier.dstAccessMask =
            VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
            VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        vkCmdPipelineBarrier(
            native_state_.command_buffer,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            0,
            1,
            &build_barrier,
            0,
            nullptr,
            0,
            nullptr);
    }
    if (!context.build_infos.empty()) {
        VkMemoryBarrier memory_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        memory_barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        memory_barrier.dstAccessMask =
            VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(
            native_state_.command_buffer,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            0,
            1,
            &memory_barrier,
            0,
            nullptr,
            0,
            nullptr);
    }

    context.top_build_info.dstAccelerationStructure = tlas->handle;
    if (context.can_update_tlas) {
        context.top_build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
        context.top_build_info.srcAccelerationStructure = tlas->handle;
    }
    context.top_build_info.scratchData.deviceAddress = scratch_address;
    const VkAccelerationStructureBuildRangeInfoKHR* top_ranges[] = {
        &context.top_build_range};
    native_state_.cmd_build_acceleration_structures(
        native_state_.command_buffer,
        1,
        &context.top_build_info,
        top_ranges);
    native_state_.diagnostics.acceleration_build_call_record_ms +=
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - build_record_start).count();
    write_timestamp_query_end(
        native_state_,
        native_state_.active_command_slot_index,
        timestamp_query_region::accel);
    native_state_.diagnostics.acceleration_command_record_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - context.command_start).count();
    context.recorded = true;
    native_state_.tlas_instance_count = instances.size();
    return true;
}

VkDescriptorType vulkan_descriptor_type(rt_descriptor_type type) {
    switch (type) {
    case rt_descriptor_type::acceleration_structure:
        return VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    case rt_descriptor_type::storage_texture:
        return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    case rt_descriptor_type::structured_buffer:
    case rt_descriptor_type::storage_buffer:
        return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    case rt_descriptor_type::uniform_buffer:
        return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    }
    return VK_DESCRIPTOR_TYPE_MAX_ENUM;
}

// =============================================================================
// Pipeline and binding state.
// =============================================================================

bool ensure_descriptor_set_layout(
    vulkan_backend_state &state,
    const rt_binding_update_request &request)
{
    if (state.descriptor_set_layout != VK_NULL_HANDLE) {
        return true;
    }

    if (request.writes == nullptr || request.write_count == 0) {
        return false;
    }
    std::vector<VkDescriptorSetLayoutBinding> bindings(request.write_count);
    const VkShaderStageFlags stages =
        VK_SHADER_STAGE_RAYGEN_BIT_KHR |
        VK_SHADER_STAGE_MISS_BIT_KHR |
        VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
        VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
    for (std::size_t index = 0; index < request.write_count; ++index) {
        const rt_binding_write &write = request.writes[index];
        const VkDescriptorType type = vulkan_descriptor_type(write.type);
        if (write.location.group != 0 || type == VK_DESCRIPTOR_TYPE_MAX_ENUM) {
            return false;
        }
        bindings[index] = {
            write.location.binding,
            type,
            1,
            stages,
            nullptr};
    }
    VkDescriptorSetLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout_info.bindingCount = static_cast<std::uint32_t>(bindings.size());
    layout_info.pBindings = bindings.data();
    return vkCreateDescriptorSetLayout(state.device, &layout_info, nullptr, &state.descriptor_set_layout) ==
        VK_SUCCESS;
}

bool ensure_descriptor_set(
    vulkan_backend_state &state,
    const rt_binding_update_request &request)
{
    if (!ensure_descriptor_set_layout(state, request)) {
        return false;
    }
    if (state.descriptor_pool == VK_NULL_HANDLE) {
        std::vector<VkDescriptorPoolSize> pool_sizes;
        for (std::size_t index = 0; index < request.write_count; ++index) {
            const VkDescriptorType type = vulkan_descriptor_type(request.writes[index].type);
            if (type == VK_DESCRIPTOR_TYPE_MAX_ENUM) {
                return false;
            }
            const auto existing = std::find_if(
                pool_sizes.begin(),
                pool_sizes.end(),
                [type](const VkDescriptorPoolSize &entry) {
                    return entry.type == type;
                });
            if (existing == pool_sizes.end()) {
                pool_sizes.push_back({type, 1});
            } else {
                ++existing->descriptorCount;
            }
        }
        VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
        pool_info.pPoolSizes = pool_sizes.data();
        if (vkCreateDescriptorPool(state.device, &pool_info, nullptr, &state.descriptor_pool) != VK_SUCCESS) {
            return false;
        }
    }
    if (state.descriptor_set == VK_NULL_HANDLE) {
        VkDescriptorSetAllocateInfo alloc_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        alloc_info.descriptorPool = state.descriptor_pool;
        alloc_info.descriptorSetCount = 1;
        alloc_info.pSetLayouts = &state.descriptor_set_layout;
        if (vkAllocateDescriptorSets(state.device, &alloc_info, &state.descriptor_set) != VK_SUCCESS) {
            return false;
        }
    }
    return true;
}

bool update_descriptor_set(vulkan_backend_state &state, const rt_binding_update_request &request) {
    if (!ensure_descriptor_set(state, request) || request.writes == nullptr || request.write_count == 0) {
        return false;
    }
    std::vector<VkDescriptorBufferInfo> buffer_infos(request.write_count);
    std::vector<VkDescriptorImageInfo> image_infos(request.write_count);
    std::vector<VkAccelerationStructureKHR> acceleration_handles(request.write_count);
    std::vector<VkWriteDescriptorSetAccelerationStructureKHR> acceleration_infos(request.write_count);
    std::vector<VkWriteDescriptorSet> writes(request.write_count);
    for (std::size_t index = 0; index < request.write_count; ++index) {
        const rt_binding_write &source = request.writes[index];
        VkWriteDescriptorSet &write = writes[index];
        write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = state.descriptor_set;
        write.dstBinding = source.location.binding;
        write.descriptorCount = 1;
        write.descriptorType = vulkan_descriptor_type(source.type);
        if (source.location.group != 0 || write.descriptorType == VK_DESCRIPTOR_TYPE_MAX_ENUM) {
            return false;
        }
        if (source.type == rt_descriptor_type::acceleration_structure) {
            const vulkan_acceleration_structure* const tlas = state.tlas_registry.get(state.tlas);
            if (tlas == nullptr || tlas->handle == VK_NULL_HANDLE) {
                return false;
            }
            acceleration_handles[index] = tlas->handle;
            acceleration_infos[index] = {
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
            acceleration_infos[index].accelerationStructureCount = 1;
            acceleration_infos[index].pAccelerationStructures = &acceleration_handles[index];
            write.pNext = &acceleration_infos[index];
        } else if (source.type == rt_descriptor_type::storage_texture) {
            const vulkan_texture* const texture = state.texture_registry.get(source.texture);
            if (texture == nullptr || texture->view == VK_NULL_HANDLE) {
                return false;
            }
            image_infos[index].imageView = texture->view;
            image_infos[index].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            write.pImageInfo = &image_infos[index];
        } else if (source.type == rt_descriptor_type::structured_buffer ||
            source.type == rt_descriptor_type::storage_buffer ||
            source.type == rt_descriptor_type::uniform_buffer) {
            const vulkan_buffer* const buffer = vulkan_api_buffer(state, source.resource);
            if (buffer == nullptr || buffer->buffer == VK_NULL_HANDLE) {
                return false;
            }
            buffer_infos[index].buffer = buffer->buffer;
            buffer_infos[index].offset = 0;
            buffer_infos[index].range = source.type == rt_descriptor_type::uniform_buffer
                ? source.element_stride
                : VK_WHOLE_SIZE;
            write.pBufferInfo = &buffer_infos[index];
        } else {
            return false;
        }
    }
    vkUpdateDescriptorSets(
        state.device,
        static_cast<std::uint32_t>(writes.size()),
        writes.data(),
        0,
        nullptr);
    return true;
}
bool ensure_pipeline_layout(vulkan_backend_state &state) {
    if (state.pipeline_layout != VK_NULL_HANDLE) {
        return true;
    }
    if (state.descriptor_set_layout == VK_NULL_HANDLE) {
        return false;
    }
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &state.descriptor_set_layout;
    return vkCreatePipelineLayout(state.device, &layout_info, nullptr, &state.pipeline_layout) == VK_SUCCESS;
}

VkShaderStageFlagBits vulkan_shader_stage(rt_shader_stage stage) {
    switch (stage) {
    case rt_shader_stage::ray_generation:
        return VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    case rt_shader_stage::miss:
        return VK_SHADER_STAGE_MISS_BIT_KHR;
    case rt_shader_stage::closest_hit:
        return VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    case rt_shader_stage::any_hit:
        return VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
    case rt_shader_stage::intersection:
        return VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
    case rt_shader_stage::callable:
        return VK_SHADER_STAGE_CALLABLE_BIT_KHR;
    }
    return VK_SHADER_STAGE_FLAG_BITS_MAX_ENUM;
}

VkShaderModule vulkan_shader_module(
    const vulkan_backend_state &state,
    rt_shader_module_handle module)
{
    const VkShaderModule* const shader_module = state.shader_module_registry.get(module);
    return shader_module != nullptr ? *shader_module : VK_NULL_HANDLE;
}

bool ensure_pipeline(vulkan_backend_state &state, const rt_pipeline_desc &desc) {
    if (state.pipeline != VK_NULL_HANDLE) {
        return true;
    }
    if (!ensure_pipeline_layout(state)) {
        append_rt_startup_log("Vulkan pipeline creation aborted: ensure_pipeline_layout failed");
        return false;
    }

    std::vector<VkPipelineShaderStageCreateInfo> stages(desc.shader_count);
    for (std::size_t index = 0; index < desc.shader_count; ++index) {
        const rt_shader_entry_desc &source = desc.shaders[index];
        VkPipelineShaderStageCreateInfo &stage = stages[index];
        stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = vulkan_shader_stage(source.stage);
        stage.module = vulkan_shader_module(state, source.module);
        stage.pName = source.entry_point;
        if (stage.stage == VK_SHADER_STAGE_FLAG_BITS_MAX_ENUM ||
            stage.module == VK_NULL_HANDLE) {
            return false;
        }
    }

    std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups(desc.group_count);
    for (std::size_t index = 0; index < desc.group_count; ++index) {
        const rt_shader_group_desc &source = desc.groups[index];
        VkRayTracingShaderGroupCreateInfoKHR &group = groups[index];
        group = {VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
        group.generalShader = VK_SHADER_UNUSED_KHR;
        group.closestHitShader = VK_SHADER_UNUSED_KHR;
        group.anyHitShader = VK_SHADER_UNUSED_KHR;
        group.intersectionShader = VK_SHADER_UNUSED_KHR;
        if (source.type == rt_shader_group_type::general) {
            group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
            group.generalShader = source.general_shader;
        } else {
            group.type = source.type == rt_shader_group_type::triangles_hit_group
                ? VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR
                : VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
            group.closestHitShader = source.closest_hit_shader;
            group.anyHitShader = source.any_hit_shader == kRtUnusedShaderIndex
                ? VK_SHADER_UNUSED_KHR
                : source.any_hit_shader;
            group.intersectionShader = source.intersection_shader == kRtUnusedShaderIndex
                ? VK_SHADER_UNUSED_KHR
                : source.intersection_shader;
        }
    }

    VkRayTracingPipelineCreateInfoKHR pipeline_info{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    pipeline_info.stageCount = static_cast<std::uint32_t>(stages.size());
    pipeline_info.pStages = stages.data();
    pipeline_info.groupCount = static_cast<std::uint32_t>(groups.size());
    pipeline_info.pGroups = groups.data();
    pipeline_info.maxPipelineRayRecursionDepth = desc.max_recursion_depth;
    pipeline_info.layout = state.pipeline_layout;
    const VkResult result = state.create_ray_tracing_pipelines(
        state.device,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        1,
        &pipeline_info,
        nullptr,
        &state.pipeline);
    if (result != VK_SUCCESS) {
        append_rt_startup_log("Vulkan pipeline creation failed: vkCreateRayTracingPipelinesKHR returned error");
        return false;
    }
    state.pipeline_chunk_count = static_cast<std::uint32_t>(groups.size());
    return true;
}

bool ensure_shader_binding_table(
    vulkan_backend_state &state,
    const rt_shader_table_desc &desc)
{
    if (state.shader_binding_table.buffer != VK_NULL_HANDLE) {
        return true;
    }
    const std::size_t record_count =
        desc.ray_generation.group_count +
        desc.miss.group_count +
        desc.hit.group_count +
        desc.callable.group_count;
    if (state.pipeline == VK_NULL_HANDLE ||
        state.pipeline_chunk_count == 0 ||
        record_count == 0) {
        append_rt_startup_log("Vulkan SBT creation aborted: pipeline is unavailable");
        return false;
    }

    const std::uint32_t handle_size = state.rt_pipeline_properties.shaderGroupHandleSize;
    const std::uint32_t aligned_handle_size =
        (handle_size + state.rt_pipeline_properties.shaderGroupHandleAlignment - 1u) &
        ~(state.rt_pipeline_properties.shaderGroupHandleAlignment - 1u);
    const std::uint32_t stride = (aligned_handle_size + state.rt_pipeline_properties.shaderGroupBaseAlignment - 1u) &
        ~(state.rt_pipeline_properties.shaderGroupBaseAlignment - 1u);
    const VkDeviceSize sbt_size = static_cast<VkDeviceSize>(stride) * record_count;

    std::vector<std::uint8_t> handles(handle_size * state.pipeline_chunk_count);
    if (state.get_ray_tracing_shader_group_handles(
            state.device,
            state.pipeline,
            0,
            state.pipeline_chunk_count,
            handles.size(),
            handles.data()) != VK_SUCCESS) {
        append_rt_startup_log("Vulkan SBT creation failed: vkGetRayTracingShaderGroupHandlesKHR returned error");
        return false;
    }

    if (!create_native_buffer(
            state,
            sbt_size,
            VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &state.shader_binding_table)) {
        append_rt_startup_log("Vulkan SBT creation failed: create_buffer returned false");
        return false;
    }

    std::vector<std::uint8_t> sbt_bytes(static_cast<std::size_t>(sbt_size), 0);
    std::size_t record_index = 0;
    const auto copy_section = [&](const rt_shader_table_section_desc &section) {
        if (section.group_count != 0 && section.groups == nullptr) {
            return false;
        }
        for (std::size_t index = 0; index < section.group_count; ++index) {
            const std::uint32_t group_index = section.groups[index];
            if (group_index >= state.pipeline_chunk_count) {
                return false;
            }
            std::memcpy(
                sbt_bytes.data() + stride * record_index,
                handles.data() + handle_size * group_index,
                handle_size);
            ++record_index;
        }
        return true;
    };
    if (!copy_section(desc.ray_generation) ||
        !copy_section(desc.miss) ||
        !copy_section(desc.hit) ||
        !copy_section(desc.callable)) {
        return false;
    }
    if (!upload_buffer_data(state, &state.shader_binding_table, 0, sbt_bytes.data(), sbt_bytes.size())) {
        append_rt_startup_log("Vulkan SBT creation failed: upload_buffer_data returned false");
        return false;
    }

    const VkDeviceAddress sbt_address = buffer_device_address(state, state.shader_binding_table);
    VkDeviceSize section_offset = 0;
    state.raygen_regions.resize(desc.ray_generation.group_count);
    for (std::size_t index = 0; index < state.raygen_regions.size(); ++index) {
        state.raygen_regions[index] = {
            sbt_address + section_offset + stride * index,
            stride,
            stride};
    }
    section_offset += stride * desc.ray_generation.group_count;
    state.miss_region = desc.miss.group_count != 0
        ? VkStridedDeviceAddressRegionKHR{
            sbt_address + section_offset,
            stride,
            stride * desc.miss.group_count}
        : VkStridedDeviceAddressRegionKHR{};
    section_offset += stride * desc.miss.group_count;
    state.hit_region = desc.hit.group_count != 0
        ? VkStridedDeviceAddressRegionKHR{
            sbt_address + section_offset,
            stride,
            stride * desc.hit.group_count}
        : VkStridedDeviceAddressRegionKHR{};
    section_offset += stride * desc.hit.group_count;
    state.callable_region = desc.callable.group_count != 0
        ? VkStridedDeviceAddressRegionKHR{
            sbt_address + section_offset,
            stride,
            stride * desc.callable.group_count}
        : VkStridedDeviceAddressRegionKHR{};
    return true;
}

// =============================================================================
// Command submission, native presentation, and trace dispatch.
// =============================================================================

void transition_image(
    VkCommandBuffer command_buffer,
    VkImage image,
    VkImageLayout old_layout,
    VkImageLayout new_layout,
    VkAccessFlags src_access_mask,
    VkAccessFlags dst_access_mask,
    VkPipelineStageFlags src_stage_mask,
    VkPipelineStageFlags dst_stage_mask)
{
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.srcAccessMask = src_access_mask;
    barrier.dstAccessMask = dst_access_mask;
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(
        command_buffer,
        src_stage_mask,
        dst_stage_mask,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &barrier);
}

void collect_command_slot_timestamps(vulkan_backend_state &state, std::uint32_t slot_index) {
    vulkan_command_slot &slot = state.command_slots[slot_index];
    if ((slot.timestamp_regions & 1u) != 0u) {
        (void)read_timestamp_query_ms(
            state,
            slot_index,
            timestamp_query_region::accel,
            &state.diagnostics.acceleration_gpu_ms);
    }
    if ((slot.timestamp_regions & 2u) != 0u) {
        (void)read_timestamp_query_ms(
            state,
            slot_index,
            timestamp_query_region::dispatch,
            &state.diagnostics.dispatch_gpu_ms);
    }
    slot.timestamp_regions = 0;
}

void complete_command_slot(vulkan_backend_state &state, std::uint32_t slot_index) {
    vulkan_command_slot &slot = state.command_slots[slot_index];
    state.completed_submission_serial = (std::max)(
        state.completed_submission_serial,
        slot.submission.serial);
    collect_command_slot_timestamps(state, slot_index);
    slot.submitted = false;
    slot.submission = {};
    collect_deferred_releases(state);
}

bool is_submission_complete(vulkan_backend_state &state, rt_submission_token submission) {
    if (!submission || submission.serial <= state.completed_submission_serial) {
        return true;
    }
    for (std::uint32_t slot_index = 0; slot_index < kRtCommandSlotCount; ++slot_index) {
        vulkan_command_slot &slot = state.command_slots[slot_index];
        if (slot.submission != submission) {
            continue;
        }
        const VkResult status = vkGetFenceStatus(state.device, slot.fence);
        if (status == VK_SUCCESS) {
            complete_command_slot(state, slot_index);
            return true;
        }
        return false;
    }
    return false;
}

bool wait_for_submission(
    vulkan_backend_state &state,
    rt_submission_token submission,
    double* out_wait_ms = nullptr)
{
    const auto wait_start = std::chrono::steady_clock::now();
    bool completed = is_submission_complete(state, submission);
    if (!completed) {
        for (std::uint32_t slot_index = 0; slot_index < kRtCommandSlotCount; ++slot_index) {
            vulkan_command_slot &slot = state.command_slots[slot_index];
            if (slot.submission != submission) {
                continue;
            }
            completed =
                vkWaitForFences(state.device, 1, &slot.fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
            if (completed) {
                complete_command_slot(state, slot_index);
            }
            break;
        }
    }
    if (out_wait_ms != nullptr) {
        *out_wait_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - wait_start).count();
    }
    return completed;
}

bool begin_command_buffer(vulkan_backend_state &state) {
    const std::uint32_t slot_index = state.command_slot_index % kRtCommandSlotCount;
    vulkan_command_slot &slot = state.command_slots[slot_index];
    if (slot.submitted) {
        const auto reuse_wait_start = std::chrono::steady_clock::now();
        if (vkWaitForFences(state.device, 1, &slot.fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
            return false;
        }
        state.diagnostics.command_slot_reuse_wait_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - reuse_wait_start).count();
        complete_command_slot(state, slot_index);
    }
    if (vkResetFences(state.device, 1, &slot.fence) != VK_SUCCESS ||
        vkResetCommandPool(state.device, slot.pool, 0) != VK_SUCCESS) {
        return false;
    }
    state.command_pool = slot.pool;
    state.command_buffer = slot.buffer;
    state.command_fence = slot.fence;
    state.active_command_slot_index = slot_index;
    VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    return vkBeginCommandBuffer(state.command_buffer, &begin_info) == VK_SUCCESS;
}

bool submit_command_buffer(
    vulkan_backend_state &state,
    double* out_submit_cpu_ms,
    rt_submission_token* out_submission)
{
    if (out_submission != nullptr) {
        *out_submission = {};
    }
    const auto submit_start = std::chrono::steady_clock::now();
    VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &state.command_buffer;
    const bool submitted = vkQueueSubmit(state.graphics_queue, 1, &submit_info, state.command_fence) == VK_SUCCESS;
    if (submitted) {
        vulkan_command_slot &slot = state.command_slots[state.active_command_slot_index];
        slot.submitted = true;
        slot.submission = {state.next_submission_serial++};
        state.command_slot_index = (state.active_command_slot_index + 1u) % kRtCommandSlotCount;
        if (out_submission != nullptr) {
            *out_submission = slot.submission;
        }
    }
    if (out_submit_cpu_ms != nullptr) {
        *out_submit_cpu_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - submit_start).count();
    }
    return submitted;
}

bool wait_for_all_command_slots(vulkan_backend_state &state) {
    for (std::uint32_t slot_index = 0; slot_index < kRtCommandSlotCount; ++slot_index) {
        vulkan_command_slot &slot = state.command_slots[slot_index];
        if (!slot.submitted) {
            continue;
        }
        if (vkWaitForFences(state.device, 1, &slot.fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
            return false;
        }
        complete_command_slot(state, slot_index);
    }
    return true;
}

bool end_and_submit_command_buffer(
    vulkan_backend_state &state,
    double* out_submit_cpu_ms = nullptr,
    double* out_wait_ms = nullptr,
    rt_submission_token* out_submission = nullptr)
{
    if (vkEndCommandBuffer(state.command_buffer) != VK_SUCCESS) {
        return false;
    }
    if (out_wait_ms != nullptr) {
        *out_wait_ms = 0.0;
    }
    return submit_command_buffer(state, out_submit_cpu_ms, out_submission);
}

bool timestamp_queries_enabled(const vulkan_backend_state &state) {
    return state.timestamp_queries_supported &&
        state.timestamp_query_pool != VK_NULL_HANDLE &&
        state.timestamp_period_ns > 0.0f;
}

void reset_timestamp_query_region(
    vulkan_backend_state &state,
    std::uint32_t slot_index,
    timestamp_query_region region)
{
    if (!timestamp_queries_enabled(state) || state.command_buffer == VK_NULL_HANDLE) {
        return;
    }
    const std::uint32_t query_index =
        slot_index * kTimestampQueryCountPerCommandSlot +
        static_cast<std::uint32_t>(region);
    vkCmdResetQueryPool(
        state.command_buffer,
        state.timestamp_query_pool,
        query_index,
        kTimestampQueryCountPerRegion);
    state.command_slots[slot_index].timestamp_regions |=
        region == timestamp_query_region::accel ? 1u : 2u;
}

void write_timestamp_query_begin(
    vulkan_backend_state &state,
    std::uint32_t slot_index,
    timestamp_query_region region)
{
    if (!timestamp_queries_enabled(state) || state.command_buffer == VK_NULL_HANDLE) {
        return;
    }
    vkCmdWriteTimestamp(
        state.command_buffer,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        state.timestamp_query_pool,
        slot_index * kTimestampQueryCountPerCommandSlot +
        static_cast<std::uint32_t>(region));
}

void write_timestamp_query_end(
    vulkan_backend_state &state,
    std::uint32_t slot_index,
    timestamp_query_region region)
{
    if (!timestamp_queries_enabled(state) || state.command_buffer == VK_NULL_HANDLE) {
        return;
    }
    vkCmdWriteTimestamp(
        state.command_buffer,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        state.timestamp_query_pool,
        slot_index * kTimestampQueryCountPerCommandSlot +
            static_cast<std::uint32_t>(region) + 1u);
}

bool read_timestamp_query_ms(
    vulkan_backend_state &state,
    std::uint32_t slot_index,
    timestamp_query_region region,
    double* out_ms)
{
    if (out_ms == nullptr || !timestamp_queries_enabled(state)) {
        return false;
    }
    std::uint64_t timestamps[2]{};
    const VkResult result = vkGetQueryPoolResults(
        state.device,
        state.timestamp_query_pool,
        slot_index * kTimestampQueryCountPerCommandSlot +
            static_cast<std::uint32_t>(region),
        kTimestampQueryCountPerRegion,
        sizeof(timestamps),
        timestamps,
        sizeof(std::uint64_t),
        VK_QUERY_RESULT_64_BIT);
    if (result != VK_SUCCESS || timestamps[1] <= timestamps[0]) {
        return false;
    }
    *out_ms = static_cast<double>(timestamps[1] - timestamps[0]) * static_cast<double>(state.timestamp_period_ns) /
        1000000.0;
    return true;
}

// =============================================================================
// Device lifecycle and operation entry points.
// =============================================================================

bool initialize_vulkan_native(vulkan_backend_state &state, const rt_rhi_device_desc &) {
    reset_vulkan_state_locked(state);
    append_rt_startup_log("Vulkan native initialization begin");
    VkApplicationInfo app_info{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app_info.pApplicationName = "rtvdb_viewer";
    app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.pEngineName = "rtvdb";
    app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.apiVersion = VK_API_VERSION_1_3;

    std::vector<const char*> required_instance_extensions(
        kRequiredInstanceExtensions.begin(),
        kRequiredInstanceExtensions.end());
#if defined(__FreeBSD__)
    if (!viewer_shell::copy_vulkan_instance_extensions(&required_instance_extensions)) {
        append_rt_startup_log("Vulkan startup failed: SDL did not provide the required instance extensions");
        reset_vulkan_state_locked(state);
        return false;
    }
#endif

    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &app_info;
    instance_info.enabledExtensionCount = static_cast<std::uint32_t>(required_instance_extensions.size());
    instance_info.ppEnabledExtensionNames = required_instance_extensions.data();
    const VkResult instance_result = vkCreateInstance(&instance_info, nullptr, &state.instance);
    if (instance_result != VK_SUCCESS) {
        append_rt_startup_log(
            std::string("Vulkan startup failed: vkCreateInstance result=") +
            std::to_string(static_cast<int>(instance_result)));
        reset_vulkan_state_locked(state);
        return false;
    }
    if (!choose_physical_device(state) || !create_logical_device(state)) {
        append_rt_startup_log("Vulkan startup failed: device selection or creation");
        reset_vulkan_state_locked(state);
        return false;
    }
    if (!create_command_objects(state)) {
        append_rt_startup_log("Vulkan startup failed: command objects");
        reset_vulkan_state_locked(state);
        return false;
    }
    state.initialized = true;
    return true;
}

void shutdown_vulkan_native(vulkan_backend_state &state) {
    reset_vulkan_state_locked(state);
}

vulkan_rhi_device::vulkan_rhi_device() {
    device_.kind = rt_device_kind::vulkan_rt;
    device_.api = this;
    device_.native_state = &native_state_;
}

vulkan_rhi_device::~vulkan_rhi_device() {
    if (device_.api == this) {
        device_.api = nullptr;
    }
}

rt_rhi_device_info vulkan_rhi_device::info() const {
    std::scoped_lock lock(device_.access_mutex);
    return vulkan_backend_info_locked(native_state_);
}

rt_device* vulkan_rhi_device::device() {
    return &device_;
}

rt_native_texture_extension* vulkan_rhi_device::native_texture_extension() {
    return this;
}

rt_vulkan_interop_extension* vulkan_rhi_device::vulkan_interop_extension() {
    return this;
}

bool vulkan_rhi_device::initialize(
    const rt_rhi_device_desc &desc,
    rt_device_error* out_error)
{
    if (out_error != nullptr) {
        *out_error = {rt_device_operation::initialize, 0, {}};
    }
    if (device_.native_state != &native_state_) {
        if (out_error != nullptr) {
            out_error->detail = "Vulkan device state is unavailable";
        }
        return false;
    }
    std::scoped_lock lock(device_.access_mutex);
    if (!initialize_vulkan_native(native_state_, desc)) {
        if (out_error != nullptr) {
            out_error->detail = "Vulkan native initialization failed";
        }
        return false;
    }

    device_.capabilities.hardware_ray_tracing = native_state_.hardware_ray_tracing;
    device_.capabilities.timestamp_queries = native_state_.timestamp_queries_supported;
    device_.capabilities.native_vulkan_target = true;
    device_.capabilities.bgra_capture = true;
    device_.capabilities.bgra_readback = true;
    device_.capabilities.shader_binary_format = rt_shader_binary_format::spirv;
    device_.capabilities.output_format = rt_texture_format::rgba8_unorm;
    device_.capabilities.accumulation_format = rt_texture_format::rgba16_float;
    return true;
}

bool vulkan_rhi_device::wait_idle(
    rt_device_timing* out_timing,
    rt_device_error* out_error)
{
    if (out_timing != nullptr) {
        *out_timing = {};
    }
    if (out_error != nullptr) {
        *out_error = {rt_device_operation::wait_idle, 0, {}};
    }
    if (device_.native_state != &native_state_) {
        if (out_error != nullptr) {
            out_error->detail = "Vulkan device state is unavailable";
        }
        return false;
    }

    std::scoped_lock lock(device_.access_mutex);
    if (native_state_.device == VK_NULL_HANDLE) {
        return true;
    }

    const auto wait_start = std::chrono::steady_clock::now();
    const VkResult result = vkDeviceWaitIdle(native_state_.device);
    if (result == VK_SUCCESS) {
        native_state_.completed_submission_serial =
            native_state_.next_submission_serial > 1
                ? native_state_.next_submission_serial - 1u
                : 0;
        collect_deferred_releases(native_state_);
    }
    if (out_timing != nullptr) {
        out_timing->gpu_wait_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - wait_start).count();
    }
    if (result != VK_SUCCESS && out_error != nullptr) {
        out_error->native_code = static_cast<std::int64_t>(result);
        out_error->detail = "vkDeviceWaitIdle failed";
    }
    return result == VK_SUCCESS;
}

bool vulkan_rhi_device::begin_commands(
    rt_queue_class queue,
    rt_command_encoder* out_encoder,
    rt_device_error* out_error)
{
    if (out_encoder != nullptr) {
        *out_encoder = {};
    }
    if (device_.native_state != &native_state_ ||
        out_encoder == nullptr || queue != rt_queue_class::graphics ||
        native_state_.active_encoder_id != 0 || !begin_command_buffer(native_state_)) {
        if (out_error != nullptr) {
            *out_error = {rt_device_operation::begin_commands, 0, "Vulkan command begin failed"};
        }
        return false;
    }
    native_state_.active_encoder_id = native_state_.next_encoder_id++;
    *out_encoder = {
        native_state_.active_encoder_id,
        native_state_.active_command_slot_index,
        queue};
    return true;
}

bool vulkan_rhi_device::submit_commands(
    rt_command_encoder encoder,
    rt_submission_token* out_submission,
    rt_device_timing* out_timing,
    rt_device_error* out_error)
{
    if (out_submission != nullptr) {
        *out_submission = {};
    }
    if (out_timing != nullptr) {
        *out_timing = {};
    }
    if (device_.native_state != &native_state_ ||
        !encoder || encoder.id != native_state_.active_encoder_id ||
        out_submission == nullptr) {
        if (out_error != nullptr) {
            *out_error = {rt_device_operation::submit_commands, 0, "Vulkan command submit is invalid"};
        }
        return false;
    }
    double submit_cpu_ms = 0.0;
    const bool submitted = end_and_submit_command_buffer(
        native_state_,
        &submit_cpu_ms,
        nullptr,
        out_submission);
    vulkan_acceleration_build_context &acceleration = native_state_.acceleration_build;
    if (acceleration.active) {
        native_state_.diagnostics.acceleration_submit_cpu_ms = submit_cpu_ms;
        native_state_.diagnostics.acceleration_gpu_wait_ms = 0.0;
        native_state_.diagnostics.acceleration_cpu_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - acceleration.total_start).count();
        if (submitted && acceleration.recorded) {
            vulkan_acceleration_structure* const tlas =
                native_state_.tlas_registry.get(native_state_.tlas);
            if (tlas != nullptr && tlas->handle != VK_NULL_HANDLE) {
                VkAccelerationStructureDeviceAddressInfoKHR address_info{
                    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
                address_info.accelerationStructure = tlas->handle;
                tlas->device_address =
                    native_state_.get_acceleration_structure_device_address(
                        native_state_.device,
                        &address_info);
            }
        }
        acceleration = {};
    }
    native_state_.active_encoder_id = 0;
    if (out_timing != nullptr) {
        out_timing->submit_cpu_ms = submit_cpu_ms;
        out_timing->gpu_ms = native_state_.diagnostics.dispatch_gpu_ms;
    }
    if (!submitted && out_error != nullptr) {
        *out_error = {rt_device_operation::submit_commands, 0, "Vulkan command submit failed"};
    }
    return submitted;
}

void vulkan_rhi_device::discard_commands(
    rt_command_encoder encoder)
{
    if (device_.native_state != &native_state_ ||
        !encoder || encoder.id != native_state_.active_encoder_id) {
        return;
    }
    native_state_.acceleration_build = {};
    native_state_.active_encoder_id = 0;
}

bool vulkan_rhi_device::is_complete(
    rt_submission_token submission,
    bool* out_complete,
    rt_device_error* out_error)
{
    if (out_complete != nullptr) {
        *out_complete = false;
    }
    if (device_.native_state != &native_state_ ||
        !submission || out_complete == nullptr) {
        if (out_error != nullptr) {
            *out_error = {rt_device_operation::query_submission, 0, "Vulkan submission query is invalid"};
        }
        return false;
    }
    *out_complete = is_submission_complete(native_state_, submission);
    return true;
}

bool vulkan_rhi_device::wait(
    rt_submission_token submission,
    rt_device_timing* out_timing,
    rt_device_error* out_error)
{
    if (out_timing != nullptr) {
        *out_timing = {};
    }
    if (device_.native_state != &native_state_ || !submission) {
        if (out_error != nullptr) {
            *out_error = {rt_device_operation::wait_submission, 0, "Vulkan submission wait is invalid"};
        }
        return false;
    }
    double wait_ms = 0.0;
    const bool completed = wait_for_submission(native_state_, submission, &wait_ms);
    if (out_timing != nullptr) {
        out_timing->gpu_wait_ms = wait_ms;
    }
    if (!completed && out_error != nullptr) {
        *out_error = {rt_device_operation::wait_submission, 0, "Vulkan submission wait failed"};
    }
    return completed;
}

VkPipelineStageFlags vulkan_resource_stage(rt_resource_usage usage) {
    switch (usage) {
    case rt_resource_usage::undefined:
        return VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    case rt_resource_usage::shader_write:
    case rt_resource_usage::shader_read:
        return VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR |
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    case rt_resource_usage::acceleration_build_input:
    case rt_resource_usage::acceleration_storage:
        return VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    case rt_resource_usage::copy_source:
    case rt_resource_usage::copy_destination:
    case rt_resource_usage::clear_destination:
        return VK_PIPELINE_STAGE_TRANSFER_BIT;
    case rt_resource_usage::host_read:
        return VK_PIPELINE_STAGE_HOST_BIT;
    }
    return VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
}

VkAccessFlags vulkan_resource_access(rt_resource_usage usage) {
    switch (usage) {
    case rt_resource_usage::undefined:
        return 0;
    case rt_resource_usage::shader_write:
        return VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    case rt_resource_usage::shader_read:
        return VK_ACCESS_SHADER_READ_BIT;
    case rt_resource_usage::acceleration_build_input:
        return VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    case rt_resource_usage::acceleration_storage:
        return VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
            VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    case rt_resource_usage::copy_source:
        return VK_ACCESS_TRANSFER_READ_BIT;
    case rt_resource_usage::copy_destination:
    case rt_resource_usage::clear_destination:
        return VK_ACCESS_TRANSFER_WRITE_BIT;
    case rt_resource_usage::host_read:
        return VK_ACCESS_HOST_READ_BIT;
    }
    return 0;
}

VkImageLayout vulkan_resource_layout(rt_resource_usage usage) {
    switch (usage) {
    case rt_resource_usage::undefined:
        return VK_IMAGE_LAYOUT_UNDEFINED;
    case rt_resource_usage::shader_write:
        return VK_IMAGE_LAYOUT_GENERAL;
    case rt_resource_usage::shader_read:
        return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    case rt_resource_usage::copy_source:
        return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    case rt_resource_usage::copy_destination:
    case rt_resource_usage::clear_destination:
        return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    default:
        return VK_IMAGE_LAYOUT_GENERAL;
    }
}

rt_resource_usage vulkan_image_usage(VkImageLayout layout) {
    switch (layout) {
    case VK_IMAGE_LAYOUT_GENERAL:
        return rt_resource_usage::shader_write;
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
        return rt_resource_usage::shader_read;
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
        return rt_resource_usage::copy_source;
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
        return rt_resource_usage::copy_destination;
    default:
        return rt_resource_usage::undefined;
    }
}

bool vulkan_rhi_device::barrier(
    rt_command_encoder encoder,
    const rt_resource_barrier* barriers,
    std::size_t barrier_count,
    rt_device_error* out_error)
{
    if (device_.native_state != &native_state_ ||
        !encoder || encoder.id != native_state_.active_encoder_id ||
        (barrier_count != 0 && barriers == nullptr)) {
        if (out_error != nullptr) {
            *out_error = {rt_device_operation::transition_resource, 0, "Vulkan barrier request is invalid"};
        }
        return false;
    }

    for (std::size_t index = 0; index < barrier_count; ++index) {
        const rt_resource_barrier &source = barriers[index];
        const VkPipelineStageFlags destination_stage = vulkan_resource_stage(source.after);
        if (source.kind == rt_resource_kind::texture) {
            vulkan_texture* const texture =
                native_state_.texture_registry.get(source.texture);
            if (texture == nullptr || texture->image == VK_NULL_HANDLE) {
                if (out_error != nullptr) {
                    *out_error = {
                        rt_device_operation::transition_resource,
                        0,
                        "Vulkan texture barrier resource is invalid"};
                }
                return false;
            }
            const rt_resource_usage before = vulkan_image_usage(texture->layout);
            const VkPipelineStageFlags source_stage = vulkan_resource_stage(before);
            VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            barrier.srcAccessMask = vulkan_resource_access(before);
            barrier.dstAccessMask = vulkan_resource_access(source.after);
            barrier.oldLayout = texture->layout;
            barrier.newLayout = vulkan_resource_layout(source.after);
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = texture->image;
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.layerCount = 1;
            vkCmdPipelineBarrier(
                native_state_.command_buffer,
                source_stage,
                destination_stage,
                0,
                0,
                nullptr,
                0,
                nullptr,
                1,
                &barrier);
            texture->layout = barrier.newLayout;
        } else {
            vulkan_buffer* const buffer =
                vulkan_api_buffer(native_state_, source.buffer);
            if (buffer == nullptr || buffer->buffer == VK_NULL_HANDLE) {
                if (out_error != nullptr) {
                    *out_error = {
                        rt_device_operation::transition_resource,
                        0,
                        "Vulkan buffer barrier resource is invalid"};
                }
                return false;
            }
            const rt_resource_usage before =
                buffer->usage != rt_resource_usage::undefined
                    ? buffer->usage
                    : source.before;
            const VkPipelineStageFlags source_stage = vulkan_resource_stage(before);
            VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            barrier.srcAccessMask = vulkan_resource_access(before);
            barrier.dstAccessMask = vulkan_resource_access(source.after);
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = buffer->buffer;
            barrier.size = VK_WHOLE_SIZE;
            vkCmdPipelineBarrier(
                native_state_.command_buffer,
                source_stage,
                destination_stage,
                0,
                0,
                nullptr,
                1,
                &barrier,
                0,
                nullptr);
            buffer->usage = source.after;
        }
    }
    return true;
}

bool vulkan_rhi_device::copy_buffer(
    rt_command_encoder encoder,
    rt_buffer_handle source,
    rt_buffer_handle destination,
    const rt_buffer_copy_region &region,
    rt_device_error* out_error)
{
    const vulkan_buffer* const source_buffer = vulkan_api_buffer(native_state_, source);
    const vulkan_buffer* const destination_buffer =
        vulkan_api_buffer(native_state_, destination);
    const bool valid = device_.native_state == &native_state_ &&
        encoder && encoder.id == native_state_.active_encoder_id &&
        source_buffer != nullptr && destination_buffer != nullptr &&
        region.size > 0 &&
        region.source_offset <= source_buffer->size &&
        region.size <= source_buffer->size - region.source_offset &&
        region.destination_offset <= destination_buffer->size &&
        region.size <= destination_buffer->size - region.destination_offset;
    if (!valid) {
        if (out_error != nullptr) {
            *out_error = {rt_device_operation::copy_resource, 0, "Vulkan buffer copy request is invalid"};
        }
        return false;
    }
    const VkBufferCopy copy{
        region.source_offset,
        region.destination_offset,
        region.size};
    vkCmdCopyBuffer(
        native_state_.command_buffer,
        source_buffer->buffer,
        destination_buffer->buffer,
        1,
        &copy);
    return true;
}

bool vulkan_rhi_device::copy_texture_to_buffer(
    rt_command_encoder encoder,
    rt_texture_handle source,
    rt_buffer_handle destination,
    const rt_texture_buffer_copy_region &region,
    rt_device_error* out_error)
{
    const vulkan_texture* const source_texture =
        native_state_.texture_registry.get(source);
    const vulkan_buffer* const destination_buffer =
        vulkan_api_buffer(native_state_, destination);
    const std::size_t bytes_per_pixel = source_texture != nullptr
        ? rt_texture_format_bytes_per_pixel(source_texture->desc.format)
        : 0;
    const std::size_t required_size = region.buffer_offset +
        region.buffer_row_pitch * static_cast<std::size_t>(region.height);
    const bool valid = device_.native_state == &native_state_ &&
        encoder && encoder.id == native_state_.active_encoder_id &&
        source_texture != nullptr && destination_buffer != nullptr &&
        region.width > 0 && region.height > 0 && bytes_per_pixel > 0 &&
        region.width <= source_texture->desc.width &&
        region.height <= source_texture->desc.height &&
        region.buffer_row_pitch >= static_cast<std::size_t>(region.width) * bytes_per_pixel &&
        region.buffer_row_pitch % bytes_per_pixel == 0 &&
        required_size <= destination_buffer->size;
    if (!valid) {
        if (out_error != nullptr) {
            *out_error = {
                rt_device_operation::copy_resource,
                0,
                "Vulkan texture-to-buffer copy request is invalid"};
        }
        return false;
    }

    VkBufferImageCopy copy{};
    copy.bufferOffset = region.buffer_offset;
    copy.bufferRowLength =
        static_cast<std::uint32_t>(region.buffer_row_pitch / bytes_per_pixel);
    copy.bufferImageHeight = region.height;
    copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy.imageSubresource.layerCount = 1;
    copy.imageExtent = {region.width, region.height, 1};
    vkCmdCopyImageToBuffer(
        native_state_.command_buffer,
        source_texture->image,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        destination_buffer->buffer,
        1,
        &copy);
    return true;
}

bool vulkan_rhi_device::clear_texture(
    rt_command_encoder encoder,
    rt_texture_handle texture,
    const float color[4],
    rt_device_error* out_error)
{
    const vulkan_texture* const texture_object =
        native_state_.texture_registry.get(texture);
    const bool valid = device_.native_state == &native_state_ &&
        encoder && encoder.id == native_state_.active_encoder_id &&
        texture_object != nullptr && texture_object->image != VK_NULL_HANDLE &&
        color != nullptr &&
        (texture_object->desc.usage & rt_texture_usage_copy_destination) != 0u;
    if (!valid) {
        if (out_error != nullptr) {
            *out_error = {rt_device_operation::clear_texture, 0, "Vulkan texture clear request is invalid"};
        }
        return false;
    }

    VkClearColorValue clear_color{};
    std::memcpy(clear_color.float32, color, sizeof(clear_color.float32));
    VkImageSubresourceRange range{};
    range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    range.levelCount = 1;
    range.layerCount = 1;
    vkCmdClearColorImage(
        native_state_.command_buffer,
        texture_object->image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        &clear_color,
        1,
        &range);
    return true;
}

bool vulkan_rhi_device::trace_rays(
    rt_command_encoder encoder,
    const rt_trace_rays_desc &desc,
    rt_device_error* out_error)
{
    const VkPipeline* const pipeline =
        native_state_.pipeline_registry.get(desc.pipeline);
    const VkBuffer* const shader_table =
        native_state_.shader_table_registry.get(desc.shader_table);
    const VkStridedDeviceAddressRegionKHR* const ray_generation_region =
        desc.ray_generation_record < native_state_.raygen_regions.size()
            ? &native_state_.raygen_regions[desc.ray_generation_record]
            : nullptr;
    const bool valid = device_.native_state == &native_state_ &&
        encoder && encoder.id == native_state_.active_encoder_id &&
        pipeline != nullptr && *pipeline != VK_NULL_HANDLE &&
        shader_table != nullptr && *shader_table == native_state_.shader_binding_table.buffer &&
        ray_generation_region != nullptr && ray_generation_region->deviceAddress != 0u &&
        native_state_.miss_region.deviceAddress != 0u &&
        native_state_.hit_region.deviceAddress != 0u &&
        native_state_.pipeline_layout != VK_NULL_HANDLE &&
        native_state_.descriptor_set != VK_NULL_HANDLE &&
        native_state_.cmd_trace_rays != nullptr &&
        desc.width > 0u && desc.height > 0u && desc.depth > 0u;
    if (!valid) {
        if (out_error != nullptr) {
            *out_error = {rt_device_operation::trace_rays, 0, "Vulkan ray trace request is invalid"};
        }
        return false;
    }

    vkCmdBindPipeline(
        native_state_.command_buffer,
        VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
        *pipeline);
    const std::uint32_t viewer_constant_offset =
        native_state_.active_command_slot_index *
        static_cast<std::uint32_t>(kRtViewerConstantSlotStride);
    vkCmdBindDescriptorSets(
        native_state_.command_buffer,
        VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
        native_state_.pipeline_layout,
        0,
        1,
        &native_state_.descriptor_set,
        1,
        &viewer_constant_offset);
    if (desc.measure_gpu_time) {
        reset_timestamp_query_region(
            native_state_,
            native_state_.active_command_slot_index,
            timestamp_query_region::dispatch);
        write_timestamp_query_begin(
            native_state_,
            native_state_.active_command_slot_index,
            timestamp_query_region::dispatch);
    }
    native_state_.cmd_trace_rays(
        native_state_.command_buffer,
        ray_generation_region,
        &native_state_.miss_region,
        &native_state_.hit_region,
        &native_state_.callable_region,
        desc.width,
        desc.height,
        desc.depth);
    if (desc.measure_gpu_time) {
        write_timestamp_query_end(
            native_state_,
            native_state_.active_command_slot_index,
            timestamp_query_region::dispatch);
    }
    return true;
}

bool vulkan_rhi_device::shutdown(rt_device_error* out_error) {
    if (out_error != nullptr) {
        *out_error = {rt_device_operation::shutdown, 0, {}};
    }

    std::scoped_lock lock(device_.access_mutex);
    shutdown_vulkan_native(native_state_);
    device_.capabilities = {};
    return true;
}

bool vulkan_rhi_device::update_bindings(
    const rt_binding_update_request &request,
    rt_device_error* out_error)
{
    if (device_.native_state != &native_state_ || !update_descriptor_set(native_state_, request)) {
        if (out_error != nullptr) {
            out_error->detail = "Vulkan descriptor binding update failed";
        }
        return false;
    }
    return true;
}

bool vulkan_rhi_device::create_shader_module(
    const rt_shader_module_desc &desc,
    rt_shader_module_handle* out_module,
    rt_device_error* out_error)
{
    if (out_module != nullptr) {
        *out_module = {};
    }
    if (device_.native_state != &native_state_ ||
        desc.format != rt_shader_binary_format::spirv ||
        desc.data == nullptr || desc.size == 0 || out_module == nullptr) {
        if (out_error != nullptr) {
            out_error->detail = "Vulkan shader module descriptor is invalid";
        }
        return false;
    }
    VkShaderModule shader_module = VK_NULL_HANDLE;
    if (!create_native_shader_module(
            native_state_,
            static_cast<const std::uint8_t*>(desc.data),
            desc.size,
            &shader_module) ||
        !native_state_.shader_module_registry.insert(shader_module, out_module)) {
        destroy_native_shader_module(native_state_, &shader_module);
        if (out_error != nullptr) {
            out_error->detail = "Vulkan shader module creation failed";
        }
        return false;
    }
    return true;
}

void vulkan_rhi_device::destroy_shader_module(
    rt_shader_module_handle module)
{
    if (device_.native_state != &native_state_) {
        return;
    }
    VkShaderModule shader_module = VK_NULL_HANDLE;
    if (native_state_.shader_module_registry.erase(module, &shader_module)) {
        destroy_native_shader_module(native_state_, &shader_module);
    }
}

bool vulkan_rhi_device::create_pipeline(
    const rt_pipeline_desc &desc,
    rt_pipeline_handle* out_pipeline,
    rt_device_error* out_error)
{
    if (out_pipeline != nullptr) {
        *out_pipeline = {};
    }
    if (device_.native_state != &native_state_ ||
        out_pipeline == nullptr || desc.bindings == nullptr ||
        desc.shaders == nullptr || desc.groups == nullptr) {
        if (out_error != nullptr) {
            out_error->detail = "Vulkan pipeline build request is invalid";
        }
        return false;
    }
    if (!ensure_pipeline(native_state_, desc)) {
        if (out_error != nullptr) {
            out_error->detail = "Vulkan pipeline creation failed";
        }
        return false;
    }
    VkPipeline* registered_pipeline =
        native_state_.pipeline_registry.get(native_state_.pipeline_handle);
    if (registered_pipeline == nullptr || *registered_pipeline != native_state_.pipeline) {
        native_state_.pipeline_registry.clear([](VkPipeline &) {});
        native_state_.pipeline_handle = {};
        if (!native_state_.pipeline_registry.insert(
                native_state_.pipeline,
                &native_state_.pipeline_handle)) {
            if (out_error != nullptr) {
                out_error->detail = "Vulkan pipeline handle allocation failed";
            }
            return false;
        }
    }
    *out_pipeline = native_state_.pipeline_handle;
    return true;
}

bool vulkan_rhi_device::create_shader_table(
    const rt_shader_table_desc &desc,
    rt_shader_table_handle* out_shader_table,
    rt_device_error* out_error)
{
    if (out_shader_table != nullptr) {
        *out_shader_table = {};
    }
    const VkPipeline* const registered_pipeline =
        native_state_.pipeline_registry.get(desc.pipeline);
    if (device_.native_state != &native_state_ ||
        registered_pipeline == nullptr || *registered_pipeline != native_state_.pipeline ||
        out_shader_table == nullptr ||
        !ensure_shader_binding_table(native_state_, desc)) {
        if (out_error != nullptr) {
            out_error->detail = "Vulkan shader table creation failed";
        }
        return false;
    }
    VkBuffer* registered_shader_table =
        native_state_.shader_table_registry.get(native_state_.shader_table_handle);
    if (registered_shader_table == nullptr ||
        *registered_shader_table != native_state_.shader_binding_table.buffer) {
        native_state_.shader_table_registry.clear([](VkBuffer &) {});
        native_state_.shader_table_handle = {};
        if (!native_state_.shader_table_registry.insert(
                native_state_.shader_binding_table.buffer,
                &native_state_.shader_table_handle)) {
            if (out_error != nullptr) {
                out_error->detail = "Vulkan shader table handle allocation failed";
            }
            return false;
        }
    }
    *out_shader_table = native_state_.shader_table_handle;
    return true;
}

void vulkan_rhi_device::get_diagnostics(rt_rhi_diagnostics* out_diagnostics) const {
    if (out_diagnostics != nullptr) {
        *out_diagnostics = native_state_.diagnostics;
    }
}

bool vulkan_rhi_device::get_interop(vulkan_renderer_interop* out_interop) {
    if (out_interop == nullptr) {
        return false;
    }
    if (!native_state_.initialized ||
        native_state_.instance == VK_NULL_HANDLE ||
        native_state_.physical_device == VK_NULL_HANDLE ||
        native_state_.device == VK_NULL_HANDLE ||
        native_state_.graphics_queue_family == VK_QUEUE_FAMILY_IGNORED ||
        native_state_.present_queue_family == VK_QUEUE_FAMILY_IGNORED) {
        *out_interop = {};
        return false;
    }
    *out_interop = {
        native_state_.instance,
        native_state_.physical_device,
        native_state_.device,
        native_state_.graphics_queue_family,
        native_state_.present_queue_family,
    };
    return true;
}

bool vulkan_rhi_device::publish_texture(
    rt_texture_handle texture,
    const rt_native_texture_publish_desc &desc,
    rt_device_timing* out_timing,
    rt_device_error* out_error)
{
    if (out_timing != nullptr) {
        *out_timing = {};
    }
    vulkan_texture* const source = native_state_.texture_registry.get(texture);
    if (device_.native_state != &native_state_ ||
        texture != device_.output_texture || desc.out_target == nullptr ||
        source == nullptr || source->image == VK_NULL_HANDLE ||
        source->layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        if (out_error != nullptr) {
            out_error->operation = rt_device_operation::native_texture;
            out_error->detail = "Vulkan native output request is invalid";
        }
        return false;
    }
    *desc.out_target = reinterpret_cast<void*>(source->image);
    return true;
}

} // namespace

std::unique_ptr<rt_rhi_device> create_vulkan_rhi_device() {
    return std::make_unique<vulkan_rhi_device>();
}

} // namespace rtvdb::viewer_backend
