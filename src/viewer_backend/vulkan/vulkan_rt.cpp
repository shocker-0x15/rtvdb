#include "viewer_backend/backend_internal.h"
#include "viewer_capture/png.h"
#include "viewer_diagnostics/output.h"
#include "viewer_shell/shell.h"
#include "rtvdb_vulkan_rt_pick_rgen_spv.h"
#include "rtvdb_vulkan_rt_point_rchit_spv.h"
#include "rtvdb_vulkan_rt_point_rint_spv.h"
#include "rtvdb_vulkan_rt_rchit_spv.h"
#include "rtvdb_vulkan_rt_rgen_spv.h"
#include "rtvdb_vulkan_rt_rmiss_spv.h"
#include "rtvdb_vulkan_rt_line_rchit_spv.h"
#include "rtvdb_vulkan_rt_line_rint_spv.h"

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
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

namespace rtvdb::viewer_backend {
namespace {

constexpr std::uint32_t kMaxAccumulationSamples = 64;
constexpr VkFormat kOutputFormat = VK_FORMAT_R8G8B8A8_UNORM;
constexpr std::uint32_t kShaderGroupCount = 6;
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

std::filesystem::path capture_log_path(const char* filename) {
    try {
        if (!rtvdb::viewer_diagnostics::output_enabled()) {
            return {};
        }
        const std::filesystem::path dir = rtvdb::viewer_diagnostics::output_directory();
        std::filesystem::create_directories(dir);
        return dir / (filename != nullptr ? filename : "startup.log");
    } catch (...) {
        return std::filesystem::path(filename != nullptr ? filename : "startup.log");
    }
}

void append_startup_log(const std::string &message) {
    if (!rtvdb::viewer_diagnostics::output_enabled()) {
        return;
    }
    std::ofstream file(capture_log_path("startup.log"), std::ios::binary | std::ios::app);
    if (!file) {
        return;
    }
#if defined(_WIN32)
    SYSTEMTIME system_time{};
    GetLocalTime(&system_time);
    file << '['
         << system_time.wYear << '-'
         << (system_time.wMonth < 10 ? "0" : "") << system_time.wMonth << '-'
         << (system_time.wDay < 10 ? "0" : "") << system_time.wDay << ' '
         << (system_time.wHour < 10 ? "0" : "") << system_time.wHour << ':'
         << (system_time.wMinute < 10 ? "0" : "") << system_time.wMinute << ':'
         << (system_time.wSecond < 10 ? "0" : "") << system_time.wSecond << '.'
         << system_time.wMilliseconds << "] ";
#endif
    file << message << '\n';
}

#if defined(_WIN32)
void append_vulkan_fault_log(const char* stage, unsigned int exception_code) {
    char line[256]{};
    std::snprintf(
        line,
        sizeof(line),
        "Vulkan runtime fault: stage=%s exception=0x%08x",
        stage != nullptr ? stage : "unknown",
        exception_code);
    append_startup_log(line);
}
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

struct vec4_gpu {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 0.0f;
};

struct point_gpu {
    vec4_gpu position_radius{};
    vec4_gpu color{};
};

struct line_gpu {
    vec4_gpu a_radius{};
    vec4_gpu b_pad{};
    vec4_gpu color{};
    std::uint32_t flags = 0;
    float pad[3]{};
};

struct geometry_metadata_gpu {
    std::uint32_t first_triangle = 0;
    std::uint32_t index_offset = 0;
    std::uint32_t primitive_offset = 0;
    std::uint32_t primitive_count = 0;
};

struct viewer_constants_gpu {
    float origin[4]{};
    float forward[4]{};
    float right[4]{};
    float up[4]{};
    float scene_bounds_min[4]{};
    float scene_bounds_max[4]{};
    std::uint32_t size_and_mode[4]{};
    float projection_from[4]{};
    float projection_to[4]{};
    std::uint32_t projection_modes[4]{};
    float blend_and_jitter[4]{};
    std::uint32_t pick_and_flags[4]{};
    std::uint32_t pick_params[4]{};
};

static_assert(sizeof(viewer_constants_gpu) == 208, "viewer_constants_gpu must match GLSL std140 layout");

struct gpu_pick_result {
    std::uint32_t primitive_kind = 0;
    std::uint32_t primitive_index = 0;
    float distance = 0.0f;
    std::uint32_t hit = 0;
};

struct accumulation_key {
    std::uint64_t build_revision = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t display_mode = 0;
    std::uint32_t has_frame = 0;
    std::uint32_t hover_highlight_kind = 0;
    std::uint32_t hover_primitive_index = 0;
    float camera_origin[3]{};
    float camera_target[3]{};
    float camera_up[3]{};
    float camera_vertical_fov_degrees = 0.0f;
    std::uint32_t camera_projection = 0;
    float camera_fisheye_theta_degrees = 0.0f;
    float camera_fisheye_phi_degrees = 0.0f;
    float camera_orthographic_height = 0.0f;
    std::uint32_t camera_projection_blend_from = 0;
    std::uint32_t camera_projection_blend_to = 0;
    float camera_projection_blend_t = 1.0f;
};

struct vulkan_buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
};

struct vulkan_acceleration_structure {
    vulkan_buffer storage{};
    VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
    VkDeviceAddress device_address = 0;
};

struct vulkan_chunk_blas_entry {
    std::size_t first_triangle = 0;
    std::size_t triangle_count = 0;
    std::uint64_t fingerprint = 0;
    std::size_t vertex_offset = 0;
    std::size_t vertex_count = 0;
    std::size_t index_offset = 0;
    std::size_t index_count = 0;
    vulkan_acceleration_structure accel{};
};

struct vulkan_procedural_blas_entry {
    vulkan_acceleration_structure accel{};
    std::uint64_t geometry_fingerprint = 0;
    std::size_t first_primitive = 0;
    std::size_t primitive_count = 0;
};

struct vulkan_backend_state {
    backend_config config{};
    bool initialized = false;
    bool hardware_ray_tracing = false;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphics_queue = VK_NULL_HANDLE;
    std::uint32_t graphics_queue_family = VK_QUEUE_FAMILY_IGNORED;
    std::uint32_t present_queue_family = VK_QUEUE_FAMILY_IGNORED;
    VkShaderModule raygen_shader = VK_NULL_HANDLE;
    VkShaderModule pick_raygen_shader = VK_NULL_HANDLE;
    VkShaderModule miss_shader = VK_NULL_HANDLE;
    VkShaderModule closest_hit_shader = VK_NULL_HANDLE;
    VkShaderModule point_closest_hit_shader = VK_NULL_HANDLE;
    VkShaderModule point_intersection_shader = VK_NULL_HANDLE;
    VkShaderModule line_closest_hit_shader = VK_NULL_HANDLE;
    VkShaderModule line_intersection_shader = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkFence command_fence = VK_NULL_HANDLE;
    VkQueryPool timestamp_query_pool = VK_NULL_HANDLE;
    float timestamp_period_ns = 0.0f;
    bool timestamp_queries_supported = false;
    VkImage output_image = VK_NULL_HANDLE;
    VkDeviceMemory output_image_memory = VK_NULL_HANDLE;
    VkImageView output_image_view = VK_NULL_HANDLE;
    VkImageLayout output_image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImage accumulation_image = VK_NULL_HANDLE;
    VkDeviceMemory accumulation_image_memory = VK_NULL_HANDLE;
    VkImageView accumulation_image_view = VK_NULL_HANDLE;
    VkImageLayout accumulation_image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkBuffer readback_buffer = VK_NULL_HANDLE;
    VkDeviceMemory readback_buffer_memory = VK_NULL_HANDLE;
    int output_width = 0;
    int output_height = 0;
    bool direct_vulkan_present_requested = false;
    bool capture_readback_requested = false;
    vulkan_buffer scene_positions{};
    vulkan_buffer scene_indices{};
    vulkan_buffer triangle_colors{};
    vulkan_buffer instance_metadata{};
    vulkan_buffer point_buffer{};
    vulkan_buffer line_buffer{};
    vulkan_buffer point_aabb_buffer{};
    vulkan_buffer line_aabb_buffer{};
    vulkan_buffer viewer_constants{};
    vulkan_buffer pick_output{};
    vulkan_buffer tlas_instance_buffer{};
    vulkan_buffer scratch_buffer{};
    std::vector<vulkan_chunk_blas_entry> chunk_blas_cache;
    std::vector<vulkan_procedural_blas_entry> point_blas_entries;
    std::vector<vulkan_procedural_blas_entry> line_blas_entries;
    std::uint64_t point_geometry_fingerprint = 0;
    std::uint64_t line_geometry_fingerprint = 0;
    std::size_t point_primitive_count = 0;
    std::size_t line_primitive_count = 0;
    vulkan_acceleration_structure tlas{};
    std::size_t tlas_instance_count = 0;
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    vulkan_buffer shader_binding_table{};
    VkStridedDeviceAddressRegionKHR raygen_region{};
    VkStridedDeviceAddressRegionKHR pick_raygen_region{};
    VkStridedDeviceAddressRegionKHR miss_region{};
    VkStridedDeviceAddressRegionKHR hit_region{};
    VkStridedDeviceAddressRegionKHR callable_region{};
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
    std::uint64_t synced_build_revision = 0;
    std::size_t synced_blas_reused_count = 0;
    std::size_t synced_blas_rebuilt_count = 0;
    std::size_t synced_blas_reused_chunk_count = 0;
    std::size_t synced_blas_rebuilt_chunk_count = 0;
    std::size_t synced_tlas_rebuild_count = 0;
    std::uint32_t accumulation_sample_count = 0;
    bool accumulation_active = false;
    accumulation_key accumulation_state_key{};
    scene_build_info build_info{};
    std::mutex mutex;
} g_vulkan;

enum class timestamp_query_region : std::uint32_t {
    accel = 0,
    dispatch = 2,
};

bool timestamp_queries_enabled();
void reset_timestamp_query_region(timestamp_query_region region);
void write_timestamp_query_begin(timestamp_query_region region);
void write_timestamp_query_end(timestamp_query_region region);
bool read_timestamp_query_ms(timestamp_query_region region, double* out_ms);

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

bool capture_vulkan_to_bgra(
    int width,
    int height,
    const frame_scene &scene,
    bool has_frame,
    std::vector<std::uint8_t>* out_pixels,
    bool update_build_info);
bool ensure_pick_output_buffer();

backend_info vulkan_backend_info_locked() {
    return {
        backend_kind::vulkan_rt,
        "vulkan_rt",
        {g_vulkan.hardware_ray_tracing, false, true}
    };
}

backend_info vulkan_backend_info() {
    std::scoped_lock lock(g_vulkan.mutex);
    return vulkan_backend_info_locked();
}

void destroy_buffer(vulkan_buffer* buffer) {
    if (buffer == nullptr || g_vulkan.device == VK_NULL_HANDLE) {
        return;
    }
    if (buffer->buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(g_vulkan.device, buffer->buffer, nullptr);
        buffer->buffer = VK_NULL_HANDLE;
    }
    if (buffer->memory != VK_NULL_HANDLE) {
        vkFreeMemory(g_vulkan.device, buffer->memory, nullptr);
        buffer->memory = VK_NULL_HANDLE;
    }
    buffer->size = 0;
}

void destroy_acceleration_structure(vulkan_acceleration_structure* accel) {
    if (accel == nullptr || g_vulkan.device == VK_NULL_HANDLE) {
        return;
    }
    if (accel->handle != VK_NULL_HANDLE && g_vulkan.destroy_acceleration_structure != nullptr) {
        g_vulkan.destroy_acceleration_structure(g_vulkan.device, accel->handle, nullptr);
        accel->handle = VK_NULL_HANDLE;
    }
    destroy_buffer(&accel->storage);
    accel->device_address = 0;
}

void destroy_chunk_blas_cache(std::vector<vulkan_chunk_blas_entry>* cache) {
    if (cache == nullptr) {
        return;
    }
    for (vulkan_chunk_blas_entry &entry : *cache) {
        destroy_acceleration_structure(&entry.accel);
    }
    cache->clear();
}

void destroy_shader_module(VkShaderModule* shader_module) {
    if (shader_module == nullptr || *shader_module == VK_NULL_HANDLE || g_vulkan.device == VK_NULL_HANDLE) {
        return;
    }
    vkDestroyShaderModule(g_vulkan.device, *shader_module, nullptr);
    *shader_module = VK_NULL_HANDLE;
}

void destroy_output_resources_locked() {
    if (g_vulkan.device == VK_NULL_HANDLE) {
        return;
    }
    if (g_vulkan.output_image_view != VK_NULL_HANDLE) {
        vkDestroyImageView(g_vulkan.device, g_vulkan.output_image_view, nullptr);
        g_vulkan.output_image_view = VK_NULL_HANDLE;
    }
    if (g_vulkan.accumulation_image_view != VK_NULL_HANDLE) {
        vkDestroyImageView(g_vulkan.device, g_vulkan.accumulation_image_view, nullptr);
        g_vulkan.accumulation_image_view = VK_NULL_HANDLE;
    }
    if (g_vulkan.accumulation_image != VK_NULL_HANDLE) {
        vkDestroyImage(g_vulkan.device, g_vulkan.accumulation_image, nullptr);
        g_vulkan.accumulation_image = VK_NULL_HANDLE;
    }
    if (g_vulkan.accumulation_image_memory != VK_NULL_HANDLE) {
        vkFreeMemory(g_vulkan.device, g_vulkan.accumulation_image_memory, nullptr);
        g_vulkan.accumulation_image_memory = VK_NULL_HANDLE;
    }
    if (g_vulkan.output_image != VK_NULL_HANDLE) {
        vkDestroyImage(g_vulkan.device, g_vulkan.output_image, nullptr);
        g_vulkan.output_image = VK_NULL_HANDLE;
    }
    if (g_vulkan.output_image_memory != VK_NULL_HANDLE) {
        vkFreeMemory(g_vulkan.device, g_vulkan.output_image_memory, nullptr);
        g_vulkan.output_image_memory = VK_NULL_HANDLE;
    }
    if (g_vulkan.readback_buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(g_vulkan.device, g_vulkan.readback_buffer, nullptr);
        g_vulkan.readback_buffer = VK_NULL_HANDLE;
    }
    if (g_vulkan.readback_buffer_memory != VK_NULL_HANDLE) {
        vkFreeMemory(g_vulkan.device, g_vulkan.readback_buffer_memory, nullptr);
        g_vulkan.readback_buffer_memory = VK_NULL_HANDLE;
    }
    g_vulkan.output_width = 0;
    g_vulkan.output_height = 0;
    g_vulkan.output_image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    g_vulkan.accumulation_image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
}

void reset_vulkan_state_locked() {
    if (g_vulkan.command_pool != VK_NULL_HANDLE &&
        g_vulkan.command_buffer != VK_NULL_HANDLE &&
        g_vulkan.device != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(g_vulkan.device, g_vulkan.command_pool, 1, &g_vulkan.command_buffer);
        g_vulkan.command_buffer = VK_NULL_HANDLE;
    }
    if (g_vulkan.timestamp_query_pool != VK_NULL_HANDLE && g_vulkan.device != VK_NULL_HANDLE) {
        vkDestroyQueryPool(g_vulkan.device, g_vulkan.timestamp_query_pool, nullptr);
        g_vulkan.timestamp_query_pool = VK_NULL_HANDLE;
    }
    if (g_vulkan.command_fence != VK_NULL_HANDLE && g_vulkan.device != VK_NULL_HANDLE) {
        vkDestroyFence(g_vulkan.device, g_vulkan.command_fence, nullptr);
        g_vulkan.command_fence = VK_NULL_HANDLE;
    }
    destroy_output_resources_locked();
    destroy_buffer(&g_vulkan.shader_binding_table);
    if (g_vulkan.pipeline != VK_NULL_HANDLE && g_vulkan.device != VK_NULL_HANDLE) {
        vkDestroyPipeline(g_vulkan.device, g_vulkan.pipeline, nullptr);
        g_vulkan.pipeline = VK_NULL_HANDLE;
    }
    if (g_vulkan.pipeline_layout != VK_NULL_HANDLE && g_vulkan.device != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(g_vulkan.device, g_vulkan.pipeline_layout, nullptr);
        g_vulkan.pipeline_layout = VK_NULL_HANDLE;
    }
    if (g_vulkan.descriptor_pool != VK_NULL_HANDLE && g_vulkan.device != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(g_vulkan.device, g_vulkan.descriptor_pool, nullptr);
        g_vulkan.descriptor_pool = VK_NULL_HANDLE;
    }
    if (g_vulkan.descriptor_set_layout != VK_NULL_HANDLE && g_vulkan.device != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(g_vulkan.device, g_vulkan.descriptor_set_layout, nullptr);
        g_vulkan.descriptor_set_layout = VK_NULL_HANDLE;
    }
    destroy_acceleration_structure(&g_vulkan.tlas);
    g_vulkan.tlas_instance_count = 0;
    for (vulkan_procedural_blas_entry &entry : g_vulkan.point_blas_entries) {
        destroy_acceleration_structure(&entry.accel);
    }
    for (vulkan_procedural_blas_entry &entry : g_vulkan.line_blas_entries) {
        destroy_acceleration_structure(&entry.accel);
    }
    g_vulkan.point_blas_entries.clear();
    g_vulkan.line_blas_entries.clear();
    g_vulkan.point_geometry_fingerprint = 0;
    g_vulkan.line_geometry_fingerprint = 0;
    g_vulkan.point_primitive_count = 0;
    g_vulkan.line_primitive_count = 0;
    destroy_chunk_blas_cache(&g_vulkan.chunk_blas_cache);
    destroy_buffer(&g_vulkan.scratch_buffer);
    destroy_buffer(&g_vulkan.tlas_instance_buffer);
    destroy_buffer(&g_vulkan.viewer_constants);
    destroy_buffer(&g_vulkan.pick_output);
    destroy_buffer(&g_vulkan.line_buffer);
    destroy_buffer(&g_vulkan.point_buffer);
    destroy_buffer(&g_vulkan.line_aabb_buffer);
    destroy_buffer(&g_vulkan.point_aabb_buffer);
    destroy_buffer(&g_vulkan.triangle_colors);
    destroy_buffer(&g_vulkan.instance_metadata);
    destroy_buffer(&g_vulkan.scene_indices);
    destroy_buffer(&g_vulkan.scene_positions);
    if (g_vulkan.command_pool != VK_NULL_HANDLE && g_vulkan.device != VK_NULL_HANDLE) {
        vkDestroyCommandPool(g_vulkan.device, g_vulkan.command_pool, nullptr);
        g_vulkan.command_pool = VK_NULL_HANDLE;
    }
    destroy_shader_module(&g_vulkan.raygen_shader);
    destroy_shader_module(&g_vulkan.pick_raygen_shader);
    destroy_shader_module(&g_vulkan.miss_shader);
    destroy_shader_module(&g_vulkan.closest_hit_shader);
    destroy_shader_module(&g_vulkan.point_closest_hit_shader);
    destroy_shader_module(&g_vulkan.point_intersection_shader);
    destroy_shader_module(&g_vulkan.line_closest_hit_shader);
    destroy_shader_module(&g_vulkan.line_intersection_shader);
    if (g_vulkan.device != VK_NULL_HANDLE) {
        vkDestroyDevice(g_vulkan.device, nullptr);
        g_vulkan.device = VK_NULL_HANDLE;
    }
    if (g_vulkan.instance != VK_NULL_HANDLE) {
        vkDestroyInstance(g_vulkan.instance, nullptr);
        g_vulkan.instance = VK_NULL_HANDLE;
    }
    g_vulkan.config = {};
    g_vulkan.initialized = false;
    g_vulkan.hardware_ray_tracing = false;
    g_vulkan.physical_device = VK_NULL_HANDLE;
    g_vulkan.graphics_queue = VK_NULL_HANDLE;
    g_vulkan.graphics_queue_family = VK_QUEUE_FAMILY_IGNORED;
    g_vulkan.present_queue_family = VK_QUEUE_FAMILY_IGNORED;
    g_vulkan.capture_readback_requested = false;
    g_vulkan.rt_pipeline_properties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
    g_vulkan.accel_properties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
    g_vulkan.synced_build_revision = 0;
    g_vulkan.synced_blas_reused_count = 0;
    g_vulkan.synced_blas_rebuilt_count = 0;
    g_vulkan.synced_blas_reused_chunk_count = 0;
    g_vulkan.synced_blas_rebuilt_chunk_count = 0;
    g_vulkan.synced_tlas_rebuild_count = 0;
    g_vulkan.accumulation_sample_count = 0;
    g_vulkan.accumulation_active = false;
    g_vulkan.accumulation_state_key = {};
    g_vulkan.build_info = {};
}

bool check_device_extension_support(VkPhysicalDevice physical_device) {
    std::uint32_t extension_count = 0;
    if (vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, nullptr) != VK_SUCCESS) {
        return false;
    }
    std::vector<VkExtensionProperties> extensions(extension_count);
    if (extension_count > 0 &&
        vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, extensions.data()) != VK_SUCCESS) {
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

bool choose_physical_device() {
    bool require_matching_adapter = false;
#if defined(_WIN32)
    LUID target_adapter_luid{};
    DXGI_ADAPTER_DESC1 preferred_adapter_desc{};
    if (query_preferred_dxgi_adapter(&target_adapter_luid, &preferred_adapter_desc)) {
        require_matching_adapter = true;
        char line[512]{};
        std::snprintf(
            line,
            sizeof(line),
            "Vulkan startup preferred DXGI adapter: source=dxgi_preferred name=%ws vendor=0x%04x device=0x%04x luid=%08x:%08x",
            preferred_adapter_desc.Description,
            static_cast<unsigned>(preferred_adapter_desc.VendorId),
            static_cast<unsigned>(preferred_adapter_desc.DeviceId),
            static_cast<unsigned>(preferred_adapter_desc.AdapterLuid.HighPart),
            static_cast<unsigned>(preferred_adapter_desc.AdapterLuid.LowPart));
        append_startup_log(line);
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
        if (vkEnumeratePhysicalDevices(g_vulkan.instance, &physical_device_count, nullptr) != VK_SUCCESS ||
            physical_device_count == 0) {
            return false;
        }

        std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
        if (vkEnumeratePhysicalDevices(g_vulkan.instance, &physical_device_count, physical_devices.data()) != VK_SUCCESS) {
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

                const bool graphics_queue_presents = viewer_shell::vulkan_presentation_supported(
                    g_vulkan.instance,
                    physical_device,
                    queue_family_index);
                std::uint32_t present_queue_family_index =
                    graphics_queue_presents ? queue_family_index : VK_QUEUE_FAMILY_IGNORED;
                if (!graphics_queue_presents) {
                    for (std::uint32_t present_index = 0; present_index < queue_family_count; ++present_index) {
                        if (queue_families[present_index].queueCount == 0) {
                            continue;
                        }
                        if (viewer_shell::vulkan_presentation_supported(
                                g_vulkan.instance,
                                physical_device,
                                present_index)) {
                            present_queue_family_index = present_index;
                            break;
                        }
                    }
                }
                if (present_queue_family_index == VK_QUEUE_FAMILY_IGNORED) {
                    continue;
                }

                append_startup_log(
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
            append_startup_log(
                "Vulkan startup failed: no RT-capable Vulkan device matched the preferred DXGI adapter");
        }
        return false;
    }

    VkPhysicalDeviceProperties best_properties{};
    vkGetPhysicalDeviceProperties(best_physical_device, &best_properties);
    g_vulkan.physical_device = best_physical_device;
    g_vulkan.graphics_queue_family = best_queue_family_index;
    g_vulkan.present_queue_family = best_present_queue_family_index;
    g_vulkan.hardware_ray_tracing = true;
    g_vulkan.timestamp_queries_supported = best_timestamp_valid_bits != 0;
    append_startup_log(
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

bool load_rt_functions() {
    g_vulkan.create_acceleration_structure =
        reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(
            vkGetDeviceProcAddr(g_vulkan.device, "vkCreateAccelerationStructureKHR"));
    g_vulkan.destroy_acceleration_structure =
        reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(
            vkGetDeviceProcAddr(g_vulkan.device, "vkDestroyAccelerationStructureKHR"));
    g_vulkan.get_acceleration_structure_build_sizes =
        reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(
            vkGetDeviceProcAddr(g_vulkan.device, "vkGetAccelerationStructureBuildSizesKHR"));
    g_vulkan.cmd_build_acceleration_structures =
        reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(
            vkGetDeviceProcAddr(g_vulkan.device, "vkCmdBuildAccelerationStructuresKHR"));
    g_vulkan.get_acceleration_structure_device_address =
        reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(
            vkGetDeviceProcAddr(g_vulkan.device, "vkGetAccelerationStructureDeviceAddressKHR"));
    g_vulkan.create_ray_tracing_pipelines =
        reinterpret_cast<PFN_vkCreateRayTracingPipelinesKHR>(
            vkGetDeviceProcAddr(g_vulkan.device, "vkCreateRayTracingPipelinesKHR"));
    g_vulkan.get_ray_tracing_shader_group_handles =
        reinterpret_cast<PFN_vkGetRayTracingShaderGroupHandlesKHR>(
            vkGetDeviceProcAddr(g_vulkan.device, "vkGetRayTracingShaderGroupHandlesKHR"));
    g_vulkan.cmd_trace_rays =
        reinterpret_cast<PFN_vkCmdTraceRaysKHR>(
            vkGetDeviceProcAddr(g_vulkan.device, "vkCmdTraceRaysKHR"));

    return g_vulkan.create_acceleration_structure != nullptr &&
        g_vulkan.destroy_acceleration_structure != nullptr &&
        g_vulkan.get_acceleration_structure_build_sizes != nullptr &&
        g_vulkan.cmd_build_acceleration_structures != nullptr &&
        g_vulkan.get_acceleration_structure_device_address != nullptr &&
        g_vulkan.create_ray_tracing_pipelines != nullptr &&
        g_vulkan.get_ray_tracing_shader_group_handles != nullptr &&
        g_vulkan.cmd_trace_rays != nullptr;
}

bool create_logical_device() {
    const float queue_priority = 1.0f;
    std::array<VkDeviceQueueCreateInfo, 2> queue_infos{};
    queue_infos[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_infos[0].queueFamilyIndex = g_vulkan.graphics_queue_family;
    queue_infos[0].queueCount = 1;
    queue_infos[0].pQueuePriorities = &queue_priority;
    std::uint32_t queue_info_count = 1;
    if (g_vulkan.present_queue_family != VK_QUEUE_FAMILY_IGNORED &&
        g_vulkan.present_queue_family != g_vulkan.graphics_queue_family) {
        queue_infos[1].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_infos[1].queueFamilyIndex = g_vulkan.present_queue_family;
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
    if (vkCreateDevice(g_vulkan.physical_device, &device_info, nullptr, &g_vulkan.device) != VK_SUCCESS) {
        return false;
    }

    vkGetDeviceQueue(g_vulkan.device, g_vulkan.graphics_queue_family, 0, &g_vulkan.graphics_queue);
    if (g_vulkan.graphics_queue == VK_NULL_HANDLE || !load_rt_functions()) {
        return false;
    }

    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    properties.pNext = &g_vulkan.rt_pipeline_properties;
    g_vulkan.rt_pipeline_properties.pNext = &g_vulkan.accel_properties;
    vkGetPhysicalDeviceProperties2(g_vulkan.physical_device, &properties);
    g_vulkan.timestamp_period_ns = properties.properties.limits.timestampPeriod;
    return true;
}

std::uint32_t find_memory_type_index(std::uint32_t type_bits, VkMemoryPropertyFlags required_flags) {
    VkPhysicalDeviceMemoryProperties memory_properties{};
    vkGetPhysicalDeviceMemoryProperties(g_vulkan.physical_device, &memory_properties);
    for (std::uint32_t memory_index = 0; memory_index < memory_properties.memoryTypeCount; ++memory_index) {
        const bool supported = (type_bits & (1u << memory_index)) != 0u;
        const bool matches = (memory_properties.memoryTypes[memory_index].propertyFlags & required_flags) == required_flags;
        if (supported && matches) {
            return memory_index;
        }
    }
    return UINT32_MAX;
}

bool create_buffer(
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags memory_flags,
    vulkan_buffer* out_buffer)
{
    if (out_buffer == nullptr) {
        return false;
    }
    destroy_buffer(out_buffer);

    VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(g_vulkan.device, &buffer_info, nullptr, &out_buffer->buffer) != VK_SUCCESS) {
        return false;
    }

    VkMemoryRequirements memory_requirements{};
    vkGetBufferMemoryRequirements(g_vulkan.device, out_buffer->buffer, &memory_requirements);
    const std::uint32_t memory_index = find_memory_type_index(memory_requirements.memoryTypeBits, memory_flags);
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
    if (vkAllocateMemory(g_vulkan.device, &allocate_info, nullptr, &out_buffer->memory) != VK_SUCCESS) {
        return false;
    }
    if (vkBindBufferMemory(g_vulkan.device, out_buffer->buffer, out_buffer->memory, 0) != VK_SUCCESS) {
        return false;
    }
    out_buffer->size = size;
    return true;
}

bool upload_buffer_data(vulkan_buffer* buffer, const void* data, std::size_t size) {
    if (buffer == nullptr || buffer->memory == VK_NULL_HANDLE) {
        return false;
    }
    if (size == 0) {
        return true;
    }
    void* mapped = nullptr;
    if (vkMapMemory(g_vulkan.device, buffer->memory, 0, size, 0, &mapped) != VK_SUCCESS || mapped == nullptr) {
        return false;
    }
    std::memcpy(mapped, data, size);
    vkUnmapMemory(g_vulkan.device, buffer->memory);
    return true;
}

bool create_and_upload_buffer(
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags memory_flags,
    const void* data,
    std::size_t data_size,
    vulkan_buffer* out_buffer)
{
    const VkDeviceSize allocation_size = size == 0 ? VkDeviceSize{16} : size;
    if (!create_buffer(allocation_size, usage, memory_flags, out_buffer)) {
        return false;
    }
    return upload_buffer_data(out_buffer, data, data_size);
}

VkDeviceSize normalized_buffer_size(VkDeviceSize size) {
    return size == 0 ? VkDeviceSize{16} : size;
}

bool ensure_uploaded_buffer(
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
    if (buffer->buffer == VK_NULL_HANDLE || buffer->size != allocation_size) {
        if (!create_buffer(allocation_size, usage, memory_flags, buffer)) {
            return false;
        }
    }
    return upload_buffer_data(buffer, data, data_size);
}

bool ensure_acceleration_structure(
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

    destroy_acceleration_structure(accel);
    if (!create_buffer(
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
    if (g_vulkan.create_acceleration_structure(g_vulkan.device, &accel_create_info, nullptr, &accel->handle) !=
        VK_SUCCESS) {
        destroy_acceleration_structure(accel);
        return false;
    }

    return true;
}

VkDeviceAddress buffer_device_address(const vulkan_buffer &buffer) {
    VkBufferDeviceAddressInfo address_info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    address_info.buffer = buffer.buffer;
    return vkGetBufferDeviceAddress(g_vulkan.device, &address_info);
}

bool create_shader_module(const std::uint8_t* bytes, std::size_t size, VkShaderModule* out_shader_module) {
    if (bytes == nullptr || size == 0 || out_shader_module == nullptr || (size % 4) != 0) {
        return false;
    }
    VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shader_info.codeSize = size;
    shader_info.pCode = reinterpret_cast<const std::uint32_t*>(bytes);
    return vkCreateShaderModule(g_vulkan.device, &shader_info, nullptr, out_shader_module) == VK_SUCCESS;
}

bool create_command_objects() {
    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = g_vulkan.graphics_queue_family;
    if (vkCreateCommandPool(g_vulkan.device, &pool_info, nullptr, &g_vulkan.command_pool) != VK_SUCCESS) {
        return false;
    }

    VkCommandBufferAllocateInfo command_buffer_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_buffer_info.commandPool = g_vulkan.command_pool;
    command_buffer_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_buffer_info.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(g_vulkan.device, &command_buffer_info, &g_vulkan.command_buffer) != VK_SUCCESS) {
        return false;
    }

    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    if (vkCreateFence(g_vulkan.device, &fence_info, nullptr, &g_vulkan.command_fence) != VK_SUCCESS) {
        return false;
    }
    if (!g_vulkan.timestamp_queries_supported) {
        return true;
    }
    VkQueryPoolCreateInfo query_pool_info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    query_pool_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    query_pool_info.queryCount = 4;
    if (vkCreateQueryPool(g_vulkan.device, &query_pool_info, nullptr, &g_vulkan.timestamp_query_pool) != VK_SUCCESS) {
        g_vulkan.timestamp_queries_supported = false;
    }
    return true;
}

bool create_output_resources(int width, int height) {
    destroy_output_resources_locked();

    VkImageCreateInfo image_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = kOutputFormat;
    image_info.extent = {static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height), 1u};
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(g_vulkan.device, &image_info, nullptr, &g_vulkan.output_image) != VK_SUCCESS) {
        return false;
    }

    VkMemoryRequirements image_requirements{};
    vkGetImageMemoryRequirements(g_vulkan.device, g_vulkan.output_image, &image_requirements);
    const std::uint32_t image_memory_index =
        find_memory_type_index(image_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (image_memory_index == UINT32_MAX) {
        return false;
    }

    VkMemoryAllocateInfo image_allocate_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    image_allocate_info.allocationSize = image_requirements.size;
    image_allocate_info.memoryTypeIndex = image_memory_index;
    if (vkAllocateMemory(g_vulkan.device, &image_allocate_info, nullptr, &g_vulkan.output_image_memory) != VK_SUCCESS) {
        return false;
    }
    if (vkBindImageMemory(g_vulkan.device, g_vulkan.output_image, g_vulkan.output_image_memory, 0) != VK_SUCCESS) {
        return false;
    }

    VkImageViewCreateInfo image_view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    image_view_info.image = g_vulkan.output_image;
    image_view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    image_view_info.format = kOutputFormat;
    image_view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    image_view_info.subresourceRange.levelCount = 1;
    image_view_info.subresourceRange.layerCount = 1;
    if (vkCreateImageView(g_vulkan.device, &image_view_info, nullptr, &g_vulkan.output_image_view) != VK_SUCCESS) {
        return false;
    }

    VkImageCreateInfo accumulation_info = image_info;
    accumulation_info.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    accumulation_info.usage = VK_IMAGE_USAGE_STORAGE_BIT;
    if (vkCreateImage(g_vulkan.device, &accumulation_info, nullptr, &g_vulkan.accumulation_image) != VK_SUCCESS) {
        return false;
    }
    VkMemoryRequirements accumulation_requirements{};
    vkGetImageMemoryRequirements(g_vulkan.device, g_vulkan.accumulation_image, &accumulation_requirements);
    const std::uint32_t accumulation_memory_index = find_memory_type_index(
        accumulation_requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (accumulation_memory_index == UINT32_MAX) {
        return false;
    }
    VkMemoryAllocateInfo accumulation_allocate_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    accumulation_allocate_info.allocationSize = accumulation_requirements.size;
    accumulation_allocate_info.memoryTypeIndex = accumulation_memory_index;
    if (vkAllocateMemory(
            g_vulkan.device,
            &accumulation_allocate_info,
            nullptr,
            &g_vulkan.accumulation_image_memory) != VK_SUCCESS ||
        vkBindImageMemory(
            g_vulkan.device,
            g_vulkan.accumulation_image,
            g_vulkan.accumulation_image_memory,
            0) != VK_SUCCESS) {
        return false;
    }
    VkImageViewCreateInfo accumulation_view_info = image_view_info;
    accumulation_view_info.image = g_vulkan.accumulation_image;
    accumulation_view_info.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    if (vkCreateImageView(
            g_vulkan.device,
            &accumulation_view_info,
            nullptr,
            &g_vulkan.accumulation_image_view) != VK_SUCCESS) {
        return false;
    }

    const VkDeviceSize readback_size =
        static_cast<VkDeviceSize>(width) * static_cast<VkDeviceSize>(height) * VkDeviceSize{4};
    VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.size = readback_size;
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(g_vulkan.device, &buffer_info, nullptr, &g_vulkan.readback_buffer) != VK_SUCCESS) {
        return false;
    }

    VkMemoryRequirements buffer_requirements{};
    vkGetBufferMemoryRequirements(g_vulkan.device, g_vulkan.readback_buffer, &buffer_requirements);
    const std::uint32_t buffer_memory_index = find_memory_type_index(
        buffer_requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (buffer_memory_index == UINT32_MAX) {
        return false;
    }

    VkMemoryAllocateInfo buffer_allocate_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    buffer_allocate_info.allocationSize = buffer_requirements.size;
    buffer_allocate_info.memoryTypeIndex = buffer_memory_index;
    if (vkAllocateMemory(g_vulkan.device, &buffer_allocate_info, nullptr, &g_vulkan.readback_buffer_memory) != VK_SUCCESS) {
        return false;
    }
    if (vkBindBufferMemory(g_vulkan.device, g_vulkan.readback_buffer, g_vulkan.readback_buffer_memory, 0) != VK_SUCCESS) {
        return false;
    }

    g_vulkan.output_width = width;
    g_vulkan.output_height = height;
    g_vulkan.output_image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    g_vulkan.accumulation_image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    return true;
}

bool ensure_output_resources(int width, int height) {
    if (width <= 0 || height <= 0) {
        return false;
    }
    if (g_vulkan.output_image != VK_NULL_HANDLE &&
        g_vulkan.readback_buffer != VK_NULL_HANDLE &&
        g_vulkan.output_width == width &&
        g_vulkan.output_height == height) {
        return true;
    }
    return create_output_resources(width, height);
}


rtvdb::vec3 subtract(rtvdb::vec3 a, rtvdb::vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

rtvdb::vec3 cross(rtvdb::vec3 a, rtvdb::vec3 b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

float length_sq(rtvdb::vec3 value) {
    return value.x * value.x + value.y * value.y + value.z * value.z;
}

float dot(rtvdb::vec3 a, rtvdb::vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

rtvdb::vec3 add(rtvdb::vec3 a, rtvdb::vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

rtvdb::vec3 scale(rtvdb::vec3 value, float scalar) {
    return {value.x * scalar, value.y * scalar, value.z * scalar};
}

rtvdb::vec3 lerp(rtvdb::vec3 a, rtvdb::vec3 b, float t) {
    return add(scale(a, 1.0f - t), scale(b, t));
}

rtvdb::vec3 normalize_or(rtvdb::vec3 value, rtvdb::vec3 fallback) {
    const float len2 = length_sq(value);
    if (len2 <= 1.0e-12f) {
        return fallback;
    }
    const float inv_len = 1.0f / std::sqrt(len2);
    return {value.x * inv_len, value.y * inv_len, value.z * inv_len};
}

rtvdb::vec3 hash_color(std::uint32_t seed) {
    auto fract = [](float value) {
        return value - std::floor(value);
    };

    rtvdb::vec3 h{
        fract(static_cast<float>(seed) * 0.1031f),
        fract(static_cast<float>(seed) * 0.11369f),
        fract(static_cast<float>(seed) * 0.13787f),
    };
    const float shared = h.x * (h.y + 19.19f) + h.y * (h.z + 19.19f) + h.z * (h.x + 19.19f);
    h = {
        fract(h.x + shared),
        fract(h.y + shared),
        fract(h.z + shared),
    };
    return {
        fract((h.x + h.y) * h.z),
        fract((h.x + h.z) * h.y),
        fract((h.y + h.z) * h.x),
    };
}

float encode_srgb_channel(float value) {
    const float x = (std::clamp)(value, 0.0f, 1.0f);
    if (x <= 0.0031308f) {
        return x * 12.92f;
    }
    return 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
}

void fill_projection_parameters(
    const rtvdb::camera &camera,
    rtvdb::camera_projection projection,
    float aspect,
    float* out_param0,
    float* out_param1)
{
    if (out_param0 == nullptr || out_param1 == nullptr) {
        return;
    }

    switch (projection) {
    case rtvdb::camera_projection::fisheye:
        *out_param0 = camera.fisheye_phi_degrees * 3.1415926535f / 180.0f;
        *out_param1 = camera.fisheye_theta_degrees * 3.1415926535f / 180.0f;
        break;
    case rtvdb::camera_projection::orthographic:
        *out_param1 = camera.orthographic_height;
        *out_param0 = *out_param1 * aspect;
        break;
    case rtvdb::camera_projection::perspective:
    default: {
        const float fov_radians = camera.vertical_fov_degrees * 3.1415926535f / 180.0f;
        *out_param0 = std::tan(fov_radians * 0.5f);
        *out_param1 = 0.0f;
        break;
    }
    }
}

float halton(std::uint32_t index, std::uint32_t base) {
    float result = 0.0f;
    float factor = 1.0f / static_cast<float>(base);
    std::uint32_t value = index;
    while (value > 0) {
        result += factor * static_cast<float>(value % base);
        value /= base;
        factor /= static_cast<float>(base);
    }
    return result;
}

void fill_accumulation_jitter(std::uint32_t sample_index, float out_jitter[2]) {
    if (sample_index == 0u) {
        out_jitter[0] = 0.0f;
        out_jitter[1] = 0.0f;
        return;
    }
    out_jitter[0] = halton(sample_index + 1u, 2u) - 0.5f;
    out_jitter[1] = halton(sample_index + 1u, 3u) - 0.5f;
}

accumulation_key make_accumulation_key(
    const frame_scene &scene,
    bool has_frame,
    const rt_scene_build &build,
    int width,
    int height,
    std::uint32_t display_mode_value)
{
    accumulation_key key{};
    hover_highlight highlight{};
    get_hover_highlight(&highlight);
    key.build_revision = build.revision;
    key.width = static_cast<std::uint32_t>(width);
    key.height = static_cast<std::uint32_t>(height);
    key.display_mode = display_mode_value;
    key.has_frame = has_frame ? 1u : 0u;
    key.hover_highlight_kind = static_cast<std::uint32_t>(highlight.kind);
    key.hover_primitive_index = highlight.primitive_index;
    key.camera_origin[0] = scene.camera.origin.x;
    key.camera_origin[1] = scene.camera.origin.y;
    key.camera_origin[2] = scene.camera.origin.z;
    key.camera_target[0] = scene.camera.target.x;
    key.camera_target[1] = scene.camera.target.y;
    key.camera_target[2] = scene.camera.target.z;
    key.camera_up[0] = scene.camera.up.x;
    key.camera_up[1] = scene.camera.up.y;
    key.camera_up[2] = scene.camera.up.z;
    key.camera_vertical_fov_degrees = scene.camera.vertical_fov_degrees;
    key.camera_projection = static_cast<std::uint32_t>(scene.camera.projection);
    key.camera_fisheye_theta_degrees = scene.camera.fisheye_theta_degrees;
    key.camera_fisheye_phi_degrees = scene.camera.fisheye_phi_degrees;
    key.camera_orthographic_height = scene.camera.orthographic_height;
    key.camera_projection_blend_from = static_cast<std::uint32_t>(scene.projection_blend_from);
    key.camera_projection_blend_to = static_cast<std::uint32_t>(scene.projection_blend_to);
    key.camera_projection_blend_t = scene.projection_blend_t;
    return key;
}

bool accumulation_key_equals(const accumulation_key &a, const accumulation_key &b) {
    return std::memcmp(&a, &b, sizeof(accumulation_key)) == 0;
}

bool build_projection_ray(
    const frame_scene &scene,
    int width,
    int height,
    int pixel_x,
    int pixel_y,
    rtvdb::vec3* out_origin,
    rtvdb::vec3* out_direction)
{
    if (out_origin == nullptr || out_direction == nullptr || width <= 0 || height <= 0) {
        return false;
    }

    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    const rtvdb::camera &camera = scene.camera;
    const rtvdb::vec3 forward = normalize_or(subtract(camera.target, camera.origin), {0.0f, 0.0f, 1.0f});
    rtvdb::vec3 up = normalize_or(camera.up, {0.0f, 1.0f, 0.0f});
    const rtvdb::vec3 right = normalize_or(cross(forward, up), {1.0f, 0.0f, 0.0f});
    up = normalize_or(cross(right, forward), {0.0f, 1.0f, 0.0f});

    const float u = ((static_cast<float>(pixel_x) + 0.5f) / static_cast<float>(width)) * 2.0f - 1.0f;
    const float v = 1.0f - ((static_cast<float>(pixel_y) + 0.5f) / static_cast<float>(height)) * 2.0f;

    auto build_one = [&](rtvdb::camera_projection projection, rtvdb::vec3* ray_origin, rtvdb::vec3* ray_direction) {
        *ray_origin = camera.origin;
        *ray_direction = forward;
        switch (projection) {
        case rtvdb::camera_projection::orthographic: {
            const float ortho_height = (std::max)(camera.orthographic_height, 0.001f);
            const float ortho_width = ortho_height * aspect;
            *ray_origin = add(
                *ray_origin,
                add(scale(right, u * ortho_width * 0.5f), scale(up, v * ortho_height * 0.5f)));
            *ray_direction = forward;
            break;
        }
        case rtvdb::camera_projection::fisheye: {
            const float yaw = u * camera.fisheye_phi_degrees * 3.1415926535f / 360.0f;
            const float pitch = v * camera.fisheye_theta_degrees * 3.1415926535f / 360.0f;
            *ray_direction = normalize_or(
                add(
                    add(scale(forward, std::cos(yaw) * std::cos(pitch)), scale(right, std::sin(yaw))),
                    scale(up, std::sin(pitch))),
                forward);
            break;
        }
        case rtvdb::camera_projection::perspective:
        default: {
            const float tan_half_fov =
                std::tan(camera.vertical_fov_degrees * 3.1415926535f / 360.0f);
            *ray_direction = normalize_or(
                add(
                    forward,
                    add(scale(right, u * aspect * tan_half_fov), scale(up, v * tan_half_fov))),
                forward);
            break;
        }
        }
    };

    build_one(scene.projection_blend_from, out_origin, out_direction);
    if (scene.projection_blend_from != scene.projection_blend_to && scene.projection_blend_t < 1.0f) {
        rtvdb::vec3 origin_b{};
        rtvdb::vec3 direction_b{};
        build_one(scene.projection_blend_to, &origin_b, &direction_b);
        const float blend_t = (std::clamp)(scene.projection_blend_t, 0.0f, 1.0f);
        *out_origin = lerp(*out_origin, origin_b, blend_t);
        *out_direction = normalize_or(lerp(*out_direction, direction_b, blend_t), *out_direction);
    }
    return true;
}

bool intersect_triangle(
    rtvdb::vec3 ray_origin,
    rtvdb::vec3 ray_direction,
    const triangle &triangle_value,
    float* out_t)
{
    const rtvdb::vec3 edge1 = subtract(triangle_value.b, triangle_value.a);
    const rtvdb::vec3 edge2 = subtract(triangle_value.c, triangle_value.a);
    const rtvdb::vec3 p = cross(ray_direction, edge2);
    const float det = dot(edge1, p);
    if (std::fabs(det) <= 1.0e-8f) {
        return false;
    }
    const float inv_det = 1.0f / det;
    const rtvdb::vec3 tvec = subtract(ray_origin, triangle_value.a);
    const float u = dot(tvec, p) * inv_det;
    if (u < 0.0f || u > 1.0f) {
        return false;
    }
    const rtvdb::vec3 q = cross(tvec, edge1);
    const float v = dot(ray_direction, q) * inv_det;
    if (v < 0.0f || (u + v) > 1.0f) {
        return false;
    }
    const float t = dot(edge2, q) * inv_det;
    if (t <= 1.0e-5f) {
        return false;
    }
    if (out_t != nullptr) {
        *out_t = t;
    }
    return true;
}

bool intersect_sphere(
    rtvdb::vec3 ray_origin,
    rtvdb::vec3 ray_direction,
    const point &point_value,
    float* out_t)
{
    const rtvdb::vec3 oc = subtract(ray_origin, point_value.position);
    const float a = dot(ray_direction, ray_direction);
    const float b = dot(oc, ray_direction);
    const float c = dot(oc, oc) - point_value.radius * point_value.radius;
    const float h = b * b - a * c;
    if (h < 0.0f) {
        return false;
    }
    const float sqrt_h = std::sqrt(h);
    const float t0 = (-b - sqrt_h) / a;
    const float t1 = (-b + sqrt_h) / a;
    const float t = t0 > 1.0e-5f ? t0 : t1;
    if (t <= 1.0e-5f) {
        return false;
    }
    if (out_t != nullptr) {
        *out_t = t;
    }
    return true;
}

bool intersect_line_capsule(
    rtvdb::vec3 ray_origin,
    rtvdb::vec3 ray_direction,
    const line &line_value,
    bool is_pick_pass,
    float* out_t)
{
    if (is_pick_pass && has_line_flags(line_value.flags, line_flags::non_pickable)) {
        return false;
    }

    const rtvdb::vec3 pa = line_value.a;
    const rtvdb::vec3 pb = line_value.b;
    const float radius = line_value.radius;
    const float radius_sq = radius * radius;
    const rtvdb::vec3 ba = subtract(pb, pa);
    const float baba = dot(ba, ba);
    const float rdrd = dot(ray_direction, ray_direction);
    const float length_sq_epsilon = 1.0e-12f;
    if (rdrd <= 1.0e-20f) {
        return false;
    }

    const rtvdb::vec3 shift_target = baba <= length_sq_epsilon ? pa : lerp(pa, pb, 0.5f);
    const float projected_t =
        (std::clamp)(dot(subtract(shift_target, ray_origin), ray_direction) / rdrd, 1.0e-5f, 10000.0f);
    const rtvdb::vec3 ro = add(ray_origin, scale(ray_direction, projected_t));
    const rtvdb::vec3 oa = subtract(ro, pa);

    float best_t = 10000.0f;
    bool found = false;

    if (baba <= length_sq_epsilon) {
        const float b = dot(oa, ray_direction);
        const float c = dot(oa, oa) - radius_sq;
        const float h = b * b - rdrd * c;
        if (h < 0.0f) {
            return false;
        }
        const float s = std::sqrt(h);
        const float t0 = (-b - s) / rdrd + projected_t;
        const float t1 = (-b + s) / rdrd + projected_t;
        if (t0 > 1.0e-5f && t0 < best_t) {
            best_t = t0;
            found = true;
        }
        if (t1 > 1.0e-5f && t1 < best_t) {
            best_t = t1;
            found = true;
        }
    } else {
        const float bard = dot(ba, ray_direction);
        const float baoa = dot(ba, oa);
        const float rdoa = dot(ray_direction, oa);
        const float oaoa = dot(oa, oa);
        const float a = baba * rdrd - bard * bard;
        const float b = baba * rdoa - baoa * bard;
        const float c = baba * oaoa - baoa * baoa - radius_sq * baba;
        const float h = b * b - a * c;
        if (h >= 0.0f && std::fabs(a) > length_sq_epsilon) {
            const float s = std::sqrt(h);
            const float local_t0 = (-b - s) / a;
            const float t0 = local_t0 + projected_t;
            const float y0 = baoa + local_t0 * bard;
            if (t0 > 1.0e-5f && t0 < best_t && y0 >= 0.0f && y0 <= baba) {
                best_t = t0;
                found = true;
            }

            const float local_t1 = (-b + s) / a;
            const float t1 = local_t1 + projected_t;
            const float y1 = baoa + local_t1 * bard;
            if (t1 > 1.0e-5f && t1 < best_t && y1 >= 0.0f && y1 <= baba) {
                best_t = t1;
                found = true;
            }
        }
    }

    float sphere_t = 0.0f;
    if (intersect_sphere(ray_origin, ray_direction, point{pa, radius, line_value.color, line_value.user_data}, &sphere_t) &&
        sphere_t < best_t) {
        best_t = sphere_t;
        found = true;
    }
    if (intersect_sphere(ray_origin, ray_direction, point{pb, radius, line_value.color, line_value.user_data}, &sphere_t) &&
        sphere_t < best_t) {
        best_t = sphere_t;
        found = true;
    }

    if (found && out_t != nullptr) {
        *out_t = best_t;
    }
    return found;
}

void update_viewer_constants(
    const frame_scene &scene,
    const rt_scene_build &build,
    bool has_frame,
    int width,
    int height,
    bool is_pick_pass = false,
    int pick_pixel_x = 0,
    int pick_pixel_y = 0)
{
    viewer_constants_gpu constants{};
    hover_highlight highlight{};
    get_hover_highlight(&highlight);

    constants.size_and_mode[0] = static_cast<std::uint32_t>(width);
    constants.size_and_mode[1] = static_cast<std::uint32_t>(height);
    display_mode mode = display_mode::triangle_normal;
    get_display_mode(&mode);
    constants.size_and_mode[2] = static_cast<std::uint32_t>(mode);
    constants.size_and_mode[3] = g_vulkan.accumulation_sample_count;

    const float aspect = height == 0 ? 1.0f : static_cast<float>(width) / static_cast<float>(height);
    rtvdb::camera camera{};
    if (has_frame) {
        camera = scene.camera;
    }
    const rtvdb::vec3 forward = normalize_or(subtract(camera.target, camera.origin), {0.0f, 0.0f, 1.0f});
    rtvdb::vec3 up = normalize_or(camera.up, {0.0f, 1.0f, 0.0f});
    const rtvdb::vec3 right = normalize_or(cross(forward, up), {1.0f, 0.0f, 0.0f});
    up = normalize_or(cross(right, forward), {0.0f, 1.0f, 0.0f});

    constants.origin[0] = camera.origin.x;
    constants.origin[1] = camera.origin.y;
    constants.origin[2] = camera.origin.z;
    constants.origin[3] = 0.0f;
    constants.forward[0] = forward.x;
    constants.forward[1] = forward.y;
    constants.forward[2] = forward.z;
    constants.forward[3] = 0.0f;
    constants.right[0] = right.x;
    constants.right[1] = right.y;
    constants.right[2] = right.z;
    constants.right[3] = 0.0f;
    constants.up[0] = up.x;
    constants.up[1] = up.y;
    constants.up[2] = up.z;
    constants.up[3] = 0.0f;

    if (build.bounds.valid) {
        constants.scene_bounds_min[0] = build.bounds.min.x;
        constants.scene_bounds_min[1] = build.bounds.min.y;
        constants.scene_bounds_min[2] = build.bounds.min.z;
        constants.scene_bounds_max[0] = build.bounds.max.x;
        constants.scene_bounds_max[1] = build.bounds.max.y;
        constants.scene_bounds_max[2] = build.bounds.max.z;
        constants.scene_bounds_max[3] = 1.0f;
    }

    float projection_from0 = 0.0f;
    float projection_from1 = 0.0f;
    float projection_to0 = 0.0f;
    float projection_to1 = 0.0f;
    fill_projection_parameters(camera, scene.projection_blend_from, aspect, &projection_from0, &projection_from1);
    fill_projection_parameters(camera, scene.projection_blend_to, aspect, &projection_to0, &projection_to1);
    constants.projection_from[0] = projection_from0;
    constants.projection_from[1] = projection_from1;
    constants.projection_from[2] = static_cast<float>(static_cast<std::uint32_t>(scene.projection_blend_from));
    constants.projection_from[3] = aspect;
    constants.projection_to[0] = projection_to0;
    constants.projection_to[1] = projection_to1;
    constants.projection_to[2] = static_cast<float>(static_cast<std::uint32_t>(scene.projection_blend_to));
    constants.projection_to[3] = 0.0f;

    constants.projection_modes[0] = static_cast<std::uint32_t>(scene.projection_blend_from);
    constants.projection_modes[1] = static_cast<std::uint32_t>(scene.projection_blend_to);
    constants.projection_modes[2] = static_cast<std::uint32_t>(highlight.kind);
    constants.projection_modes[3] = highlight.primitive_index;

    float jitter[2]{};
    fill_accumulation_jitter(g_vulkan.accumulation_sample_count, jitter);
    constants.blend_and_jitter[0] = scene.projection_blend_t;
    constants.blend_and_jitter[1] = jitter[0];
    constants.blend_and_jitter[2] = jitter[1];
    constants.blend_and_jitter[3] = 0.85f;

    constants.pick_and_flags[0] = static_cast<std::uint32_t>(build.triangle_count);
    constants.pick_and_flags[1] = static_cast<std::uint32_t>(build.point_count);
    constants.pick_and_flags[2] = static_cast<std::uint32_t>(build.line_count);
    constants.pick_and_flags[3] = has_frame ? 1u : 0u;
    constants.pick_params[0] = static_cast<std::uint32_t>((std::max)(pick_pixel_x, 0));
    constants.pick_params[1] = static_cast<std::uint32_t>((std::max)(pick_pixel_y, 0));
    constants.pick_params[2] = is_pick_pass ? 1u : 0u;
    constants.pick_params[3] = 0u;

    (void)upload_buffer_data(&g_vulkan.viewer_constants, &constants, sizeof(constants));
}

bool ensure_scene_buffers(const rt_scene_build &build) {
    if (build.revision != 0 &&
        g_vulkan.synced_build_revision == build.revision &&
        g_vulkan.scene_positions.buffer != VK_NULL_HANDLE &&
        g_vulkan.scene_indices.buffer != VK_NULL_HANDLE &&
        g_vulkan.triangle_colors.buffer != VK_NULL_HANDLE &&
        g_vulkan.instance_metadata.buffer != VK_NULL_HANDLE &&
        g_vulkan.point_buffer.buffer != VK_NULL_HANDLE &&
        g_vulkan.line_buffer.buffer != VK_NULL_HANDLE) {
        if (g_vulkan.viewer_constants.buffer == VK_NULL_HANDLE &&
            !create_buffer(
                sizeof(viewer_constants_gpu),
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                &g_vulkan.viewer_constants)) {
            return false;
        }
        return ensure_pick_output_buffer();
    }

    std::vector<vec4_gpu> positions(build.vertex_count);
    std::vector<vec4_gpu> triangle_colors(build.triangle_count);
    std::vector<point_gpu> points(build.point_count);
    std::vector<line_gpu> lines(build.line_count);
    const geometry_metadata_gpu triangle_metadata{
        0u,
        0u,
        0u,
        static_cast<std::uint32_t>(build.triangle_count)};
    for (std::size_t vertex_index = 0; vertex_index < build.vertex_count; ++vertex_index) {
        positions[vertex_index] = {
            build.vertices[vertex_index].position.x,
            build.vertices[vertex_index].position.y,
            build.vertices[vertex_index].position.z,
            0.0f
        };
    }
    for (const rt_scene_chunk &chunk : build.chunks) {
        for (std::size_t local_triangle_index = 0; local_triangle_index < chunk.triangle_count; ++local_triangle_index) {
            const std::size_t index_base = chunk.index_offset + local_triangle_index * 3u;
            const std::uint32_t ia = build.indices[index_base + 0];
            const rt_scene_vertex &va = build.vertices[ia];
            triangle_colors[chunk.first_triangle + local_triangle_index] = {
                encode_srgb_channel(va.color.r),
                encode_srgb_channel(va.color.g),
                encode_srgb_channel(va.color.b),
                va.color.a
            };
        }
    }
    for (std::size_t point_index = 0; point_index < build.point_count; ++point_index) {
        const point &source = build.points[point_index];
        points[point_index] = {
            {source.position.x, source.position.y, source.position.z, source.radius},
            {source.color.r, source.color.g, source.color.b, source.color.a}
        };
    }
    for (std::size_t line_index = 0; line_index < build.line_count; ++line_index) {
        const line &source = build.lines[line_index];
        line_gpu &target = lines[line_index];
        target.a_radius = {source.a.x, source.a.y, source.a.z, source.radius};
        target.b_pad = {source.b.x, source.b.y, source.b.z, 0.0f};
        target.color = {source.color.r, source.color.g, source.color.b, source.color.a};
        target.flags = static_cast<std::uint32_t>(source.flags);
    }

    if (!ensure_uploaded_buffer(
            &g_vulkan.scene_positions,
            positions.size() * sizeof(vec4_gpu),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            positions.data(),
            positions.size() * sizeof(vec4_gpu))) {
        return false;
    }

    if (!ensure_uploaded_buffer(
            &g_vulkan.scene_indices,
            build.indices.size() * sizeof(std::uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            build.indices.data(),
            build.indices.size() * sizeof(std::uint32_t))) {
        return false;
    }

    if (!ensure_uploaded_buffer(
            &g_vulkan.triangle_colors,
            triangle_colors.size() * sizeof(vec4_gpu),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            triangle_colors.data(),
            triangle_colors.size() * sizeof(vec4_gpu))) {
        return false;
    }
    if (!ensure_uploaded_buffer(
            &g_vulkan.instance_metadata,
            sizeof(triangle_metadata),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &triangle_metadata,
            sizeof(triangle_metadata))) {
        return false;
    }
    if (!ensure_uploaded_buffer(
            &g_vulkan.point_buffer,
            points.size() * sizeof(point_gpu),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            points.data(),
            points.size() * sizeof(point_gpu))) {
        return false;
    }
    if (!ensure_uploaded_buffer(
            &g_vulkan.line_buffer,
            lines.size() * sizeof(line_gpu),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            lines.data(),
            lines.size() * sizeof(line_gpu))) {
        return false;
    }

    if (g_vulkan.viewer_constants.buffer == VK_NULL_HANDLE &&
        !create_buffer(
            sizeof(viewer_constants_gpu),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &g_vulkan.viewer_constants)) {
        return false;
    }
    if (!ensure_pick_output_buffer()) {
        return false;
    }
    return true;
}

bool ensure_pick_output_buffer() {
    if (g_vulkan.pick_output.buffer != VK_NULL_HANDLE) {
        return true;
    }
    return create_buffer(
        sizeof(gpu_pick_result),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &g_vulkan.pick_output);
}

bool ensure_scratch_buffer(VkDeviceSize size) {
    if (g_vulkan.scratch_buffer.buffer != VK_NULL_HANDLE && g_vulkan.scratch_buffer.size >= size) {
        return true;
    }
    return create_buffer(
        size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        &g_vulkan.scratch_buffer);
}

bool chunk_blas_entry_matches(
    const vulkan_chunk_blas_entry &entry,
    const rt_scene_chunk &chunk,
    VkDeviceSize required_size)
{
    return entry.accel.handle != VK_NULL_HANDLE &&
        entry.accel.storage.size >= required_size &&
        entry.first_triangle == chunk.first_triangle &&
        entry.triangle_count == chunk.triangle_count &&
        entry.fingerprint == chunk.fingerprint &&
        entry.vertex_offset == chunk.vertex_offset &&
        entry.vertex_count == chunk.vertex_count &&
        entry.index_offset == chunk.index_offset &&
        entry.index_count == chunk.index_count;
}

void hash_procedural_geometry_value(std::uint64_t* fingerprint, const void* data, std::size_t size) {
    constexpr std::uint64_t kFnvPrime = 1099511628211ull;
    const std::uint8_t* bytes = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < size; ++i) {
        *fingerprint ^= bytes[i];
        *fingerprint *= kFnvPrime;
    }
}

std::uint64_t point_geometry_fingerprint(const rt_scene_build &build) {
    constexpr std::uint64_t kFnvOffset = 14695981039346656037ull;
    std::uint64_t fingerprint = kFnvOffset;
    hash_procedural_geometry_value(&fingerprint, &build.point_count, sizeof(build.point_count));
    for (const point &value : build.points) {
        hash_procedural_geometry_value(&fingerprint, &value.position, sizeof(value.position));
        hash_procedural_geometry_value(&fingerprint, &value.radius, sizeof(value.radius));
    }
    return fingerprint;
}

std::uint64_t line_geometry_fingerprint(const rt_scene_build &build) {
    constexpr std::uint64_t kFnvOffset = 14695981039346656037ull;
    std::uint64_t fingerprint = kFnvOffset;
    hash_procedural_geometry_value(&fingerprint, &build.line_count, sizeof(build.line_count));
    for (const line &value : build.lines) {
        hash_procedural_geometry_value(&fingerprint, &value.a, sizeof(value.a));
        hash_procedural_geometry_value(&fingerprint, &value.radius, sizeof(value.radius));
        hash_procedural_geometry_value(&fingerprint, &value.b, sizeof(value.b));
    }
    return fingerprint;
}

bool build_triangle_acceleration_structures(
    const rt_scene_build &build,
    double* out_submit_cpu_ms = nullptr,
    double* out_wait_ms = nullptr)
{
    if (build.triangle_count == 0 && build.point_count == 0 && build.line_count == 0) {
        g_vulkan.synced_build_revision = build.revision;
        g_vulkan.synced_blas_reused_count = 0;
        g_vulkan.synced_blas_rebuilt_count = 0;
        g_vulkan.synced_blas_reused_chunk_count = 0;
        g_vulkan.synced_blas_rebuilt_chunk_count = 0;
        g_vulkan.synced_tlas_rebuild_count = 0;
        return true;
    }
    std::vector<geometry_metadata_gpu> instance_metadata;
    std::vector<VkAccelerationStructureInstanceKHR> instances;
    std::vector<VkAccelerationStructureGeometryKHR> chunk_geometries;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> chunk_build_ranges;
    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> chunk_build_infos;
    VkDeviceSize max_scratch_size = 0;
    std::size_t blas_reused_count = 0;
    std::size_t blas_rebuilt_count = 0;
    std::size_t blas_reused_chunk_count = 0;
    std::size_t blas_rebuilt_chunk_count = 0;

    const std::uint64_t point_fingerprint = point_geometry_fingerprint(build);
    const std::uint64_t line_fingerprint = line_geometry_fingerprint(build);
    const bool point_geometry_changed = g_vulkan.point_primitive_count != build.point_count ||
        g_vulkan.point_geometry_fingerprint != point_fingerprint;
    const bool line_geometry_changed = g_vulkan.line_primitive_count != build.line_count ||
        g_vulkan.line_geometry_fingerprint != line_fingerprint;
    if (build.point_count == 0) {
        for (vulkan_procedural_blas_entry &entry : g_vulkan.point_blas_entries) {
            destroy_acceleration_structure(&entry.accel);
        }
        g_vulkan.point_blas_entries.clear();
        destroy_buffer(&g_vulkan.point_aabb_buffer);
        g_vulkan.point_geometry_fingerprint = 0;
        g_vulkan.point_primitive_count = 0;
    } else if (point_geometry_changed) {
        std::vector<VkAabbPositionsKHR> point_aabbs(build.point_count);
        for (std::size_t point_index = 0; point_index < build.point_count; ++point_index) {
            const point &value = build.points[point_index];
            const float radius = (std::max)(value.radius, 1.0e-6f);
            point_aabbs[point_index] = {
                value.position.x - radius,
                value.position.y - radius,
                value.position.z - radius,
                value.position.x + radius,
                value.position.y + radius,
                value.position.z + radius};
        }
        if (!ensure_uploaded_buffer(
                &g_vulkan.point_aabb_buffer,
                point_aabbs.size() * sizeof(VkAabbPositionsKHR),
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                point_aabbs.data(),
                point_aabbs.size() * sizeof(VkAabbPositionsKHR))) {
            append_startup_log("Vulkan point AABB upload failed");
            return false;
        }
    }
    if (build.line_count == 0) {
        for (vulkan_procedural_blas_entry &entry : g_vulkan.line_blas_entries) {
            destroy_acceleration_structure(&entry.accel);
        }
        g_vulkan.line_blas_entries.clear();
        destroy_buffer(&g_vulkan.line_aabb_buffer);
        g_vulkan.line_geometry_fingerprint = 0;
        g_vulkan.line_primitive_count = 0;
    } else if (line_geometry_changed) {
        std::vector<VkAabbPositionsKHR> line_aabbs(build.line_count);
        for (std::size_t line_index = 0; line_index < build.line_count; ++line_index) {
            const line &value = build.lines[line_index];
            const float radius = (std::max)(value.radius, 1.0e-6f);
            line_aabbs[line_index] = {
                (std::min)(value.a.x, value.b.x) - radius,
                (std::min)(value.a.y, value.b.y) - radius,
                (std::min)(value.a.z, value.b.z) - radius,
                (std::max)(value.a.x, value.b.x) + radius,
                (std::max)(value.a.y, value.b.y) + radius,
                (std::max)(value.a.z, value.b.z) + radius};
        }
        if (!ensure_uploaded_buffer(
                &g_vulkan.line_aabb_buffer,
                line_aabbs.size() * sizeof(VkAabbPositionsKHR),
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                line_aabbs.data(),
                line_aabbs.size() * sizeof(VkAabbPositionsKHR))) {
            append_startup_log("Vulkan line AABB upload failed");
            return false;
        }
    }

    const std::size_t build_geometry_count =
        build.chunks.size() + build.point_groups.size() + build.line_groups.size();
    chunk_geometries.reserve(build_geometry_count);
    chunk_build_ranges.reserve(build_geometry_count);
    chunk_build_infos.reserve(build_geometry_count);

    {
        const VkDeviceAddress position_address = buffer_device_address(g_vulkan.scene_positions);
        const VkDeviceAddress index_address = buffer_device_address(g_vulkan.scene_indices);
        if (g_vulkan.chunk_blas_cache.size() > build.chunks.size()) {
            for (std::size_t i = build.chunks.size(); i < g_vulkan.chunk_blas_cache.size(); ++i) {
                destroy_acceleration_structure(&g_vulkan.chunk_blas_cache[i].accel);
            }
            g_vulkan.chunk_blas_cache.resize(build.chunks.size());
        } else if (g_vulkan.chunk_blas_cache.size() < build.chunks.size()) {
            g_vulkan.chunk_blas_cache.resize(build.chunks.size());
        }

        instance_metadata.reserve(build.chunks.size());
        instances.reserve(build.chunks.size());

        for (std::size_t chunk_index = 0; chunk_index < build.chunks.size(); ++chunk_index) {
            const rt_scene_chunk &chunk = build.chunks[chunk_index];
            VkAccelerationStructureGeometryTrianglesDataKHR triangles{
                VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
            triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
            triangles.vertexData.deviceAddress = position_address;
            triangles.vertexStride = sizeof(vec4_gpu);
            triangles.maxVertex = static_cast<std::uint32_t>(build.vertex_count);
            triangles.indexType = VK_INDEX_TYPE_UINT32;
            triangles.indexData.deviceAddress =
                index_address + static_cast<VkDeviceAddress>(chunk.index_offset * sizeof(std::uint32_t));

            VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
            geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
            geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
            geometry.geometry.triangles = triangles;

            VkAccelerationStructureBuildRangeInfoKHR build_range{};
            build_range.primitiveCount = static_cast<std::uint32_t>(chunk.triangle_count);

            VkAccelerationStructureBuildGeometryInfoKHR build_info{
                VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
            build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            build_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
            build_info.geometryCount = 1;
            build_info.pGeometries = &geometry;

            VkAccelerationStructureBuildSizesInfoKHR build_sizes{
                VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
            g_vulkan.get_acceleration_structure_build_sizes(
                g_vulkan.device,
                VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                &build_info,
                &build_range.primitiveCount,
                &build_sizes);

            vulkan_chunk_blas_entry &entry = g_vulkan.chunk_blas_cache[chunk_index];
            if (chunk_blas_entry_matches(entry, chunk, build_sizes.accelerationStructureSize)) {
                ++blas_reused_count;
                ++blas_reused_chunk_count;
            } else {
                if (!ensure_acceleration_structure(
                        &entry.accel,
                        build_sizes.accelerationStructureSize,
                        VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR)) {
                    append_startup_log("Vulkan triangle BLAS allocation failed");
                    return false;
                }
                entry.first_triangle = chunk.first_triangle;
                entry.triangle_count = chunk.triangle_count;
                entry.fingerprint = chunk.fingerprint;
                entry.vertex_offset = chunk.vertex_offset;
                entry.vertex_count = chunk.vertex_count;
                entry.index_offset = chunk.index_offset;
                entry.index_count = chunk.index_count;
                build_info.dstAccelerationStructure = entry.accel.handle;
                chunk_geometries.push_back(geometry);
                chunk_build_ranges.push_back(build_range);
                build_info.pGeometries = &chunk_geometries.back();
                chunk_build_infos.push_back(build_info);
                max_scratch_size = (std::max)(max_scratch_size, build_sizes.buildScratchSize);
                ++blas_rebuilt_count;
                ++blas_rebuilt_chunk_count;
            }

            VkAccelerationStructureDeviceAddressInfoKHR blas_address_info{
                VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
            blas_address_info.accelerationStructure = entry.accel.handle;
            entry.accel.device_address = g_vulkan.get_acceleration_structure_device_address(
                g_vulkan.device,
                &blas_address_info);

            instance_metadata.push_back({
                static_cast<std::uint32_t>(chunk.first_triangle),
                static_cast<std::uint32_t>(chunk.index_offset),
                0u,
                static_cast<std::uint32_t>(chunk.triangle_count)});

            VkAccelerationStructureInstanceKHR instance{};
            instance.transform.matrix[0][0] = 1.0f;
            instance.transform.matrix[1][1] = 1.0f;
            instance.transform.matrix[2][2] = 1.0f;
            instance.instanceCustomIndex = static_cast<std::uint32_t>(chunk_index);
            instance.mask = chunk.visible ? 0xff : 0x00;
            instance.instanceShaderBindingTableRecordOffset = 0;
            instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            instance.accelerationStructureReference = entry.accel.device_address;
            instances.push_back(instance);
        }
    }

    const auto append_procedural_groups = [&](const std::vector<rt_procedural_group> &groups,
                                               const vulkan_buffer &aabb_buffer,
                                               std::uint64_t geometry_fingerprint,
                                               std::uint32_t hit_group,
                                               std::vector<vulkan_procedural_blas_entry>* entries) {
        while (entries->size() > groups.size()) {
            destroy_acceleration_structure(&entries->back().accel);
            entries->pop_back();
        }
        entries->resize(groups.size());
        for (std::size_t group_index = 0; group_index < groups.size(); ++group_index) {
            const rt_procedural_group &group = groups[group_index];
            vulkan_procedural_blas_entry &entry = (*entries)[group_index];
            VkAccelerationStructureGeometryAabbsDataKHR aabb_data{
                VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR};
            aabb_data.data.deviceAddress = buffer_device_address(aabb_buffer);
            aabb_data.stride = sizeof(VkAabbPositionsKHR);
            VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
            geometry.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
            geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
            geometry.geometry.aabbs = aabb_data;
            VkAccelerationStructureBuildRangeInfoKHR build_range{};
            build_range.primitiveCount = static_cast<std::uint32_t>(group.primitive_count);
            build_range.primitiveOffset = static_cast<std::uint32_t>(
                group.first_primitive * sizeof(VkAabbPositionsKHR));
            VkAccelerationStructureBuildGeometryInfoKHR build_info{
                VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
            build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            build_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
            build_info.geometryCount = 1;
            build_info.pGeometries = &geometry;
            VkAccelerationStructureBuildSizesInfoKHR build_sizes{
                VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
            g_vulkan.get_acceleration_structure_build_sizes(
                g_vulkan.device,
                VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                &build_info,
                &build_range.primitiveCount,
                &build_sizes);
            const bool reusable = entry.accel.handle != VK_NULL_HANDLE &&
                entry.geometry_fingerprint == geometry_fingerprint &&
                entry.first_primitive == group.first_primitive &&
                entry.primitive_count == group.primitive_count;
            if (reusable) {
                ++blas_reused_count;
            } else {
                if (!ensure_acceleration_structure(
                        &entry.accel,
                        build_sizes.accelerationStructureSize,
                        VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR)) {
                    return false;
                }
                entry.geometry_fingerprint = geometry_fingerprint;
                entry.first_primitive = group.first_primitive;
                entry.primitive_count = group.primitive_count;
                build_info.dstAccelerationStructure = entry.accel.handle;
                chunk_geometries.push_back(geometry);
                chunk_build_ranges.push_back(build_range);
                build_info.pGeometries = &chunk_geometries.back();
                chunk_build_infos.push_back(build_info);
                max_scratch_size = (std::max)(max_scratch_size, build_sizes.buildScratchSize);
                ++blas_rebuilt_count;
            }
            VkAccelerationStructureDeviceAddressInfoKHR address_info{
                VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
            address_info.accelerationStructure = entry.accel.handle;
            entry.accel.device_address = g_vulkan.get_acceleration_structure_device_address(g_vulkan.device, &address_info);
            instance_metadata.push_back({0u, 0u, static_cast<std::uint32_t>(group.first_primitive),
                                         static_cast<std::uint32_t>(group.primitive_count)});
            VkAccelerationStructureInstanceKHR instance{};
            instance.transform.matrix[0][0] = 1.0f;
            instance.transform.matrix[1][1] = 1.0f;
            instance.transform.matrix[2][2] = 1.0f;
            instance.instanceCustomIndex = static_cast<std::uint32_t>(group.first_primitive);
            instance.mask = group.visible ? 0xff : 0x00;
            instance.instanceShaderBindingTableRecordOffset = hit_group;
            instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            instance.accelerationStructureReference = entry.accel.device_address;
            instances.push_back(instance);
        }
        return true;
    };
    if (!append_procedural_groups(
            build.point_groups, g_vulkan.point_aabb_buffer, point_fingerprint, 1,
            &g_vulkan.point_blas_entries) ||
        !append_procedural_groups(
            build.line_groups, g_vulkan.line_aabb_buffer, line_fingerprint, 2,
            &g_vulkan.line_blas_entries)) {
        append_startup_log("Vulkan procedural BLAS allocation failed");
        return false;
    }
    g_vulkan.point_geometry_fingerprint = point_fingerprint;
    g_vulkan.point_primitive_count = build.point_count;
    g_vulkan.line_geometry_fingerprint = line_fingerprint;
    g_vulkan.line_primitive_count = build.line_count;

    if (!ensure_uploaded_buffer(
            &g_vulkan.instance_metadata,
            instance_metadata.size() * sizeof(geometry_metadata_gpu),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            instance_metadata.data(),
            instance_metadata.size() * sizeof(geometry_metadata_gpu))) {
        append_startup_log("Vulkan instance metadata upload failed");
        return false;
    }

    if (!ensure_uploaded_buffer(
            &g_vulkan.tlas_instance_buffer,
            instances.size() * sizeof(VkAccelerationStructureInstanceKHR),
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            instances.data(),
            instances.size() * sizeof(VkAccelerationStructureInstanceKHR))) {
        append_startup_log("Vulkan TLAS instance upload failed");
        return false;
    }

    VkAccelerationStructureGeometryInstancesDataKHR instances_data{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR};
    instances_data.arrayOfPointers = VK_FALSE;
    instances_data.data.deviceAddress = buffer_device_address(g_vulkan.tlas_instance_buffer);

    VkAccelerationStructureGeometryKHR top_geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    top_geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    top_geometry.geometry.instances = instances_data;

    VkAccelerationStructureBuildRangeInfoKHR top_build_range{};
    top_build_range.primitiveCount = static_cast<std::uint32_t>(instances.size());
    VkAccelerationStructureBuildGeometryInfoKHR top_build_info{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    top_build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    top_build_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
        VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    top_build_info.geometryCount = 1;
    top_build_info.pGeometries = &top_geometry;

    VkAccelerationStructureBuildSizesInfoKHR top_build_sizes{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    const std::uint32_t top_primitive_count = static_cast<std::uint32_t>(instances.size());
    g_vulkan.get_acceleration_structure_build_sizes(
        g_vulkan.device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &top_build_info,
        &top_primitive_count,
        &top_build_sizes);

    const bool can_update_tlas = g_vulkan.tlas.handle != VK_NULL_HANDLE &&
        g_vulkan.tlas_instance_count == instances.size() &&
        g_vulkan.tlas.storage.size >= top_build_sizes.accelerationStructureSize;
    if (!ensure_acceleration_structure(
            &g_vulkan.tlas,
            top_build_sizes.accelerationStructureSize,
            VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR)) {
        append_startup_log("Vulkan TLAS allocation failed");
        return false;
    }

    max_scratch_size = (std::max)(
        max_scratch_size,
        can_update_tlas ? top_build_sizes.updateScratchSize : top_build_sizes.buildScratchSize);
    if (!ensure_scratch_buffer(max_scratch_size)) {
        append_startup_log("Vulkan scratch buffer allocation failed");
        return false;
    }

    VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkResetFences(g_vulkan.device, 1, &g_vulkan.command_fence) != VK_SUCCESS ||
        vkResetCommandPool(g_vulkan.device, g_vulkan.command_pool, 0) != VK_SUCCESS ||
        vkBeginCommandBuffer(g_vulkan.command_buffer, &begin_info) != VK_SUCCESS) {
        append_startup_log("Vulkan accel build command buffer begin failed");
        return false;
    }
    reset_timestamp_query_region(timestamp_query_region::accel);
    write_timestamp_query_begin(timestamp_query_region::accel);

    const VkDeviceAddress scratch_address = buffer_device_address(g_vulkan.scratch_buffer);
    for (std::size_t i = 0; i < chunk_build_infos.size(); ++i) {
        chunk_build_infos[i].scratchData.deviceAddress = scratch_address;
        const VkAccelerationStructureBuildRangeInfoKHR* build_ranges[] = {&chunk_build_ranges[i]};
        g_vulkan.cmd_build_acceleration_structures(g_vulkan.command_buffer, 1, &chunk_build_infos[i], build_ranges);
        VkMemoryBarrier build_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        build_barrier.srcAccessMask =
            VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        build_barrier.dstAccessMask =
            VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        vkCmdPipelineBarrier(
            g_vulkan.command_buffer,
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

    if (!chunk_build_infos.empty()) {
        VkMemoryBarrier memory_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        memory_barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        memory_barrier.dstAccessMask =
            VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(
            g_vulkan.command_buffer,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            0,
            1,
            &memory_barrier,
            0,
            nullptr,
            0,
            nullptr);
    }

    top_build_info.dstAccelerationStructure = g_vulkan.tlas.handle;
    if (can_update_tlas) {
        top_build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
        top_build_info.srcAccelerationStructure = g_vulkan.tlas.handle;
    }
    top_build_info.scratchData.deviceAddress = scratch_address;
    const VkAccelerationStructureBuildRangeInfoKHR* top_build_ranges[] = {&top_build_range};
    g_vulkan.cmd_build_acceleration_structures(g_vulkan.command_buffer, 1, &top_build_info, top_build_ranges);
    write_timestamp_query_end(timestamp_query_region::accel);

    if (vkEndCommandBuffer(g_vulkan.command_buffer) != VK_SUCCESS) {
        append_startup_log("Vulkan accel build command buffer end failed");
        return false;
    }
    VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &g_vulkan.command_buffer;
    const auto submit_start = std::chrono::steady_clock::now();
    const bool submitted = vkQueueSubmit(g_vulkan.graphics_queue, 1, &submit_info, g_vulkan.command_fence) == VK_SUCCESS;
    if (out_submit_cpu_ms != nullptr) {
        *out_submit_cpu_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - submit_start).count();
    }
    const auto wait_start = std::chrono::steady_clock::now();
    const bool waited = submitted &&
        vkWaitForFences(g_vulkan.device, 1, &g_vulkan.command_fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
    if (out_wait_ms != nullptr) {
        *out_wait_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - wait_start).count();
    }
    if (!submitted || !waited) {
        append_startup_log("Vulkan accel build submit/wait failed");
        return false;
    }

    VkAccelerationStructureDeviceAddressInfoKHR tlas_address_info{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    tlas_address_info.accelerationStructure = g_vulkan.tlas.handle;
    g_vulkan.tlas.device_address = g_vulkan.get_acceleration_structure_device_address(
        g_vulkan.device,
        &tlas_address_info);
    g_vulkan.tlas_instance_count = instances.size();
    g_vulkan.synced_build_revision = build.revision;
    g_vulkan.synced_blas_reused_count = blas_reused_count;
    g_vulkan.synced_blas_rebuilt_count = blas_rebuilt_count;
    g_vulkan.synced_blas_reused_chunk_count = blas_reused_chunk_count;
    g_vulkan.synced_blas_rebuilt_chunk_count = blas_rebuilt_chunk_count;
    g_vulkan.synced_tlas_rebuild_count = 1;
    return true;
}

bool ensure_descriptor_set_layout() {
    if (g_vulkan.descriptor_set_layout != VK_NULL_HANDLE) {
        return true;
    }

    std::array<VkDescriptorSetLayoutBinding, 11> bindings{};
    bindings[0] = {0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR, nullptr};
    bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR, nullptr};
    bindings[2] = {
        2,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        1,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
            VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
        nullptr};
    bindings[3] = {
        3,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        1,
        VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
        nullptr};
    bindings[4] = {
        4,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        1,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
        nullptr};
    bindings[5] = {
        5,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        1,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
        nullptr};
    bindings[6] = {
        6,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        1,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
            VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
        nullptr};
    bindings[7] = {
        7,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        1,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
            VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
        nullptr};
    bindings[8] = {
        8,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        1,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
            VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
        nullptr};
    bindings[9] = {
        9,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        1,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
        nullptr};
    bindings[10] = {
        10,
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        1,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR,
        nullptr};

    VkDescriptorSetLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout_info.bindingCount = static_cast<std::uint32_t>(bindings.size());
    layout_info.pBindings = bindings.data();
    return vkCreateDescriptorSetLayout(g_vulkan.device, &layout_info, nullptr, &g_vulkan.descriptor_set_layout) ==
        VK_SUCCESS;
}

bool ensure_descriptor_set() {
    if (!ensure_descriptor_set_layout()) {
        return false;
    }
    if (g_vulkan.descriptor_pool == VK_NULL_HANDLE) {
        std::array<VkDescriptorPoolSize, 4> pool_sizes{};
        pool_sizes[0] = {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1};
        pool_sizes[1] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2};
        pool_sizes[2] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7};
        pool_sizes[3] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1};
        VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
        pool_info.pPoolSizes = pool_sizes.data();
        if (vkCreateDescriptorPool(g_vulkan.device, &pool_info, nullptr, &g_vulkan.descriptor_pool) != VK_SUCCESS) {
            return false;
        }
    }
    if (g_vulkan.descriptor_set == VK_NULL_HANDLE) {
        VkDescriptorSetAllocateInfo alloc_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        alloc_info.descriptorPool = g_vulkan.descriptor_pool;
        alloc_info.descriptorSetCount = 1;
        alloc_info.pSetLayouts = &g_vulkan.descriptor_set_layout;
        if (vkAllocateDescriptorSets(g_vulkan.device, &alloc_info, &g_vulkan.descriptor_set) != VK_SUCCESS) {
            return false;
        }
    }
    return true;
}

bool update_descriptor_set() {
    if (!ensure_descriptor_set()) {
        return false;
    }

    VkWriteDescriptorSetAccelerationStructureKHR accel_info{
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    accel_info.accelerationStructureCount = 1;
    accel_info.pAccelerationStructures = &g_vulkan.tlas.handle;

    VkDescriptorImageInfo image_info{};
    image_info.imageView = g_vulkan.output_image_view;
    image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo accumulation_image_info{};
    accumulation_image_info.imageView = g_vulkan.accumulation_image_view;
    accumulation_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorBufferInfo triangle_colors_info{g_vulkan.triangle_colors.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo instance_metadata_info{g_vulkan.instance_metadata.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo positions_info{g_vulkan.scene_positions.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo indices_info{g_vulkan.scene_indices.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo points_info{g_vulkan.point_buffer.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo lines_info{g_vulkan.line_buffer.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo constants_info{g_vulkan.viewer_constants.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo pick_output_info{g_vulkan.pick_output.buffer, 0, VK_WHOLE_SIZE};

    std::array<VkWriteDescriptorSet, 11> writes{};
    writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writes[0].pNext = &accel_info;
    writes[0].dstSet = g_vulkan.descriptor_set;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

    writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writes[1].dstSet = g_vulkan.descriptor_set;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].pImageInfo = &image_info;

    writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writes[2].dstSet = g_vulkan.descriptor_set;
    writes[2].dstBinding = 2;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[2].pBufferInfo = &triangle_colors_info;

    writes[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writes[3].dstSet = g_vulkan.descriptor_set;
    writes[3].dstBinding = 3;
    writes[3].descriptorCount = 1;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[3].pBufferInfo = &instance_metadata_info;

    writes[4] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writes[4].dstSet = g_vulkan.descriptor_set;
    writes[4].dstBinding = 4;
    writes[4].descriptorCount = 1;
    writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[4].pBufferInfo = &positions_info;

    writes[5] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writes[5].dstSet = g_vulkan.descriptor_set;
    writes[5].dstBinding = 5;
    writes[5].descriptorCount = 1;
    writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[5].pBufferInfo = &indices_info;

    writes[6] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writes[6].dstSet = g_vulkan.descriptor_set;
    writes[6].dstBinding = 6;
    writes[6].descriptorCount = 1;
    writes[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[6].pBufferInfo = &points_info;

    writes[7] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writes[7].dstSet = g_vulkan.descriptor_set;
    writes[7].dstBinding = 7;
    writes[7].descriptorCount = 1;
    writes[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[7].pBufferInfo = &lines_info;

    writes[8] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writes[8].dstSet = g_vulkan.descriptor_set;
    writes[8].dstBinding = 8;
    writes[8].descriptorCount = 1;
    writes[8].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[8].pBufferInfo = &constants_info;

    writes[9] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writes[9].dstSet = g_vulkan.descriptor_set;
    writes[9].dstBinding = 9;
    writes[9].descriptorCount = 1;
    writes[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[9].pBufferInfo = &pick_output_info;
    writes[10] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writes[10].dstSet = g_vulkan.descriptor_set;
    writes[10].dstBinding = 10;
    writes[10].descriptorCount = 1;
    writes[10].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[10].pImageInfo = &accumulation_image_info;

    vkUpdateDescriptorSets(g_vulkan.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    return true;
}

bool ensure_pipeline_layout() {
    if (g_vulkan.pipeline_layout != VK_NULL_HANDLE) {
        return true;
    }
    if (!ensure_descriptor_set_layout()) {
        return false;
    }
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &g_vulkan.descriptor_set_layout;
    return vkCreatePipelineLayout(g_vulkan.device, &layout_info, nullptr, &g_vulkan.pipeline_layout) == VK_SUCCESS;
}

bool ensure_pipeline() {
    if (g_vulkan.pipeline != VK_NULL_HANDLE) {
        return true;
    }
    if (!ensure_pipeline_layout()) {
        append_startup_log("Vulkan pipeline creation aborted: ensure_pipeline_layout failed");
        return false;
    }

    std::array<VkPipelineShaderStageCreateInfo, 8> stages{};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[0].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[0].module = g_vulkan.raygen_shader;
    stages[0].pName = "RayGen";
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[1].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[1].module = g_vulkan.pick_raygen_shader;
    stages[1].pName = "PickRayGen";
    stages[2] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[2].stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[2].module = g_vulkan.miss_shader;
    stages[2].pName = "Miss";
    stages[3] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[3].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[3].module = g_vulkan.closest_hit_shader;
    stages[3].pName = "ClosestHitTriangle";
    stages[4] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[4].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[4].module = g_vulkan.point_closest_hit_shader;
    stages[4].pName = "ClosestHitPoint";
    stages[5] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[5].stage = VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
    stages[5].module = g_vulkan.point_intersection_shader;
    stages[5].pName = "IntersectionPoint";
    stages[6] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[6].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[6].module = g_vulkan.line_closest_hit_shader;
    stages[6].pName = "ClosestHitLine";
    stages[7] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[7].stage = VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
    stages[7].module = g_vulkan.line_intersection_shader;
    stages[7].pName = "IntersectionLine";

    std::array<VkRayTracingShaderGroupCreateInfoKHR, kShaderGroupCount> groups{};
    groups[0] = {VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    groups[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[0].generalShader = 0;
    groups[0].closestHitShader = VK_SHADER_UNUSED_KHR;
    groups[0].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[0].intersectionShader = VK_SHADER_UNUSED_KHR;

    groups[1] = {VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    groups[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[1].generalShader = 1;
    groups[1].closestHitShader = VK_SHADER_UNUSED_KHR;
    groups[1].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[1].intersectionShader = VK_SHADER_UNUSED_KHR;

    groups[2] = {VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    groups[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[2].generalShader = 2;
    groups[2].closestHitShader = VK_SHADER_UNUSED_KHR;
    groups[2].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[2].intersectionShader = VK_SHADER_UNUSED_KHR;

    groups[3] = {VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    groups[3].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    groups[3].generalShader = VK_SHADER_UNUSED_KHR;
    groups[3].closestHitShader = 3;
    groups[3].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[3].intersectionShader = VK_SHADER_UNUSED_KHR;

    groups[4] = {VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    groups[4].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
    groups[4].generalShader = VK_SHADER_UNUSED_KHR;
    groups[4].closestHitShader = 4;
    groups[4].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[4].intersectionShader = 5;

    groups[5] = {VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    groups[5].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
    groups[5].generalShader = VK_SHADER_UNUSED_KHR;
    groups[5].closestHitShader = 6;
    groups[5].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[5].intersectionShader = 7;

    VkRayTracingPipelineCreateInfoKHR pipeline_info{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    pipeline_info.stageCount = static_cast<std::uint32_t>(stages.size());
    pipeline_info.pStages = stages.data();
    pipeline_info.groupCount = static_cast<std::uint32_t>(groups.size());
    pipeline_info.pGroups = groups.data();
    pipeline_info.maxPipelineRayRecursionDepth = 1;
    pipeline_info.layout = g_vulkan.pipeline_layout;
    const VkResult result = g_vulkan.create_ray_tracing_pipelines(
        g_vulkan.device,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        1,
        &pipeline_info,
        nullptr,
        &g_vulkan.pipeline);
    if (result != VK_SUCCESS) {
        append_startup_log("Vulkan pipeline creation failed: vkCreateRayTracingPipelinesKHR returned error");
        return false;
    }
    return true;
}

bool ensure_shader_binding_table() {
    if (g_vulkan.shader_binding_table.buffer != VK_NULL_HANDLE) {
        return true;
    }
    if (!ensure_pipeline()) {
        append_startup_log("Vulkan SBT creation aborted: ensure_pipeline failed");
        return false;
    }

    const std::uint32_t handle_size = g_vulkan.rt_pipeline_properties.shaderGroupHandleSize;
    const std::uint32_t aligned_handle_size = (handle_size + g_vulkan.rt_pipeline_properties.shaderGroupHandleAlignment - 1u) &
        ~(g_vulkan.rt_pipeline_properties.shaderGroupHandleAlignment - 1u);
    const std::uint32_t stride = (aligned_handle_size + g_vulkan.rt_pipeline_properties.shaderGroupBaseAlignment - 1u) &
        ~(g_vulkan.rt_pipeline_properties.shaderGroupBaseAlignment - 1u);
    const VkDeviceSize sbt_size = static_cast<VkDeviceSize>(stride) * kShaderGroupCount;

    std::vector<std::uint8_t> handles(handle_size * kShaderGroupCount);
    if (g_vulkan.get_ray_tracing_shader_group_handles(
            g_vulkan.device,
            g_vulkan.pipeline,
            0,
            kShaderGroupCount,
            handles.size(),
            handles.data()) != VK_SUCCESS) {
        append_startup_log("Vulkan SBT creation failed: vkGetRayTracingShaderGroupHandlesKHR returned error");
        return false;
    }

    if (!create_buffer(
            sbt_size,
            VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &g_vulkan.shader_binding_table)) {
        append_startup_log("Vulkan SBT creation failed: create_buffer returned false");
        return false;
    }

    std::vector<std::uint8_t> sbt_bytes(static_cast<std::size_t>(sbt_size), 0);
    for (std::uint32_t group_index = 0; group_index < kShaderGroupCount; ++group_index) {
        std::memcpy(
            sbt_bytes.data() + stride * group_index,
            handles.data() + handle_size * group_index,
            handle_size);
    }
    if (!upload_buffer_data(&g_vulkan.shader_binding_table, sbt_bytes.data(), sbt_bytes.size())) {
        append_startup_log("Vulkan SBT creation failed: upload_buffer_data returned false");
        return false;
    }

    const VkDeviceAddress sbt_address = buffer_device_address(g_vulkan.shader_binding_table);
    g_vulkan.raygen_region = {sbt_address + stride * 0u, stride, stride};
    g_vulkan.pick_raygen_region = {sbt_address + stride * 1u, stride, stride};
    g_vulkan.miss_region = {sbt_address + stride * 2u, stride, stride};
    g_vulkan.hit_region = {sbt_address + stride * 3u, stride, stride * 3u};
    g_vulkan.callable_region = {};
    return true;
}

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

void transition_output_image(
    VkCommandBuffer command_buffer,
    VkImageLayout old_layout,
    VkImageLayout new_layout,
    VkAccessFlags src_access_mask,
    VkAccessFlags dst_access_mask,
    VkPipelineStageFlags src_stage_mask,
    VkPipelineStageFlags dst_stage_mask)
{
    transition_image(
        command_buffer,
        g_vulkan.output_image,
        old_layout,
        new_layout,
        src_access_mask,
        dst_access_mask,
        src_stage_mask,
        dst_stage_mask);
}

void image_layout_access_and_stage(
    VkImageLayout layout,
    VkAccessFlags* out_access_mask,
    VkPipelineStageFlags* out_stage_mask)
{
    VkAccessFlags access_mask = 0;
    VkPipelineStageFlags stage_mask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    switch (layout) {
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
        access_mask = VK_ACCESS_TRANSFER_READ_BIT;
        stage_mask = VK_PIPELINE_STAGE_TRANSFER_BIT;
        break;
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
        access_mask = VK_ACCESS_SHADER_READ_BIT;
        stage_mask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        break;
    case VK_IMAGE_LAYOUT_GENERAL:
        access_mask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        stage_mask = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
        break;
    case VK_IMAGE_LAYOUT_UNDEFINED:
    default:
        access_mask = 0;
        stage_mask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        break;
    }
    if (out_access_mask != nullptr) {
        *out_access_mask = access_mask;
    }
    if (out_stage_mask != nullptr) {
        *out_stage_mask = stage_mask;
    }
}

void scene_clear_color(const frame_scene &scene, bool has_frame, VkClearColorValue* out_clear_color) {
    if (out_clear_color == nullptr) {
        return;
    }
    const float triangle_term = static_cast<float>((scene.triangles.size() % 13u) + 1u) / 13.0f;
    const float point_term = static_cast<float>((scene.points.size() % 11u) + 1u) / 11.0f;
    const float line_term = static_cast<float>((scene.lines.size() % 7u) + 1u) / 7.0f;
    out_clear_color->float32[0] = has_frame ? triangle_term * 0.8f : 0.05f;
    out_clear_color->float32[1] = has_frame ? point_term * 0.7f : 0.05f;
    out_clear_color->float32[2] = has_frame ? line_term * 0.6f : 0.05f;
    out_clear_color->float32[3] = 1.0f;
}

rtvdb::rgba apply_hover_highlight_cpu(
    rtvdb::rgba color,
    hover_highlight highlight,
    hover_highlight_kind primitive_kind,
    std::uint32_t primitive_index)
{
    if (highlight.kind == hover_highlight_kind::none ||
        highlight.kind != primitive_kind ||
        highlight.primitive_index != primitive_index) {
        return color;
    }

    const float luminance = color.r * 0.299f + color.g * 0.587f + color.b * 0.114f;
    const rtvdb::vec3 complement{1.0f - color.r, 1.0f - color.g, 1.0f - color.b};
    const rtvdb::vec3 bias = luminance >= 0.45f ? rtvdb::vec3{0.15f, 0.15f, 0.15f} : rtvdb::vec3{0.35f, 0.35f, 0.35f};
    const float mix_t = 0.85f;
    color.r = (std::clamp)(color.r * (1.0f - mix_t) + ((std::clamp)(complement.x + bias.x, 0.0f, 1.0f)) * mix_t, 0.0f, 1.0f);
    color.g = (std::clamp)(color.g * (1.0f - mix_t) + ((std::clamp)(complement.y + bias.y, 0.0f, 1.0f)) * mix_t, 0.0f, 1.0f);
    color.b = (std::clamp)(color.b * (1.0f - mix_t) + ((std::clamp)(complement.z + bias.z, 0.0f, 1.0f)) * mix_t, 0.0f, 1.0f);
    return color;
}

rtvdb::rgba apply_display_mode_cpu(
    rtvdb::rgba client_color,
    rtvdb::vec3 normal,
    std::uint32_t primitive_seed,
    std::uint32_t geometry_index,
    std::uint32_t instance_index)
{
    display_mode mode = display_mode::triangle_normal;
    get_display_mode(&mode);
    if (mode == display_mode::client_color) {
        return {
            (std::clamp)(client_color.r, 0.0f, 1.0f),
            (std::clamp)(client_color.g, 0.0f, 1.0f),
            (std::clamp)(client_color.b, 0.0f, 1.0f),
            (std::clamp)(client_color.a, 0.0f, 1.0f)
        };
    }
    if (mode == display_mode::simple_shaded) {
        const rtvdb::vec3 light_dir = normalize_or({0.35f, 0.70f, 0.62f}, {0.0f, 1.0f, 0.0f});
        const float diffuse = (std::max)(0.0f, dot(normal, light_dir));
        const float intensity = 0.18f + (1.0f - 0.18f) * diffuse;
        const float shaded = encode_srgb_channel(intensity);
        return {shaded, shaded, shaded, 1.0f};
    }
    if (mode == display_mode::primitive_id) {
        const rtvdb::vec3 color = hash_color(primitive_seed + 1u);
        return {color.x, color.y, color.z, 1.0f};
    }
    if (mode == display_mode::geometry_index) {
        const rtvdb::vec3 color = hash_color(geometry_index + 1u);
        return {color.x, color.y, color.z, 1.0f};
    }
    if (mode == display_mode::instance_index) {
        const rtvdb::vec3 color = hash_color(instance_index + 1u);
        return {color.x, color.y, color.z, 1.0f};
    }
    return {
        normal.x * 0.5f + 0.5f,
        normal.y * 0.5f + 0.5f,
        normal.z * 0.5f + 0.5f,
        1.0f
    };
}

rtvdb::vec3 point_normal_cpu(const point &point_value, rtvdb::vec3 hit_position) {
    return normalize_or(subtract(hit_position, point_value.position), {0.0f, 1.0f, 0.0f});
}

rtvdb::vec3 line_normal_cpu(const line &line_value, rtvdb::vec3 hit_position) {
    const rtvdb::vec3 ab = subtract(line_value.b, line_value.a);
    const float ab_len_sq = dot(ab, ab);
    if (ab_len_sq <= 1.0e-12f) {
        return normalize_or(subtract(hit_position, line_value.a), {0.0f, 1.0f, 0.0f});
    }
    const float u = (std::clamp)(dot(subtract(hit_position, line_value.a), ab) / ab_len_sq, 0.0f, 1.0f);
    const rtvdb::vec3 closest = lerp(line_value.a, line_value.b, u);
    return normalize_or(subtract(hit_position, closest), {0.0f, 1.0f, 0.0f});
}

void write_bgra_pixel(std::uint8_t* pixel, rtvdb::rgba color) {
    if (pixel == nullptr) {
        return;
    }
    pixel[0] = static_cast<std::uint8_t>((std::clamp)(color.b, 0.0f, 1.0f) * 255.0f + 0.5f);
    pixel[1] = static_cast<std::uint8_t>((std::clamp)(color.g, 0.0f, 1.0f) * 255.0f + 0.5f);
    pixel[2] = static_cast<std::uint8_t>((std::clamp)(color.r, 0.0f, 1.0f) * 255.0f + 0.5f);
    pixel[3] = static_cast<std::uint8_t>((std::clamp)(color.a, 0.0f, 1.0f) * 255.0f + 0.5f);
}

bool begin_command_buffer() {
    if (vkWaitForFences(g_vulkan.device, 1, &g_vulkan.command_fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        return false;
    }
    if (vkResetFences(g_vulkan.device, 1, &g_vulkan.command_fence) != VK_SUCCESS ||
        vkResetCommandPool(g_vulkan.device, g_vulkan.command_pool, 0) != VK_SUCCESS) {
        return false;
    }
    VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    return vkBeginCommandBuffer(g_vulkan.command_buffer, &begin_info) == VK_SUCCESS;
}

bool submit_command_buffer(double* out_submit_cpu_ms) {
    const auto submit_start = std::chrono::steady_clock::now();
    VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &g_vulkan.command_buffer;
    const bool submitted = vkQueueSubmit(g_vulkan.graphics_queue, 1, &submit_info, g_vulkan.command_fence) == VK_SUCCESS;
    if (out_submit_cpu_ms != nullptr) {
        *out_submit_cpu_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - submit_start).count();
    }
    return submitted;
}

bool wait_for_command_buffer(double* out_wait_ms) {
    const auto wait_start = std::chrono::steady_clock::now();
    const bool waited = vkWaitForFences(g_vulkan.device, 1, &g_vulkan.command_fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
    if (out_wait_ms != nullptr) {
        *out_wait_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - wait_start).count();
    }
    return waited;
}

bool end_and_submit_command_buffer(double* out_submit_cpu_ms = nullptr, double* out_wait_ms = nullptr) {
    if (vkEndCommandBuffer(g_vulkan.command_buffer) != VK_SUCCESS) {
        return false;
    }
    return submit_command_buffer(out_submit_cpu_ms) && wait_for_command_buffer(out_wait_ms);
}

bool timestamp_queries_enabled() {
    return g_vulkan.timestamp_queries_supported &&
        g_vulkan.timestamp_query_pool != VK_NULL_HANDLE &&
        g_vulkan.timestamp_period_ns > 0.0f;
}

void reset_timestamp_query_region(timestamp_query_region region) {
    if (!timestamp_queries_enabled() || g_vulkan.command_buffer == VK_NULL_HANDLE) {
        return;
    }
    vkCmdResetQueryPool(g_vulkan.command_buffer, g_vulkan.timestamp_query_pool, static_cast<std::uint32_t>(region), 2);
}

void write_timestamp_query_begin(timestamp_query_region region) {
    if (!timestamp_queries_enabled() || g_vulkan.command_buffer == VK_NULL_HANDLE) {
        return;
    }
    vkCmdWriteTimestamp(
        g_vulkan.command_buffer,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        g_vulkan.timestamp_query_pool,
        static_cast<std::uint32_t>(region));
}

void write_timestamp_query_end(timestamp_query_region region) {
    if (!timestamp_queries_enabled() || g_vulkan.command_buffer == VK_NULL_HANDLE) {
        return;
    }
    vkCmdWriteTimestamp(
        g_vulkan.command_buffer,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        g_vulkan.timestamp_query_pool,
        static_cast<std::uint32_t>(region) + 1u);
}

bool read_timestamp_query_ms(timestamp_query_region region, double* out_ms) {
    if (out_ms == nullptr || !timestamp_queries_enabled()) {
        return false;
    }
    std::uint64_t timestamps[2]{};
    const VkResult result = vkGetQueryPoolResults(
        g_vulkan.device,
        g_vulkan.timestamp_query_pool,
        static_cast<std::uint32_t>(region),
        2,
        sizeof(timestamps),
        timestamps,
        sizeof(std::uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (result != VK_SUCCESS || timestamps[1] <= timestamps[0]) {
        return false;
    }
    *out_ms = static_cast<double>(timestamps[1] - timestamps[0]) * static_cast<double>(g_vulkan.timestamp_period_ns) /
        1000000.0;
    return true;
}

bool copy_output_to_cpu(
    int width,
    int height,
    std::vector<std::uint8_t>* out_pixels,
    double* out_dispatch_ms,
    double* out_readback_ms,
    double* out_dispatch_submit_cpu_ms = nullptr,
    double* out_dispatch_gpu_wait_ms = nullptr,
    double* out_dispatch_gpu_ms = nullptr)
{
    if (g_vulkan.direct_vulkan_present_requested) {
        transition_output_image(
            g_vulkan.command_buffer,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_TRANSFER_READ_BIT,
            VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        g_vulkan.output_image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        double submit_cpu_ms = 0.0;
        double wait_ms = 0.0;
        if (!end_and_submit_command_buffer(&submit_cpu_ms, &wait_ms)) {
            return false;
        }
        if (out_dispatch_submit_cpu_ms != nullptr) {
            *out_dispatch_submit_cpu_ms = submit_cpu_ms;
        }
        if (out_dispatch_gpu_wait_ms != nullptr) {
            *out_dispatch_gpu_wait_ms = wait_ms;
        }
        if (out_dispatch_gpu_ms != nullptr && !read_timestamp_query_ms(timestamp_query_region::dispatch, out_dispatch_gpu_ms)) {
            *out_dispatch_gpu_ms = 0.0;
        }
        if (out_dispatch_ms != nullptr) {
            *out_dispatch_ms = submit_cpu_ms + wait_ms;
        }
        if (out_readback_ms != nullptr) {
            *out_readback_ms = 0.0;
        }
        return true;
    }
    double submit_cpu_ms = 0.0;
    double wait_ms = 0.0;
    if (!end_and_submit_command_buffer(&submit_cpu_ms, &wait_ms)) {
        return false;
    }
    if (out_dispatch_submit_cpu_ms != nullptr) {
        *out_dispatch_submit_cpu_ms = submit_cpu_ms;
    }
    if (out_dispatch_gpu_wait_ms != nullptr) {
        *out_dispatch_gpu_wait_ms = wait_ms;
    }
    if (out_dispatch_gpu_ms != nullptr && !read_timestamp_query_ms(timestamp_query_region::dispatch, out_dispatch_gpu_ms)) {
        *out_dispatch_gpu_ms = 0.0;
    }

    const auto readback_start = std::chrono::steady_clock::now();
    void* mapped = nullptr;
    if (vkMapMemory(g_vulkan.device, g_vulkan.readback_buffer_memory, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS ||
        mapped == nullptr) {
        return false;
    }
    const std::size_t byte_count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4u;
    out_pixels->resize(byte_count);
    std::memcpy(out_pixels->data(), mapped, byte_count);
    vkUnmapMemory(g_vulkan.device, g_vulkan.readback_buffer_memory);
    for (std::size_t offset = 0; offset + 3u < byte_count; offset += 4u) {
        std::swap((*out_pixels)[offset], (*out_pixels)[offset + 2u]);
    }
    const auto readback_end = std::chrono::steady_clock::now();

    if (out_dispatch_ms != nullptr) {
        *out_dispatch_ms = submit_cpu_ms + wait_ms;
    }
    if (out_readback_ms != nullptr) {
        *out_readback_ms = std::chrono::duration<double, std::milli>(readback_end - readback_start).count();
    }
    return true;
}


bool dispatch_clear_frame(
    int width,
    int height,
    const frame_scene &scene,
    bool has_frame,
    std::vector<std::uint8_t>* out_pixels,
    void* native_target_resource,
    double* out_dispatch_ms,
    double* out_readback_ms,
    double* out_dispatch_submit_cpu_ms = nullptr,
    double* out_dispatch_gpu_wait_ms = nullptr,
    double* out_dispatch_gpu_ms = nullptr)
{
    if (!begin_command_buffer()) {
        return false;
    }
    const VkAccessFlags prior_access_mask = g_vulkan.output_image_layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
        ? VK_ACCESS_TRANSFER_READ_BIT
        : (g_vulkan.output_image_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL ? VK_ACCESS_SHADER_READ_BIT : 0);
    const VkPipelineStageFlags prior_stage_mask = g_vulkan.output_image_layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
        ? VK_PIPELINE_STAGE_TRANSFER_BIT
        : (g_vulkan.output_image_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
            : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
    transition_output_image(
        g_vulkan.command_buffer,
        g_vulkan.output_image_layout,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        prior_access_mask,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        prior_stage_mask,
        VK_PIPELINE_STAGE_TRANSFER_BIT);
    g_vulkan.output_image_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

    VkClearColorValue clear_color{};
    scene_clear_color(scene, has_frame, &clear_color);
    VkImageSubresourceRange range{};
    range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    range.levelCount = 1;
    range.layerCount = 1;
    vkCmdClearColorImage(
        g_vulkan.command_buffer,
        g_vulkan.output_image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        &clear_color,
        1,
        &range);

    transition_output_image(
        g_vulkan.command_buffer,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_TRANSFER_READ_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT);
    g_vulkan.output_image_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

    VkBufferImageCopy copy_region{};
    copy_region.bufferRowLength = static_cast<std::uint32_t>(width);
    copy_region.bufferImageHeight = static_cast<std::uint32_t>(height);
    copy_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy_region.imageSubresource.layerCount = 1;
    copy_region.imageExtent = {static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height), 1u};
    if (!g_vulkan.direct_vulkan_present_requested) {
        vkCmdCopyImageToBuffer(
            g_vulkan.command_buffer,
            g_vulkan.output_image,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            g_vulkan.readback_buffer,
            1,
            &copy_region);
    }

    (void)native_target_resource;
    return copy_output_to_cpu(
        width,
        height,
        out_pixels,
        out_dispatch_ms,
        out_readback_ms,
        out_dispatch_submit_cpu_ms,
        out_dispatch_gpu_wait_ms,
        out_dispatch_gpu_ms);
}

bool dispatch_pick_query(
    int width,
    int height,
    int pixel_x,
    int pixel_y,
    const frame_scene &scene,
    const rt_scene_build &build,
    bool has_frame,
    pick_result* out_result)
{
    if (out_result == nullptr || width <= 0 || height <= 0) {
        return false;
    }
    if (!ensure_output_resources(width, height)) {
        return false;
    }
    if (!ensure_pick_output_buffer()) {
        return false;
    }
    if (!ensure_scene_buffers(build)) {
        return false;
    }
    if (g_vulkan.synced_build_revision != build.revision) {
        if (!build_triangle_acceleration_structures(build)) {
            return false;
        }
        destroy_buffer(&g_vulkan.shader_binding_table);
    }
    if ((build.triangle_count == 0 && build.point_count == 0 && build.line_count == 0) ||
        g_vulkan.tlas.handle == VK_NULL_HANDLE) {
        *out_result = {};
        return true;
    }

    update_viewer_constants(scene, build, has_frame, width, height, true, pixel_x, pixel_y);
    if (!update_descriptor_set() || !ensure_pipeline() || !ensure_shader_binding_table()) {
        return false;
    }

    const gpu_pick_result cleared_pick{};
    if (!upload_buffer_data(&g_vulkan.pick_output, &cleared_pick, sizeof(cleared_pick))) {
        return false;
    }

    if (!begin_command_buffer()) {
        return false;
    }

    VkBufferMemoryBarrier pre_barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    pre_barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    pre_barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    pre_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    pre_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    pre_barrier.buffer = g_vulkan.pick_output.buffer;
    pre_barrier.offset = 0;
    pre_barrier.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(
        g_vulkan.command_buffer,
        VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        0,
        0,
        nullptr,
        1,
        &pre_barrier,
        0,
        nullptr);

    vkCmdBindPipeline(g_vulkan.command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, g_vulkan.pipeline);
    vkCmdBindDescriptorSets(
        g_vulkan.command_buffer,
        VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
        g_vulkan.pipeline_layout,
        0,
        1,
        &g_vulkan.descriptor_set,
        0,
        nullptr);
    g_vulkan.cmd_trace_rays(
        g_vulkan.command_buffer,
        &g_vulkan.pick_raygen_region,
        &g_vulkan.miss_region,
        &g_vulkan.hit_region,
        &g_vulkan.callable_region,
        1,
        1,
        1);

    VkBufferMemoryBarrier post_barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    post_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    post_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    post_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    post_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    post_barrier.buffer = g_vulkan.pick_output.buffer;
    post_barrier.offset = 0;
    post_barrier.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(
        g_vulkan.command_buffer,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        VK_PIPELINE_STAGE_HOST_BIT,
        0,
        0,
        nullptr,
        1,
        &post_barrier,
        0,
        nullptr);

    if (!end_and_submit_command_buffer()) {
        return false;
    }

    gpu_pick_result mapped_pick{};
    void* mapped_data = nullptr;
    if (vkMapMemory(g_vulkan.device, g_vulkan.pick_output.memory, 0, sizeof(mapped_pick), 0, &mapped_data) != VK_SUCCESS ||
        mapped_data == nullptr) {
        return false;
    }
    std::memcpy(&mapped_pick, mapped_data, sizeof(mapped_pick));
    vkUnmapMemory(g_vulkan.device, g_vulkan.pick_output.memory);

    out_result->kind = static_cast<hover_highlight_kind>(mapped_pick.primitive_kind);
    out_result->primitive_index = mapped_pick.primitive_index;
    out_result->distance = mapped_pick.hit != 0u ? mapped_pick.distance : 0.0f;
    return true;
}

bool dispatch_triangle_rt_frame(
    int width,
    int height,
    const frame_scene &scene,
    const rt_scene_build &build,
    bool has_frame,
    std::vector<std::uint8_t>* out_pixels,
    void* native_target_resource,
    double* out_accel_build_ms,
    double* out_dispatch_ms,
    double* out_readback_ms,
    double* out_accel_submit_cpu_ms = nullptr,
    double* out_accel_gpu_wait_ms = nullptr,
    double* out_dispatch_submit_cpu_ms = nullptr,
    double* out_dispatch_gpu_wait_ms = nullptr,
    double* out_accel_gpu_ms = nullptr,
    double* out_dispatch_gpu_ms = nullptr)
{
    const auto accel_start = std::chrono::steady_clock::now();
    if (!ensure_scene_buffers(build)) {
        return false;
    }
    if (g_vulkan.synced_build_revision != build.revision) {
        if (!build_triangle_acceleration_structures(build, out_accel_submit_cpu_ms, out_accel_gpu_wait_ms)) {
            return false;
        }
        if (out_accel_gpu_ms != nullptr && !read_timestamp_query_ms(timestamp_query_region::accel, out_accel_gpu_ms)) {
            *out_accel_gpu_ms = 0.0;
        }
        destroy_buffer(&g_vulkan.shader_binding_table);
    }
    if ((build.triangle_count == 0 && build.point_count == 0 && build.line_count == 0) ||
        g_vulkan.tlas.handle == VK_NULL_HANDLE) {
        return dispatch_clear_frame(
            width,
            height,
            scene,
            has_frame,
            out_pixels,
            native_target_resource,
            out_dispatch_ms,
            out_readback_ms,
            out_dispatch_submit_cpu_ms,
            out_dispatch_gpu_wait_ms,
            out_dispatch_gpu_ms);
    }
    if (out_accel_build_ms != nullptr) {
        *out_accel_build_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - accel_start).count();
    }
    update_viewer_constants(scene, build, has_frame, width, height);
    if (!update_descriptor_set() || !ensure_pipeline() || !ensure_shader_binding_table()) {
        return false;
    }

    if (!begin_command_buffer()) {
        return false;
    }
    reset_timestamp_query_region(timestamp_query_region::dispatch);
    write_timestamp_query_begin(timestamp_query_region::dispatch);

    if (g_vulkan.accumulation_image_layout == VK_IMAGE_LAYOUT_UNDEFINED) {
        VkImageMemoryBarrier accumulation_barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        accumulation_barrier.srcAccessMask = 0;
        accumulation_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        accumulation_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        accumulation_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        accumulation_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        accumulation_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        accumulation_barrier.image = g_vulkan.accumulation_image;
        accumulation_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        accumulation_barrier.subresourceRange.levelCount = 1;
        accumulation_barrier.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(
            g_vulkan.command_buffer,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            0,
            0,
            nullptr,
            0,
            nullptr,
            1,
            &accumulation_barrier);
        g_vulkan.accumulation_image_layout = VK_IMAGE_LAYOUT_GENERAL;
    } else {
        VkMemoryBarrier accumulation_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        accumulation_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        accumulation_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(
            g_vulkan.command_buffer,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            0,
            1,
            &accumulation_barrier,
            0,
            nullptr,
            0,
            nullptr);
    }

    const VkAccessFlags prior_access_mask = g_vulkan.output_image_layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
        ? VK_ACCESS_TRANSFER_READ_BIT
        : (g_vulkan.output_image_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL ? VK_ACCESS_SHADER_READ_BIT : 0);
    const VkPipelineStageFlags prior_stage_mask = g_vulkan.output_image_layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
        ? VK_PIPELINE_STAGE_TRANSFER_BIT
        : (g_vulkan.output_image_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
            : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
    transition_output_image(
        g_vulkan.command_buffer,
        g_vulkan.output_image_layout,
        VK_IMAGE_LAYOUT_GENERAL,
        prior_access_mask,
        VK_ACCESS_SHADER_WRITE_BIT,
        prior_stage_mask,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR);
    g_vulkan.output_image_layout = VK_IMAGE_LAYOUT_GENERAL;

    vkCmdBindPipeline(g_vulkan.command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, g_vulkan.pipeline);
    vkCmdBindDescriptorSets(
        g_vulkan.command_buffer,
        VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
        g_vulkan.pipeline_layout,
        0,
        1,
        &g_vulkan.descriptor_set,
        0,
        nullptr);
    g_vulkan.cmd_trace_rays(
        g_vulkan.command_buffer,
        &g_vulkan.raygen_region,
        &g_vulkan.miss_region,
        &g_vulkan.hit_region,
        &g_vulkan.callable_region,
        static_cast<std::uint32_t>(width),
        static_cast<std::uint32_t>(height),
        1);
    write_timestamp_query_end(timestamp_query_region::dispatch);

    transition_output_image(
        g_vulkan.command_buffer,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_TRANSFER_READ_BIT,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        VK_PIPELINE_STAGE_TRANSFER_BIT);
    g_vulkan.output_image_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

    VkBufferImageCopy copy_region{};
    copy_region.bufferRowLength = static_cast<std::uint32_t>(width);
    copy_region.bufferImageHeight = static_cast<std::uint32_t>(height);
    copy_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy_region.imageSubresource.layerCount = 1;
    copy_region.imageExtent = {static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height), 1u};
    if (!g_vulkan.direct_vulkan_present_requested) {
        vkCmdCopyImageToBuffer(
            g_vulkan.command_buffer,
            g_vulkan.output_image,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            g_vulkan.readback_buffer,
            1,
            &copy_region);
    }

    (void)native_target_resource;
    return copy_output_to_cpu(
        width,
        height,
        out_pixels,
        out_dispatch_ms,
        out_readback_ms,
        out_dispatch_submit_cpu_ms,
        out_dispatch_gpu_wait_ms,
        out_dispatch_gpu_ms);
}

bool render_vulkan_frame_locked(
    int width,
    int height,
    const frame_scene &scene,
    bool has_frame,
    std::vector<std::uint8_t>* out_pixels,
    void* native_target_resource,
    double* out_accel_build_ms,
    double* out_dispatch_ms,
    double* out_readback_ms,
    double* out_accel_submit_cpu_ms = nullptr,
    double* out_accel_gpu_wait_ms = nullptr,
    double* out_dispatch_submit_cpu_ms = nullptr,
    double* out_dispatch_gpu_wait_ms = nullptr,
    double* out_accel_gpu_ms = nullptr,
    double* out_dispatch_gpu_ms = nullptr)
{
    if (!g_vulkan.initialized || !ensure_output_resources(width, height)) {
        return false;
    }

    rt_scene_build build{};
    copy_present_render_rt_scene_build(&build);
    const bool has_primitives = build.triangle_count > 0 || build.point_count > 0 || build.line_count > 0;
    display_mode mode = display_mode::triangle_normal;
    get_display_mode(&mode);
    const accumulation_key next_key =
        make_accumulation_key(scene, has_frame, build, width, height, static_cast<std::uint32_t>(mode));
    const bool accumulation_changed = !accumulation_key_equals(g_vulkan.accumulation_state_key, next_key);
    if (accumulation_changed) {
        g_vulkan.accumulation_state_key = next_key;
        g_vulkan.accumulation_sample_count = 0;
    }

    if (g_vulkan.config.continuous_render &&
        !accumulation_changed &&
        g_vulkan.accumulation_sample_count >= kMaxAccumulationSamples) {
        g_vulkan.accumulation_sample_count = 0;
    }

    if (!accumulation_changed &&
        !g_vulkan.config.continuous_render &&
        g_vulkan.direct_vulkan_present_requested &&
        g_vulkan.accumulation_sample_count >= kMaxAccumulationSamples &&
        g_vulkan.output_image != VK_NULL_HANDLE) {
        bool transitioned_for_present = false;
        if (g_vulkan.output_image_layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            if (!begin_command_buffer()) {
                return false;
            }
            VkAccessFlags prior_access_mask = 0;
            VkPipelineStageFlags prior_stage_mask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            image_layout_access_and_stage(
                g_vulkan.output_image_layout,
                &prior_access_mask,
                &prior_stage_mask);
            transition_output_image(
                g_vulkan.command_buffer,
                g_vulkan.output_image_layout,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                prior_access_mask,
                VK_ACCESS_SHADER_READ_BIT,
                prior_stage_mask,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
            g_vulkan.output_image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            transitioned_for_present = true;
            const auto transition_start = std::chrono::steady_clock::now();
            if (!end_and_submit_command_buffer()) {
                return false;
            }
            if (out_dispatch_ms != nullptr) {
                *out_dispatch_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - transition_start).count();
            }
        }
        g_vulkan.accumulation_active = false;
        g_vulkan.build_info.readback_ms = 0.0;
        g_vulkan.build_info.accumulation_in_progress = false;
        if (out_dispatch_ms != nullptr && !transitioned_for_present) {
            *out_dispatch_ms = g_vulkan.build_info.dispatch_ms;
        }
        if (out_dispatch_submit_cpu_ms != nullptr) {
            *out_dispatch_submit_cpu_ms = g_vulkan.build_info.dispatch_submit_cpu_ms;
        }
        if (out_dispatch_gpu_wait_ms != nullptr) {
            *out_dispatch_gpu_wait_ms = g_vulkan.build_info.dispatch_gpu_wait_ms;
        }
        if (out_dispatch_gpu_ms != nullptr) {
            *out_dispatch_gpu_ms = g_vulkan.build_info.dispatch_gpu_ms;
        }
        if (out_readback_ms != nullptr) {
            *out_readback_ms = 0.0;
        }
        return true;
    }

    g_vulkan.accumulation_active = false;
    const bool accel_was_stale = has_primitives && g_vulkan.synced_build_revision != build.revision;
    const bool rendered = has_primitives
        ? dispatch_triangle_rt_frame(
            width,
            height,
            scene,
            build,
            has_frame,
            out_pixels,
            native_target_resource,
            out_accel_build_ms,
            out_dispatch_ms,
            out_readback_ms,
            out_accel_submit_cpu_ms,
            out_accel_gpu_wait_ms,
            out_dispatch_submit_cpu_ms,
            out_dispatch_gpu_wait_ms,
            out_accel_gpu_ms,
            out_dispatch_gpu_ms)
        : dispatch_clear_frame(
            width,
            height,
            scene,
            has_frame,
            out_pixels,
            native_target_resource,
            out_dispatch_ms,
            out_readback_ms,
            out_dispatch_submit_cpu_ms,
            out_dispatch_gpu_wait_ms,
            out_dispatch_gpu_ms);
    if (!rendered) {
        return false;
    }

    if (!has_primitives) {
        g_vulkan.synced_blas_reused_count = 0;
        g_vulkan.synced_blas_rebuilt_count = 0;
        g_vulkan.synced_blas_reused_chunk_count = 0;
        g_vulkan.synced_blas_rebuilt_chunk_count = 0;
        g_vulkan.synced_tlas_rebuild_count = 0;
    } else if (!accel_was_stale) {
        g_vulkan.synced_blas_reused_count =
            build.chunks.size() + (build.point_count > 0 ? 1u : 0u) + (build.line_count > 0 ? 1u : 0u);
        g_vulkan.synced_blas_rebuilt_count = 0;
        g_vulkan.synced_blas_reused_chunk_count = build.chunks.size();
        g_vulkan.synced_blas_rebuilt_chunk_count = 0;
        g_vulkan.synced_tlas_rebuild_count = 0;
    }
    g_vulkan.build_info.blas_reused_count = g_vulkan.synced_blas_reused_count;
    g_vulkan.build_info.blas_rebuilt_count = g_vulkan.synced_blas_rebuilt_count;
    g_vulkan.build_info.blas_reused_chunk_count = g_vulkan.synced_blas_reused_chunk_count;
    g_vulkan.build_info.blas_rebuilt_chunk_count = g_vulkan.synced_blas_rebuilt_chunk_count;
    g_vulkan.build_info.tlas_rebuild_count = g_vulkan.synced_tlas_rebuild_count;
    if (!g_vulkan.capture_readback_requested) {
        if (accel_was_stale) {
            g_vulkan.build_info.accel_build_ms = out_accel_build_ms != nullptr ? *out_accel_build_ms : 0.0;
            g_vulkan.build_info.accel_submit_cpu_ms =
                out_accel_submit_cpu_ms != nullptr ? *out_accel_submit_cpu_ms : 0.0;
            g_vulkan.build_info.accel_gpu_wait_ms =
                out_accel_gpu_wait_ms != nullptr ? *out_accel_gpu_wait_ms : 0.0;
            g_vulkan.build_info.accel_gpu_ms = out_accel_gpu_ms != nullptr ? *out_accel_gpu_ms : 0.0;
        }
        g_vulkan.build_info.dispatch_ms = out_dispatch_ms != nullptr ? *out_dispatch_ms : 0.0;
        g_vulkan.build_info.dispatch_submit_cpu_ms =
            out_dispatch_submit_cpu_ms != nullptr ? *out_dispatch_submit_cpu_ms : 0.0;
        g_vulkan.build_info.dispatch_gpu_wait_ms =
            out_dispatch_gpu_wait_ms != nullptr ? *out_dispatch_gpu_wait_ms : 0.0;
        g_vulkan.build_info.dispatch_gpu_ms = out_dispatch_gpu_ms != nullptr ? *out_dispatch_gpu_ms : 0.0;
        g_vulkan.build_info.readback_ms = out_readback_ms != nullptr ? *out_readback_ms : 0.0;
    }
    if (build.revision != 0 && has_primitives) {
        if (g_vulkan.accumulation_sample_count < kMaxAccumulationSamples) {
            ++g_vulkan.accumulation_sample_count;
        }
        g_vulkan.accumulation_active =
            g_vulkan.config.continuous_render || g_vulkan.accumulation_sample_count < kMaxAccumulationSamples;
    }
    g_vulkan.build_info.accumulation_sample_count = g_vulkan.accumulation_sample_count;
    g_vulkan.build_info.accumulation_target_sample_count = kMaxAccumulationSamples;
    g_vulkan.build_info.accumulation_in_progress = g_vulkan.accumulation_active;
    return true;
}

#if defined(_WIN32)
bool try_render_vulkan_frame_locked_with_seh(
    const char* stage,
    int width,
    int height,
    const frame_scene* scene,
    bool has_frame,
    std::vector<std::uint8_t>* out_pixels,
    void* native_target_resource,
    double* out_accel_build_ms,
    double* out_dispatch_ms,
    double* out_readback_ms,
    double* out_accel_submit_cpu_ms = nullptr,
    double* out_accel_gpu_wait_ms = nullptr,
    double* out_dispatch_submit_cpu_ms = nullptr,
    double* out_dispatch_gpu_wait_ms = nullptr,
    double* out_accel_gpu_ms = nullptr,
    double* out_dispatch_gpu_ms = nullptr)
{
    if (scene == nullptr) {
        return false;
    }

    bool rendered = false;
    __try {
        rendered = render_vulkan_frame_locked(
            width,
            height,
            *scene,
            has_frame,
            out_pixels,
            native_target_resource,
            out_accel_build_ms,
            out_dispatch_ms,
            out_readback_ms,
            out_accel_submit_cpu_ms,
            out_accel_gpu_wait_ms,
            out_dispatch_submit_cpu_ms,
            out_dispatch_gpu_wait_ms,
            out_accel_gpu_ms,
            out_dispatch_gpu_ms);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        append_vulkan_fault_log(stage, GetExceptionCode());
        rendered = false;
    }
    return rendered;
}
#endif

bool initialize_vulkan_backend(const backend_config &config) {
    std::scoped_lock lock(g_vulkan.mutex);
    reset_vulkan_state_locked();
    g_vulkan.config = config;

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
        append_startup_log("Vulkan startup failed: SDL did not provide the required instance extensions");
        reset_vulkan_state_locked();
        return false;
    }
#endif

    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &app_info;
    instance_info.enabledExtensionCount = static_cast<std::uint32_t>(required_instance_extensions.size());
    instance_info.ppEnabledExtensionNames = required_instance_extensions.data();
    if (vkCreateInstance(&instance_info, nullptr, &g_vulkan.instance) != VK_SUCCESS) {
        reset_vulkan_state_locked();
        return false;
    }
    if (!choose_physical_device() || !create_logical_device()) {
        reset_vulkan_state_locked();
        return false;
    }
    if (!create_shader_module(kVulkanRtRaygenSpirv, kVulkanRtRaygenSpirvSize, &g_vulkan.raygen_shader) ||
        !create_shader_module(
            kVulkanRtPickRaygenSpirv,
            kVulkanRtPickRaygenSpirvSize,
            &g_vulkan.pick_raygen_shader) ||
        !create_shader_module(kVulkanRtMissSpirv, kVulkanRtMissSpirvSize, &g_vulkan.miss_shader) ||
        !create_shader_module(kVulkanRtClosestHitSpirv, kVulkanRtClosestHitSpirvSize, &g_vulkan.closest_hit_shader) ||
        !create_shader_module(
            kVulkanRtPointClosestHitSpirv,
            kVulkanRtPointClosestHitSpirvSize,
            &g_vulkan.point_closest_hit_shader) ||
        !create_shader_module(
            kVulkanRtPointIntersectionSpirv,
            kVulkanRtPointIntersectionSpirvSize,
            &g_vulkan.point_intersection_shader) ||
        !create_shader_module(
            kVulkanRtLineClosestHitSpirv,
            kVulkanRtLineClosestHitSpirvSize,
            &g_vulkan.line_closest_hit_shader) ||
        !create_shader_module(
            kVulkanRtLineIntersectionSpirv,
            kVulkanRtLineIntersectionSpirvSize,
            &g_vulkan.line_intersection_shader) ||
        !create_command_objects() ||
        !create_buffer(
            sizeof(viewer_constants_gpu),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &g_vulkan.viewer_constants)) {
        reset_vulkan_state_locked();
        return false;
    }

    g_vulkan.build_info.accumulation_target_sample_count = kMaxAccumulationSamples;
    g_vulkan.initialized = true;
    return true;
}

void shutdown_vulkan_backend() {
    std::scoped_lock lock(g_vulkan.mutex);
    reset_vulkan_state_locked();
}

bool render_vulkan_to_window(viewer_shell::native_window_handle, const frame_scene&, bool) {
    frame_scene render_scene{};
    bool render_has_frame = false;
    copy_present_render_scene(&render_scene, &render_has_frame);

    int width = 0;
    int height = 0;
    if (!viewer_shell::render_window_size(&width, &height) || width <= 0 || height <= 0) {
        return false;
    }

    std::vector<std::uint8_t> pixels;
    if (!capture_vulkan_to_bgra(width, height, render_scene, render_has_frame, &pixels, false) || pixels.empty()) {
        return false;
    }

    return viewer_shell::upload_bgra_frame(width, height, width * 4, pixels.data());
}

bool render_vulkan_to_native_metal_texture(int, int, const frame_scene&, bool, void*) {
    return false;
}

bool render_vulkan_to_native_vulkan_texture(
    int width,
    int height,
    const frame_scene &scene,
    bool has_frame,
    void** out_image)
{
    if (out_image == nullptr || width <= 0 || height <= 0 || !has_frame) {
        return false;
    }
    *out_image = nullptr;
    std::scoped_lock lock(g_vulkan.mutex);
    g_vulkan.direct_vulkan_present_requested = true;
    std::vector<std::uint8_t> unused_pixels;
    double accel_build_ms = 0.0;
    double dispatch_ms = 0.0;
    double readback_ms = 0.0;
    double accel_submit_cpu_ms = 0.0;
    double accel_gpu_wait_ms = 0.0;
    double dispatch_submit_cpu_ms = 0.0;
    double dispatch_gpu_wait_ms = 0.0;
    double accel_gpu_ms = 0.0;
    double dispatch_gpu_ms = 0.0;
#if defined(_WIN32)
    const bool rendered = try_render_vulkan_frame_locked_with_seh(
        "render_vulkan_to_native_vulkan_texture",
        width,
        height,
        &scene,
        has_frame,
        &unused_pixels,
        nullptr,
        &accel_build_ms,
        &dispatch_ms,
        &readback_ms,
        &accel_submit_cpu_ms,
        &accel_gpu_wait_ms,
        &dispatch_submit_cpu_ms,
        &dispatch_gpu_wait_ms,
        &accel_gpu_ms,
        &dispatch_gpu_ms);
#else
    const bool rendered = render_vulkan_frame_locked(
        width,
        height,
        scene,
        has_frame,
        &unused_pixels,
        nullptr,
        &accel_build_ms,
        &dispatch_ms,
        &readback_ms,
        &accel_submit_cpu_ms,
        &accel_gpu_wait_ms,
        &dispatch_submit_cpu_ms,
        &dispatch_gpu_wait_ms,
        &accel_gpu_ms,
        &dispatch_gpu_ms);
#endif
    g_vulkan.direct_vulkan_present_requested = false;
    if (!rendered || g_vulkan.output_image_layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        return false;
    }
    *out_image = reinterpret_cast<void*>(g_vulkan.output_image);
    return true;
}

bool capture_vulkan_to_bgra(
    int width,
    int height,
    const frame_scene &scene,
    bool has_frame,
    std::vector<std::uint8_t>* out_pixels,
    bool update_build_info)
{
    if (out_pixels == nullptr || width <= 0 || height <= 0) {
        return false;
    }

    double accel_build_ms = 0.0;
    double dispatch_ms = 0.0;
    double readback_ms = 0.0;
    double accel_submit_cpu_ms = 0.0;
    double accel_gpu_wait_ms = 0.0;
    double dispatch_submit_cpu_ms = 0.0;
    double dispatch_gpu_wait_ms = 0.0;
    double accel_gpu_ms = 0.0;
    double dispatch_gpu_ms = 0.0;
    {
        std::scoped_lock lock(g_vulkan.mutex);
        g_vulkan.capture_readback_requested = !update_build_info;
#if defined(_WIN32)
        const bool rendered = try_render_vulkan_frame_locked_with_seh(
            "capture_vulkan_to_bgra",
            width,
            height,
            &scene,
            has_frame,
            out_pixels,
            nullptr,
            &accel_build_ms,
            &dispatch_ms,
            &readback_ms,
            &accel_submit_cpu_ms,
            &accel_gpu_wait_ms,
            &dispatch_submit_cpu_ms,
            &dispatch_gpu_wait_ms,
            &accel_gpu_ms,
            &dispatch_gpu_ms);
#else
        const bool rendered = render_vulkan_frame_locked(
            width,
            height,
            scene,
            has_frame,
            out_pixels,
            nullptr,
            &accel_build_ms,
            &dispatch_ms,
            &readback_ms,
            &accel_submit_cpu_ms,
            &accel_gpu_wait_ms,
            &dispatch_submit_cpu_ms,
            &dispatch_gpu_wait_ms,
            &accel_gpu_ms,
            &dispatch_gpu_ms);
#endif
        g_vulkan.capture_readback_requested = false;
        if (!rendered) {
            return false;
        }
    }

    return true;
}

bool capture_vulkan_to_png(const wchar_t* path, int width, int height, const frame_scene &scene, bool has_frame) {
    if (path == nullptr) {
        return false;
    }
    std::vector<std::uint8_t> pixels;
    if (!capture_vulkan_to_bgra(width, height, scene, has_frame, &pixels, false)) {
        return false;
    }
    return viewer_capture::write_png_bgra8(path, pixels.data(), width, height, width * 4);
}

void fill_vulkan_build_info(scene_build_info* out_info) {
    if (out_info == nullptr) {
        return;
    }
    std::scoped_lock lock(g_vulkan.mutex);
    out_info->blas_reused_count = g_vulkan.build_info.blas_reused_count;
    out_info->blas_rebuilt_count = g_vulkan.build_info.blas_rebuilt_count;
    out_info->blas_reused_chunk_count = g_vulkan.build_info.blas_reused_chunk_count;
    out_info->blas_rebuilt_chunk_count = g_vulkan.build_info.blas_rebuilt_chunk_count;
    out_info->tlas_rebuild_count = g_vulkan.build_info.tlas_rebuild_count;
    out_info->accel_build_ms = g_vulkan.build_info.accel_build_ms;
    out_info->accel_submit_cpu_ms = g_vulkan.build_info.accel_submit_cpu_ms;
    out_info->accel_gpu_wait_ms = g_vulkan.build_info.accel_gpu_wait_ms;
    out_info->accel_gpu_ms = g_vulkan.build_info.accel_gpu_ms;
    out_info->dispatch_ms = g_vulkan.build_info.dispatch_ms;
    out_info->dispatch_submit_cpu_ms = g_vulkan.build_info.dispatch_submit_cpu_ms;
    out_info->dispatch_gpu_wait_ms = g_vulkan.build_info.dispatch_gpu_wait_ms;
    out_info->dispatch_gpu_ms = g_vulkan.build_info.dispatch_gpu_ms;
    out_info->readback_ms = g_vulkan.build_info.readback_ms;
    out_info->accumulation_sample_count = g_vulkan.build_info.accumulation_sample_count;
    out_info->accumulation_target_sample_count = g_vulkan.build_info.accumulation_target_sample_count;
    out_info->accumulation_in_progress = g_vulkan.build_info.accumulation_in_progress;
}

bool pick_vulkan(
    int width,
    int height,
    int pixel_x,
    int pixel_y,
    const frame_scene &scene,
    bool has_frame,
    pick_result* out_result)
{
    if (out_result == nullptr || !has_frame || width <= 0 || height <= 0 || pixel_x < 0 || pixel_y < 0) {
        return false;
    }
    rt_scene_build build{};
    copy_present_render_rt_scene_build(&build);
    std::scoped_lock lock(g_vulkan.mutex);
    if (!g_vulkan.initialized) {
        return false;
    }
    return dispatch_pick_query(width, height, pixel_x, pixel_y, scene, build, has_frame, out_result);
}

bool vulkan_accumulation_in_progress() {
    std::scoped_lock lock(g_vulkan.mutex);
    return g_vulkan.accumulation_active;
}

bool get_vulkan_renderer_interop(vulkan_renderer_interop* out_interop) {
    if (out_interop == nullptr) {
        return false;
    }
    std::scoped_lock lock(g_vulkan.mutex);
    if (!g_vulkan.initialized ||
        g_vulkan.instance == VK_NULL_HANDLE ||
        g_vulkan.physical_device == VK_NULL_HANDLE ||
        g_vulkan.device == VK_NULL_HANDLE ||
        g_vulkan.graphics_queue_family == VK_QUEUE_FAMILY_IGNORED ||
        g_vulkan.present_queue_family == VK_QUEUE_FAMILY_IGNORED) {
        *out_interop = {};
        return false;
    }
    *out_interop = {
        g_vulkan.instance,
        g_vulkan.physical_device,
        g_vulkan.device,
        g_vulkan.graphics_queue_family,
        g_vulkan.present_queue_family,
    };
    return true;
}

void vulkan_notify_shell_post_present() {
}

const backend_ops kVulkanBackendOps{
    vulkan_backend_info,
    initialize_vulkan_backend,
    shutdown_vulkan_backend,
    nullptr,
    render_vulkan_to_native_metal_texture,
    render_vulkan_to_native_vulkan_texture,
    capture_vulkan_to_bgra,
    capture_vulkan_to_png,
    fill_vulkan_build_info,
    pick_vulkan,
    vulkan_accumulation_in_progress,
    nullptr,
    get_vulkan_renderer_interop,
    vulkan_notify_shell_post_present,
};

} // namespace

const backend_ops* vulkan_rt_backend_ops() {
    return &kVulkanBackendOps;
}

} // namespace rtvdb::viewer_backend
