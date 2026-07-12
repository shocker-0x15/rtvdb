#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>

#include "viewer_backend/backend_internal.h"
#include "viewer_capture/png.h"
#include "viewer_diagnostics/output.h"
#include "rtvdb_d3d12_raytracing_dxil.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

namespace rtvdb::viewer_backend {
namespace {

constexpr UINT kFrameCount = 2;
constexpr UINT kDescriptorHeapCount = 11;
constexpr UINT kOutputUavDescriptorIndex = 0;
constexpr UINT kAccumulationUavDescriptorIndex = 1;
constexpr UINT kPickOutputUavDescriptorIndex = 2;
constexpr UINT kSceneSrvDescriptorIndex = 3;
constexpr UINT kTriangleColorSrvDescriptorIndex = 4;
constexpr UINT kInstanceMetadataSrvDescriptorIndex = 5;
constexpr UINT kScenePositionSrvDescriptorIndex = 6;
constexpr UINT kSceneIndexSrvDescriptorIndex = 7;
constexpr UINT kPointSrvDescriptorIndex = 8;
constexpr UINT kLineSrvDescriptorIndex = 9;
constexpr UINT kCameraCbvDescriptorIndex = 10;
constexpr std::size_t kCameraConstantBufferBytes = 256;
constexpr std::size_t kShaderRecordSize = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
constexpr std::size_t kInvalidCacheIndex = (std::numeric_limits<std::size_t>::max)();
constexpr std::uint32_t kMaxAccumulationSamples = 64;
constexpr std::size_t kBlasGroupChunkCount = 4;
constexpr float kBlasGroupCenterDistanceScale = 1.5f;
constexpr float kBlasGroupMergedDiagonalScale = 2.5f;
constexpr DXGI_FORMAT kOutputColorFormat = DXGI_FORMAT_B8G8R8A8_UNORM;
constexpr UINT kTriangleHitGroupContributionIndex = 0;
constexpr UINT kPointHitGroupContributionIndex = 1;
constexpr UINT kLineHitGroupContributionIndex = 2;
constexpr std::uint32_t kLineFlagFixedColor = 1u << 0;
constexpr std::uint32_t kLineFlagNonPickable = 1u << 1;


struct viewer_constants {
    float origin[3];
    float pad0 = 0.0f;
    float forward[3];
    float pad1 = 0.0f;
    float right[3];
    float pad2 = 0.0f;
    float up[3];
    float pad3 = 0.0f;
    float scene_bounds_min[3];
    std::uint32_t scene_bounds_valid = 0;
    float scene_bounds_max[3];
    float pad4 = 0.0f;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t projection = 0;
    float aspect = 1.0f;
    float projection_param_from0 = 0.0f;
    float projection_param_from1 = 0.0f;
    float projection_param_to0 = 0.0f;
    float projection_param_to1 = 0.0f;
    std::uint32_t projection_blend_from = 0;
    std::uint32_t projection_blend_to = 0;
    float projection_blend_t = 1.0f;
    std::uint32_t pick_pixel_x = 0;
    std::uint32_t pick_pixel_y = 0;
    std::uint32_t display_mode = 0;
    std::uint32_t accumulation_sample_index = 0;
    std::uint32_t pad5 = 0;
    float accumulation_jitter[2]{};
    std::uint32_t hover_highlight_kind = 0;
    std::uint32_t hover_primitive_index = 0;
    float hover_highlight_mix = 0.0f;
    std::uint32_t is_pick_pass = 0;
    float pad6[2]{};
};

static_assert(sizeof(viewer_constants) == 192, "viewer_constants layout must match HLSL packing");

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

struct gpu_pick_result {
    std::uint32_t primitive_kind = 0;
    std::uint32_t primitive_index = 0;
    float distance = 0.0f;
    float pad = 0.0f;
};

struct instance_metadata {
    std::uint32_t first_triangle = 0;
    std::uint32_t index_offset = 0;
    std::uint32_t primitive_offset = 0;
    std::uint32_t primitive_count = 0;
};

struct point_gpu {
    rtvdb::vec3 position{};
    float radius = 0.0f;
    rtvdb::rgba color{};
};

struct line_gpu {
    rtvdb::vec3 a{};
    float radius = 0.0f;
    rtvdb::vec3 b{};
    float pad = 0.0f;
    rtvdb::rgba color{};
    std::uint32_t flags = 0;
    float pad_flags[3]{};
};

struct pending_chunk_copy {
    std::size_t cache_index = kInvalidCacheIndex;
    std::size_t vertex_bytes = 0;
    std::size_t index_bytes = 0;
};


struct gpu_chunk_cache_entry {
    std::size_t first_triangle = 0;
    std::size_t triangle_count = 0;
    std::uint64_t fingerprint = 0;
    std::size_t vertex_count = 0;
    std::size_t index_count = 0;
    ID3D12Resource* vertex_upload = nullptr;
    ID3D12Resource* index_upload = nullptr;
    ID3D12Resource* vertex_buffer = nullptr;
    ID3D12Resource* index_buffer = nullptr;
    ID3D12Resource* blas_scratch = nullptr;
    ID3D12Resource* blas_result = nullptr;
};

struct gpu_group_cache_entry {
    std::array<std::size_t, kBlasGroupChunkCount> first_triangle{};
    std::array<std::uint64_t, kBlasGroupChunkCount> chunk_fingerprints{};
    std::size_t chunk_count = 0;
    std::size_t vertex_count = 0;
    std::size_t index_count = 0;
    ID3D12Resource* vertex_upload = nullptr;
    ID3D12Resource* index_upload = nullptr;
    ID3D12Resource* blas_scratch = nullptr;
    ID3D12Resource* blas_result = nullptr;
};

struct scene_blas_instance {
    std::array<std::size_t, kBlasGroupChunkCount> chunk_indices{};
    std::size_t chunk_count = 0;
    std::size_t cache_index = kInvalidCacheIndex;
    bool uses_group_blas = false;
};

struct procedural_blas_entry {
    std::size_t first_primitive = 0;
    std::size_t primitive_count = 0;
    std::uint64_t geometry_fingerprint = 0;
    ID3D12Resource* blas_scratch = nullptr;
    ID3D12Resource* blas_result = nullptr;
};

struct dxr_backend_state {
    backend_config config{};
    bool initialized = false;
    bool raytracing_supported = false;
    bool shader_library_ready = false;
    bool output_typed_uav_store_supported = false;
    bool accumulation_typed_uav_load_supported = false;
    bool accumulation_typed_uav_store_supported = false;
    bool native_d3d12_texture_present_supported = false;
    HWND hwnd = nullptr;
    UINT width = 0;
    UINT height = 0;
    UINT output_width = 0;
    UINT output_height = 0;
    UINT frame_index = 0;
    float clear_color[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    std::uint64_t synced_build_revision = 0;
    std::uint64_t synced_accel_revision = 0;
    std::size_t synced_chunk_count = 0;
    std::size_t synced_reused_chunk_count = 0;
    std::size_t synced_rebuilt_chunk_count = 0;
    std::size_t synced_blas_reused_count = 0;
    std::size_t synced_blas_rebuilt_count = 0;
    std::size_t synced_blas_reused_chunk_count = 0;
    std::size_t synced_blas_rebuilt_chunk_count = 0;
    std::size_t synced_tlas_rebuild_count = 0;
    double last_accel_build_ms = 0.0;
    double last_accel_host_prep_ms = 0.0;
    double last_accel_instance_build_ms = 0.0;
    double last_accel_procedural_aabb_ms = 0.0;
    double last_accel_command_record_ms = 0.0;
    double last_accel_resource_alloc_ms = 0.0;
    double last_accel_build_call_record_ms = 0.0;
    double last_accel_prebuild_info_ms = 0.0;
    double last_accel_tlas_instance_upload_ms = 0.0;
    double last_accel_submit_cpu_ms = 0.0;
    double last_accel_gpu_wait_ms = 0.0;
    double last_accel_gpu_ms = 0.0;
    double last_dispatch_ms = 0.0;
    double last_dispatch_submit_cpu_ms = 0.0;
    double last_dispatch_gpu_wait_ms = 0.0;
    double last_dispatch_gpu_ms = 0.0;
    double last_readback_ms = 0.0;
    std::size_t synced_vertex_bytes = 0;
    std::size_t synced_index_bytes = 0;
    std::uint32_t last_chunk_sync_error = 0;
    std::size_t last_chunk_sync_error_index = kInvalidCacheIndex;
    std::vector<gpu_chunk_cache_entry> chunk_cache;
    std::vector<gpu_group_cache_entry> group_cache;
    std::vector<std::size_t> scene_chunk_cache_indices;
    std::vector<scene_blas_instance> scene_blas_instances;
    ID3D12Resource* tlas_instance_upload = nullptr;
    ID3D12Resource* tlas_scratch = nullptr;
    ID3D12Resource* tlas_result = nullptr;
    std::size_t tlas_instance_count = 0;
    ID3D12Resource* point_aabb_buffer = nullptr;
    ID3D12Resource* line_aabb_buffer = nullptr;
    std::uint64_t point_geometry_fingerprint = 0;
    std::uint64_t line_geometry_fingerprint = 0;
    std::size_t point_primitive_count = 0;
    std::size_t line_primitive_count = 0;

    IDXGIFactory7* factory = nullptr;
    IDXGIAdapter1* adapter = nullptr;
    ID3D12Device5* device = nullptr;
    ID3D12CommandQueue* queue = nullptr;
    IDXGISwapChain3* swapchain = nullptr;
    ID3D12DescriptorHeap* rtv_heap = nullptr;
    UINT rtv_descriptor_size = 0;
    std::array<ID3D12Resource*, kFrameCount> render_targets{};
    std::array<ID3D12CommandAllocator*, kFrameCount> command_allocators{};
    UINT command_allocator_index = 0;
    ID3D12GraphicsCommandList4* command_list = nullptr;
    ID3D12Fence* fence = nullptr;
    std::array<std::uint64_t, kFrameCount> command_allocator_fence_values{};
    std::uint64_t submitted_fence_value = 0;
    HANDLE fence_event = nullptr;
    ID3D12QueryHeap* timestamp_query_heap = nullptr;
    ID3D12Resource* timestamp_query_readback = nullptr;
    std::uint64_t timestamp_frequency = 0;
    std::array<std::uint64_t, kFrameCount> dispatch_timestamp_fence_values{};
    std::array<bool, kFrameCount> dispatch_timestamp_pending{};

    ID3D12DescriptorHeap* srv_uav_cbv_heap = nullptr;
    UINT srv_uav_cbv_descriptor_size = 0;
    ID3D12RootSignature* global_root_signature = nullptr;
    ID3D12StateObject* raytracing_state_object = nullptr;
    ID3D12StateObjectProperties* raytracing_state_props = nullptr;
    ID3D12Resource* output_texture = nullptr;
    ID3D12Resource* accumulation_texture = nullptr;
    ID3D12Resource* pick_output_buffer = nullptr;
    ID3D12Resource* pick_readback_buffer = nullptr;
    ID3D12Resource* output_readback_buffer = nullptr;
    std::size_t output_readback_size = 0;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT output_readback_footprint{};
    ID3D12Resource* native_display_target = nullptr;
    D3D12_RESOURCE_STATES native_display_target_state = D3D12_RESOURCE_STATE_COPY_DEST;
    ID3D12Resource* triangle_color_buffer = nullptr;
    ID3D12Resource* instance_metadata_buffer = nullptr;
    ID3D12Resource* scene_position_buffer = nullptr;
    ID3D12Resource* scene_index_buffer = nullptr;
    ID3D12Resource* point_buffer = nullptr;
    ID3D12Resource* line_buffer = nullptr;
    std::vector<procedural_blas_entry> point_blas_entries;
    std::vector<procedural_blas_entry> line_blas_entries;
    ID3D12Resource* camera_constant_buffer = nullptr;
    void* camera_constant_buffer_mapped = nullptr;
    ID3D12Resource* raygen_shader_table = nullptr;
    ID3D12Resource* pick_raygen_shader_table = nullptr;
    ID3D12Resource* miss_shader_table = nullptr;
    ID3D12Resource* hitgroup_shader_table = nullptr;
    std::vector<std::uint8_t> shader_library;
    accumulation_key accumulation_key{};
    std::uint32_t accumulation_sample_count = 0;
    bool accumulation_active = false;
    bool owns_device_queue = false;
} g_dxr_backend;

std::mutex g_dxr_backend_api_mutex;

std::string capture_log_path(const char* filename) {
    try {
        if (!rtvdb::viewer_diagnostics::output_enabled()) {
            return {};
        }
        const std::filesystem::path dir = rtvdb::viewer_diagnostics::output_directory();
        std::filesystem::create_directories(dir);
        return (dir / (filename != nullptr ? filename : "dxr.log")).string();
    } catch (...) {
        return filename != nullptr ? filename : "dxr.log";
    }
}

void append_timestamped_log_line(const char* filename, const char* text) {
    if (!rtvdb::viewer_diagnostics::output_enabled() || text == nullptr || *text == '\0') {
        return;
    }

    std::ofstream file(capture_log_path(filename), std::ios::binary | std::ios::app);
    if (!file.is_open()) {
        return;
    }

    SYSTEMTIME system_time{};
    GetLocalTime(&system_time);
    char line[2048]{};
    std::snprintf(
        line,
        sizeof(line),
        "[%04u-%02u-%02u %02u:%02u:%02u.%03u] %s\n",
        static_cast<unsigned>(system_time.wYear),
        static_cast<unsigned>(system_time.wMonth),
        static_cast<unsigned>(system_time.wDay),
        static_cast<unsigned>(system_time.wHour),
        static_cast<unsigned>(system_time.wMinute),
        static_cast<unsigned>(system_time.wSecond),
        static_cast<unsigned>(system_time.wMilliseconds),
        text);
    file << line;
}

void append_startup_log(const char* message) {
    append_timestamped_log_line("startup.log", message);
}

void append_startup_hresult_log(const char* message, HRESULT hr) {
    char line[512]{};
    std::snprintf(line, sizeof(line), "%s hr=0x%08lx", message != nullptr ? message : "DXR startup failed", hr);
    append_startup_log(line);
}

void record_dxr_failure(const char* stage, HRESULT hr = S_OK, const char* detail = nullptr) {
    char line[1024]{};
    if (detail != nullptr && *detail != '\0') {
        std::snprintf(
            line,
            sizeof(line),
            "stage=%s hr=0x%08lx detail=%s",
            stage != nullptr ? stage : "unknown",
            static_cast<unsigned long>(hr),
            detail);
    } else {
        std::snprintf(
            line,
            sizeof(line),
            "stage=%s hr=0x%08lx",
            stage != nullptr ? stage : "unknown",
            static_cast<unsigned long>(hr));
    }
    append_timestamped_log_line("dxr_failure.log", line);
}

bool device_supports_raytracing(ID3D12Device5* device, HRESULT* out_hr) {
    if (device == nullptr) {
        if (out_hr != nullptr) {
            *out_hr = E_POINTER;
        }
        return false;
    }

    D3D12_FEATURE_DATA_D3D12_OPTIONS5 options5{};
    const HRESULT hr = device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &options5, sizeof(options5));
    if (out_hr != nullptr) {
        *out_hr = hr;
    }
    if (FAILED(hr)) {
        return false;
    }
    return options5.RaytracingTier != D3D12_RAYTRACING_TIER_NOT_SUPPORTED;
}

bool query_format_support(
    ID3D12Device5* device,
    DXGI_FORMAT format,
    D3D12_FORMAT_SUPPORT1* out_support1,
    D3D12_FORMAT_SUPPORT2* out_support2,
    HRESULT* out_hr)
{
    if (device == nullptr) {
        if (out_hr != nullptr) {
            *out_hr = E_POINTER;
        }
        return false;
    }

    D3D12_FEATURE_DATA_FORMAT_SUPPORT support{};
    support.Format = format;
    const HRESULT hr = device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support));
    if (out_hr != nullptr) {
        *out_hr = hr;
    }
    if (FAILED(hr)) {
        return false;
    }
    if (out_support1 != nullptr) {
        *out_support1 = support.Support1;
    }
    if (out_support2 != nullptr) {
        *out_support2 = support.Support2;
    }
    return true;
}

bool ensure_required_uav_format_support(HRESULT* out_hr) {
    D3D12_FORMAT_SUPPORT1 output_support1 = D3D12_FORMAT_SUPPORT1_NONE;
    D3D12_FORMAT_SUPPORT2 output_support2 = D3D12_FORMAT_SUPPORT2_NONE;
    HRESULT hr = S_OK;
    if (!query_format_support(
            g_dxr_backend.device,
            kOutputColorFormat,
            &output_support1,
            &output_support2,
            &hr)) {
        if (out_hr != nullptr) {
            *out_hr = hr;
        }
        return false;
    }
    g_dxr_backend.output_typed_uav_store_supported =
        (output_support1 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW) != 0 &&
        (output_support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE) != 0;

    D3D12_FORMAT_SUPPORT1 accumulation_support1 = D3D12_FORMAT_SUPPORT1_NONE;
    D3D12_FORMAT_SUPPORT2 accumulation_support2 = D3D12_FORMAT_SUPPORT2_NONE;
    if (!query_format_support(
            g_dxr_backend.device,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            &accumulation_support1,
            &accumulation_support2,
            &hr)) {
        if (out_hr != nullptr) {
            *out_hr = hr;
        }
        return false;
    }
    const bool accumulation_has_uav =
        (accumulation_support1 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW) != 0;
    g_dxr_backend.accumulation_typed_uav_load_supported =
        accumulation_has_uav &&
        (accumulation_support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD) != 0;
    g_dxr_backend.accumulation_typed_uav_store_supported =
        accumulation_has_uav &&
        (accumulation_support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE) != 0;

    if (out_hr != nullptr) {
        *out_hr = S_OK;
    }
    return g_dxr_backend.output_typed_uav_store_supported &&
        g_dxr_backend.accumulation_typed_uav_load_supported &&
        g_dxr_backend.accumulation_typed_uav_store_supported;
}

void safe_release(IUnknown* &object) {
    if (object != nullptr) {
        object->Release();
        object = nullptr;
    }
}

template <typename T>
void safe_release(T* &object) {
    if (object != nullptr) {
        object->Release();
        object = nullptr;
    }
}

void release_procedural_blas_entry(procedural_blas_entry* entry) {
    if (entry == nullptr) {
        return;
    }
    safe_release(entry->blas_scratch);
    safe_release(entry->blas_result);
    entry->first_primitive = 0;
    entry->primitive_count = 0;
    entry->geometry_fingerprint = 0;
}

void release_procedural_blas_entries(std::vector<procedural_blas_entry>* entries) {
    if (entries == nullptr) {
        return;
    }
    for (procedural_blas_entry &entry : *entries) {
        release_procedural_blas_entry(&entry);
    }
    entries->clear();
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1u) / alignment * alignment;
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

float dot(rtvdb::vec3 a, rtvdb::vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

float length3(rtvdb::vec3 v) {
    return std::sqrt(dot(v, v));
}

rtvdb::vec3 normalize_or(rtvdb::vec3 v, rtvdb::vec3 fallback) {
    const float len = length3(v);
    if (len <= 1.0e-6f) {
        return fallback;
    }
    const float inv = 1.0f / len;
    return {v.x * inv, v.y * inv, v.z * inv};
}

HWND as_hwnd(viewer_shell::native_window_handle window) {
    if (window.kind != viewer_shell::native_window_kind::win32_hwnd) {
        return nullptr;
    }
    return static_cast<HWND>(window.value);
}

bool environment_flag_enabled(const char* name) {
    if (name == nullptr || *name == '\0') {
        return false;
    }

    char value[16]{};
    const DWORD size = GetEnvironmentVariableA(name, value, static_cast<DWORD>(std::size(value)));
    if (size == 0 || size >= std::size(value)) {
        return false;
    }
    return value[0] == '1' || value[0] == 't' || value[0] == 'T' || value[0] == 'y' || value[0] == 'Y';
}

std::size_t environment_group_size(
    const char* name,
    std::size_t default_value,
    std::size_t min_value,
    std::size_t max_value)
{
    if (name == nullptr || *name == '\0') {
        return default_value;
    }

    char value[16]{};
    const DWORD size = GetEnvironmentVariableA(name, value, static_cast<DWORD>(std::size(value)));
    if (size == 0 || size >= std::size(value)) {
        return default_value;
    }

    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value || *end != '\0') {
        return default_value;
    }

    const std::size_t group_size = static_cast<std::size_t>(parsed);
    return (std::min)((std::max)(group_size, min_value), max_value);
}

std::size_t current_blas_group_chunk_limit() {
    return environment_group_size("RTVDB_DXR_BLAS_GROUP_SIZE", kBlasGroupChunkCount, 1u, 16u);
}

backend_info dxr_backend_info() {
    return {
        backend_kind::d3d12_dxr,
        "d3d12_dxr",
        {true, true, false}
    };
}

void update_clear_color(const frame_scene &scene, bool has_frame) {
    (void)scene;
    (void)has_frame;
    g_dxr_backend.clear_color[0] = 0.0f;
    g_dxr_backend.clear_color[1] = 0.0f;
    g_dxr_backend.clear_color[2] = 0.0f;
    g_dxr_backend.clear_color[3] = 1.0f;
}

void fill_solid_bgra(std::vector<std::uint8_t>* out_pixels, int width, int height, const float clear_color[4]) {
    if (out_pixels == nullptr || width <= 0 || height <= 0) {
        return;
    }

    const std::uint8_t b = static_cast<std::uint8_t>(
        (clear_color[2] < 0.0f ? 0.0f : (clear_color[2] > 1.0f ? 1.0f : clear_color[2])) * 255.0f);
    const std::uint8_t g = static_cast<std::uint8_t>(
        (clear_color[1] < 0.0f ? 0.0f : (clear_color[1] > 1.0f ? 1.0f : clear_color[1])) * 255.0f);
    const std::uint8_t r = static_cast<std::uint8_t>(
        (clear_color[0] < 0.0f ? 0.0f : (clear_color[0] > 1.0f ? 1.0f : clear_color[0])) * 255.0f);
    const std::uint8_t a = 255;

    out_pixels->resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4u);
    for (std::size_t i = 0; i < out_pixels->size(); i += 4) {
        (*out_pixels)[i + 0] = b;
        (*out_pixels)[i + 1] = g;
        (*out_pixels)[i + 2] = r;
        (*out_pixels)[i + 3] = a;
    }
}
bool write_solid_png(const wchar_t* path, int width, int height, const float clear_color[4]) {
    if (path == nullptr || width <= 0 || height <= 0) {
        return false;
    }

    const std::uint8_t b = static_cast<std::uint8_t>(
        (clear_color[2] < 0.0f ? 0.0f : (clear_color[2] > 1.0f ? 1.0f : clear_color[2])) * 255.0f);
    const std::uint8_t g = static_cast<std::uint8_t>(
        (clear_color[1] < 0.0f ? 0.0f : (clear_color[1] > 1.0f ? 1.0f : clear_color[1])) * 255.0f);
    const std::uint8_t r = static_cast<std::uint8_t>(
        (clear_color[0] < 0.0f ? 0.0f : (clear_color[0] > 1.0f ? 1.0f : clear_color[0])) * 255.0f);
    const std::uint8_t a = 255;

    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4u);
    for (std::size_t i = 0; i < pixels.size(); i += 4) {
        pixels[i + 0] = b;
        pixels[i + 1] = g;
        pixels[i + 2] = r;
        pixels[i + 3] = a;
    }

    return viewer_capture::write_png_bgra8(path, pixels.data(), width, height, width * 4);
}

bool write_bgra_png(const wchar_t* path, int width, int height, const std::vector<std::uint8_t> &bgra_pixels) {
    if (path == nullptr || width <= 0 || height <= 0) {
        return false;
    }
    const std::size_t expected_size = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4u;
    if (bgra_pixels.size() != expected_size) {
        return false;
    }

    return viewer_capture::write_png_bgra8(path, bgra_pixels.data(), width, height, width * 4);
}

bool reset_command_list();
bool close_and_execute_command_list(double* out_submit_cpu_ms = nullptr);

bool wait_for_fence_value(std::uint64_t value) {
    if (value == 0) {
        return true;
    }
    if (g_dxr_backend.fence == nullptr || g_dxr_backend.fence_event == nullptr) {
        return false;
    }
    if (g_dxr_backend.fence->GetCompletedValue() >= value) {
        return true;
    }
    if (FAILED(g_dxr_backend.fence->SetEventOnCompletion(value, g_dxr_backend.fence_event))) {
        return false;
    }
    return WaitForSingleObject(g_dxr_backend.fence_event, INFINITE) == WAIT_OBJECT_0;
}

bool wait_for_fence_value_timed(std::uint64_t value, double* out_wait_ms) {
    const auto wait_start = std::chrono::steady_clock::now();
    const bool waited = wait_for_fence_value(value);
    if (out_wait_ms != nullptr) {
        *out_wait_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - wait_start).count();
    }
    return waited;
}

void wait_for_gpu() {
    if (g_dxr_backend.queue == nullptr || g_dxr_backend.fence == nullptr || g_dxr_backend.fence_event == nullptr) {
        return;
    }
    (void)wait_for_fence_value(g_dxr_backend.submitted_fence_value);
}

UINT current_timestamp_query_slot() {
    return g_dxr_backend.command_allocator_index % kFrameCount;
}

bool ensure_timestamp_query_resources() {
    if (g_dxr_backend.device == nullptr || g_dxr_backend.queue == nullptr) {
        return false;
    }
    if (g_dxr_backend.timestamp_query_heap != nullptr &&
        g_dxr_backend.timestamp_query_readback != nullptr &&
        g_dxr_backend.timestamp_frequency != 0) {
        return true;
    }

    safe_release(g_dxr_backend.timestamp_query_heap);
    safe_release(g_dxr_backend.timestamp_query_readback);
    g_dxr_backend.timestamp_frequency = 0;
    g_dxr_backend.dispatch_timestamp_fence_values.fill(0);
    g_dxr_backend.dispatch_timestamp_pending.fill(false);

    D3D12_QUERY_HEAP_DESC heap_desc{};
    heap_desc.Count = kFrameCount * 2u;
    heap_desc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    HRESULT hr = g_dxr_backend.device->CreateQueryHeap(&heap_desc, IID_PPV_ARGS(&g_dxr_backend.timestamp_query_heap));
    if (FAILED(hr)) {
        record_dxr_failure("ensure_timestamp_query_resources.CreateQueryHeap", hr);
        return false;
    }

    D3D12_HEAP_PROPERTIES heap_properties{};
    heap_properties.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC resource_desc{};
    resource_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    resource_desc.Width = static_cast<UINT64>(heap_desc.Count) * sizeof(std::uint64_t);
    resource_desc.Height = 1;
    resource_desc.DepthOrArraySize = 1;
    resource_desc.MipLevels = 1;
    resource_desc.SampleDesc.Count = 1;
    resource_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    hr = g_dxr_backend.device->CreateCommittedResource(
        &heap_properties,
        D3D12_HEAP_FLAG_NONE,
        &resource_desc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(&g_dxr_backend.timestamp_query_readback));
    if (FAILED(hr)) {
        record_dxr_failure("ensure_timestamp_query_resources.CreateCommittedResource", hr);
        safe_release(g_dxr_backend.timestamp_query_heap);
        return false;
    }

    hr = g_dxr_backend.queue->GetTimestampFrequency(&g_dxr_backend.timestamp_frequency);
    if (FAILED(hr) || g_dxr_backend.timestamp_frequency == 0) {
        record_dxr_failure("ensure_timestamp_query_resources.GetTimestampFrequency", hr);
        safe_release(g_dxr_backend.timestamp_query_heap);
        safe_release(g_dxr_backend.timestamp_query_readback);
        g_dxr_backend.timestamp_frequency = 0;
        return false;
    }
    return true;
}

void write_timestamp_query_range(UINT slot) {
    if (g_dxr_backend.command_list == nullptr ||
        g_dxr_backend.timestamp_query_heap == nullptr ||
        g_dxr_backend.timestamp_query_readback == nullptr ||
        slot >= kFrameCount) {
        return;
    }

    const UINT start_index = slot * 2u;
    g_dxr_backend.command_list->EndQuery(g_dxr_backend.timestamp_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, start_index);
    g_dxr_backend.command_list->EndQuery(g_dxr_backend.timestamp_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, start_index + 1u);
}

void write_timestamp_query_begin(UINT slot) {
    if (g_dxr_backend.command_list == nullptr || g_dxr_backend.timestamp_query_heap == nullptr || slot >= kFrameCount) {
        return;
    }
    g_dxr_backend.command_list->EndQuery(g_dxr_backend.timestamp_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, slot * 2u);
}

void write_timestamp_query_end_and_resolve(UINT slot) {
    if (g_dxr_backend.command_list == nullptr ||
        g_dxr_backend.timestamp_query_heap == nullptr ||
        g_dxr_backend.timestamp_query_readback == nullptr ||
        slot >= kFrameCount) {
        return;
    }
    const UINT start_index = slot * 2u;
    g_dxr_backend.command_list->EndQuery(g_dxr_backend.timestamp_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, start_index + 1u);
    g_dxr_backend.command_list->ResolveQueryData(
        g_dxr_backend.timestamp_query_heap,
        D3D12_QUERY_TYPE_TIMESTAMP,
        start_index,
        2,
        g_dxr_backend.timestamp_query_readback,
        static_cast<UINT64>(start_index) * sizeof(std::uint64_t));
}

bool read_timestamp_query_ms(UINT slot, double* out_ms) {
    if (out_ms == nullptr ||
        g_dxr_backend.timestamp_query_readback == nullptr ||
        g_dxr_backend.timestamp_frequency == 0 ||
        slot >= kFrameCount) {
        return false;
    }

    void* mapped = nullptr;
    const UINT64 offset_bytes = static_cast<UINT64>(slot * 2u) * sizeof(std::uint64_t);
    D3D12_RANGE read_range{offset_bytes, offset_bytes + sizeof(std::uint64_t) * 2u};
    const HRESULT hr = g_dxr_backend.timestamp_query_readback->Map(0, &read_range, &mapped);
    if (FAILED(hr) || mapped == nullptr) {
        record_dxr_failure("read_timestamp_query_ms.Map", hr);
        return false;
    }

    const auto* timestamps =
        reinterpret_cast<const std::uint64_t*>(static_cast<const std::uint8_t*>(mapped) + offset_bytes);
    const std::uint64_t start = timestamps[0];
    const std::uint64_t end = timestamps[1];
    g_dxr_backend.timestamp_query_readback->Unmap(0, nullptr);

    if (end <= start) {
        return false;
    }

    *out_ms = static_cast<double>(end - start) * 1000.0 / static_cast<double>(g_dxr_backend.timestamp_frequency);
    return true;
}

void collect_completed_dispatch_timestamp_queries() {
    if (g_dxr_backend.fence == nullptr) {
        return;
    }

    const std::uint64_t completed_value = g_dxr_backend.fence->GetCompletedValue();
    for (UINT slot = 0; slot < kFrameCount; ++slot) {
        if (!g_dxr_backend.dispatch_timestamp_pending[slot]) {
            continue;
        }
        if (completed_value < g_dxr_backend.dispatch_timestamp_fence_values[slot]) {
            continue;
        }
        double measured_ms = 0.0;
        if (read_timestamp_query_ms(slot, &measured_ms)) {
            g_dxr_backend.last_dispatch_gpu_ms = measured_ms;
        }
        g_dxr_backend.dispatch_timestamp_pending[slot] = false;
        g_dxr_backend.dispatch_timestamp_fence_values[slot] = 0;
    }
}

bool create_buffer(
    D3D12_HEAP_TYPE heap_type,
    std::size_t size_bytes,
    D3D12_RESOURCE_STATES initial_state,
    D3D12_RESOURCE_FLAGS flags,
    ID3D12Resource** out_resource)
{
    if (out_resource == nullptr || g_dxr_backend.device == nullptr) {
        return false;
    }
    *out_resource = nullptr;
    if (size_bytes == 0) {
        return true;
    }

    D3D12_HEAP_PROPERTIES heap_properties{};
    heap_properties.Type = heap_type;

    D3D12_RESOURCE_DESC resource_desc{};
    resource_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    resource_desc.Width = size_bytes;
    resource_desc.Height = 1;
    resource_desc.DepthOrArraySize = 1;
    resource_desc.MipLevels = 1;
    resource_desc.SampleDesc.Count = 1;
    resource_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    resource_desc.Flags = flags;

    const HRESULT hr = g_dxr_backend.device->CreateCommittedResource(
        &heap_properties,
        D3D12_HEAP_FLAG_NONE,
        &resource_desc,
        initial_state,
        nullptr,
        IID_PPV_ARGS(out_resource));
    if (FAILED(hr)) {
        record_dxr_failure("create_buffer.CreateCommittedResource", hr);
    }
    return SUCCEEDED(hr);
}

bool upload_buffer_data(ID3D12Resource* resource, const void* data, std::size_t size_bytes) {
    if (size_bytes == 0) {
        return true;
    }
    if (resource == nullptr || data == nullptr) {
        return false;
    }

    void* mapped = nullptr;
    D3D12_RANGE read_range{0, 0};
    const HRESULT hr = resource->Map(0, &read_range, &mapped);
    if (FAILED(hr)) {
        record_dxr_failure("upload_buffer_data.Map", hr);
        return false;
    }
    std::memcpy(mapped, data, size_bytes);
    D3D12_RANGE written_range{0, size_bytes};
    resource->Unmap(0, &written_range);
    return true;
}

bool has_renderable_primitives(const rt_scene_build &build) {
    return !build.chunks.empty() || build.point_count > 0 || build.line_count > 0;
}

rtvdb::vec3 bounds_center(const scene_bounds &bounds) {
    return {
        (bounds.min.x + bounds.max.x) * 0.5f,
        (bounds.min.y + bounds.max.y) * 0.5f,
        (bounds.min.z + bounds.max.z) * 0.5f,
    };
}

float bounds_diagonal_length(const scene_bounds &bounds) {
    if (!bounds.valid) {
        return 0.0f;
    }
    return length3(subtract(bounds.max, bounds.min));
}

scene_bounds merge_bounds(const scene_bounds &a, const scene_bounds &b) {
    if (!a.valid) {
        return b;
    }
    if (!b.valid) {
        return a;
    }

    scene_bounds merged{};
    merged.valid = true;
    merged.min.x = (std::min)(a.min.x, b.min.x);
    merged.min.y = (std::min)(a.min.y, b.min.y);
    merged.min.z = (std::min)(a.min.z, b.min.z);
    merged.max.x = (std::max)(a.max.x, b.max.x);
    merged.max.y = (std::max)(a.max.y, b.max.y);
    merged.max.z = (std::max)(a.max.z, b.max.z);
    return merged;
}

void release_chunk_cache_entry(gpu_chunk_cache_entry* entry) {
    if (entry == nullptr) {
        return;
    }
    safe_release(entry->vertex_upload);
    safe_release(entry->index_upload);
    safe_release(entry->vertex_buffer);
    safe_release(entry->index_buffer);
    safe_release(entry->blas_scratch);
    safe_release(entry->blas_result);
    *entry = {};
}

void release_group_cache_entry(gpu_group_cache_entry* entry) {
    if (entry == nullptr) {
        return;
    }
    safe_release(entry->vertex_upload);
    safe_release(entry->index_upload);
    safe_release(entry->blas_scratch);
    safe_release(entry->blas_result);
    *entry = {};
}

void clear_chunk_cache() {
    for (gpu_chunk_cache_entry &entry : g_dxr_backend.chunk_cache) {
        release_chunk_cache_entry(&entry);
    }
    g_dxr_backend.chunk_cache.clear();
    for (gpu_group_cache_entry &entry : g_dxr_backend.group_cache) {
        release_group_cache_entry(&entry);
    }
    g_dxr_backend.group_cache.clear();
    g_dxr_backend.scene_chunk_cache_indices.clear();
    g_dxr_backend.scene_blas_instances.clear();
    safe_release(g_dxr_backend.tlas_instance_upload);
    safe_release(g_dxr_backend.tlas_scratch);
    safe_release(g_dxr_backend.tlas_result);
    g_dxr_backend.tlas_instance_count = 0;
    release_procedural_blas_entries(&g_dxr_backend.point_blas_entries);
    release_procedural_blas_entries(&g_dxr_backend.line_blas_entries);
    g_dxr_backend.synced_build_revision = 0;
    g_dxr_backend.synced_accel_revision = 0;
    g_dxr_backend.synced_chunk_count = 0;
    g_dxr_backend.synced_reused_chunk_count = 0;
    g_dxr_backend.synced_rebuilt_chunk_count = 0;
    g_dxr_backend.synced_blas_reused_count = 0;
    g_dxr_backend.synced_blas_rebuilt_count = 0;
    g_dxr_backend.synced_blas_reused_chunk_count = 0;
    g_dxr_backend.synced_blas_rebuilt_chunk_count = 0;
    g_dxr_backend.synced_tlas_rebuild_count = 0;
    g_dxr_backend.last_accel_host_prep_ms = 0.0;
    g_dxr_backend.last_accel_instance_build_ms = 0.0;
    g_dxr_backend.last_accel_procedural_aabb_ms = 0.0;
    g_dxr_backend.last_accel_command_record_ms = 0.0;
    g_dxr_backend.last_accel_resource_alloc_ms = 0.0;
    g_dxr_backend.last_accel_build_call_record_ms = 0.0;
    g_dxr_backend.last_accel_prebuild_info_ms = 0.0;
    g_dxr_backend.last_accel_tlas_instance_upload_ms = 0.0;
    g_dxr_backend.synced_vertex_bytes = 0;
    g_dxr_backend.synced_index_bytes = 0;
    g_dxr_backend.last_chunk_sync_error = 0;
    g_dxr_backend.last_chunk_sync_error_index = kInvalidCacheIndex;
}

void reset_accumulation_state() {
    g_dxr_backend.accumulation_key = {};
    g_dxr_backend.accumulation_sample_count = 0;
    g_dxr_backend.accumulation_active = false;
}

accumulation_key make_accumulation_key(
    const frame_scene &scene,
    bool has_frame,
    const rt_scene_build &build,
    UINT width, UINT height,
    std::uint32_t display_mode)
{
    accumulation_key key{};
    hover_highlight highlight{};
    get_hover_highlight(&highlight);
    key.build_revision = build.revision;
    key.width = width;
    key.height = height;
    key.display_mode = display_mode;
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

bool cache_entry_matches_chunk_upload(const gpu_chunk_cache_entry &entry, const rt_scene_chunk &chunk) {
    return entry.fingerprint == chunk.fingerprint &&
        entry.triangle_count == chunk.triangle_count &&
        entry.vertex_count == chunk.vertex_count &&
        entry.index_count == chunk.index_count &&
        entry.vertex_buffer != nullptr &&
        entry.index_buffer != nullptr;
}

bool cache_entry_matches_chunk_blas(const gpu_chunk_cache_entry &entry, const rt_scene_chunk &chunk) {
    return cache_entry_matches_chunk_upload(entry, chunk) &&
        entry.blas_result != nullptr;
}

bool cache_entry_matches_group_blas(
    const gpu_group_cache_entry &entry,
    const scene_blas_instance &instance,
    const rt_scene_build &build)
{
    if (entry.chunk_count != instance.chunk_count ||
        entry.blas_result == nullptr) {
        return false;
    }
    for (std::size_t i = 0; i < instance.chunk_count; ++i) {
        const std::size_t chunk_index = instance.chunk_indices[i];
        if (chunk_index >= build.chunks.size()) {
            return false;
        }
        const rt_scene_chunk &chunk = build.chunks[chunk_index];
        if (entry.first_triangle[i] != chunk.first_triangle || entry.chunk_fingerprints[i] != chunk.fingerprint) {
            return false;
        }
    }
    std::size_t total_vertex_count = 0;
    std::size_t total_index_count = 0;
    for (std::size_t i = 0; i < instance.chunk_count; ++i) {
        const rt_scene_chunk &chunk = build.chunks[instance.chunk_indices[i]];
        total_vertex_count += chunk.vertex_count;
        total_index_count += chunk.index_count;
    }
    return entry.vertex_count == total_vertex_count && entry.index_count == total_index_count;
}

bool upload_pending_chunk_copies(const std::vector<pending_chunk_copy> &pending_copies) {
    if (pending_copies.empty()) {
        return true;
    }
    if (!reset_command_list()) {
        return false;
    }

    std::vector<D3D12_RESOURCE_BARRIER> barriers;
    barriers.reserve(pending_copies.size() * 2u);
    for (const pending_chunk_copy &pending : pending_copies) {
        if (pending.cache_index >= g_dxr_backend.chunk_cache.size()) {
            return false;
        }
        gpu_chunk_cache_entry &cache_entry = g_dxr_backend.chunk_cache[pending.cache_index];
        if (cache_entry.vertex_upload == nullptr || cache_entry.index_upload == nullptr ||
            cache_entry.vertex_buffer == nullptr || cache_entry.index_buffer == nullptr) {
            return false;
        }

        if (pending.vertex_bytes != 0) {
            g_dxr_backend.command_list->CopyBufferRegion(
                cache_entry.vertex_buffer,
                0,
                cache_entry.vertex_upload,
                0,
                pending.vertex_bytes);
            D3D12_RESOURCE_BARRIER barrier{};
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Transition.pResource = cache_entry.vertex_buffer;
            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            barriers.push_back(barrier);
        }
        if (pending.index_bytes != 0) {
            g_dxr_backend.command_list->CopyBufferRegion(
                cache_entry.index_buffer,
                0,
                cache_entry.index_upload,
                0,
                pending.index_bytes);
            D3D12_RESOURCE_BARRIER barrier{};
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Transition.pResource = cache_entry.index_buffer;
            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            barriers.push_back(barrier);
        }
    }

    if (!barriers.empty()) {
        g_dxr_backend.command_list->ResourceBarrier(static_cast<UINT>(barriers.size()), barriers.data());
    }
    if (!close_and_execute_command_list()) {
        return false;
    }
    wait_for_gpu();

    for (const pending_chunk_copy &pending : pending_copies) {
        if (pending.cache_index >= g_dxr_backend.chunk_cache.size()) {
            continue;
        }
        gpu_chunk_cache_entry &cache_entry = g_dxr_backend.chunk_cache[pending.cache_index];
        safe_release(cache_entry.vertex_upload);
        safe_release(cache_entry.index_upload);
    }
    return true;
}

std::size_t find_reusable_chunk_cache_entry(const rt_scene_chunk &chunk, const std::vector<bool> &claimed_entries) {
    for (std::size_t i = 0; i < g_dxr_backend.chunk_cache.size(); ++i) {
        if (i < claimed_entries.size() && claimed_entries[i]) {
            continue;
        }
        const gpu_chunk_cache_entry &cache_entry = g_dxr_backend.chunk_cache[i];
        if (cache_entry.first_triangle != chunk.first_triangle) {
            continue;
        }
        if (cache_entry_matches_chunk_upload(cache_entry, chunk)) {
            return i;
        }
    }
    return kInvalidCacheIndex;
}

std::size_t append_chunk_cache_entry() {
    g_dxr_backend.chunk_cache.push_back({});
    return g_dxr_backend.chunk_cache.size() - 1;
}

std::size_t allocate_chunk_cache_entry(const std::vector<bool> &claimed_entries) {
    for (std::size_t i = 0; i < g_dxr_backend.chunk_cache.size(); ++i) {
        if (i < claimed_entries.size() && claimed_entries[i]) {
            continue;
        }
        return i;
    }
    return append_chunk_cache_entry();
}

std::size_t find_reusable_group_cache_entry(
    const scene_blas_instance &instance,
    const rt_scene_build &build,
    const std::vector<bool> &claimed_entries)
{
    for (std::size_t i = 0; i < g_dxr_backend.group_cache.size(); ++i) {
        if (i < claimed_entries.size() && claimed_entries[i]) {
            continue;
        }
        if (cache_entry_matches_group_blas(g_dxr_backend.group_cache[i], instance, build)) {
            return i;
        }
    }
    return kInvalidCacheIndex;
}

std::size_t append_group_cache_entry() {
    g_dxr_backend.group_cache.push_back({});
    return g_dxr_backend.group_cache.size() - 1;
}

std::size_t allocate_group_cache_entry(const std::vector<bool> &claimed_entries) {
    for (std::size_t i = 0; i < g_dxr_backend.group_cache.size(); ++i) {
        if (i < claimed_entries.size() && claimed_entries[i]) {
            continue;
        }
        return i;
    }
    return append_group_cache_entry();
}

bool can_append_chunk_to_group(
    const scene_bounds &current_group_bounds,
    std::size_t current_group_size,
    const rt_scene_chunk &candidate)
{
    if (current_group_size == 0) {
        return true;
    }
    const scene_bounds merged_bounds = merge_bounds(current_group_bounds, candidate.bounds);
    const float current_diagonal = bounds_diagonal_length(current_group_bounds);
    const float candidate_diagonal = bounds_diagonal_length(candidate.bounds);
    const float merged_diagonal = bounds_diagonal_length(merged_bounds);
    const float scale = (std::max)(0.001f, (std::max)(current_diagonal, candidate_diagonal));
    const rtvdb::vec3 current_center = bounds_center(current_group_bounds);
    const rtvdb::vec3 candidate_center = bounds_center(candidate.bounds);
    const float center_distance = length3(subtract(candidate_center, current_center));
    return center_distance <= scale * kBlasGroupCenterDistanceScale &&
        merged_diagonal <= scale * kBlasGroupMergedDiagonalScale;
}

void append_single_chunk_instance(std::size_t chunk_index) {
    scene_blas_instance instance{};
    instance.chunk_indices[0] = chunk_index;
    instance.chunk_count = 1;
    instance.uses_group_blas = false;
    g_dxr_backend.scene_blas_instances.push_back(instance);
}

void append_group_instance(
    const std::array<std::size_t, kBlasGroupChunkCount> &chunk_indices,
    std::size_t chunk_count)
{
    if (chunk_count == 0) {
        return;
    }
    if (chunk_count == 1) {
        append_single_chunk_instance(chunk_indices[0]);
        return;
    }

    scene_blas_instance instance{};
    instance.chunk_indices = chunk_indices;
    instance.chunk_count = chunk_count;
    instance.uses_group_blas = true;
    g_dxr_backend.scene_blas_instances.push_back(instance);
}

bool rebuild_scene_blas_instances(const rt_scene_build &build) {
    g_dxr_backend.scene_blas_instances.clear();
    const std::size_t group_chunk_limit = current_blas_group_chunk_limit();

    std::array<std::size_t, kBlasGroupChunkCount> pending_group_indices{};
    std::size_t pending_group_count = 0;
    scene_bounds pending_group_bounds{};

    auto flush_pending_group = [&]() {
        append_group_instance(pending_group_indices, pending_group_count);
        pending_group_indices = {};
        pending_group_count = 0;
        pending_group_bounds = {};
    };

    for (std::size_t chunk_index = 0; chunk_index < build.chunks.size(); ++chunk_index) {
        const rt_scene_chunk &chunk = build.chunks[chunk_index];
        if (!chunk.sealed) {
            flush_pending_group();
            append_single_chunk_instance(chunk_index);
            continue;
        }

        if (pending_group_count == 0) {
            pending_group_indices[0] = chunk_index;
            pending_group_count = 1;
            pending_group_bounds = chunk.bounds;
            continue;
        }

        const rt_scene_chunk &first_group_chunk = build.chunks[pending_group_indices[0]];
        if (pending_group_count < group_chunk_limit &&
            chunk.layer == first_group_chunk.layer &&
            can_append_chunk_to_group(pending_group_bounds, pending_group_count, chunk)) {
            pending_group_indices[pending_group_count] = chunk_index;
            ++pending_group_count;
            pending_group_bounds = merge_bounds(pending_group_bounds, chunk.bounds);
            continue;
        }

        flush_pending_group();
        pending_group_indices[0] = chunk_index;
        pending_group_count = 1;
        pending_group_bounds = chunk.bounds;
    }

    flush_pending_group();

    for (scene_blas_instance &instance : g_dxr_backend.scene_blas_instances) {
        if (instance.chunk_count == 0) {
            return false;
        }
        if (instance.uses_group_blas && instance.chunk_count < 2) {
            instance.uses_group_blas = false;
        }
    }
    return true;
}

bool sync_chunk_upload_cache(const rt_scene_build &build) {
    g_dxr_backend.last_chunk_sync_error = 0;
    g_dxr_backend.last_chunk_sync_error_index = kInvalidCacheIndex;
    if (!g_dxr_backend.initialized || g_dxr_backend.device == nullptr) {
        g_dxr_backend.last_chunk_sync_error = 1;
        return false;
    }
    if (build.revision == 0 || g_dxr_backend.synced_build_revision == build.revision) {
        return true;
    }

    const bool disable_upload_reuse = environment_flag_enabled("RTVDB_DXR_DISABLE_UPLOAD_REUSE");
    g_dxr_backend.scene_chunk_cache_indices.assign(build.chunks.size(), kInvalidCacheIndex);
    std::vector<bool> claimed_entries(g_dxr_backend.chunk_cache.size(), false);
    std::vector<pending_chunk_copy> pending_copies;

    for (std::size_t i = 0; i < build.chunks.size(); ++i) {
        const rt_scene_chunk &chunk = build.chunks[i];
        std::size_t cache_index = disable_upload_reuse
            ? kInvalidCacheIndex
            : find_reusable_chunk_cache_entry(chunk, claimed_entries);
        if (cache_index != kInvalidCacheIndex) {
            claimed_entries[cache_index] = true;
            g_dxr_backend.scene_chunk_cache_indices[i] = cache_index;
            continue;
        }

        cache_index = allocate_chunk_cache_entry(claimed_entries);
        if (cache_index >= claimed_entries.size()) {
            claimed_entries.resize(cache_index + 1u, false);
        }
        claimed_entries[cache_index] = true;
        g_dxr_backend.scene_chunk_cache_indices[i] = cache_index;

        gpu_chunk_cache_entry &cache_entry = g_dxr_backend.chunk_cache[cache_index];
        release_chunk_cache_entry(&cache_entry);

        const std::size_t vertex_bytes = chunk.vertex_count * sizeof(rt_scene_vertex);
        const std::size_t index_bytes = chunk.index_count * sizeof(std::uint32_t);
        std::vector<std::uint32_t> local_indices(chunk.index_count);
        for (std::size_t index = 0; index < chunk.index_count; ++index) {
            local_indices[index] =
                build.indices[chunk.index_offset + index] - static_cast<std::uint32_t>(chunk.vertex_offset);
        }
        if (!create_buffer(
                D3D12_HEAP_TYPE_UPLOAD,
                vertex_bytes,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                D3D12_RESOURCE_FLAG_NONE,
                &cache_entry.vertex_upload)) {
            g_dxr_backend.last_chunk_sync_error = 2;
            g_dxr_backend.last_chunk_sync_error_index = i;
            return false;
        }
        if (!create_buffer(
                D3D12_HEAP_TYPE_DEFAULT,
                vertex_bytes,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_FLAG_NONE,
                &cache_entry.vertex_buffer)) {
            release_chunk_cache_entry(&cache_entry);
            g_dxr_backend.last_chunk_sync_error = 2;
            g_dxr_backend.last_chunk_sync_error_index = i;
            return false;
        }
        if (!create_buffer(
                D3D12_HEAP_TYPE_UPLOAD,
                index_bytes,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                D3D12_RESOURCE_FLAG_NONE,
                &cache_entry.index_upload)) {
            release_chunk_cache_entry(&cache_entry);
            g_dxr_backend.last_chunk_sync_error = 3;
            g_dxr_backend.last_chunk_sync_error_index = i;
            return false;
        }
        if (!create_buffer(
                D3D12_HEAP_TYPE_DEFAULT,
                index_bytes,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_FLAG_NONE,
                &cache_entry.index_buffer)) {
            release_chunk_cache_entry(&cache_entry);
            g_dxr_backend.last_chunk_sync_error = 3;
            g_dxr_backend.last_chunk_sync_error_index = i;
            return false;
        }
        if (!upload_buffer_data(
                cache_entry.vertex_upload,
                build.vertices.data() + static_cast<std::ptrdiff_t>(chunk.vertex_offset),
                vertex_bytes)) {
            release_chunk_cache_entry(&cache_entry);
            g_dxr_backend.last_chunk_sync_error = 4;
            g_dxr_backend.last_chunk_sync_error_index = i;
            return false;
        }
        if (!upload_buffer_data(cache_entry.index_upload, local_indices.data(), index_bytes)) {
            release_chunk_cache_entry(&cache_entry);
            g_dxr_backend.last_chunk_sync_error = 5;
            g_dxr_backend.last_chunk_sync_error_index = i;
            return false;
        }

        cache_entry.first_triangle = chunk.first_triangle;
        cache_entry.triangle_count = chunk.triangle_count;
        cache_entry.fingerprint = chunk.fingerprint;
        cache_entry.vertex_count = chunk.vertex_count;
        cache_entry.index_count = chunk.index_count;
        pending_copies.push_back({cache_index, vertex_bytes, index_bytes});
    }

    if (!upload_pending_chunk_copies(pending_copies)) {
        g_dxr_backend.last_chunk_sync_error = 6;
        g_dxr_backend.last_chunk_sync_error_index = pending_copies.empty()
            ? kInvalidCacheIndex
            : pending_copies.front().cache_index;
        return false;
    }

    for (std::size_t i = 0; i < g_dxr_backend.chunk_cache.size(); ++i) {
        if (i < claimed_entries.size() && claimed_entries[i]) {
            continue;
        }
        release_chunk_cache_entry(&g_dxr_backend.chunk_cache[i]);
    }

    g_dxr_backend.synced_build_revision = build.revision;
    g_dxr_backend.synced_chunk_count = build.chunks.size();
    g_dxr_backend.synced_reused_chunk_count = build.reused_chunk_count;
    g_dxr_backend.synced_rebuilt_chunk_count = build.rebuilt_chunk_count;
    g_dxr_backend.synced_vertex_bytes = build.vertex_count * sizeof(rt_scene_vertex);
    g_dxr_backend.synced_index_bytes = build.index_count * sizeof(std::uint32_t);
    return true;
}

bool build_bottom_level_as(const rt_scene_chunk &chunk, gpu_chunk_cache_entry* cache_entry) {
    if (cache_entry == nullptr || cache_entry->vertex_buffer == nullptr || cache_entry->index_buffer == nullptr) {
        return false;
    }

    D3D12_RAYTRACING_GEOMETRY_DESC geometry_desc{};
    geometry_desc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
    geometry_desc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
    geometry_desc.Triangles.Transform3x4 = 0;
    geometry_desc.Triangles.IndexFormat = DXGI_FORMAT_R32_UINT;
    geometry_desc.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
    geometry_desc.Triangles.IndexCount = static_cast<UINT>(chunk.index_count);
    geometry_desc.Triangles.VertexCount = static_cast<UINT>(chunk.vertex_count);
    geometry_desc.Triangles.IndexBuffer = cache_entry->index_buffer->GetGPUVirtualAddress();
    geometry_desc.Triangles.VertexBuffer.StartAddress = cache_entry->vertex_buffer->GetGPUVirtualAddress();
    geometry_desc.Triangles.VertexBuffer.StrideInBytes = sizeof(rt_scene_vertex);

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs{};
    inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    inputs.NumDescs = 1;
    inputs.pGeometryDescs = &geometry_desc;
    inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuild_info{};
    const auto prebuild_start = std::chrono::steady_clock::now();
    g_dxr_backend.device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &prebuild_info);
    g_dxr_backend.last_accel_prebuild_info_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - prebuild_start).count();
    if (prebuild_info.ResultDataMaxSizeInBytes == 0) {
        return false;
    }

    safe_release(cache_entry->blas_scratch);
    safe_release(cache_entry->blas_result);
    const auto alloc_start = std::chrono::steady_clock::now();
    if (!create_buffer(
            D3D12_HEAP_TYPE_DEFAULT,
            static_cast<std::size_t>(prebuild_info.ScratchDataSizeInBytes),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            &cache_entry->blas_scratch)) {
        return false;
    }
    if (!create_buffer(
            D3D12_HEAP_TYPE_DEFAULT,
            static_cast<std::size_t>(prebuild_info.ResultDataMaxSizeInBytes),
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            &cache_entry->blas_result)) {
        safe_release(cache_entry->blas_scratch);
        return false;
    }
    g_dxr_backend.last_accel_resource_alloc_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - alloc_start).count();

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build_desc{};
    build_desc.Inputs = inputs;
    build_desc.ScratchAccelerationStructureData = cache_entry->blas_scratch->GetGPUVirtualAddress();
    build_desc.DestAccelerationStructureData = cache_entry->blas_result->GetGPUVirtualAddress();
    const auto build_record_start = std::chrono::steady_clock::now();
    g_dxr_backend.command_list->BuildRaytracingAccelerationStructure(&build_desc, 0, nullptr);

    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = cache_entry->blas_result;
    g_dxr_backend.command_list->ResourceBarrier(1, &barrier);
    g_dxr_backend.last_accel_build_call_record_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - build_record_start).count();
    return true;
}

bool build_group_bottom_level_as(
    const rt_scene_build &build,
    const scene_blas_instance &instance,
    gpu_group_cache_entry* cache_entry)
{
    if (cache_entry == nullptr || instance.chunk_count < 2 || instance.chunk_count > kBlasGroupChunkCount) {
        return false;
    }

    std::array<D3D12_RAYTRACING_GEOMETRY_DESC, kBlasGroupChunkCount> geometry_descs{};
    for (std::size_t i = 0; i < instance.chunk_count; ++i) {
        const std::size_t chunk_index = instance.chunk_indices[i];
        if (chunk_index >= build.chunks.size() || chunk_index >= g_dxr_backend.scene_chunk_cache_indices.size()) {
            return false;
        }
        const std::size_t chunk_cache_index = g_dxr_backend.scene_chunk_cache_indices[chunk_index];
        if (chunk_cache_index == kInvalidCacheIndex || chunk_cache_index >= g_dxr_backend.chunk_cache.size()) {
            return false;
        }
        const rt_scene_chunk &chunk = build.chunks[chunk_index];
        const gpu_chunk_cache_entry &chunk_cache_entry = g_dxr_backend.chunk_cache[chunk_cache_index];
        if (chunk_cache_entry.vertex_buffer == nullptr || chunk_cache_entry.index_buffer == nullptr) {
            return false;
        }

        D3D12_RAYTRACING_GEOMETRY_DESC &geometry_desc = geometry_descs[i];
        geometry_desc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
        geometry_desc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
        geometry_desc.Triangles.Transform3x4 = 0;
        geometry_desc.Triangles.IndexFormat = DXGI_FORMAT_R32_UINT;
        geometry_desc.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
        geometry_desc.Triangles.IndexCount = static_cast<UINT>(chunk.index_count);
        geometry_desc.Triangles.VertexCount = static_cast<UINT>(chunk.vertex_count);
        geometry_desc.Triangles.IndexBuffer = chunk_cache_entry.index_buffer->GetGPUVirtualAddress();
        geometry_desc.Triangles.VertexBuffer.StartAddress = chunk_cache_entry.vertex_buffer->GetGPUVirtualAddress();
        geometry_desc.Triangles.VertexBuffer.StrideInBytes = sizeof(rt_scene_vertex);
    }

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs{};
    inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    inputs.NumDescs = static_cast<UINT>(instance.chunk_count);
    inputs.pGeometryDescs = geometry_descs.data();
    inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuild_info{};
    const auto prebuild_start = std::chrono::steady_clock::now();
    g_dxr_backend.device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &prebuild_info);
    g_dxr_backend.last_accel_prebuild_info_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - prebuild_start).count();
    if (prebuild_info.ResultDataMaxSizeInBytes == 0) {
        return false;
    }

    const auto alloc_start = std::chrono::steady_clock::now();
    safe_release(cache_entry->blas_scratch);
    safe_release(cache_entry->blas_result);
    if (!create_buffer(
            D3D12_HEAP_TYPE_DEFAULT,
            static_cast<std::size_t>(prebuild_info.ScratchDataSizeInBytes),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            &cache_entry->blas_scratch)) {
        return false;
    }
    if (!create_buffer(
            D3D12_HEAP_TYPE_DEFAULT,
            static_cast<std::size_t>(prebuild_info.ResultDataMaxSizeInBytes),
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            &cache_entry->blas_result)) {
        safe_release(cache_entry->blas_scratch);
        return false;
    }
    g_dxr_backend.last_accel_resource_alloc_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - alloc_start).count();

    cache_entry->chunk_count = instance.chunk_count;
    cache_entry->vertex_count = 0;
    cache_entry->index_count = 0;
    for (std::size_t i = 0; i < instance.chunk_count; ++i) {
        const rt_scene_chunk &chunk = build.chunks[instance.chunk_indices[i]];
        cache_entry->first_triangle[i] = chunk.first_triangle;
        cache_entry->chunk_fingerprints[i] = chunk.fingerprint;
        cache_entry->vertex_count += chunk.vertex_count;
        cache_entry->index_count += chunk.index_count;
    }

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build_desc{};
    build_desc.Inputs = inputs;
    build_desc.ScratchAccelerationStructureData = cache_entry->blas_scratch->GetGPUVirtualAddress();
    build_desc.DestAccelerationStructureData = cache_entry->blas_result->GetGPUVirtualAddress();
    const auto build_record_start = std::chrono::steady_clock::now();
    g_dxr_backend.command_list->BuildRaytracingAccelerationStructure(&build_desc, 0, nullptr);

    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = cache_entry->blas_result;
    g_dxr_backend.command_list->ResourceBarrier(1, &barrier);
    g_dxr_backend.last_accel_build_call_record_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - build_record_start).count();
    return true;
}

bool build_point_bottom_level_as(
    const rt_scene_build &build,
    const rt_procedural_group &group,
    procedural_blas_entry* entry)
{
    if (entry == nullptr || group.primitive_count == 0 || g_dxr_backend.point_aabb_buffer == nullptr) {
        return false;
    }
    release_procedural_blas_entry(entry);

    D3D12_RAYTRACING_GEOMETRY_DESC geometry_desc{};
    geometry_desc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS;
    geometry_desc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
    geometry_desc.AABBs.AABBCount = static_cast<UINT>(group.primitive_count);
    geometry_desc.AABBs.AABBs.StartAddress =
        g_dxr_backend.point_aabb_buffer->GetGPUVirtualAddress() +
        static_cast<D3D12_GPU_VIRTUAL_ADDRESS>(group.first_primitive * sizeof(D3D12_RAYTRACING_AABB));
    geometry_desc.AABBs.AABBs.StrideInBytes = sizeof(D3D12_RAYTRACING_AABB);

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs{};
    inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    inputs.NumDescs = 1;
    inputs.pGeometryDescs = &geometry_desc;
    inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuild_info{};
    const auto prebuild_start = std::chrono::steady_clock::now();
    g_dxr_backend.device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &prebuild_info);
    g_dxr_backend.last_accel_prebuild_info_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - prebuild_start).count();
    if (prebuild_info.ResultDataMaxSizeInBytes == 0) {
        return false;
    }

    const auto alloc_start = std::chrono::steady_clock::now();
    if (!create_buffer(
            D3D12_HEAP_TYPE_DEFAULT,
            static_cast<std::size_t>(prebuild_info.ScratchDataSizeInBytes),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            &entry->blas_scratch)) {
        return false;
    }
    if (!create_buffer(
            D3D12_HEAP_TYPE_DEFAULT,
            static_cast<std::size_t>(prebuild_info.ResultDataMaxSizeInBytes),
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            &entry->blas_result)) {
        safe_release(entry->blas_scratch);
        return false;
    }
    g_dxr_backend.last_accel_resource_alloc_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - alloc_start).count();

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build_desc{};
    build_desc.Inputs = inputs;
    build_desc.ScratchAccelerationStructureData = entry->blas_scratch->GetGPUVirtualAddress();
    build_desc.DestAccelerationStructureData = entry->blas_result->GetGPUVirtualAddress();
    const auto build_record_start = std::chrono::steady_clock::now();
    g_dxr_backend.command_list->BuildRaytracingAccelerationStructure(&build_desc, 0, nullptr);

    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = entry->blas_result;
    g_dxr_backend.command_list->ResourceBarrier(1, &barrier);
    g_dxr_backend.last_accel_build_call_record_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - build_record_start).count();
    entry->first_primitive = group.first_primitive;
    entry->primitive_count = group.primitive_count;
    return true;
}

std::uint64_t procedural_group_fingerprint(
    const rt_scene_build &build,
    const rt_procedural_group &group,
    bool points)
{
    std::uint64_t hash = 1469598103934665603ull;
    const auto append_bytes = [&hash](const void* data, std::size_t size) {
        const auto* bytes = static_cast<const std::uint8_t*>(data);
        for (std::size_t i = 0; i < size; ++i) {
            hash ^= bytes[i];
            hash *= 1099511628211ull;
        }
    };
    for (std::size_t i = 0; i < group.primitive_count; ++i) {
        if (points) {
            const point &value = build.points[group.first_primitive + i];
            append_bytes(&value.position, sizeof(value.position));
            append_bytes(&value.radius, sizeof(value.radius));
        } else {
            const line &value = build.lines[group.first_primitive + i];
            append_bytes(&value.a, sizeof(value.a));
            append_bytes(&value.radius, sizeof(value.radius));
            append_bytes(&value.b, sizeof(value.b));
        }
    }
    return hash;
}

std::uint64_t point_geometry_fingerprint(const rt_scene_build &build) {
    std::uint64_t hash = 1469598103934665603ull;
    const auto append_bytes = [&hash](const void* data, std::size_t size) {
        const auto* bytes = static_cast<const std::uint8_t*>(data);
        for (std::size_t i = 0; i < size; ++i) {
            hash ^= bytes[i];
            hash *= 1099511628211ull;
        }
    };
    append_bytes(&build.point_count, sizeof(build.point_count));
    for (const point &value : build.points) {
        append_bytes(&value.position, sizeof(value.position));
        append_bytes(&value.radius, sizeof(value.radius));
    }
    return hash;
}

std::uint64_t line_geometry_fingerprint(const rt_scene_build &build) {
    std::uint64_t hash = 1469598103934665603ull;
    const auto append_bytes = [&hash](const void* data, std::size_t size) {
        const auto* bytes = static_cast<const std::uint8_t*>(data);
        for (std::size_t i = 0; i < size; ++i) {
            hash ^= bytes[i];
            hash *= 1099511628211ull;
        }
    };
    append_bytes(&build.line_count, sizeof(build.line_count));
    for (const line &value : build.lines) {
        append_bytes(&value.a, sizeof(value.a));
        append_bytes(&value.radius, sizeof(value.radius));
        append_bytes(&value.b, sizeof(value.b));
    }
    return hash;
}

bool sync_procedural_aabb_buffers(const rt_scene_build &build) {
    const std::uint64_t next_point_fingerprint = point_geometry_fingerprint(build);
    const bool point_geometry_changed = g_dxr_backend.point_primitive_count != build.point_count ||
        g_dxr_backend.point_geometry_fingerprint != next_point_fingerprint;
    if (build.point_count == 0) {
        safe_release(g_dxr_backend.point_aabb_buffer);
        g_dxr_backend.point_geometry_fingerprint = 0;
        g_dxr_backend.point_primitive_count = 0;
    } else if (point_geometry_changed) {
        std::vector<D3D12_RAYTRACING_AABB> aabbs(build.point_count);
        for (std::size_t point_index = 0; point_index < build.point_count; ++point_index) {
            const point &value = build.points[point_index];
            D3D12_RAYTRACING_AABB &aabb = aabbs[point_index];
            aabb.MinX = value.position.x - value.radius;
            aabb.MinY = value.position.y - value.radius;
            aabb.MinZ = value.position.z - value.radius;
            aabb.MaxX = value.position.x + value.radius;
            aabb.MaxY = value.position.y + value.radius;
            aabb.MaxZ = value.position.z + value.radius;
        }
        safe_release(g_dxr_backend.point_aabb_buffer);
        if (!create_buffer(
                D3D12_HEAP_TYPE_UPLOAD,
                aabbs.size() * sizeof(D3D12_RAYTRACING_AABB),
                D3D12_RESOURCE_STATE_GENERIC_READ,
                D3D12_RESOURCE_FLAG_NONE,
                &g_dxr_backend.point_aabb_buffer)) {
            return false;
        }
        if (!upload_buffer_data(
                g_dxr_backend.point_aabb_buffer,
                aabbs.data(),
                aabbs.size() * sizeof(D3D12_RAYTRACING_AABB))) {
            return false;
        }
        g_dxr_backend.point_geometry_fingerprint = next_point_fingerprint;
        g_dxr_backend.point_primitive_count = build.point_count;
    }

    const std::uint64_t next_line_fingerprint = line_geometry_fingerprint(build);
    const bool line_geometry_changed = g_dxr_backend.line_primitive_count != build.line_count ||
        g_dxr_backend.line_geometry_fingerprint != next_line_fingerprint;
    if (build.line_count == 0) {
        safe_release(g_dxr_backend.line_aabb_buffer);
        g_dxr_backend.line_geometry_fingerprint = 0;
        g_dxr_backend.line_primitive_count = 0;
    } else if (line_geometry_changed) {
        std::vector<D3D12_RAYTRACING_AABB> aabbs(build.line_count);
        for (std::size_t line_index = 0; line_index < build.line_count; ++line_index) {
            const line &value = build.lines[line_index];
            D3D12_RAYTRACING_AABB &aabb = aabbs[line_index];
            aabb.MinX = (std::min)(value.a.x, value.b.x) - value.radius;
            aabb.MinY = (std::min)(value.a.y, value.b.y) - value.radius;
            aabb.MinZ = (std::min)(value.a.z, value.b.z) - value.radius;
            aabb.MaxX = (std::max)(value.a.x, value.b.x) + value.radius;
            aabb.MaxY = (std::max)(value.a.y, value.b.y) + value.radius;
            aabb.MaxZ = (std::max)(value.a.z, value.b.z) + value.radius;
        }
        safe_release(g_dxr_backend.line_aabb_buffer);
        if (!create_buffer(
                D3D12_HEAP_TYPE_UPLOAD,
                aabbs.size() * sizeof(D3D12_RAYTRACING_AABB),
                D3D12_RESOURCE_STATE_GENERIC_READ,
                D3D12_RESOURCE_FLAG_NONE,
                &g_dxr_backend.line_aabb_buffer)) {
            return false;
        }
        if (!upload_buffer_data(
                g_dxr_backend.line_aabb_buffer,
                aabbs.data(),
                aabbs.size() * sizeof(D3D12_RAYTRACING_AABB))) {
            return false;
        }
        g_dxr_backend.line_geometry_fingerprint = next_line_fingerprint;
        g_dxr_backend.line_primitive_count = build.line_count;
    }
    return true;
}

bool build_line_bottom_level_as(
    const rt_scene_build &build,
    const rt_procedural_group &group,
    procedural_blas_entry* entry)
{
    if (entry == nullptr || group.primitive_count == 0 || g_dxr_backend.line_aabb_buffer == nullptr) {
        return false;
    }
    release_procedural_blas_entry(entry);

    D3D12_RAYTRACING_GEOMETRY_DESC geometry_desc{};
    geometry_desc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS;
    geometry_desc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
    geometry_desc.AABBs.AABBCount = static_cast<UINT>(group.primitive_count);
    geometry_desc.AABBs.AABBs.StartAddress =
        g_dxr_backend.line_aabb_buffer->GetGPUVirtualAddress() +
        static_cast<D3D12_GPU_VIRTUAL_ADDRESS>(group.first_primitive * sizeof(D3D12_RAYTRACING_AABB));
    geometry_desc.AABBs.AABBs.StrideInBytes = sizeof(D3D12_RAYTRACING_AABB);

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs{};
    inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    inputs.NumDescs = 1;
    inputs.pGeometryDescs = &geometry_desc;
    inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuild_info{};
    const auto prebuild_start = std::chrono::steady_clock::now();
    g_dxr_backend.device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &prebuild_info);
    g_dxr_backend.last_accel_prebuild_info_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - prebuild_start).count();
    if (prebuild_info.ResultDataMaxSizeInBytes == 0) {
        return false;
    }

    const auto alloc_start = std::chrono::steady_clock::now();
    if (!create_buffer(
            D3D12_HEAP_TYPE_DEFAULT,
            static_cast<std::size_t>(prebuild_info.ScratchDataSizeInBytes),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            &entry->blas_scratch)) {
        return false;
    }
    if (!create_buffer(
            D3D12_HEAP_TYPE_DEFAULT,
            static_cast<std::size_t>(prebuild_info.ResultDataMaxSizeInBytes),
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            &entry->blas_result)) {
        safe_release(entry->blas_scratch);
        return false;
    }
    g_dxr_backend.last_accel_resource_alloc_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - alloc_start).count();

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build_desc{};
    build_desc.Inputs = inputs;
    build_desc.ScratchAccelerationStructureData = entry->blas_scratch->GetGPUVirtualAddress();
    build_desc.DestAccelerationStructureData = entry->blas_result->GetGPUVirtualAddress();
    const auto build_record_start = std::chrono::steady_clock::now();
    g_dxr_backend.command_list->BuildRaytracingAccelerationStructure(&build_desc, 0, nullptr);

    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = entry->blas_result;
    g_dxr_backend.command_list->ResourceBarrier(1, &barrier);
    g_dxr_backend.last_accel_build_call_record_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - build_record_start).count();
    entry->first_primitive = group.first_primitive;
    entry->primitive_count = group.primitive_count;
    return true;
}

bool build_top_level_as(const rt_scene_build &build) {
    const std::size_t instance_count =
        g_dxr_backend.scene_blas_instances.size() +
        build.point_groups.size() +
        build.line_groups.size();
    if (instance_count == 0) {
        safe_release(g_dxr_backend.tlas_instance_upload);
        safe_release(g_dxr_backend.tlas_scratch);
        safe_release(g_dxr_backend.tlas_result);
        g_dxr_backend.tlas_instance_count = 0;
        return true;
    }

    std::vector<D3D12_RAYTRACING_INSTANCE_DESC> instances(instance_count);
    std::size_t write_index = 0;
    for (std::size_t i = 0; i < g_dxr_backend.scene_blas_instances.size(); ++i) {
        const scene_blas_instance &instance = g_dxr_backend.scene_blas_instances[i];
        if (instance.chunk_count == 0 || instance.cache_index == kInvalidCacheIndex) {
            return false;
        }
        D3D12_GPU_VIRTUAL_ADDRESS accel_address = 0;
        if (instance.uses_group_blas) {
            if (instance.cache_index >= g_dxr_backend.group_cache.size()) {
                return false;
            }
            const gpu_group_cache_entry &cache_entry = g_dxr_backend.group_cache[instance.cache_index];
            if (cache_entry.blas_result == nullptr) {
                return false;
            }
            accel_address = cache_entry.blas_result->GetGPUVirtualAddress();
        } else {
            if (instance.cache_index >= g_dxr_backend.chunk_cache.size()) {
                return false;
            }
            const gpu_chunk_cache_entry &cache_entry = g_dxr_backend.chunk_cache[instance.cache_index];
            if (cache_entry.blas_result == nullptr) {
                return false;
            }
            accel_address = cache_entry.blas_result->GetGPUVirtualAddress();
        }

        D3D12_RAYTRACING_INSTANCE_DESC desc{};
        desc.Transform[0][0] = 1.0f;
        desc.Transform[1][1] = 1.0f;
        desc.Transform[2][2] = 1.0f;
        desc.InstanceID = static_cast<UINT>(write_index * kBlasGroupChunkCount);
        desc.InstanceContributionToHitGroupIndex = kTriangleHitGroupContributionIndex;
        const rt_scene_chunk &first_chunk = build.chunks[instance.chunk_indices[0]];
        desc.InstanceMask = first_chunk.visible ? 0xFF : 0x00;
        desc.Flags = D3D12_RAYTRACING_INSTANCE_FLAG_NONE;
        desc.AccelerationStructure = accel_address;
        instances[write_index++] = desc;
    }

    for (std::size_t group_index = 0; group_index < build.point_groups.size(); ++group_index) {
        const rt_procedural_group &group = build.point_groups[group_index];
        if (group_index >= g_dxr_backend.point_blas_entries.size() ||
            g_dxr_backend.point_blas_entries[group_index].blas_result == nullptr) {
            return false;
        }
        D3D12_RAYTRACING_INSTANCE_DESC desc{};
        desc.Transform[0][0] = 1.0f;
        desc.Transform[1][1] = 1.0f;
        desc.Transform[2][2] = 1.0f;
        desc.InstanceID = static_cast<UINT>(write_index * kBlasGroupChunkCount);
        desc.InstanceContributionToHitGroupIndex = kPointHitGroupContributionIndex;
        desc.InstanceMask = group.visible ? 0xFF : 0x00;
        desc.Flags = D3D12_RAYTRACING_INSTANCE_FLAG_NONE;
        desc.AccelerationStructure = g_dxr_backend.point_blas_entries[group_index].blas_result->GetGPUVirtualAddress();
        instances[write_index++] = desc;
    }

    for (std::size_t group_index = 0; group_index < build.line_groups.size(); ++group_index) {
        const rt_procedural_group &group = build.line_groups[group_index];
        if (group_index >= g_dxr_backend.line_blas_entries.size() ||
            g_dxr_backend.line_blas_entries[group_index].blas_result == nullptr) {
            return false;
        }
        D3D12_RAYTRACING_INSTANCE_DESC desc{};
        desc.Transform[0][0] = 1.0f;
        desc.Transform[1][1] = 1.0f;
        desc.Transform[2][2] = 1.0f;
        desc.InstanceID = static_cast<UINT>(write_index * kBlasGroupChunkCount);
        desc.InstanceContributionToHitGroupIndex = kLineHitGroupContributionIndex;
        desc.InstanceMask = group.visible ? 0xFF : 0x00;
        desc.Flags = D3D12_RAYTRACING_INSTANCE_FLAG_NONE;
        desc.AccelerationStructure = g_dxr_backend.line_blas_entries[group_index].blas_result->GetGPUVirtualAddress();
        instances[write_index++] = desc;
    }

    const auto alloc_start = std::chrono::steady_clock::now();
    safe_release(g_dxr_backend.tlas_instance_upload);
    if (!create_buffer(
            D3D12_HEAP_TYPE_UPLOAD,
            instances.size() * sizeof(D3D12_RAYTRACING_INSTANCE_DESC),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            D3D12_RESOURCE_FLAG_NONE,
            &g_dxr_backend.tlas_instance_upload)) {
        return false;
    }
    const auto tlas_upload_start = std::chrono::steady_clock::now();
    if (!upload_buffer_data(
            g_dxr_backend.tlas_instance_upload,
            instances.data(),
            instances.size() * sizeof(D3D12_RAYTRACING_INSTANCE_DESC))) {
        return false;
    }
    g_dxr_backend.last_accel_tlas_instance_upload_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tlas_upload_start).count();

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs{};
    inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    inputs.NumDescs = static_cast<UINT>(instance_count);
    inputs.InstanceDescs = g_dxr_backend.tlas_instance_upload->GetGPUVirtualAddress();
    inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE |
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuild_info{};
    const auto prebuild_start = std::chrono::steady_clock::now();
    g_dxr_backend.device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &prebuild_info);
    g_dxr_backend.last_accel_prebuild_info_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - prebuild_start).count();
    if (prebuild_info.ResultDataMaxSizeInBytes == 0) {
        return false;
    }

    const bool can_update = g_dxr_backend.tlas_result != nullptr &&
        g_dxr_backend.tlas_instance_count == instance_count &&
        g_dxr_backend.tlas_result->GetDesc().Width >= prebuild_info.ResultDataMaxSizeInBytes;
    const UINT64 required_scratch_size = can_update
        ? prebuild_info.UpdateScratchDataSizeInBytes
        : prebuild_info.ScratchDataSizeInBytes;
    if (g_dxr_backend.tlas_scratch == nullptr ||
        g_dxr_backend.tlas_scratch->GetDesc().Width < required_scratch_size) {
        safe_release(g_dxr_backend.tlas_scratch);
    }
    if (g_dxr_backend.tlas_scratch == nullptr && !create_buffer(
            D3D12_HEAP_TYPE_DEFAULT,
            static_cast<std::size_t>(required_scratch_size),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            &g_dxr_backend.tlas_scratch)) {
        return false;
    }
    if (!can_update) {
        safe_release(g_dxr_backend.tlas_result);
    }
    if (g_dxr_backend.tlas_result == nullptr && !create_buffer(
            D3D12_HEAP_TYPE_DEFAULT,
            static_cast<std::size_t>(prebuild_info.ResultDataMaxSizeInBytes),
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            &g_dxr_backend.tlas_result)) {
        safe_release(g_dxr_backend.tlas_scratch);
        return false;
    }
    g_dxr_backend.last_accel_resource_alloc_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - alloc_start).count();

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build_desc{};
    build_desc.Inputs = inputs;
    if (can_update) {
        build_desc.Inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
        build_desc.SourceAccelerationStructureData = g_dxr_backend.tlas_result->GetGPUVirtualAddress();
    }
    build_desc.ScratchAccelerationStructureData = g_dxr_backend.tlas_scratch->GetGPUVirtualAddress();
    build_desc.DestAccelerationStructureData = g_dxr_backend.tlas_result->GetGPUVirtualAddress();
    const auto build_record_start = std::chrono::steady_clock::now();
    g_dxr_backend.command_list->BuildRaytracingAccelerationStructure(&build_desc, 0, nullptr);

    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = g_dxr_backend.tlas_result;
    g_dxr_backend.command_list->ResourceBarrier(1, &barrier);
    g_dxr_backend.last_accel_build_call_record_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - build_record_start).count();
    g_dxr_backend.tlas_instance_count = instance_count;
    return true;
}

bool sync_acceleration_structures(const rt_scene_build &build) {
    if (!g_dxr_backend.raytracing_supported) {
        g_dxr_backend.synced_blas_reused_count = 0;
        g_dxr_backend.synced_blas_rebuilt_count = 0;
        g_dxr_backend.synced_blas_reused_chunk_count = 0;
        g_dxr_backend.synced_blas_rebuilt_chunk_count = 0;
        g_dxr_backend.synced_tlas_rebuild_count = 0;
        g_dxr_backend.last_accel_submit_cpu_ms = 0.0;
        g_dxr_backend.last_accel_gpu_wait_ms = 0.0;
        g_dxr_backend.last_accel_gpu_ms = 0.0;
        return true;
    }
    if (build.revision == 0 || g_dxr_backend.synced_accel_revision == build.revision) {
        return true;
    }
    const auto total_start = std::chrono::steady_clock::now();
    auto stage_start = total_start;
    g_dxr_backend.last_accel_host_prep_ms = 0.0;
    g_dxr_backend.last_accel_instance_build_ms = 0.0;
    g_dxr_backend.last_accel_procedural_aabb_ms = 0.0;
    g_dxr_backend.last_accel_command_record_ms = 0.0;
    g_dxr_backend.last_accel_resource_alloc_ms = 0.0;
    g_dxr_backend.last_accel_build_call_record_ms = 0.0;
    g_dxr_backend.last_accel_prebuild_info_ms = 0.0;
    g_dxr_backend.last_accel_tlas_instance_upload_ms = 0.0;
    if (!sync_chunk_upload_cache(build)) {
        return false;
    }
    const auto after_chunk_upload = std::chrono::steady_clock::now();
    g_dxr_backend.last_accel_host_prep_ms =
        std::chrono::duration<double, std::milli>(after_chunk_upload - stage_start).count();
    stage_start = after_chunk_upload;
    if (!rebuild_scene_blas_instances(build)) {
        return false;
    }
    const auto after_instance_build = std::chrono::steady_clock::now();
    g_dxr_backend.last_accel_instance_build_ms =
        std::chrono::duration<double, std::milli>(after_instance_build - stage_start).count();
    stage_start = after_instance_build;
    if (!sync_procedural_aabb_buffers(build)) {
        return false;
    }
    const auto after_procedural_aabb = std::chrono::steady_clock::now();
    g_dxr_backend.last_accel_procedural_aabb_ms =
        std::chrono::duration<double, std::milli>(after_procedural_aabb - stage_start).count();
    stage_start = after_procedural_aabb;

    if (!reset_command_list()) {
        return false;
    }
    if (!ensure_timestamp_query_resources()) {
        return false;
    }
    const UINT timestamp_slot = current_timestamp_query_slot();
    write_timestamp_query_begin(timestamp_slot);

    const bool disable_blas_reuse = environment_flag_enabled("RTVDB_DXR_DISABLE_BLAS_REUSE");
    std::vector<bool> claimed_group_entries(g_dxr_backend.group_cache.size(), false);
    std::size_t blas_reused_count = 0;
    std::size_t blas_rebuilt_count = 0;
    std::size_t blas_reused_chunk_count = 0;
    std::size_t blas_rebuilt_chunk_count = 0;

    for (scene_blas_instance &instance : g_dxr_backend.scene_blas_instances) {
        if (instance.chunk_count == 0) {
            return false;
        }
        if (instance.uses_group_blas) {
            std::size_t cache_index = disable_blas_reuse
                ? kInvalidCacheIndex
                : find_reusable_group_cache_entry(instance, build, claimed_group_entries);
            if (cache_index == kInvalidCacheIndex) {
                cache_index = allocate_group_cache_entry(claimed_group_entries);
            }
            if (cache_index >= claimed_group_entries.size()) {
                claimed_group_entries.resize(cache_index + 1u, false);
            }
            claimed_group_entries[cache_index] = true;
            instance.cache_index = cache_index;
            gpu_group_cache_entry &cache_entry = g_dxr_backend.group_cache[cache_index];
            const bool blas_reusable =
                !disable_blas_reuse && cache_entry_matches_group_blas(cache_entry, instance, build);
            if (blas_reusable) {
                ++blas_reused_count;
                blas_reused_chunk_count += instance.chunk_count;
                continue;
            }
            if (!build_group_bottom_level_as(build, instance, &cache_entry)) {
                return false;
            }
            ++blas_rebuilt_count;
            blas_rebuilt_chunk_count += instance.chunk_count;
            continue;
        }

        const std::size_t chunk_index = instance.chunk_indices[0];
        if (chunk_index >= build.chunks.size() || chunk_index >= g_dxr_backend.scene_chunk_cache_indices.size()) {
            return false;
        }
        const std::size_t cache_index = g_dxr_backend.scene_chunk_cache_indices[chunk_index];
        if (cache_index == kInvalidCacheIndex || cache_index >= g_dxr_backend.chunk_cache.size()) {
            return false;
        }
        instance.cache_index = cache_index;
        const rt_scene_chunk &chunk = build.chunks[chunk_index];
        gpu_chunk_cache_entry &cache_entry = g_dxr_backend.chunk_cache[cache_index];
        const bool blas_reusable = !disable_blas_reuse && cache_entry_matches_chunk_blas(cache_entry, chunk);
        if (blas_reusable) {
            ++blas_reused_count;
            ++blas_reused_chunk_count;
            continue;
        }
        if (!build_bottom_level_as(chunk, &cache_entry)) {
            return false;
        }
        ++blas_rebuilt_count;
        ++blas_rebuilt_chunk_count;
    }

    for (std::size_t i = 0; i < g_dxr_backend.group_cache.size(); ++i) {
        if (i < claimed_group_entries.size() && claimed_group_entries[i]) {
            continue;
        }
        release_group_cache_entry(&g_dxr_backend.group_cache[i]);
    }

    const auto sync_procedural_groups = [&](const std::vector<rt_procedural_group> &groups,
                                             bool points,
                                             std::vector<procedural_blas_entry>* entries) {
        while (entries->size() > groups.size()) {
            release_procedural_blas_entry(&entries->back());
            entries->pop_back();
        }
        entries->resize(groups.size());
        for (std::size_t i = 0; i < groups.size(); ++i) {
            const rt_procedural_group &group = groups[i];
            procedural_blas_entry &entry = (*entries)[i];
            const std::uint64_t fingerprint = procedural_group_fingerprint(build, group, points);
            if (entry.blas_result != nullptr && entry.first_primitive == group.first_primitive &&
                entry.primitive_count == group.primitive_count && entry.geometry_fingerprint == fingerprint) {
                ++blas_reused_count;
                continue;
            }
            const bool built = points
                ? build_point_bottom_level_as(build, group, &entry)
                : build_line_bottom_level_as(build, group, &entry);
            if (!built) {
                return false;
            }
            entry.geometry_fingerprint = fingerprint;
            ++blas_rebuilt_count;
        }
        return true;
    };
    if (!sync_procedural_groups(build.point_groups, true, &g_dxr_backend.point_blas_entries) ||
        !sync_procedural_groups(build.line_groups, false, &g_dxr_backend.line_blas_entries)) {
        return false;
    }
    if (!build_top_level_as(build)) {
        return false;
    }
    write_timestamp_query_end_and_resolve(timestamp_slot);
    const auto after_command_record = std::chrono::steady_clock::now();
    g_dxr_backend.last_accel_command_record_ms =
        std::chrono::duration<double, std::milli>(after_command_record - stage_start).count();

    if (!close_and_execute_command_list(&g_dxr_backend.last_accel_submit_cpu_ms)) {
        return false;
    }
    if (!wait_for_fence_value_timed(g_dxr_backend.submitted_fence_value, &g_dxr_backend.last_accel_gpu_wait_ms)) {
        return false;
    }
    if (!read_timestamp_query_ms(timestamp_slot, &g_dxr_backend.last_accel_gpu_ms)) {
        g_dxr_backend.last_accel_gpu_ms = 0.0;
    }
    g_dxr_backend.synced_blas_reused_count = blas_reused_count;
    g_dxr_backend.synced_blas_rebuilt_count = blas_rebuilt_count;
    g_dxr_backend.synced_blas_reused_chunk_count = blas_reused_chunk_count;
    g_dxr_backend.synced_blas_rebuilt_chunk_count = blas_rebuilt_chunk_count;
    g_dxr_backend.synced_tlas_rebuild_count =
        (build.chunks.empty() && build.point_count == 0 && build.line_count == 0) ? 0 : 1;
    g_dxr_backend.synced_accel_revision = build.revision;
    g_dxr_backend.last_accel_build_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - total_start).count();
    return true;
}
void destroy_swapchain_resources() {
    for (ID3D12Resource* &target : g_dxr_backend.render_targets) {
        safe_release(target);
    }
    safe_release(g_dxr_backend.swapchain);
    safe_release(g_dxr_backend.rtv_heap);
    g_dxr_backend.frame_index = 0;
    g_dxr_backend.width = 0;
    g_dxr_backend.height = 0;
    g_dxr_backend.hwnd = nullptr;
    reset_accumulation_state();
}

bool create_swapchain_resources(HWND hwnd) {
    RECT rect{};
    if (!GetClientRect(hwnd, &rect)) {
        return false;
    }

    const UINT width = rect.right > rect.left
        ? static_cast<UINT>(rect.right - rect.left)
        : static_cast<UINT>(g_dxr_backend.config.capture_width);
    const UINT height = rect.bottom > rect.top
        ? static_cast<UINT>(rect.bottom - rect.top)
        : static_cast<UINT>(g_dxr_backend.config.capture_height);
    if (width == 0 || height == 0) {
        return false;
    }

    DXGI_SWAP_CHAIN_DESC1 swapchain_desc{};
    swapchain_desc.BufferCount = kFrameCount;
    swapchain_desc.Width = width;
    swapchain_desc.Height = height;
    swapchain_desc.Format = kOutputColorFormat;
    swapchain_desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapchain_desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    swapchain_desc.SampleDesc.Count = 1;

    IDXGISwapChain1* swapchain1 = nullptr;
    HRESULT hr = g_dxr_backend.factory->CreateSwapChainForHwnd(
        g_dxr_backend.queue,
        hwnd,
        &swapchain_desc,
        nullptr,
        nullptr,
        &swapchain1);
    if (FAILED(hr)) {
        record_dxr_failure("create_swapchain_resources.CreateSwapChainForHwnd", hr);
        return false;
    }

    hr = swapchain1->QueryInterface(IID_PPV_ARGS(&g_dxr_backend.swapchain));
    swapchain1->Release();
    if (FAILED(hr)) {
        record_dxr_failure("create_swapchain_resources.QueryInterface", hr);
        return false;
    }

    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
    heap_desc.NumDescriptors = kFrameCount;
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    hr = g_dxr_backend.device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&g_dxr_backend.rtv_heap));
    if (FAILED(hr)) {
        record_dxr_failure("create_swapchain_resources.CreateDescriptorHeap", hr);
        return false;
    }

    g_dxr_backend.rtv_descriptor_size =
        g_dxr_backend.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    D3D12_CPU_DESCRIPTOR_HANDLE handle = g_dxr_backend.rtv_heap->GetCPUDescriptorHandleForHeapStart();
    for (UINT i = 0; i < kFrameCount; ++i) {
        hr = g_dxr_backend.swapchain->GetBuffer(i, IID_PPV_ARGS(&g_dxr_backend.render_targets[i]));
        if (FAILED(hr)) {
            record_dxr_failure("create_swapchain_resources.GetBuffer", hr);
            return false;
        }
        g_dxr_backend.device->CreateRenderTargetView(g_dxr_backend.render_targets[i], nullptr, handle);
        handle.ptr += g_dxr_backend.rtv_descriptor_size;
    }

    g_dxr_backend.frame_index = g_dxr_backend.swapchain->GetCurrentBackBufferIndex();
    g_dxr_backend.width = width;
    g_dxr_backend.height = height;
    g_dxr_backend.hwnd = hwnd;
    return true;
}

bool ensure_swapchain(HWND hwnd) {
    if (g_dxr_backend.swapchain != nullptr && g_dxr_backend.hwnd == hwnd) {
        RECT rect{};
        if (GetClientRect(hwnd, &rect)) {
            const UINT width = rect.right > rect.left ? static_cast<UINT>(rect.right - rect.left) : 0;
            const UINT height = rect.bottom > rect.top ? static_cast<UINT>(rect.bottom - rect.top) : 0;
            if (width == g_dxr_backend.width && height == g_dxr_backend.height) {
                return true;
            }
        }
        wait_for_gpu();
        destroy_swapchain_resources();
    }

    return create_swapchain_resources(hwnd);
}

void release_shader_compiler_state() {
    g_dxr_backend.shader_library.clear();
    g_dxr_backend.shader_library_ready = false;
}

bool load_shader_library() {
    if (!g_dxr_backend.raytracing_supported) {
        return false;
    }
    if (g_dxr_backend.shader_library_ready) {
        return true;
    }
    g_dxr_backend.shader_library.assign(
        kD3d12RaytracingDxil,
        kD3d12RaytracingDxil + kD3d12RaytracingDxilSize);
    g_dxr_backend.shader_library_ready = !g_dxr_backend.shader_library.empty();
    return g_dxr_backend.shader_library_ready;
}
void release_raytracing_runtime_state() {
    safe_release(g_dxr_backend.raygen_shader_table);
    safe_release(g_dxr_backend.pick_raygen_shader_table);
    safe_release(g_dxr_backend.miss_shader_table);
    safe_release(g_dxr_backend.hitgroup_shader_table);
    if (g_dxr_backend.camera_constant_buffer != nullptr) {
        g_dxr_backend.camera_constant_buffer->Unmap(0, nullptr);
        g_dxr_backend.camera_constant_buffer_mapped = nullptr;
    }
    safe_release(g_dxr_backend.camera_constant_buffer);
    safe_release(g_dxr_backend.scene_index_buffer);
    safe_release(g_dxr_backend.scene_position_buffer);
    safe_release(g_dxr_backend.instance_metadata_buffer);
    safe_release(g_dxr_backend.triangle_color_buffer);
    safe_release(g_dxr_backend.point_buffer);
    safe_release(g_dxr_backend.line_buffer);
    safe_release(g_dxr_backend.point_aabb_buffer);
    safe_release(g_dxr_backend.line_aabb_buffer);
    release_procedural_blas_entries(&g_dxr_backend.point_blas_entries);
    release_procedural_blas_entries(&g_dxr_backend.line_blas_entries);
    safe_release(g_dxr_backend.pick_output_buffer);
    safe_release(g_dxr_backend.pick_readback_buffer);
    safe_release(g_dxr_backend.output_readback_buffer);
    g_dxr_backend.output_readback_size = 0;
    g_dxr_backend.output_readback_footprint = {};
    safe_release(g_dxr_backend.output_texture);
    safe_release(g_dxr_backend.accumulation_texture);
    safe_release(g_dxr_backend.raytracing_state_props);
    safe_release(g_dxr_backend.raytracing_state_object);
    safe_release(g_dxr_backend.global_root_signature);
    safe_release(g_dxr_backend.srv_uav_cbv_heap);
    g_dxr_backend.last_accel_gpu_ms = 0.0;
    g_dxr_backend.last_dispatch_gpu_ms = 0.0;
    g_dxr_backend.point_geometry_fingerprint = 0;
    g_dxr_backend.line_geometry_fingerprint = 0;
    g_dxr_backend.point_primitive_count = 0;
    g_dxr_backend.line_primitive_count = 0;
    reset_accumulation_state();
}

D3D12_CPU_DESCRIPTOR_HANDLE descriptor_cpu_handle(UINT index) {
    D3D12_CPU_DESCRIPTOR_HANDLE handle = g_dxr_backend.srv_uav_cbv_heap->GetCPUDescriptorHandleForHeapStart();
    handle.ptr += static_cast<SIZE_T>(index) * g_dxr_backend.srv_uav_cbv_descriptor_size;
    return handle;
}

D3D12_GPU_DESCRIPTOR_HANDLE descriptor_gpu_handle(UINT index) {
    D3D12_GPU_DESCRIPTOR_HANDLE handle = g_dxr_backend.srv_uav_cbv_heap->GetGPUDescriptorHandleForHeapStart();
    handle.ptr += static_cast<UINT64>(index) * g_dxr_backend.srv_uav_cbv_descriptor_size;
    return handle;
}

bool ensure_descriptor_heap() {
    if (g_dxr_backend.srv_uav_cbv_heap != nullptr) {
        return true;
    }
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
    heap_desc.NumDescriptors = kDescriptorHeapCount;
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(g_dxr_backend.device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&g_dxr_backend.srv_uav_cbv_heap)))) {
        record_dxr_failure("ensure_descriptor_heap.CreateDescriptorHeap", E_FAIL);
        return false;
    }
    g_dxr_backend.srv_uav_cbv_descriptor_size =
        g_dxr_backend.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    return true;
}

float encode_srgb_channel(float value) {
    const float x = value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
    if (x <= 0.0031308f) {
        return x * 12.92f;
    }
    return 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
}

rtvdb::vec3 hash_color(std::size_t seed) {
    const float x = std::fmod(seed * 0.1031f, 1.0f);
    const float y = std::fmod(seed * 0.11369f, 1.0f);
    const float z = std::fmod(seed * 0.13787f, 1.0f);
    const rtvdb::vec3 h0{x, y, z};
    const float d = dot(h0, {h0.y + 19.19f, h0.z + 19.19f, h0.x + 19.19f});
    const rtvdb::vec3 h1{std::fmod(h0.x + d, 1.0f), std::fmod(h0.y + d, 1.0f), std::fmod(h0.z + d, 1.0f)};
    return {
        std::fmod((h1.x + h1.y) * h1.z, 1.0f),
        std::fmod((h1.x + h1.z) * h1.y, 1.0f),
        std::fmod((h1.y + h1.z) * h1.x, 1.0f),
    };
}

bool ensure_scene_data_buffers(const rt_scene_build &build) {
    safe_release(g_dxr_backend.triangle_color_buffer);
    safe_release(g_dxr_backend.instance_metadata_buffer);
    safe_release(g_dxr_backend.scene_position_buffer);
    safe_release(g_dxr_backend.scene_index_buffer);
    safe_release(g_dxr_backend.point_buffer);
    safe_release(g_dxr_backend.line_buffer);

    std::vector<rgba> triangle_colors(build.triangle_count);
    const std::size_t total_instance_count =
        g_dxr_backend.scene_blas_instances.size() +
        build.point_groups.size() +
        build.line_groups.size();
    std::vector<instance_metadata> instance_metadata_list(total_instance_count * kBlasGroupChunkCount);
    std::vector<rtvdb::vec3> scene_positions(build.vertex_count);
    std::vector<point_gpu> points(build.point_count);
    std::vector<line_gpu> lines(build.line_count);

    for (std::size_t vertex_index = 0; vertex_index < build.vertex_count; ++vertex_index) {
        scene_positions[vertex_index] = build.vertices[vertex_index].position;
    }

    for (std::size_t chunk_index = 0; chunk_index < build.chunks.size(); ++chunk_index) {
        const rt_scene_chunk &chunk = build.chunks[chunk_index];
        for (std::size_t local_triangle_index = 0; local_triangle_index < chunk.triangle_count; ++local_triangle_index) {
            const std::size_t index_base = chunk.index_offset + local_triangle_index * 3u;
            const std::uint32_t ia = build.indices[index_base + 0];
            const rt_scene_vertex &va = build.vertices[ia];
            triangle_colors[chunk.first_triangle + local_triangle_index] = {
                encode_srgb_channel(va.color.r),
                encode_srgb_channel(va.color.g),
                encode_srgb_channel(va.color.b),
                va.color.a,
            };
        }
    }

    for (std::size_t instance_index = 0; instance_index < g_dxr_backend.scene_blas_instances.size(); ++instance_index) {
        const scene_blas_instance &instance = g_dxr_backend.scene_blas_instances[instance_index];
        if (instance.chunk_count == 0 || instance.chunk_indices[0] >= build.chunks.size()) {
            return false;
        }
        std::uint32_t primitive_offset = 0;
        for (std::size_t geometry_index = 0; geometry_index < instance.chunk_count; ++geometry_index) {
            const rt_scene_chunk &chunk = build.chunks[instance.chunk_indices[geometry_index]];
            const std::size_t metadata_index = instance_index * kBlasGroupChunkCount + geometry_index;
            instance_metadata_list[metadata_index].first_triangle = static_cast<std::uint32_t>(chunk.first_triangle);
            instance_metadata_list[metadata_index].index_offset = static_cast<std::uint32_t>(chunk.index_offset);
            instance_metadata_list[metadata_index].primitive_offset = primitive_offset;
            instance_metadata_list[metadata_index].primitive_count = static_cast<std::uint32_t>(chunk.triangle_count);
            primitive_offset += static_cast<std::uint32_t>(chunk.triangle_count);
        }
    }

    for (std::size_t point_index = 0; point_index < build.point_count; ++point_index) {
        const point &source = build.points[point_index];
        points[point_index] = {source.position, source.radius, source.color};
    }

    std::size_t procedural_instance_index = g_dxr_backend.scene_blas_instances.size();
    for (const rt_procedural_group &group : build.point_groups) {
        const std::size_t metadata_index = procedural_instance_index++ * kBlasGroupChunkCount;
        instance_metadata_list[metadata_index].primitive_offset = static_cast<std::uint32_t>(group.first_primitive);
        instance_metadata_list[metadata_index].primitive_count = static_cast<std::uint32_t>(group.primitive_count);
    }
    for (const rt_procedural_group &group : build.line_groups) {
        const std::size_t metadata_index = procedural_instance_index++ * kBlasGroupChunkCount;
        instance_metadata_list[metadata_index].primitive_offset = static_cast<std::uint32_t>(group.first_primitive);
        instance_metadata_list[metadata_index].primitive_count = static_cast<std::uint32_t>(group.primitive_count);
    }

    for (std::size_t line_index = 0; line_index < build.line_count; ++line_index) {
        const line &source = build.lines[line_index];
        lines[line_index] = {
            source.a,
            source.radius,
            source.b,
            0.0f,
            source.color,
            static_cast<std::uint32_t>(source.flags),
            {}
        };
    }

    if (!create_buffer(
            D3D12_HEAP_TYPE_UPLOAD,
            triangle_colors.size() * sizeof(rgba),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            D3D12_RESOURCE_FLAG_NONE,
            &g_dxr_backend.triangle_color_buffer)) {
        return false;
    }
    if (!upload_buffer_data(
            g_dxr_backend.triangle_color_buffer,
            triangle_colors.data(),
            triangle_colors.size() * sizeof(rgba))) {
        return false;
    }
    if (!create_buffer(
            D3D12_HEAP_TYPE_UPLOAD,
            instance_metadata_list.size() * sizeof(instance_metadata),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            D3D12_RESOURCE_FLAG_NONE,
            &g_dxr_backend.instance_metadata_buffer)) {
        return false;
    }
    if (!upload_buffer_data(
            g_dxr_backend.instance_metadata_buffer,
            instance_metadata_list.data(),
            instance_metadata_list.size() * sizeof(instance_metadata))) {
        return false;
    }
    if (!create_buffer(
            D3D12_HEAP_TYPE_UPLOAD,
            scene_positions.size() * sizeof(rtvdb::vec3),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            D3D12_RESOURCE_FLAG_NONE,
            &g_dxr_backend.scene_position_buffer)) {
        return false;
    }
    if (!upload_buffer_data(
            g_dxr_backend.scene_position_buffer,
            scene_positions.data(),
            scene_positions.size() * sizeof(rtvdb::vec3))) {
        return false;
    }
    if (!create_buffer(
            D3D12_HEAP_TYPE_UPLOAD,
            build.indices.size() * sizeof(std::uint32_t),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            D3D12_RESOURCE_FLAG_NONE,
            &g_dxr_backend.scene_index_buffer)) {
        return false;
    }
    if (!upload_buffer_data(
            g_dxr_backend.scene_index_buffer,
            build.indices.data(),
            build.indices.size() * sizeof(std::uint32_t))) {
        return false;
    }
    if (!create_buffer(
            D3D12_HEAP_TYPE_UPLOAD,
            points.size() * sizeof(point_gpu),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            D3D12_RESOURCE_FLAG_NONE,
            &g_dxr_backend.point_buffer)) {
        return false;
    }
    if (!upload_buffer_data(g_dxr_backend.point_buffer, points.data(), points.size() * sizeof(point_gpu))) {
        return false;
    }
    if (!create_buffer(
            D3D12_HEAP_TYPE_UPLOAD,
            lines.size() * sizeof(line_gpu),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            D3D12_RESOURCE_FLAG_NONE,
            &g_dxr_backend.line_buffer)) {
        return false;
    }
    if (!upload_buffer_data(g_dxr_backend.line_buffer, lines.data(), lines.size() * sizeof(line_gpu))) {
        return false;
    }

    D3D12_SHADER_RESOURCE_VIEW_DESC tri_srv{};
    tri_srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    tri_srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    tri_srv.Format = DXGI_FORMAT_UNKNOWN;
    tri_srv.Buffer.FirstElement = 0;
    tri_srv.Buffer.NumElements = static_cast<UINT>(triangle_colors.size());
    tri_srv.Buffer.StructureByteStride = sizeof(rgba);
    tri_srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
    g_dxr_backend.device->CreateShaderResourceView(
        g_dxr_backend.triangle_color_buffer,
        &tri_srv,
        descriptor_cpu_handle(kTriangleColorSrvDescriptorIndex));

    D3D12_SHADER_RESOURCE_VIEW_DESC instance_srv{};
    instance_srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    instance_srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    instance_srv.Format = DXGI_FORMAT_UNKNOWN;
    instance_srv.Buffer.FirstElement = 0;
    instance_srv.Buffer.NumElements = static_cast<UINT>(instance_metadata_list.size());
    instance_srv.Buffer.StructureByteStride = sizeof(instance_metadata);
    instance_srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
    g_dxr_backend.device->CreateShaderResourceView(
        g_dxr_backend.instance_metadata_buffer,
        &instance_srv,
        descriptor_cpu_handle(kInstanceMetadataSrvDescriptorIndex));

    D3D12_SHADER_RESOURCE_VIEW_DESC position_srv{};
    position_srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    position_srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    position_srv.Format = DXGI_FORMAT_UNKNOWN;
    position_srv.Buffer.FirstElement = 0;
    position_srv.Buffer.NumElements = static_cast<UINT>(scene_positions.size());
    position_srv.Buffer.StructureByteStride = sizeof(rtvdb::vec3);
    position_srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
    g_dxr_backend.device->CreateShaderResourceView(
        g_dxr_backend.scene_position_buffer,
        &position_srv,
        descriptor_cpu_handle(kScenePositionSrvDescriptorIndex));

    D3D12_SHADER_RESOURCE_VIEW_DESC index_srv{};
    index_srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    index_srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    index_srv.Format = DXGI_FORMAT_UNKNOWN;
    index_srv.Buffer.FirstElement = 0;
    index_srv.Buffer.NumElements = static_cast<UINT>(build.indices.size());
    index_srv.Buffer.StructureByteStride = sizeof(std::uint32_t);
    index_srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
    g_dxr_backend.device->CreateShaderResourceView(
        g_dxr_backend.scene_index_buffer,
        &index_srv,
        descriptor_cpu_handle(kSceneIndexSrvDescriptorIndex));

    D3D12_SHADER_RESOURCE_VIEW_DESC point_srv{};
    point_srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    point_srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    point_srv.Format = DXGI_FORMAT_UNKNOWN;
    point_srv.Buffer.FirstElement = 0;
    point_srv.Buffer.NumElements = static_cast<UINT>(points.size());
    point_srv.Buffer.StructureByteStride = sizeof(point_gpu);
    point_srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
    g_dxr_backend.device->CreateShaderResourceView(
        g_dxr_backend.point_buffer,
        &point_srv,
        descriptor_cpu_handle(kPointSrvDescriptorIndex));

    D3D12_SHADER_RESOURCE_VIEW_DESC line_srv{};
    line_srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    line_srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    line_srv.Format = DXGI_FORMAT_UNKNOWN;
    line_srv.Buffer.FirstElement = 0;
    line_srv.Buffer.NumElements = static_cast<UINT>(lines.size());
    line_srv.Buffer.StructureByteStride = sizeof(line_gpu);
    line_srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
    g_dxr_backend.device->CreateShaderResourceView(
        g_dxr_backend.line_buffer,
        &line_srv,
        descriptor_cpu_handle(kLineSrvDescriptorIndex));
    return true;
}

bool ensure_camera_constant_buffer() {
    if (g_dxr_backend.camera_constant_buffer != nullptr && g_dxr_backend.camera_constant_buffer_mapped != nullptr) {
        return true;
    }
    if (!create_buffer(
            D3D12_HEAP_TYPE_UPLOAD,
            kCameraConstantBufferBytes,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            D3D12_RESOURCE_FLAG_NONE,
            &g_dxr_backend.camera_constant_buffer)) {
        return false;
    }
    D3D12_RANGE read_range{0, 0};
    const HRESULT hr = g_dxr_backend.camera_constant_buffer->Map(0, &read_range, &g_dxr_backend.camera_constant_buffer_mapped);
    if (FAILED(hr)) {
        record_dxr_failure("ensure_camera_constant_buffer.Map", hr);
        safe_release(g_dxr_backend.camera_constant_buffer);
    safe_release(g_dxr_backend.scene_index_buffer);
    safe_release(g_dxr_backend.scene_position_buffer);
    safe_release(g_dxr_backend.instance_metadata_buffer);
    safe_release(g_dxr_backend.triangle_color_buffer);
        return false;
    }
    D3D12_CONSTANT_BUFFER_VIEW_DESC cbv_desc{};
    cbv_desc.BufferLocation = g_dxr_backend.camera_constant_buffer->GetGPUVirtualAddress();
    cbv_desc.SizeInBytes = static_cast<UINT>(kCameraConstantBufferBytes);
    g_dxr_backend.device->CreateConstantBufferView(&cbv_desc, descriptor_cpu_handle(kCameraCbvDescriptorIndex));
    return true;
}

bool ensure_output_texture(UINT width, UINT height) {
    if (g_dxr_backend.output_texture != nullptr &&
        g_dxr_backend.accumulation_texture != nullptr &&
        g_dxr_backend.output_width == width &&
        g_dxr_backend.output_height == height) {
        return true;
    }
    safe_release(g_dxr_backend.output_texture);
    safe_release(g_dxr_backend.accumulation_texture);
    safe_release(g_dxr_backend.output_readback_buffer);
    g_dxr_backend.output_readback_size = 0;
    g_dxr_backend.output_readback_footprint = {};
    g_dxr_backend.output_width = 0;
    g_dxr_backend.output_height = 0;

    D3D12_HEAP_PROPERTIES heap_properties{};
    heap_properties.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = kOutputColorFormat;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    HRESULT hr = g_dxr_backend.device->CreateCommittedResource(
            &heap_properties,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr,
            IID_PPV_ARGS(&g_dxr_backend.output_texture));
    if (FAILED(hr)) {
        record_dxr_failure("ensure_output_texture.CreateCommittedResource.output", hr);
        return false;
    }

    D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc{};
    uav_desc.Format = kOutputColorFormat;
    uav_desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    g_dxr_backend.device->CreateUnorderedAccessView(
        g_dxr_backend.output_texture,
        nullptr,
        &uav_desc,
        descriptor_cpu_handle(kOutputUavDescriptorIndex));

    desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    hr = g_dxr_backend.device->CreateCommittedResource(
            &heap_properties,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr,
            IID_PPV_ARGS(&g_dxr_backend.accumulation_texture));
    if (FAILED(hr)) {
        record_dxr_failure("ensure_output_texture.CreateCommittedResource.accumulation", hr);
        safe_release(g_dxr_backend.output_texture);
        return false;
    }

    D3D12_UNORDERED_ACCESS_VIEW_DESC accumulation_uav_desc{};
    accumulation_uav_desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    accumulation_uav_desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    g_dxr_backend.device->CreateUnorderedAccessView(
        g_dxr_backend.accumulation_texture,
        nullptr,
        &accumulation_uav_desc,
        descriptor_cpu_handle(kAccumulationUavDescriptorIndex));
    g_dxr_backend.output_width = width;
    g_dxr_backend.output_height = height;
    reset_accumulation_state();
    return true;
}

bool ensure_output_readback_buffer() {
    if (g_dxr_backend.output_texture == nullptr) {
        return false;
    }

    D3D12_RESOURCE_DESC output_desc = g_dxr_backend.output_texture->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT num_rows = 0;
    UINT64 row_size_bytes = 0;
    UINT64 total_bytes = 0;
    g_dxr_backend.device->GetCopyableFootprints(
        &output_desc,
        0,
        1,
        0,
        &footprint,
        &num_rows,
        &row_size_bytes,
        &total_bytes);
    if (total_bytes == 0) {
        return false;
    }

    if (g_dxr_backend.output_readback_buffer != nullptr &&
        g_dxr_backend.output_readback_size == static_cast<std::size_t>(total_bytes)) {
        g_dxr_backend.output_readback_footprint = footprint;
        return true;
    }

    safe_release(g_dxr_backend.output_readback_buffer);
    if (!create_buffer(
            D3D12_HEAP_TYPE_READBACK,
            static_cast<std::size_t>(total_bytes),
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_FLAG_NONE,
            &g_dxr_backend.output_readback_buffer)) {
        return false;
    }
    g_dxr_backend.output_readback_size = static_cast<std::size_t>(total_bytes);
    g_dxr_backend.output_readback_footprint = footprint;
    return true;
}

bool ensure_pick_buffers() {
    if (g_dxr_backend.pick_output_buffer == nullptr) {
        if (!create_buffer(
                D3D12_HEAP_TYPE_DEFAULT,
                sizeof(gpu_pick_result),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                &g_dxr_backend.pick_output_buffer)) {
            return false;
        }
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc{};
        uav_desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uav_desc.Format = DXGI_FORMAT_UNKNOWN;
        uav_desc.Buffer.NumElements = 1;
        uav_desc.Buffer.StructureByteStride = sizeof(gpu_pick_result);
        g_dxr_backend.device->CreateUnorderedAccessView(
            g_dxr_backend.pick_output_buffer,
            nullptr,
            &uav_desc,
            descriptor_cpu_handle(kPickOutputUavDescriptorIndex));
    }

    if (g_dxr_backend.pick_readback_buffer == nullptr) {
        if (!create_buffer(
                D3D12_HEAP_TYPE_READBACK,
                sizeof(gpu_pick_result),
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_FLAG_NONE,
                &g_dxr_backend.pick_readback_buffer)) {
            return false;
        }
    }
    return true;
}

void update_tlas_srv() {
    if (g_dxr_backend.tlas_result == nullptr || g_dxr_backend.srv_uav_cbv_heap == nullptr) {
        return;
    }
    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc{};
    srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
    srv_desc.RaytracingAccelerationStructure.Location = g_dxr_backend.tlas_result->GetGPUVirtualAddress();
    g_dxr_backend.device->CreateShaderResourceView(nullptr, &srv_desc, descriptor_cpu_handle(kSceneSrvDescriptorIndex));
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

void update_viewer_constants(
    const frame_scene &scene,
    bool has_frame,
    UINT width,
    UINT height,
    UINT pick_pixel_x = 0,
    UINT pick_pixel_y = 0,
    bool is_pick_pass = false)
{
    viewer_constants constants{};
    hover_highlight highlight{};
    get_hover_highlight(&highlight);
    constants.width = width;
    constants.height = height;
    constants.aspect = constants.height == 0
        ? 1.0f
        : static_cast<float>(constants.width) / static_cast<float>(constants.height);

    display_mode mode = display_mode::triangle_normal;
    get_display_mode(&mode);
    constants.display_mode = static_cast<std::uint32_t>(mode);
    constants.accumulation_sample_index = g_dxr_backend.accumulation_sample_count;
    fill_accumulation_jitter(g_dxr_backend.accumulation_sample_count, constants.accumulation_jitter);
    constants.hover_highlight_kind = static_cast<std::uint32_t>(highlight.kind);
    constants.hover_primitive_index = highlight.primitive_index;
    constants.hover_highlight_mix = 0.85f;
    constants.is_pick_pass = is_pick_pass ? 1u : 0u;

    rtvdb::camera camera{};
    if (has_frame) {
        camera = scene.camera;
    }
    const rtvdb::vec3 forward = normalize_or(subtract(camera.target, camera.origin), {0.0f, 0.0f, 1.0f});
    rtvdb::vec3 up = normalize_or(camera.up, {0.0f, 1.0f, 0.0f});
    const rtvdb::vec3 right = normalize_or(cross(forward, up), {1.0f, 0.0f, 0.0f});
    up = normalize_or(cross(right, forward), {0.0f, 1.0f, 0.0f});
    rt_scene_build build{};
    copy_present_client_rt_scene_build(&build);

    constants.origin[0] = camera.origin.x;
    constants.origin[1] = camera.origin.y;
    constants.origin[2] = camera.origin.z;
    constants.forward[0] = forward.x;
    constants.forward[1] = forward.y;
    constants.forward[2] = forward.z;
    constants.right[0] = right.x;
    constants.right[1] = right.y;
    constants.right[2] = right.z;
    constants.up[0] = up.x;
    constants.up[1] = up.y;
    constants.up[2] = up.z;
    if (build.bounds.valid) {
        constants.scene_bounds_min[0] = build.bounds.min.x;
        constants.scene_bounds_min[1] = build.bounds.min.y;
        constants.scene_bounds_min[2] = build.bounds.min.z;
        constants.scene_bounds_max[0] = build.bounds.max.x;
        constants.scene_bounds_max[1] = build.bounds.max.y;
        constants.scene_bounds_max[2] = build.bounds.max.z;
        constants.scene_bounds_valid = 1;
    }
    constants.projection = static_cast<std::uint32_t>(camera.projection);
    constants.projection_blend_from = static_cast<std::uint32_t>(scene.projection_blend_from);
    constants.projection_blend_to = static_cast<std::uint32_t>(scene.projection_blend_to);
    constants.projection_blend_t = scene.projection_blend_t;
    constants.pick_pixel_x = pick_pixel_x;
    constants.pick_pixel_y = pick_pixel_y;
    fill_projection_parameters(
        camera,
        scene.projection_blend_from,
        constants.aspect,
        &constants.projection_param_from0,
        &constants.projection_param_from1);
    fill_projection_parameters(
        camera,
        scene.projection_blend_to,
        constants.aspect,
        &constants.projection_param_to0,
        &constants.projection_param_to1);

    std::memcpy(g_dxr_backend.camera_constant_buffer_mapped, &constants, sizeof(constants));
}

bool ensure_global_root_signature() {
    if (g_dxr_backend.global_root_signature != nullptr) {
        return true;
    }
    D3D12_DESCRIPTOR_RANGE1 ranges[2]{};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[0].NumDescriptors = 3;
    ranges[0].BaseShaderRegister = 0;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[1].NumDescriptors = 7;
    ranges[1].BaseShaderRegister = 0;

    D3D12_ROOT_PARAMETER1 parameters[3]{};
    parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameters[0].DescriptorTable.NumDescriptorRanges = 1;
    parameters[0].DescriptorTable.pDescriptorRanges = &ranges[0];
    parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameters[1].DescriptorTable.NumDescriptorRanges = 1;
    parameters[1].DescriptorTable.pDescriptorRanges = &ranges[1];
    parameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    parameters[2].Descriptor.ShaderRegister = 0;

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    root_desc.Desc_1_1.NumParameters = static_cast<UINT>(std::size(parameters));
    root_desc.Desc_1_1.pParameters = parameters;
    root_desc.Desc_1_1.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ID3DBlob* serialized = nullptr;
    ID3DBlob* errors = nullptr;
    const HRESULT serialize_hr = D3D12SerializeVersionedRootSignature(&root_desc, &serialized, &errors);
    if (FAILED(serialize_hr)) {
        safe_release(errors);
        safe_release(serialized);
        return false;
    }
    safe_release(errors);
    const HRESULT hr = g_dxr_backend.device->CreateRootSignature(
        0,
        serialized->GetBufferPointer(),
        serialized->GetBufferSize(),
        IID_PPV_ARGS(&g_dxr_backend.global_root_signature));
    safe_release(serialized);
    return SUCCEEDED(hr);
}

bool ensure_state_object() {
    if (g_dxr_backend.raytracing_state_object != nullptr && g_dxr_backend.raytracing_state_props != nullptr) {
        return true;
    }
    if (!g_dxr_backend.shader_library_ready || !ensure_global_root_signature()) {
        return false;
    }

    D3D12_EXPORT_DESC exports[8]{};
    exports[0].Name = L"RayGen";
    exports[1].Name = L"PickRayGen";
    exports[2].Name = L"Miss";
    exports[3].Name = L"ClosestHitTriangle";
    exports[4].Name = L"IntersectionPoint";
    exports[5].Name = L"ClosestHitPoint";
    exports[6].Name = L"IntersectionLine";
    exports[7].Name = L"ClosestHitLine";

    D3D12_DXIL_LIBRARY_DESC dxil_library_desc{};
    dxil_library_desc.DXILLibrary.pShaderBytecode = g_dxr_backend.shader_library.data();
    dxil_library_desc.DXILLibrary.BytecodeLength = g_dxr_backend.shader_library.size();
    dxil_library_desc.NumExports = static_cast<UINT>(std::size(exports));
    dxil_library_desc.pExports = exports;

    D3D12_HIT_GROUP_DESC triangle_hit_group_desc{};
    triangle_hit_group_desc.HitGroupExport = L"TriangleHitGroup";
    triangle_hit_group_desc.ClosestHitShaderImport = L"ClosestHitTriangle";
    triangle_hit_group_desc.Type = D3D12_HIT_GROUP_TYPE_TRIANGLES;

    D3D12_HIT_GROUP_DESC point_hit_group_desc{};
    point_hit_group_desc.HitGroupExport = L"PointHitGroup";
    point_hit_group_desc.ClosestHitShaderImport = L"ClosestHitPoint";
    point_hit_group_desc.IntersectionShaderImport = L"IntersectionPoint";
    point_hit_group_desc.Type = D3D12_HIT_GROUP_TYPE_PROCEDURAL_PRIMITIVE;

    D3D12_HIT_GROUP_DESC line_hit_group_desc{};
    line_hit_group_desc.HitGroupExport = L"LineHitGroup";
    line_hit_group_desc.ClosestHitShaderImport = L"ClosestHitLine";
    line_hit_group_desc.IntersectionShaderImport = L"IntersectionLine";
    line_hit_group_desc.Type = D3D12_HIT_GROUP_TYPE_PROCEDURAL_PRIMITIVE;

    D3D12_RAYTRACING_SHADER_CONFIG shader_config{};
    shader_config.MaxPayloadSizeInBytes = sizeof(float) * 8;
    shader_config.MaxAttributeSizeInBytes = sizeof(float) * 2;

    D3D12_GLOBAL_ROOT_SIGNATURE global_root_signature{};
    global_root_signature.pGlobalRootSignature = g_dxr_backend.global_root_signature;

    D3D12_RAYTRACING_PIPELINE_CONFIG pipeline_config{};
    pipeline_config.MaxTraceRecursionDepth = 1;

    D3D12_STATE_SUBOBJECT subobjects[7]{};
    subobjects[0].Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY;
    subobjects[0].pDesc = &dxil_library_desc;
    subobjects[1].Type = D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP;
    subobjects[1].pDesc = &triangle_hit_group_desc;
    subobjects[2].Type = D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP;
    subobjects[2].pDesc = &point_hit_group_desc;
    subobjects[3].Type = D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP;
    subobjects[3].pDesc = &line_hit_group_desc;
    subobjects[4].Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG;
    subobjects[4].pDesc = &shader_config;
    subobjects[5].Type = D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE;
    subobjects[5].pDesc = &global_root_signature;
    subobjects[6].Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG;
    subobjects[6].pDesc = &pipeline_config;

    D3D12_STATE_OBJECT_DESC state_object_desc{};
    state_object_desc.Type = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE;
    state_object_desc.NumSubobjects = static_cast<UINT>(std::size(subobjects));
    state_object_desc.pSubobjects = subobjects;
    if (FAILED(g_dxr_backend.device->CreateStateObject(
            &state_object_desc,
            IID_PPV_ARGS(&g_dxr_backend.raytracing_state_object)))) {
        return false;
    }
    return SUCCEEDED(
        g_dxr_backend.raytracing_state_object->QueryInterface(IID_PPV_ARGS(&g_dxr_backend.raytracing_state_props)));
}

bool create_shader_table_resource(
    ID3D12Resource** out_resource,
    const void* const* shader_identifiers,
    std::size_t shader_identifier_count)
{
    if (out_resource == nullptr || shader_identifiers == nullptr || shader_identifier_count == 0) {
        return false;
    }
    for (std::size_t i = 0; i < shader_identifier_count; ++i) {
        if (shader_identifiers[i] == nullptr) {
            return false;
        }
    }

    safe_release(*out_resource);
    const std::size_t table_bytes = align_up(
        kShaderRecordSize * shader_identifier_count,
        D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);
    if (!create_buffer(
            D3D12_HEAP_TYPE_UPLOAD,
            table_bytes,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            D3D12_RESOURCE_FLAG_NONE,
            out_resource)) {
        return false;
    }

    std::vector<std::uint8_t> table_data(table_bytes);
    for (std::size_t i = 0; i < shader_identifier_count; ++i) {
        std::memcpy(table_data.data() + i * kShaderRecordSize, shader_identifiers[i], kShaderRecordSize);
    }
    return upload_buffer_data(*out_resource, table_data.data(), table_data.size());
}

bool ensure_shader_tables() {
    if (g_dxr_backend.raygen_shader_table != nullptr &&
        g_dxr_backend.pick_raygen_shader_table != nullptr &&
        g_dxr_backend.miss_shader_table != nullptr &&
        g_dxr_backend.hitgroup_shader_table != nullptr) {
        return true;
    }
    if (g_dxr_backend.raytracing_state_props == nullptr) {
        return false;
    }
    const void* raygen_identifiers[] = {
        g_dxr_backend.raytracing_state_props->GetShaderIdentifier(L"RayGen"),
    };
    const void* pick_raygen_identifiers[] = {
        g_dxr_backend.raytracing_state_props->GetShaderIdentifier(L"PickRayGen"),
    };
    const void* miss_identifiers[] = {
        g_dxr_backend.raytracing_state_props->GetShaderIdentifier(L"Miss"),
    };
    const void* hitgroup_identifiers[] = {
        g_dxr_backend.raytracing_state_props->GetShaderIdentifier(L"TriangleHitGroup"),
        g_dxr_backend.raytracing_state_props->GetShaderIdentifier(L"PointHitGroup"),
        g_dxr_backend.raytracing_state_props->GetShaderIdentifier(L"LineHitGroup"),
    };
    return create_shader_table_resource(
            &g_dxr_backend.raygen_shader_table,
            raygen_identifiers,
            std::size(raygen_identifiers)) &&
        create_shader_table_resource(
            &g_dxr_backend.pick_raygen_shader_table,
            pick_raygen_identifiers,
            std::size(pick_raygen_identifiers)) &&
        create_shader_table_resource(&g_dxr_backend.miss_shader_table, miss_identifiers, std::size(miss_identifiers)) &&
        create_shader_table_resource(
            &g_dxr_backend.hitgroup_shader_table,
            hitgroup_identifiers,
            std::size(hitgroup_identifiers));
}

bool ensure_raytracing_runtime(UINT width, UINT height);

bool reset_command_list() {
    const UINT allocator_index = g_dxr_backend.command_allocator_index % kFrameCount;
    if (!wait_for_fence_value(g_dxr_backend.command_allocator_fence_values[allocator_index])) {
        record_dxr_failure("reset_command_list.wait_for_fence");
        return false;
    }

    ID3D12CommandAllocator* allocator = g_dxr_backend.command_allocators[allocator_index];
    const HRESULT allocator_hr = allocator->Reset();
    if (FAILED(allocator_hr)) {
        record_dxr_failure("reset_command_list.allocator_reset", allocator_hr);
        return false;
    }
    const HRESULT list_hr = g_dxr_backend.command_list->Reset(allocator, nullptr);
    if (FAILED(list_hr)) {
        record_dxr_failure("reset_command_list.command_list_reset", list_hr);
        return false;
    }
    return true;
}

bool close_and_execute_command_list(double* out_submit_cpu_ms) {
    const HRESULT close_hr = g_dxr_backend.command_list->Close();
    if (FAILED(close_hr)) {
        record_dxr_failure("close_and_execute_command_list.Close", close_hr);
        return false;
    }
    ID3D12CommandList* command_lists[] = {g_dxr_backend.command_list};
    const auto submit_start = std::chrono::steady_clock::now();
    g_dxr_backend.queue->ExecuteCommandLists(1, command_lists);
    const std::uint64_t fence_value = g_dxr_backend.submitted_fence_value + 1u;
    const HRESULT signal_hr = g_dxr_backend.queue->Signal(g_dxr_backend.fence, fence_value);
    if (FAILED(signal_hr)) {
        record_dxr_failure("close_and_execute_command_list.Signal", signal_hr);
        return false;
    }
    if (out_submit_cpu_ms != nullptr) {
        *out_submit_cpu_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - submit_start).count();
    }
    const UINT allocator_index = g_dxr_backend.command_allocator_index % kFrameCount;
    g_dxr_backend.command_allocator_fence_values[allocator_index] = fence_value;
    g_dxr_backend.submitted_fence_value = fence_value;
    g_dxr_backend.command_allocator_index = (allocator_index + 1u) % kFrameCount;
    return true;
}

bool record_ray_dispatch(const frame_scene &scene, bool has_frame) {
    if (!ensure_raytracing_runtime(g_dxr_backend.width, g_dxr_backend.height)) {
        return false;
    }
    if (!ensure_timestamp_query_resources()) {
        return false;
    }

    update_tlas_srv();
    update_viewer_constants(scene, has_frame, g_dxr_backend.width, g_dxr_backend.height);

    if (!reset_command_list()) {
        return false;
    }

    ID3D12DescriptorHeap* heaps[] = {g_dxr_backend.srv_uav_cbv_heap};
    g_dxr_backend.command_list->SetDescriptorHeaps(1, heaps);
    g_dxr_backend.command_list->SetComputeRootSignature(g_dxr_backend.global_root_signature);
    g_dxr_backend.command_list->SetComputeRootDescriptorTable(0, descriptor_gpu_handle(kOutputUavDescriptorIndex));
    g_dxr_backend.command_list->SetComputeRootDescriptorTable(1, descriptor_gpu_handle(kSceneSrvDescriptorIndex));
    g_dxr_backend.command_list->SetComputeRootConstantBufferView(
        2,
        g_dxr_backend.camera_constant_buffer->GetGPUVirtualAddress());
    g_dxr_backend.command_list->SetPipelineState1(g_dxr_backend.raytracing_state_object);
    write_timestamp_query_begin(current_timestamp_query_slot());

    D3D12_DISPATCH_RAYS_DESC dispatch_desc{};
    dispatch_desc.RayGenerationShaderRecord.StartAddress = g_dxr_backend.raygen_shader_table->GetGPUVirtualAddress();
    dispatch_desc.RayGenerationShaderRecord.SizeInBytes = kShaderRecordSize;
    dispatch_desc.MissShaderTable.StartAddress = g_dxr_backend.miss_shader_table->GetGPUVirtualAddress();
    dispatch_desc.MissShaderTable.SizeInBytes = kShaderRecordSize;
    dispatch_desc.MissShaderTable.StrideInBytes = kShaderRecordSize;
    dispatch_desc.HitGroupTable.StartAddress = g_dxr_backend.hitgroup_shader_table->GetGPUVirtualAddress();
    dispatch_desc.HitGroupTable.SizeInBytes = kShaderRecordSize * 3u;
    dispatch_desc.HitGroupTable.StrideInBytes = kShaderRecordSize;
    dispatch_desc.Width = g_dxr_backend.width;
    dispatch_desc.Height = g_dxr_backend.height;
    dispatch_desc.Depth = 1;
    g_dxr_backend.command_list->DispatchRays(&dispatch_desc);

    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = g_dxr_backend.output_texture;
    g_dxr_backend.command_list->ResourceBarrier(1, &barrier);
    write_timestamp_query_end_and_resolve(current_timestamp_query_slot());
    return true;
}

bool dispatch_pick_query(
    int width,
    int height,
    int pixel_x,
    int pixel_y,
    const frame_scene &scene,
    bool has_frame,
    pick_result* out_result)
{
    if (width <= 0 || height <= 0 || pixel_x < 0 || pixel_y < 0 || out_result == nullptr) {
        return false;
    }
    if (!ensure_raytracing_runtime(static_cast<UINT>(width), static_cast<UINT>(height))) {
        return false;
    }

    update_tlas_srv();
    update_viewer_constants(
        scene,
        has_frame,
        static_cast<UINT>(width),
        static_cast<UINT>(height),
        static_cast<UINT>(pixel_x),
        static_cast<UINT>(pixel_y),
        true);

    if (!reset_command_list()) {
        return false;
    }

    ID3D12DescriptorHeap* heaps[] = {g_dxr_backend.srv_uav_cbv_heap};
    g_dxr_backend.command_list->SetDescriptorHeaps(1, heaps);
    g_dxr_backend.command_list->SetComputeRootSignature(g_dxr_backend.global_root_signature);
    g_dxr_backend.command_list->SetComputeRootDescriptorTable(0, descriptor_gpu_handle(kOutputUavDescriptorIndex));
    g_dxr_backend.command_list->SetComputeRootDescriptorTable(1, descriptor_gpu_handle(kSceneSrvDescriptorIndex));
    g_dxr_backend.command_list->SetComputeRootConstantBufferView(2, g_dxr_backend.camera_constant_buffer->GetGPUVirtualAddress());
    g_dxr_backend.command_list->SetPipelineState1(g_dxr_backend.raytracing_state_object);

    D3D12_DISPATCH_RAYS_DESC dispatch_desc{};
    dispatch_desc.RayGenerationShaderRecord.StartAddress = g_dxr_backend.pick_raygen_shader_table->GetGPUVirtualAddress();
    dispatch_desc.RayGenerationShaderRecord.SizeInBytes = kShaderRecordSize;
    dispatch_desc.MissShaderTable.StartAddress = g_dxr_backend.miss_shader_table->GetGPUVirtualAddress();
    dispatch_desc.MissShaderTable.SizeInBytes = kShaderRecordSize;
    dispatch_desc.MissShaderTable.StrideInBytes = kShaderRecordSize;
    dispatch_desc.HitGroupTable.StartAddress = g_dxr_backend.hitgroup_shader_table->GetGPUVirtualAddress();
    dispatch_desc.HitGroupTable.SizeInBytes = kShaderRecordSize * 3u;
    dispatch_desc.HitGroupTable.StrideInBytes = kShaderRecordSize;
    dispatch_desc.Width = 1;
    dispatch_desc.Height = 1;
    dispatch_desc.Depth = 1;
    g_dxr_backend.command_list->DispatchRays(&dispatch_desc);

    D3D12_RESOURCE_BARRIER barriers[3]{};
    barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barriers[0].UAV.pResource = g_dxr_backend.pick_output_buffer;
    barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barriers[1].Transition.pResource = g_dxr_backend.pick_output_buffer;
    barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barriers[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    g_dxr_backend.command_list->ResourceBarrier(2, barriers);
    g_dxr_backend.command_list->CopyResource(g_dxr_backend.pick_readback_buffer, g_dxr_backend.pick_output_buffer);
    barriers[2] = barriers[1];
    barriers[2].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barriers[2].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    g_dxr_backend.command_list->ResourceBarrier(1, &barriers[2]);

    if (!close_and_execute_command_list()) {
        return false;
    }
    wait_for_gpu();

    gpu_pick_result mapped{};
    void* mapped_data = nullptr;
    const D3D12_RANGE read_range{0, sizeof(mapped)};
    if (FAILED(g_dxr_backend.pick_readback_buffer->Map(0, &read_range, &mapped_data)) || mapped_data == nullptr) {
        return false;
    }
    std::memcpy(&mapped, mapped_data, sizeof(mapped));
    g_dxr_backend.pick_readback_buffer->Unmap(0, nullptr);

    out_result->kind = static_cast<hover_highlight_kind>(mapped.primitive_kind);
    out_result->primitive_index = mapped.primitive_index;
    out_result->distance = mapped.distance;
    return true;
}

bool ensure_raytracing_runtime(UINT width, UINT height) {
    rt_scene_build build{};
    copy_present_render_rt_scene_build(&build);
    return ensure_descriptor_heap() &&
        ensure_camera_constant_buffer() &&
        ensure_output_texture(width, height) &&
        ensure_pick_buffers() &&
        ensure_scene_data_buffers(build) &&
        ensure_state_object() &&
        ensure_shader_tables();
}

bool dispatch_rays_to_output_texture(const frame_scene &scene, bool has_frame, double* out_submit_cpu_ms = nullptr) {
    if (!record_ray_dispatch(scene, has_frame)) {
        return false;
    }
    const UINT timestamp_slot = current_timestamp_query_slot();
    if (!close_and_execute_command_list(out_submit_cpu_ms)) {
        return false;
    }
    g_dxr_backend.dispatch_timestamp_fence_values[timestamp_slot] = g_dxr_backend.submitted_fence_value;
    g_dxr_backend.dispatch_timestamp_pending[timestamp_slot] = true;
    return true;
}

bool dispatch_rays_to_swapchain(const frame_scene &scene, bool has_frame, double* out_submit_cpu_ms = nullptr) {
    if (!record_ray_dispatch(scene, has_frame)) {
        return false;
    }

    D3D12_RESOURCE_BARRIER barriers[3]{};
    barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barriers[0].UAV.pResource = g_dxr_backend.output_texture;
    barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barriers[1].Transition.pResource = g_dxr_backend.output_texture;
    barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barriers[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barriers[2].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barriers[2].Transition.pResource = g_dxr_backend.render_targets[g_dxr_backend.frame_index];
    barriers[2].Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    barriers[2].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    barriers[2].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    g_dxr_backend.command_list->ResourceBarrier(3, barriers);

    g_dxr_backend.command_list->CopyResource(g_dxr_backend.render_targets[g_dxr_backend.frame_index], g_dxr_backend.output_texture);

    barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barriers[0].Transition.pResource = g_dxr_backend.render_targets[g_dxr_backend.frame_index];
    barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    barriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barriers[1].Transition.pResource = g_dxr_backend.output_texture;
    barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barriers[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    g_dxr_backend.command_list->ResourceBarrier(2, barriers);
    const UINT timestamp_slot = current_timestamp_query_slot();
    if (!close_and_execute_command_list(out_submit_cpu_ms)) {
        return false;
    }
    g_dxr_backend.dispatch_timestamp_fence_values[timestamp_slot] = g_dxr_backend.submitted_fence_value;
    g_dxr_backend.dispatch_timestamp_pending[timestamp_slot] = true;
    return true;
}

bool copy_output_texture_to_native_display_target(
    ID3D12Resource* texture_resource,
    double* out_submit_cpu_ms = nullptr)
{
    if (g_dxr_backend.output_texture == nullptr || texture_resource == nullptr) {
        return false;
    }
    if (g_dxr_backend.native_display_target != texture_resource) {
        g_dxr_backend.native_display_target = texture_resource;
        g_dxr_backend.native_display_target_state = D3D12_RESOURCE_STATE_COPY_DEST;
        const D3D12_RESOURCE_DESC desc = texture_resource->GetDesc();
        char detail[256]{};
        std::snprintf(
            detail,
            sizeof(detail),
            "native_target_changed ptr=0x%p width=%llu height=%u format=%u flags=0x%x",
            texture_resource,
            static_cast<unsigned long long>(desc.Width),
            static_cast<unsigned>(desc.Height),
            static_cast<unsigned>(desc.Format),
            static_cast<unsigned>(desc.Flags));
        append_timestamped_log_line("dxr_failure.log", detail);
    }
    if (!reset_command_list()) {
        return false;
    }

    D3D12_RESOURCE_BARRIER barriers[4]{};
    UINT barrier_count = 0;
    if (g_dxr_backend.native_display_target_state != D3D12_RESOURCE_STATE_COPY_DEST) {
        barriers[barrier_count].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barriers[barrier_count].Transition.pResource = texture_resource;
        barriers[barrier_count].Transition.StateBefore = g_dxr_backend.native_display_target_state;
        barriers[barrier_count].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
        barriers[barrier_count].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        ++barrier_count;
    }
    barriers[barrier_count].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barriers[barrier_count].Transition.pResource = g_dxr_backend.output_texture;
    barriers[barrier_count].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barriers[barrier_count].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barriers[barrier_count].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    ++barrier_count;
    g_dxr_backend.command_list->ResourceBarrier(barrier_count, barriers);

    g_dxr_backend.command_list->CopyResource(texture_resource, g_dxr_backend.output_texture);

    D3D12_RESOURCE_BARRIER end_barriers[2]{};
    end_barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    end_barriers[0].Transition.pResource = texture_resource;
    end_barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    end_barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    end_barriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    end_barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    end_barriers[1].Transition.pResource = g_dxr_backend.output_texture;
    end_barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    end_barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    end_barriers[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    g_dxr_backend.command_list->ResourceBarrier(2, end_barriers);

    if (!close_and_execute_command_list(out_submit_cpu_ms)) {
        return false;
    }
    g_dxr_backend.native_display_target_state = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    return true;
}

bool readback_output_texture_to_png(const wchar_t* path, int width, int height) {
    if (path == nullptr || width <= 0 || height <= 0 || g_dxr_backend.output_texture == nullptr) {
        return false;
    }
    if (!ensure_output_readback_buffer()) {
        return false;
    }

    if (!reset_command_list()) {
        return false;
    }

    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource = g_dxr_backend.output_texture;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.SubresourceIndex = 0;

    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = g_dxr_backend.output_readback_buffer;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = g_dxr_backend.output_readback_footprint;

    D3D12_RESOURCE_BARRIER begin_barrier{};
    begin_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    begin_barrier.Transition.pResource = g_dxr_backend.output_texture;
    begin_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    begin_barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    begin_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    g_dxr_backend.command_list->ResourceBarrier(1, &begin_barrier);
    g_dxr_backend.command_list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

    D3D12_RESOURCE_BARRIER end_barrier{};
    end_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    end_barrier.Transition.pResource = g_dxr_backend.output_texture;
    end_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    end_barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    end_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    g_dxr_backend.command_list->ResourceBarrier(1, &end_barrier);

    if (!close_and_execute_command_list()) {
        return false;
    }
    wait_for_gpu();

    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4u);
    void* mapped = nullptr;
    D3D12_RANGE read_range{0, static_cast<SIZE_T>(g_dxr_backend.output_readback_size)};
    const HRESULT map_hr = g_dxr_backend.output_readback_buffer->Map(0, &read_range, &mapped);
    if (FAILED(map_hr)) {
        record_dxr_failure("readback_output_texture_to_png.Map", map_hr);
        return false;
    }

    const std::uint8_t* src_bytes = static_cast<const std::uint8_t*>(mapped);
    const std::size_t dst_row_bytes = static_cast<std::size_t>(width) * 4u;
    for (int y = 0; y < height; ++y) {
        const std::size_t src_offset = static_cast<std::size_t>(y) *
            static_cast<std::size_t>(g_dxr_backend.output_readback_footprint.Footprint.RowPitch);
        const std::size_t dst_offset = static_cast<std::size_t>(y) * dst_row_bytes;
        const std::uint8_t* src_row = src_bytes + src_offset;
        std::uint8_t* dst_row = pixels.data() + dst_offset;
        std::memcpy(dst_row, src_row, dst_row_bytes);
    }
    g_dxr_backend.output_readback_buffer->Unmap(0, nullptr);
    return write_bgra_png(path, width, height, pixels);
}
bool readback_output_texture_to_bgra(int width, int height, std::vector<std::uint8_t>* out_pixels) {
    if (width <= 0 || height <= 0 || out_pixels == nullptr || g_dxr_backend.output_texture == nullptr) {
        return false;
    }
    if (!ensure_output_readback_buffer()) {
        return false;
    }

    if (!reset_command_list()) {
        return false;
    }

    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource = g_dxr_backend.output_texture;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.SubresourceIndex = 0;

    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = g_dxr_backend.output_readback_buffer;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = g_dxr_backend.output_readback_footprint;

    D3D12_RESOURCE_BARRIER begin_barrier{};
    begin_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    begin_barrier.Transition.pResource = g_dxr_backend.output_texture;
    begin_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    begin_barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    begin_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    g_dxr_backend.command_list->ResourceBarrier(1, &begin_barrier);
    g_dxr_backend.command_list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

    D3D12_RESOURCE_BARRIER end_barrier{};
    end_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    end_barrier.Transition.pResource = g_dxr_backend.output_texture;
    end_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    end_barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    end_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    g_dxr_backend.command_list->ResourceBarrier(1, &end_barrier);

    if (!close_and_execute_command_list()) {
        return false;
    }
    wait_for_gpu();

    out_pixels->resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4u);
    void* mapped = nullptr;
    D3D12_RANGE read_range{0, static_cast<SIZE_T>(g_dxr_backend.output_readback_size)};
    const HRESULT map_hr = g_dxr_backend.output_readback_buffer->Map(0, &read_range, &mapped);
    if (FAILED(map_hr)) {
        record_dxr_failure("readback_output_texture_to_bgra.Map", map_hr);
        return false;
    }

    const std::uint8_t* src_bytes = static_cast<const std::uint8_t*>(mapped);
    const std::size_t dst_row_bytes = static_cast<std::size_t>(width) * 4u;
    for (int y = 0; y < height; ++y) {
        const std::size_t src_offset = static_cast<std::size_t>(y) *
            static_cast<std::size_t>(g_dxr_backend.output_readback_footprint.Footprint.RowPitch);
        const std::size_t dst_offset = static_cast<std::size_t>(y) * dst_row_bytes;
        const std::uint8_t* src_row = src_bytes + src_offset;
        std::uint8_t* dst_row = out_pixels->data() + dst_offset;
        std::memcpy(dst_row, src_row, dst_row_bytes);
    }
    g_dxr_backend.output_readback_buffer->Unmap(0, nullptr);
    return true;
}
bool initialize_dxr_backend(const backend_config &config) {
    std::scoped_lock lock(g_dxr_backend_api_mutex);
    if (config.capture_width <= 0 || config.capture_height <= 0) {
        append_startup_log("DXR startup failed: invalid capture size");
        return false;
    }

    HRESULT hr = CreateDXGIFactory2(0, IID_PPV_ARGS(&g_dxr_backend.factory));
    if (FAILED(hr)) {
        append_startup_hresult_log("DXR startup failed: CreateDXGIFactory2", hr);
        return false;
    }

    if (config.d3d12.device != nullptr && config.d3d12.command_queue != nullptr) {
        IUnknown* device_unknown = static_cast<IUnknown*>(config.d3d12.device);
        IUnknown* queue_unknown = static_cast<IUnknown*>(config.d3d12.command_queue);
        hr = device_unknown->QueryInterface(IID_PPV_ARGS(&g_dxr_backend.device));
        if (FAILED(hr)) {
            append_startup_hresult_log("DXR startup failed: external device QueryInterface", hr);
            return false;
        }
        hr = queue_unknown->QueryInterface(IID_PPV_ARGS(&g_dxr_backend.queue));
        if (FAILED(hr)) {
            append_startup_hresult_log("DXR startup failed: external command queue QueryInterface", hr);
            return false;
        }
        g_dxr_backend.raytracing_supported = device_supports_raytracing(g_dxr_backend.device, &hr);
    } else {
        for (UINT index = 0;; ++index) {
            IDXGIAdapter1* candidate = nullptr;
            if (g_dxr_backend.factory->EnumAdapters1(index, &candidate) == DXGI_ERROR_NOT_FOUND) {
                break;
            }

            DXGI_ADAPTER_DESC1 desc{};
            candidate->GetDesc1(&desc);
            if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0) {
                candidate->Release();
                continue;
            }

            ID3D12Device5* candidate_device = nullptr;
            hr = D3D12CreateDevice(candidate, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&candidate_device));
            if (SUCCEEDED(hr) && device_supports_raytracing(candidate_device, &hr)) {
                g_dxr_backend.device = candidate_device;
                g_dxr_backend.adapter = candidate;
                g_dxr_backend.raytracing_supported = true;
                break;
            }
            safe_release(candidate_device);
            candidate->Release();
        }

        if (g_dxr_backend.device == nullptr) {
            IDXGIAdapter* warp_adapter = nullptr;
            hr = g_dxr_backend.factory->EnumWarpAdapter(IID_PPV_ARGS(&warp_adapter));
            if (FAILED(hr)) {
                append_startup_hresult_log("DXR startup failed: EnumWarpAdapter", hr);
                return false;
            }
            hr = D3D12CreateDevice(warp_adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&g_dxr_backend.device));
            warp_adapter->Release();
            if (FAILED(hr)) {
                append_startup_hresult_log("DXR startup failed: D3D12CreateDevice", hr);
                return false;
            }
            g_dxr_backend.raytracing_supported = device_supports_raytracing(g_dxr_backend.device, &hr);
        }
    }

    if (FAILED(hr)) {
        append_startup_hresult_log("DXR startup failed: CheckFeatureSupport D3D12_OPTIONS5", hr);
    }
    if (!g_dxr_backend.raytracing_supported) {
        append_startup_log("DXR startup failed: no D3D12 adapter with raytracing tier support was found");
        return false;
    }
    if (!ensure_required_uav_format_support(&hr)) {
        if (FAILED(hr)) {
            append_startup_hresult_log("DXR startup failed: CheckFeatureSupport D3D12_FORMAT_SUPPORT", hr);
        } else {
            char detail[512]{};
            std::snprintf(
                detail,
                sizeof(detail),
                "DXR startup failed: required UAV format support missing "
                "(bgra8_store=%u rgba16f_load=%u rgba16f_store=%u)",
                static_cast<unsigned>(g_dxr_backend.output_typed_uav_store_supported),
                static_cast<unsigned>(g_dxr_backend.accumulation_typed_uav_load_supported),
                static_cast<unsigned>(g_dxr_backend.accumulation_typed_uav_store_supported));
            append_startup_log(detail);
        }
        return false;
    }
    if (!load_shader_library()) {
        append_startup_log("DXR startup failed: load_shader_library");
        return false;
    }

    g_dxr_backend.native_d3d12_texture_present_supported =
        config.d3d12.device != nullptr &&
        config.d3d12.command_queue != nullptr &&
        g_dxr_backend.device != nullptr &&
        g_dxr_backend.queue != nullptr;

    if (g_dxr_backend.queue == nullptr) {
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        hr = g_dxr_backend.device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&g_dxr_backend.queue));
        if (FAILED(hr)) {
            append_startup_hresult_log("DXR startup failed: CreateCommandQueue", hr);
            return false;
        }
        g_dxr_backend.owns_device_queue = true;
        g_dxr_backend.native_d3d12_texture_present_supported = false;
    }

    for (UINT i = 0; i < kFrameCount; ++i) {
        hr = g_dxr_backend.device->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(&g_dxr_backend.command_allocators[i]));
        if (FAILED(hr)) {
            append_startup_hresult_log("DXR startup failed: CreateCommandAllocator", hr);
            return false;
        }
    }

    hr = g_dxr_backend.device->CreateCommandList(
        0,
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        g_dxr_backend.command_allocators[0],
        nullptr,
        IID_PPV_ARGS(&g_dxr_backend.command_list));
    if (FAILED(hr)) {
        append_startup_hresult_log("DXR startup failed: CreateCommandList", hr);
        return false;
    }
    g_dxr_backend.command_list->Close();

    hr = g_dxr_backend.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g_dxr_backend.fence));
    if (FAILED(hr)) {
        append_startup_hresult_log("DXR startup failed: CreateFence", hr);
        return false;
    }

    g_dxr_backend.fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (g_dxr_backend.fence_event == nullptr) {
        append_startup_log("DXR startup failed: CreateEventW");
        return false;
    }
    if (!ensure_timestamp_query_resources()) {
        append_startup_log("DXR startup failed: ensure_timestamp_query_resources");
        return false;
    }

    g_dxr_backend.config = config;
    g_dxr_backend.initialized = true;
    return true;
}

void shutdown_dxr_backend() {
    std::scoped_lock lock(g_dxr_backend_api_mutex);
    if (!g_dxr_backend.initialized &&
        g_dxr_backend.factory == nullptr &&
        g_dxr_backend.device == nullptr &&
        g_dxr_backend.queue == nullptr &&
        g_dxr_backend.command_list == nullptr &&
        g_dxr_backend.command_allocators[0] == nullptr &&
        g_dxr_backend.fence == nullptr) {
        return;
    }

    wait_for_gpu();
    clear_chunk_cache();
    destroy_swapchain_resources();
    release_shader_compiler_state();
    safe_release(g_dxr_backend.command_list);
    for (ID3D12CommandAllocator* &allocator : g_dxr_backend.command_allocators) {
        safe_release(allocator);
    }
    safe_release(g_dxr_backend.timestamp_query_readback);
    safe_release(g_dxr_backend.timestamp_query_heap);
    safe_release(g_dxr_backend.fence);
    safe_release(g_dxr_backend.queue);
    safe_release(g_dxr_backend.device);
    safe_release(g_dxr_backend.adapter);
    safe_release(g_dxr_backend.factory);
    if (g_dxr_backend.fence_event != nullptr) {
        CloseHandle(g_dxr_backend.fence_event);
        g_dxr_backend.fence_event = nullptr;
    }
    g_dxr_backend.timestamp_frequency = 0;
    g_dxr_backend.dispatch_timestamp_fence_values.fill(0);
    g_dxr_backend.dispatch_timestamp_pending.fill(false);
    g_dxr_backend = {};
}

bool render_dxr_to_window(viewer_shell::native_window_handle window, const frame_scene &scene, bool has_frame) {
    std::scoped_lock lock(g_dxr_backend_api_mutex);
    collect_completed_dispatch_timestamp_queries();
    HWND hwnd = as_hwnd(window);
    if (!g_dxr_backend.initialized || hwnd == nullptr) {
        return false;
    }
    if (!ensure_swapchain(hwnd)) {
        return false;
    }

    rt_scene_build build{};
    copy_present_render_rt_scene_build(&build);
    if (!sync_chunk_upload_cache(build)) {
        return false;
    }
    if (!sync_acceleration_structures(build)) {
        return false;
    }
    update_clear_color(scene, has_frame);

    g_dxr_backend.last_dispatch_ms = 0.0;
    g_dxr_backend.last_dispatch_submit_cpu_ms = 0.0;
    g_dxr_backend.last_dispatch_gpu_wait_ms = 0.0;
    g_dxr_backend.last_dispatch_gpu_ms = 0.0;
    g_dxr_backend.last_readback_ms = 0.0;
    bool dispatched = false;
    if (build.revision != 0 && has_renderable_primitives(build)) {
        dispatched = dispatch_rays_to_swapchain(scene, has_frame, &g_dxr_backend.last_dispatch_submit_cpu_ms);
        g_dxr_backend.last_dispatch_ms = g_dxr_backend.last_dispatch_submit_cpu_ms;
    }

    if (!dispatched) {
        if (!reset_command_list()) {
            return false;
        }

        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = g_dxr_backend.render_targets[g_dxr_backend.frame_index];
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        g_dxr_backend.command_list->ResourceBarrier(1, &barrier);

        D3D12_CPU_DESCRIPTOR_HANDLE rtv = g_dxr_backend.rtv_heap->GetCPUDescriptorHandleForHeapStart();
        rtv.ptr += static_cast<SIZE_T>(g_dxr_backend.frame_index) * g_dxr_backend.rtv_descriptor_size;
        g_dxr_backend.command_list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
        g_dxr_backend.command_list->ClearRenderTargetView(rtv, g_dxr_backend.clear_color, 0, nullptr);

        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
        g_dxr_backend.command_list->ResourceBarrier(1, &barrier);

        if (!close_and_execute_command_list()) {
            return false;
        }
    }

    const HRESULT present_hr = g_dxr_backend.swapchain->Present(1, 0);
    if (FAILED(present_hr)) {
        record_dxr_failure("render_dxr_to_window.Present", present_hr);
        return false;
    }

    if (!wait_for_fence_value_timed(g_dxr_backend.submitted_fence_value, &g_dxr_backend.last_dispatch_gpu_wait_ms)) {
        return false;
    }
    collect_completed_dispatch_timestamp_queries();
    g_dxr_backend.frame_index = g_dxr_backend.swapchain->GetCurrentBackBufferIndex();
    g_dxr_backend.last_dispatch_ms += g_dxr_backend.last_dispatch_gpu_wait_ms;
    return true;
}

bool render_dxr_to_native_d3d12_texture(
    int width,
    int height,
    const frame_scene &scene,
    bool has_frame,
    void* texture_resource)
{
    std::scoped_lock lock(g_dxr_backend_api_mutex);
    collect_completed_dispatch_timestamp_queries();
    if (width <= 0 || height <= 0 || texture_resource == nullptr || g_dxr_backend.device == nullptr || g_dxr_backend.queue == nullptr) {
        return false;
    }

    rt_scene_build build{};
    copy_present_render_rt_scene_build(&build);
    if (!sync_chunk_upload_cache(build)) {
        char detail[128]{};
        std::snprintf(
            detail,
            sizeof(detail),
            "chunk_error=%u chunk_index=%llu",
            static_cast<unsigned>(g_dxr_backend.last_chunk_sync_error),
            static_cast<unsigned long long>(g_dxr_backend.last_chunk_sync_error_index));
        record_dxr_failure("render_dxr_to_native_d3d12_texture.sync_chunk_upload_cache", S_OK, detail);
        return false;
    }
    if (!sync_acceleration_structures(build)) {
        record_dxr_failure("render_dxr_to_native_d3d12_texture.sync_acceleration_structures");
        return false;
    }
    update_clear_color(scene, has_frame);

    const UINT previous_width = g_dxr_backend.width;
    const UINT previous_height = g_dxr_backend.height;
    g_dxr_backend.width = static_cast<UINT>(width);
    g_dxr_backend.height = static_cast<UINT>(height);

    display_mode mode = display_mode::triangle_normal;
    get_display_mode(&mode);
    const accumulation_key next_key = make_accumulation_key(
        scene,
        has_frame,
        build,
        static_cast<UINT>(width),
        static_cast<UINT>(height),
        static_cast<std::uint32_t>(mode));
    if (!accumulation_key_equals(g_dxr_backend.accumulation_key, next_key)) {
        g_dxr_backend.accumulation_key = next_key;
        g_dxr_backend.accumulation_sample_count = 0;
    }

    if (g_dxr_backend.config.continuous_render &&
        g_dxr_backend.accumulation_sample_count >= kMaxAccumulationSamples) {
        g_dxr_backend.accumulation_sample_count = 0;
    }

    g_dxr_backend.accumulation_active = false;
    g_dxr_backend.last_dispatch_ms = 0.0;
    g_dxr_backend.last_dispatch_submit_cpu_ms = 0.0;
    g_dxr_backend.last_dispatch_gpu_wait_ms = 0.0;
    g_dxr_backend.last_readback_ms = 0.0;
    if (build.revision != 0 && has_renderable_primitives(build)) {
        const bool dispatched =
            dispatch_rays_to_output_texture(scene, has_frame, &g_dxr_backend.last_dispatch_submit_cpu_ms);
        if (!dispatched) {
            record_dxr_failure("render_dxr_to_native_d3d12_texture.dispatch_rays_to_output_texture");
            g_dxr_backend.width = previous_width;
            g_dxr_backend.height = previous_height;
            return false;
        }
        double copy_submit_cpu_ms = 0.0;
        if (!copy_output_texture_to_native_display_target(
                static_cast<ID3D12Resource*>(texture_resource),
                &copy_submit_cpu_ms)) {
            record_dxr_failure("render_dxr_to_native_d3d12_texture.copy_output_texture_to_native_display_target");
            g_dxr_backend.width = previous_width;
            g_dxr_backend.height = previous_height;
            return false;
        }
        g_dxr_backend.last_dispatch_submit_cpu_ms += copy_submit_cpu_ms;
        g_dxr_backend.last_dispatch_ms = g_dxr_backend.last_dispatch_submit_cpu_ms;
        if (!wait_for_fence_value_timed(
                g_dxr_backend.submitted_fence_value,
                &g_dxr_backend.last_dispatch_gpu_wait_ms)) {
            g_dxr_backend.width = previous_width;
            g_dxr_backend.height = previous_height;
            return false;
        }
        collect_completed_dispatch_timestamp_queries();
        g_dxr_backend.last_dispatch_ms += g_dxr_backend.last_dispatch_gpu_wait_ms;
        if (g_dxr_backend.accumulation_sample_count < kMaxAccumulationSamples) {
            ++g_dxr_backend.accumulation_sample_count;
        }
        g_dxr_backend.accumulation_active =
            g_dxr_backend.config.continuous_render ||
            g_dxr_backend.accumulation_sample_count < kMaxAccumulationSamples;
        g_dxr_backend.width = previous_width;
        g_dxr_backend.height = previous_height;
        return true;
    }

    g_dxr_backend.width = previous_width;
    g_dxr_backend.height = previous_height;
    return false;
}

bool capture_dxr_to_bgra(
    int width, int height,
    const frame_scene &scene,
    bool has_frame,
    std::vector<std::uint8_t>* out_pixels,
    bool update_build_info)
{
    (void)update_build_info;
    std::scoped_lock lock(g_dxr_backend_api_mutex);
    collect_completed_dispatch_timestamp_queries();
    if (width <= 0 || height <= 0 || out_pixels == nullptr) {
        return false;
    }

    rt_scene_build build{};
    copy_present_render_rt_scene_build(&build);
    if (!sync_chunk_upload_cache(build)) {
        char detail[128]{};
        std::snprintf(
            detail,
            sizeof(detail),
            "chunk_error=%u chunk_index=%llu",
            static_cast<unsigned>(g_dxr_backend.last_chunk_sync_error),
            static_cast<unsigned long long>(g_dxr_backend.last_chunk_sync_error_index));
        record_dxr_failure("capture_dxr_to_bgra.sync_chunk_upload_cache", S_OK, detail);
        return false;
    }
    if (!sync_acceleration_structures(build)) {
        record_dxr_failure("capture_dxr_to_bgra.sync_acceleration_structures");
        return false;
    }
    update_clear_color(scene, has_frame);

    const UINT previous_width = g_dxr_backend.width;
    const UINT previous_height = g_dxr_backend.height;
    g_dxr_backend.width = static_cast<UINT>(width);
    g_dxr_backend.height = static_cast<UINT>(height);

    display_mode mode = display_mode::triangle_normal;
    get_display_mode(&mode);
    const accumulation_key next_key = make_accumulation_key(
        scene,
        has_frame,
        build,
        static_cast<UINT>(width),
        static_cast<UINT>(height),
        static_cast<std::uint32_t>(mode));
    if (!accumulation_key_equals(g_dxr_backend.accumulation_key, next_key)) {
        g_dxr_backend.accumulation_key = next_key;
        g_dxr_backend.accumulation_sample_count = 0;
    }

    if (g_dxr_backend.config.continuous_render &&
        g_dxr_backend.accumulation_sample_count >= kMaxAccumulationSamples) {
        g_dxr_backend.accumulation_sample_count = 0;
    }

    bool wrote = false;
    g_dxr_backend.accumulation_active = false;
    g_dxr_backend.last_dispatch_ms = 0.0;
    g_dxr_backend.last_dispatch_submit_cpu_ms = 0.0;
    g_dxr_backend.last_dispatch_gpu_wait_ms = 0.0;
    g_dxr_backend.last_dispatch_gpu_ms = 0.0;
    g_dxr_backend.last_readback_ms = 0.0;
    if (build.revision != 0 && has_renderable_primitives(build)) {
        const bool dispatched =
            dispatch_rays_to_output_texture(scene, has_frame, &g_dxr_backend.last_dispatch_submit_cpu_ms);
        g_dxr_backend.last_dispatch_ms = g_dxr_backend.last_dispatch_submit_cpu_ms;
        if (!dispatched) {
            record_dxr_failure("capture_dxr_to_bgra.dispatch_rays_to_output_texture");
            g_dxr_backend.width = previous_width;
            g_dxr_backend.height = previous_height;
            return false;
        }
        if (g_dxr_backend.accumulation_sample_count < kMaxAccumulationSamples) {
            ++g_dxr_backend.accumulation_sample_count;
        }
        g_dxr_backend.accumulation_active =
            g_dxr_backend.config.continuous_render ||
            g_dxr_backend.accumulation_sample_count < kMaxAccumulationSamples;
        const auto readback_start = std::chrono::steady_clock::now();
        wrote = readback_output_texture_to_bgra(width, height, out_pixels);
        g_dxr_backend.last_readback_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - readback_start).count();
        if (!wrote) {
            record_dxr_failure("capture_dxr_to_bgra.readback_output_texture_to_bgra");
            g_dxr_backend.width = previous_width;
            g_dxr_backend.height = previous_height;
            return false;
        }
        collect_completed_dispatch_timestamp_queries();
    }

    g_dxr_backend.width = previous_width;
    g_dxr_backend.height = previous_height;

    if (wrote) {
        return true;
    }
    fill_solid_bgra(out_pixels, width, height, g_dxr_backend.clear_color);
    return true;
}
void fill_dxr_build_info(scene_build_info* out_info) {
    std::scoped_lock lock(g_dxr_backend_api_mutex);
    if (out_info == nullptr) {
        return;
    }
    out_info->blas_reused_count = g_dxr_backend.synced_blas_reused_count;
    out_info->blas_rebuilt_count = g_dxr_backend.synced_blas_rebuilt_count;
    out_info->blas_reused_chunk_count = g_dxr_backend.synced_blas_reused_chunk_count;
    out_info->blas_rebuilt_chunk_count = g_dxr_backend.synced_blas_rebuilt_chunk_count;
    out_info->tlas_rebuild_count = g_dxr_backend.synced_tlas_rebuild_count;
    out_info->accel_build_ms = g_dxr_backend.last_accel_build_ms;
    out_info->accel_host_prep_ms = g_dxr_backend.last_accel_host_prep_ms;
    out_info->accel_instance_build_ms = g_dxr_backend.last_accel_instance_build_ms;
    out_info->accel_procedural_aabb_ms = g_dxr_backend.last_accel_procedural_aabb_ms;
    out_info->accel_command_record_ms = g_dxr_backend.last_accel_command_record_ms;
    out_info->accel_resource_alloc_ms = g_dxr_backend.last_accel_resource_alloc_ms;
    out_info->accel_build_call_record_ms = g_dxr_backend.last_accel_build_call_record_ms;
    out_info->accel_prebuild_info_ms = g_dxr_backend.last_accel_prebuild_info_ms;
    out_info->accel_tlas_instance_upload_ms = g_dxr_backend.last_accel_tlas_instance_upload_ms;
    out_info->accel_submit_cpu_ms = g_dxr_backend.last_accel_submit_cpu_ms;
    out_info->accel_gpu_wait_ms = g_dxr_backend.last_accel_gpu_wait_ms;
    out_info->accel_gpu_ms = g_dxr_backend.last_accel_gpu_ms;
    out_info->dispatch_ms = g_dxr_backend.last_dispatch_ms;
    out_info->dispatch_submit_cpu_ms = g_dxr_backend.last_dispatch_submit_cpu_ms;
    out_info->dispatch_gpu_wait_ms = g_dxr_backend.last_dispatch_gpu_wait_ms;
    out_info->dispatch_gpu_ms = g_dxr_backend.last_dispatch_gpu_ms;
    out_info->readback_ms = g_dxr_backend.last_readback_ms;
    out_info->accumulation_sample_count = g_dxr_backend.accumulation_sample_count;
    out_info->accumulation_target_sample_count = kMaxAccumulationSamples;
    out_info->accumulation_in_progress = g_dxr_backend.accumulation_active;
}

bool capture_dxr_to_png(const wchar_t* path, int width, int height, const frame_scene &scene, bool has_frame) {
    std::scoped_lock lock(g_dxr_backend_api_mutex);
    collect_completed_dispatch_timestamp_queries();
    if (path == nullptr || width <= 0 || height <= 0) {
        return false;
    }

    rt_scene_build build{};
    copy_present_render_rt_scene_build(&build);
    if (!sync_chunk_upload_cache(build)) {
        return false;
    }
    if (!sync_acceleration_structures(build)) {
        return false;
    }
    update_clear_color(scene, has_frame);

    const UINT previous_width = g_dxr_backend.width;
    const UINT previous_height = g_dxr_backend.height;
    g_dxr_backend.width = static_cast<UINT>(width);
    g_dxr_backend.height = static_cast<UINT>(height);

    display_mode mode = display_mode::triangle_normal;
    get_display_mode(&mode);
    const accumulation_key next_key = make_accumulation_key(
        scene,
        has_frame,
        build,
        static_cast<UINT>(width),
        static_cast<UINT>(height),
        static_cast<std::uint32_t>(mode));
    if (!accumulation_key_equals(g_dxr_backend.accumulation_key, next_key)) {
        g_dxr_backend.accumulation_key = next_key;
        g_dxr_backend.accumulation_sample_count = 0;
    }

    if (g_dxr_backend.config.continuous_render &&
        g_dxr_backend.accumulation_sample_count >= kMaxAccumulationSamples) {
        g_dxr_backend.accumulation_sample_count = 0;
    }

    bool wrote = false;
    g_dxr_backend.accumulation_active = false;
    g_dxr_backend.last_dispatch_ms = 0.0;
    g_dxr_backend.last_dispatch_submit_cpu_ms = 0.0;
    g_dxr_backend.last_dispatch_gpu_wait_ms = 0.0;
    g_dxr_backend.last_dispatch_gpu_ms = 0.0;
    g_dxr_backend.last_readback_ms = 0.0;
    if (build.revision != 0 && !build.chunks.empty()) {
        const bool dispatched =
            dispatch_rays_to_output_texture(scene, has_frame, &g_dxr_backend.last_dispatch_submit_cpu_ms);
        g_dxr_backend.last_dispatch_ms = g_dxr_backend.last_dispatch_submit_cpu_ms;
        if (dispatched) {
            if (g_dxr_backend.accumulation_sample_count < kMaxAccumulationSamples) {
                ++g_dxr_backend.accumulation_sample_count;
            }
            g_dxr_backend.accumulation_active =
                g_dxr_backend.config.continuous_render ||
                g_dxr_backend.accumulation_sample_count < kMaxAccumulationSamples;
            const auto readback_start = std::chrono::steady_clock::now();
            wrote = readback_output_texture_to_png(path, width, height);
            g_dxr_backend.last_readback_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - readback_start).count();
            collect_completed_dispatch_timestamp_queries();
        }
    }

    g_dxr_backend.width = previous_width;
    g_dxr_backend.height = previous_height;

    if (wrote) {
        return true;
    }
    return write_solid_png(path, width, height, g_dxr_backend.clear_color);
}

bool pick_dxr(
    int width,
    int height,
    int pixel_x,
    int pixel_y,
    const frame_scene &scene,
    bool has_frame,
    pick_result* out_result)
{
    std::scoped_lock lock(g_dxr_backend_api_mutex);
    if (width <= 0 || height <= 0 || pixel_x < 0 || pixel_y < 0 || out_result == nullptr) {
        return false;
    }

    rt_scene_build build{};
    copy_present_render_rt_scene_build(&build);
    if (!sync_chunk_upload_cache(build) || !sync_acceleration_structures(build)) {
        return false;
    }

    const UINT previous_width = g_dxr_backend.width;
    const UINT previous_height = g_dxr_backend.height;
    g_dxr_backend.width = static_cast<UINT>(width);
    g_dxr_backend.height = static_cast<UINT>(height);

    pick_result result{};
    bool picked = false;
    if (build.revision != 0 && has_renderable_primitives(build)) {
        picked = dispatch_pick_query(width, height, pixel_x, pixel_y, scene, has_frame, &result);
    }

    g_dxr_backend.width = previous_width;
    g_dxr_backend.height = previous_height;
    if (!picked) {
        return false;
    }
    *out_result = result;
    return true;
}

bool dxr_accumulation_in_progress() {
    std::scoped_lock lock(g_dxr_backend_api_mutex);
    return g_dxr_backend.accumulation_active;
}

bool dxr_native_d3d12_texture_present_supported() {
    std::scoped_lock lock(g_dxr_backend_api_mutex);
    return g_dxr_backend.native_d3d12_texture_present_supported;
}

void dxr_notify_shell_post_present() {
    std::scoped_lock lock(g_dxr_backend_api_mutex);
    if (!g_dxr_backend.initialized || g_dxr_backend.queue == nullptr || g_dxr_backend.fence == nullptr) {
        return;
    }
    wait_for_gpu();
}

const backend_ops kDxrBackendOps{
    dxr_backend_info,
    initialize_dxr_backend,
    shutdown_dxr_backend,
    render_dxr_to_native_d3d12_texture,
    nullptr,
    nullptr,
    capture_dxr_to_bgra,
    capture_dxr_to_png,
    fill_dxr_build_info,
    pick_dxr,
    dxr_accumulation_in_progress,
    dxr_native_d3d12_texture_present_supported,
    nullptr,
    dxr_notify_shell_post_present,
};

} // namespace

const backend_ops* d3d12_dxr_backend_ops() {
    return &kDxrBackendOps;
}

} // namespace rtvdb::viewer_backend
