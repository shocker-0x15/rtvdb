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
#include "viewer_backend/rt_backend_common.h"
#include "viewer_backend/rt_object_registry.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

namespace rtvdb::viewer_backend {
namespace {

constexpr UINT kTimestampQueryCountPerRegion = 2;
constexpr UINT kTimestampQueryRegionCount = 2;
constexpr UINT kTimestampQueryCountPerCommandSlot =
    kTimestampQueryCountPerRegion * kTimestampQueryRegionCount;
constexpr UINT kDescriptorBindingCapacity = 11;
constexpr UINT kUavDescriptorBase = 0;
constexpr UINT kSrvDescriptorBase = kUavDescriptorBase + kDescriptorBindingCapacity;
constexpr UINT kScratchUavDescriptorIndex = kSrvDescriptorBase + kDescriptorBindingCapacity;
constexpr UINT kDescriptorSlotStride = kScratchUavDescriptorIndex + 1;
constexpr UINT kDescriptorHeapCount = kDescriptorSlotStride * kRtCommandSlotCount;
constexpr std::size_t kShaderRecordSize = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
constexpr std::size_t kInvalidCacheIndex = (std::numeric_limits<std::size_t>::max)();
constexpr DXGI_FORMAT kOutputFormat = DXGI_FORMAT_B8G8R8A8_UNORM;

static_assert(sizeof(rt_scene_gpu_aabb) == sizeof(D3D12_RAYTRACING_AABB));

struct dxr_buffer {
    ID3D12Resource* resource = nullptr;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
};

struct dxr_acceleration_structure {
    ID3D12Resource* result = nullptr;
    std::size_t instance_count = 0;
};

struct dxr_texture {
    ID3D12Resource* resource = nullptr;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
    rt_texture_desc desc{};
};

struct dxr_deferred_resource_release {
    rt_submission_token submission{};
    ID3D12Resource* resource = nullptr;
};

struct dxr_blas_build_record {
    std::array<D3D12_RAYTRACING_GEOMETRY_DESC, kRtBlasChunkSetChunkCount> geometry_descs{};
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs{};
    ID3D12Resource* destination = nullptr;
    UINT64 scratch_size = 0;
};

struct dxr_acceleration_build_context {
    bool active = false;
    bool recording = false;
    bool recorded = false;
    UINT timestamp_slot = 0;
    UINT64 max_scratch_size = 0;
    std::chrono::steady_clock::time_point total_start{};
    std::chrono::steady_clock::time_point command_start{};
    std::vector<dxr_blas_build_record> blas_builds;
};

struct dxr_command_slot {
    ID3D12CommandAllocator* allocator = nullptr;
    ID3D12Resource* tlas_instance_buffer = nullptr;
    std::uint64_t fence_value = 0;
    rt_submission_token submission{};
    rt_submission_token dispatch_timestamp_submission{};
    bool dispatch_timestamp_pending = false;
    rt_submission_token accel_timestamp_submission{};
    bool accel_timestamp_pending = false;
};

struct dxr_backend_state {
    bool initialized = false;
    bool raytracing_supported = false;
    bool output_typed_uav_store_supported = false;
    bool accumulation_typed_uav_load_supported = false;
    bool accumulation_typed_uav_store_supported = false;
    bool native_d3d12_texture_present_supported = false;
    UINT width = 0;
    UINT height = 0;
    UINT output_width = 0;
    UINT output_height = 0;
    float clear_color[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    rt_rhi_diagnostics diagnostics{};
    rt_object_registry<dxr_buffer, rt_buffer_handle> buffer_registry;
    rt_object_registry<dxr_texture, rt_texture_handle> texture_registry;
    rt_object_registry<dxr_acceleration_structure, rt_blas_handle> blas_registry;
    rt_object_registry<dxr_acceleration_structure, rt_tlas_handle> tlas_registry;
    rt_object_registry<std::vector<std::uint8_t>, rt_shader_module_handle> shader_module_registry;
    rt_object_registry<ID3D12StateObject*, rt_pipeline_handle> pipeline_registry;
    rt_object_registry<ID3D12Resource*, rt_shader_table_handle> shader_table_registry;
    std::vector<dxr_deferred_resource_release> deferred_resource_releases;
    rt_tlas_handle tlas{};
    rt_pipeline_handle pipeline{};
    rt_shader_table_handle shader_table{};
    dxr_acceleration_build_context acceleration_build{};

    IDXGIFactory7* factory = nullptr;
    IDXGIAdapter1* adapter = nullptr;
    ID3D12Device5* device = nullptr;
    ID3D12CommandQueue* queue = nullptr;
    std::array<dxr_command_slot, kRtCommandSlotCount> command_slots{};
    UINT command_slot_index = 0;
    UINT active_command_slot_index = 0;
    ID3D12GraphicsCommandList4* command_list = nullptr;
    ID3D12Resource* scratch_buffer = nullptr;
    ID3D12Fence* fence = nullptr;
    std::uint64_t submitted_fence_value = 0;
    std::uint64_t next_submission_serial = 1;
    std::uint64_t completed_submission_serial = 0;
    std::uint64_t next_encoder_id = 1;
    std::uint64_t active_encoder_id = 0;
    bool active_dispatch_timestamp_recorded = false;
    HANDLE fence_event = nullptr;
    ID3D12QueryHeap* timestamp_query_heap = nullptr;
    ID3D12Resource* timestamp_query_readback = nullptr;
    std::uint64_t timestamp_frequency = 0;
    std::vector<rt_binding_write> pending_binding_writes;
    std::uint64_t binding_generation = 0;
    std::array<std::uint64_t, kRtCommandSlotCount> command_slot_binding_generations{};

    ID3D12DescriptorHeap* srv_uav_cbv_heap = nullptr;
    UINT srv_uav_cbv_descriptor_size = 0;
    ID3D12RootSignature* global_root_signature = nullptr;
    ID3D12StateObject* raytracing_state_object = nullptr;
    ID3D12StateObjectProperties* raytracing_state_props = nullptr;
    ID3D12Resource* native_display_target = nullptr;
    D3D12_RESOURCE_STATES native_display_target_state = D3D12_RESOURCE_STATE_COPY_DEST;
    rt_buffer_handle camera_constant_buffer_handle{};
    ID3D12Resource* camera_constant_buffer = nullptr;
    ID3D12Resource* raygen_shader_table = nullptr;
    ID3D12Resource* miss_shader_table = nullptr;
    ID3D12Resource* hitgroup_shader_table = nullptr;
    ID3D12Resource* callable_shader_table = nullptr;
    std::size_t raygen_shader_record_count = 0;
    std::size_t miss_shader_record_count = 0;
    std::size_t hitgroup_shader_record_count = 0;
    std::size_t callable_shader_record_count = 0;
    std::vector<std::wstring> shader_group_export_names;

    bool owns_device_queue = false;
};

class d3d12_dxr_rhi_device final :
    public rt_rhi_device,
    public rt_native_texture_extension
{
public:
    d3d12_dxr_rhi_device();
    ~d3d12_dxr_rhi_device() override;

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

private:
    dxr_backend_state native_state_{};
    rt_device device_{};
};

enum class timestamp_query_region : UINT {
    accel = 0,
    dispatch = 2,
};

bool timestamp_queries_enabled(const dxr_backend_state &state);
UINT current_timestamp_query_slot(dxr_backend_state &state);
bool ensure_timestamp_query_resources(dxr_backend_state &state);
void write_timestamp_query_begin(
    dxr_backend_state &state,
    UINT slot,
    timestamp_query_region region);
void write_timestamp_query_end_and_resolve(
    dxr_backend_state &state,
    UINT slot,
    timestamp_query_region region);
bool read_timestamp_query_ms(
    dxr_backend_state &state,
    UINT slot,
    timestamp_query_region region,
    double* out_ms);
void collect_completed_dispatch_timestamp_queries(dxr_backend_state &state);
bool reset_command_list(dxr_backend_state &state);
bool close_and_execute_command_list(
    dxr_backend_state &state,
    double* out_submit_cpu_ms = nullptr,
    rt_submission_token* out_submission = nullptr);

void append_startup_hresult_log(const char* message, HRESULT hr) {
    char line[512]{};
    std::snprintf(line, sizeof(line), "%s hr=0x%08lx", message != nullptr ? message : "DXR startup failed", hr);
    append_rt_startup_log(line);
}

void append_device_removed_reason_log(const char* message, ID3D12Device* device) {
    if (device == nullptr) {
        return;
    }
    const HRESULT reason = device->GetDeviceRemovedReason();
    if (FAILED(reason)) {
        append_startup_hresult_log(message, reason);
    }
}

void append_dred_breadcrumb_log(ID3D12Device* device) {
    if (device == nullptr) {
        return;
    }
    ID3D12DeviceRemovedExtendedData1* dred = nullptr;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&dred))) || dred == nullptr) {
        return;
    }
    D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT1 breadcrumbs{};
    if (SUCCEEDED(dred->GetAutoBreadcrumbsOutput1(&breadcrumbs))) {
        std::size_t node_count = 0;
        for (const D3D12_AUTO_BREADCRUMB_NODE1* node = breadcrumbs.pHeadAutoBreadcrumbNode;
             node != nullptr && node_count < 8;
             node = node->pNext) {
            const UINT last_value = node->pLastBreadcrumbValue != nullptr ? *node->pLastBreadcrumbValue : 0u;
            char line[256]{};
            std::snprintf(
                line,
                sizeof(line),
                "DXR DRED breadcrumb node=%zu command_count=%u last=%u",
                node_count,
                node->BreadcrumbCount,
                last_value);
            append_rt_diagnostics_log_line("dxr_failure.log", line);
            ++node_count;
        }
    }
    dred->Release();
}

void record_dxr_failure(dxr_backend_state &state, const char* stage, HRESULT hr = S_OK, const char* detail = nullptr) {
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
    append_rt_diagnostics_log_line("dxr_failure.log", line);
    if (hr == DXGI_ERROR_DEVICE_HUNG || hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET) {
        append_dred_breadcrumb_log(state.device);
    }
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

bool ensure_required_uav_format_support(dxr_backend_state &state, HRESULT* out_hr) {
    D3D12_FORMAT_SUPPORT1 output_support1 = D3D12_FORMAT_SUPPORT1_NONE;
    D3D12_FORMAT_SUPPORT2 output_support2 = D3D12_FORMAT_SUPPORT2_NONE;
    HRESULT hr = S_OK;
    if (!query_format_support(
            state.device,
            kOutputFormat,
            &output_support1,
            &output_support2,
            &hr)) {
        if (out_hr != nullptr) {
            *out_hr = hr;
        }
        return false;
    }
    state.output_typed_uav_store_supported =
        (output_support1 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW) != 0 &&
        (output_support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE) != 0;

    D3D12_FORMAT_SUPPORT1 accumulation_support1 = D3D12_FORMAT_SUPPORT1_NONE;
    D3D12_FORMAT_SUPPORT2 accumulation_support2 = D3D12_FORMAT_SUPPORT2_NONE;
    if (!query_format_support(
            state.device,
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
    state.accumulation_typed_uav_load_supported =
        accumulation_has_uav &&
        (accumulation_support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD) != 0;
    state.accumulation_typed_uav_store_supported =
        accumulation_has_uav &&
        (accumulation_support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE) != 0;

    if (out_hr != nullptr) {
        *out_hr = S_OK;
    }
    return state.output_typed_uav_store_supported &&
        state.accumulation_typed_uav_load_supported &&
        state.accumulation_typed_uav_store_supported;
}

// =============================================================================
// Resource lifetime and allocation.
// =============================================================================

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

rt_submission_token latest_submission(const dxr_backend_state &state) {
    return state.next_submission_serial > 1
        ? rt_submission_token{state.next_submission_serial - 1u}
        : rt_submission_token{};
}

void defer_resource_release(dxr_backend_state &state, ID3D12Resource* &resource) {
    if (resource == nullptr) {
        return;
    }
    const rt_submission_token submission = state.active_encoder_id != 0
        ? rt_submission_token{state.next_submission_serial}
        : latest_submission(state);
    if (!submission || submission.serial <= state.completed_submission_serial) {
        safe_release(resource);
        return;
    }
    state.deferred_resource_releases.push_back({submission, resource});
    resource = nullptr;
}

void collect_deferred_resource_releases(dxr_backend_state &state, bool release_all = false) {
    auto &releases = state.deferred_resource_releases;
    for (std::size_t index = 0; index < releases.size();) {
        dxr_deferred_resource_release &release = releases[index];
        if (!release_all && release.submission.serial > state.completed_submission_serial) {
            ++index;
            continue;
        }
        safe_release(release.resource);
        releases[index] = releases.back();
        releases.pop_back();
    }
}

void release_acceleration_structure(dxr_acceleration_structure* object) {
    if (object == nullptr) {
        return;
    }
    safe_release(object->result);
}

void defer_acceleration_structure(dxr_backend_state &state, dxr_acceleration_structure* object) {
    if (object == nullptr) {
        return;
    }
    defer_resource_release(state, object->result);
    object->instance_count = 0;
}

void release_tlas_handle(dxr_backend_state &state) {
    dxr_acceleration_structure object{};
    if (state.tlas_registry.erase(state.tlas, &object)) {
        defer_acceleration_structure(state, &object);
    }
    state.tlas = {};
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1u) / alignment * alignment;
}

rt_rhi_device_info dxr_backend_info(const dxr_backend_state &state) {
    return {
        rt_rhi_backend_kind::d3d12_dxr,
        "d3d12_dxr",
        state.raytracing_supported,
    };
}

bool wait_for_fence_value(dxr_backend_state &state, std::uint64_t value) {
    if (value == 0) {
        return true;
    }
    if (state.fence == nullptr || state.fence_event == nullptr) {
        return false;
    }
    if (state.fence->GetCompletedValue() >= value) {
        return true;
    }
    if (FAILED(state.fence->SetEventOnCompletion(value, state.fence_event))) {
        return false;
    }
    return WaitForSingleObject(state.fence_event, INFINITE) == WAIT_OBJECT_0;
}

bool wait_for_fence_value_timed(dxr_backend_state &state, std::uint64_t value, double* out_wait_ms) {
    const auto wait_start = std::chrono::steady_clock::now();
    const bool waited = wait_for_fence_value(state, value);
    if (out_wait_ms != nullptr) {
        *out_wait_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - wait_start).count();
    }
    return waited;
}

void complete_command_slot(dxr_backend_state &state, UINT slot_index) {
    dxr_command_slot &slot = state.command_slots[slot_index];
    state.completed_submission_serial = (std::max)(
        state.completed_submission_serial,
        slot.submission.serial);
    slot.submission = {};
    collect_deferred_resource_releases(state);
}

bool is_submission_complete(dxr_backend_state &state, rt_submission_token submission) {
    if (!submission || submission.serial <= state.completed_submission_serial) {
        return true;
    }
    for (std::size_t slot = 0; slot < kRtCommandSlotCount; ++slot) {
        if (state.command_slots[slot].submission != submission) {
            continue;
        }
        const std::uint64_t fence_value = state.command_slots[slot].fence_value;
        if (state.fence != nullptr && state.fence->GetCompletedValue() >= fence_value) {
            complete_command_slot(state, static_cast<UINT>(slot));
            return true;
        }
        return false;
    }
    return false;
}

bool wait_for_submission(
    dxr_backend_state &state,
    rt_submission_token submission,
    double* out_wait_ms = nullptr)
{
    const auto wait_start = std::chrono::steady_clock::now();
    bool completed = is_submission_complete(state, submission);
    if (!completed) {
        for (std::size_t slot = 0; slot < kRtCommandSlotCount; ++slot) {
            if (state.command_slots[slot].submission != submission) {
                continue;
            }
            completed = wait_for_fence_value(state, state.command_slots[slot].fence_value);
            if (completed) {
                complete_command_slot(state, static_cast<UINT>(slot));
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

UINT current_timestamp_query_slot(dxr_backend_state &state) {
    return state.command_slot_index % kRtCommandSlotCount;
}

bool timestamp_queries_enabled(const dxr_backend_state &state) {
    return state.timestamp_query_heap != nullptr &&
        state.timestamp_query_readback != nullptr &&
        state.timestamp_frequency != 0;
}

bool ensure_timestamp_query_resources(dxr_backend_state &state) {
    if (state.device == nullptr || state.queue == nullptr) {
        return false;
    }
    if (timestamp_queries_enabled(state)) {
        return true;
    }

    safe_release(state.timestamp_query_heap);
    safe_release(state.timestamp_query_readback);
    state.timestamp_frequency = 0;
    for (dxr_command_slot &slot : state.command_slots) {
        slot.dispatch_timestamp_submission = {};
        slot.dispatch_timestamp_pending = false;
        slot.accel_timestamp_submission = {};
        slot.accel_timestamp_pending = false;
    }

    D3D12_QUERY_HEAP_DESC heap_desc{};
    heap_desc.Count = kRtCommandSlotCount * kTimestampQueryCountPerCommandSlot;
    heap_desc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    HRESULT hr = state.device->CreateQueryHeap(&heap_desc, IID_PPV_ARGS(&state.timestamp_query_heap));
    if (FAILED(hr)) {
        record_dxr_failure(state, "ensure_timestamp_query_resources.CreateQueryHeap", hr);
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
    hr = state.device->CreateCommittedResource(
        &heap_properties,
        D3D12_HEAP_FLAG_NONE,
        &resource_desc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(&state.timestamp_query_readback));
    if (FAILED(hr)) {
        record_dxr_failure(state, "ensure_timestamp_query_resources.CreateCommittedResource", hr);
        safe_release(state.timestamp_query_heap);
        return false;
    }

    hr = state.queue->GetTimestampFrequency(&state.timestamp_frequency);
    if (FAILED(hr) || state.timestamp_frequency == 0) {
        record_dxr_failure(state, "ensure_timestamp_query_resources.GetTimestampFrequency", hr);
        safe_release(state.timestamp_query_heap);
        safe_release(state.timestamp_query_readback);
        state.timestamp_frequency = 0;
        return false;
    }
    return true;
}

void write_timestamp_query_begin(
    dxr_backend_state &state,
    UINT slot,
    timestamp_query_region region)
{
    if (!timestamp_queries_enabled(state) || state.command_list == nullptr || slot >= kRtCommandSlotCount) {
        return;
    }
    state.command_list->EndQuery(
        state.timestamp_query_heap,
        D3D12_QUERY_TYPE_TIMESTAMP,
        slot * kTimestampQueryCountPerCommandSlot + static_cast<UINT>(region));
}

void write_timestamp_query_end_and_resolve(
    dxr_backend_state &state,
    UINT slot,
    timestamp_query_region region)
{
    if (!timestamp_queries_enabled(state) ||
        state.command_list == nullptr ||
        slot >= kRtCommandSlotCount) {
        return;
    }
    const UINT start_index = slot * kTimestampQueryCountPerCommandSlot + static_cast<UINT>(region);
    state.command_list->EndQuery(state.timestamp_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, start_index + 1u);
    state.command_list->ResolveQueryData(
        state.timestamp_query_heap,
        D3D12_QUERY_TYPE_TIMESTAMP,
        start_index,
        kTimestampQueryCountPerRegion,
        state.timestamp_query_readback,
        static_cast<UINT64>(start_index) * sizeof(std::uint64_t));
}

bool read_timestamp_query_ms(
    dxr_backend_state &state,
    UINT slot,
    timestamp_query_region region,
    double* out_ms)
{
    if (out_ms == nullptr ||
        !timestamp_queries_enabled(state) ||
        slot >= kRtCommandSlotCount) {
        return false;
    }

    void* mapped = nullptr;
    const UINT64 offset_bytes = static_cast<UINT64>(
        slot * kTimestampQueryCountPerCommandSlot + static_cast<UINT>(region)) * sizeof(std::uint64_t);
    D3D12_RANGE read_range{
        offset_bytes,
        offset_bytes + sizeof(std::uint64_t) * kTimestampQueryCountPerRegion};
    const HRESULT hr = state.timestamp_query_readback->Map(0, &read_range, &mapped);
    if (FAILED(hr) || mapped == nullptr) {
        record_dxr_failure(state, "read_timestamp_query_ms.Map", hr);
        return false;
    }

    const auto* timestamps =
        reinterpret_cast<const std::uint64_t*>(static_cast<const std::uint8_t*>(mapped) + offset_bytes);
    const std::uint64_t start = timestamps[0];
    const std::uint64_t end = timestamps[1];
    state.timestamp_query_readback->Unmap(0, nullptr);

    if (end <= start) {
        return false;
    }

    *out_ms = static_cast<double>(end - start) * 1000.0 / static_cast<double>(state.timestamp_frequency);
    return true;
}

void collect_completed_dispatch_timestamp_queries(dxr_backend_state &state) {
    if (state.fence == nullptr) {
        return;
    }

    for (UINT slot = 0; slot < kRtCommandSlotCount; ++slot) {
        if (state.command_slots[slot].dispatch_timestamp_pending &&
            is_submission_complete(state, state.command_slots[slot].dispatch_timestamp_submission)) {
            double measured_ms = 0.0;
            if (read_timestamp_query_ms(state, slot, timestamp_query_region::dispatch, &measured_ms)) {
                state.diagnostics.dispatch_gpu_ms = measured_ms;
            }
            state.command_slots[slot].dispatch_timestamp_pending = false;
            state.command_slots[slot].dispatch_timestamp_submission = {};
        }
        if (state.command_slots[slot].accel_timestamp_pending &&
            is_submission_complete(state, state.command_slots[slot].accel_timestamp_submission)) {
            double measured_ms = 0.0;
            if (read_timestamp_query_ms(state, slot, timestamp_query_region::accel, &measured_ms)) {
                state.diagnostics.acceleration_gpu_ms = measured_ms;
            }
            state.command_slots[slot].accel_timestamp_pending = false;
            state.command_slots[slot].accel_timestamp_submission = {};
        }
    }
}

bool create_native_buffer(
    dxr_backend_state &state,
    D3D12_HEAP_TYPE heap_type,
    std::size_t size_bytes,
    D3D12_RESOURCE_STATES initial_state,
    D3D12_RESOURCE_FLAGS flags,
    ID3D12Resource** out_resource)
{
    if (out_resource == nullptr || state.device == nullptr) {
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

    const HRESULT hr = state.device->CreateCommittedResource(
        &heap_properties,
        D3D12_HEAP_FLAG_NONE,
        &resource_desc,
        initial_state,
        nullptr,
        IID_PPV_ARGS(out_resource));
    if (FAILED(hr)) {
        record_dxr_failure(state, "create_buffer.CreateCommittedResource", hr);
    }
    return SUCCEEDED(hr);
}

bool ensure_buffer_capacity(
    dxr_backend_state &state,
    D3D12_HEAP_TYPE heap_type,
    std::size_t size_bytes,
    D3D12_RESOURCE_STATES initial_state,
    D3D12_RESOURCE_FLAGS flags,
    ID3D12Resource** out_resource)
{
    if (out_resource == nullptr) {
        return false;
    }
    if (*out_resource != nullptr && (*out_resource)->GetDesc().Width >= size_bytes) {
        return true;
    }
    defer_resource_release(state, *out_resource);
    return create_native_buffer(state, heap_type, size_bytes, initial_state, flags, out_resource);
}

bool upload_buffer_data(
    dxr_backend_state &state,
    ID3D12Resource* resource,
    std::size_t offset,
    const void* data,
    std::size_t size_bytes)
{
    if (size_bytes == 0) {
        return true;
    }
    if (resource == nullptr || data == nullptr ||
        offset > resource->GetDesc().Width ||
        size_bytes > resource->GetDesc().Width - offset) {
        return false;
    }

    void* mapped = nullptr;
    D3D12_RANGE read_range{0, 0};
    const HRESULT hr = resource->Map(0, &read_range, &mapped);
    if (FAILED(hr)) {
        record_dxr_failure(state, "upload_buffer_data.Map", hr);
        return false;
    }
    std::memcpy(static_cast<std::uint8_t*>(mapped) + offset, data, size_bytes);
    D3D12_RANGE written_range{offset, offset + size_bytes};
    resource->Unmap(0, &written_range);
    return true;
}

ID3D12Resource* dxr_buffer_resource(dxr_backend_state &state, rt_buffer_handle handle) {
    dxr_buffer* const resource = state.buffer_registry.get(handle);
    return resource != nullptr ? resource->resource : nullptr;
}

bool d3d12_dxr_rhi_device::create_buffer(
    const rt_buffer_desc &desc,
    rt_buffer_handle* out_buffer,
    rt_device_error* out_error)
{
    if (out_buffer != nullptr) {
        *out_buffer = {};
    }
    if (device_.native_state != &native_state_ || out_buffer == nullptr) {
        if (out_error != nullptr) {
            out_error->detail = "DXR API buffer request is invalid";
        }
        return false;
    }

    if (desc.size == 0) {
        return true;
    }
    const D3D12_HEAP_TYPE heap_type =
        desc.memory_domain == rt_memory_domain::device
            ? D3D12_HEAP_TYPE_DEFAULT
            : desc.memory_domain == rt_memory_domain::readback
                ? D3D12_HEAP_TYPE_READBACK
                : D3D12_HEAP_TYPE_UPLOAD;
    const D3D12_RESOURCE_STATES initial_state =
        desc.memory_domain == rt_memory_domain::device
            ? D3D12_RESOURCE_STATE_COMMON
            : desc.memory_domain == rt_memory_domain::readback
                ? D3D12_RESOURCE_STATE_COPY_DEST
                : D3D12_RESOURCE_STATE_GENERIC_READ;
    const D3D12_RESOURCE_FLAGS flags =
        (desc.usage & rt_buffer_usage_shader_write) != 0u
            ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
            : D3D12_RESOURCE_FLAG_NONE;
    ID3D12Resource* resource = nullptr;
    if (!create_native_buffer(
            native_state_,
            heap_type,
            desc.size,
            initial_state,
            flags,
            &resource)) {
        if (out_error != nullptr) {
            out_error->detail = "DXR buffer creation failed";
        }
        return false;
    }
    if (!native_state_.buffer_registry.insert({resource, initial_state}, out_buffer)) {
        safe_release(resource);
        if (out_error != nullptr) {
            out_error->detail = "DXR buffer registry allocation failed";
        }
        return false;
    }
    return static_cast<bool>(*out_buffer);
}

bool d3d12_dxr_rhi_device::upload_buffer(
    rt_buffer_handle buffer,
    std::size_t offset,
    const void* data,
    std::size_t size,
    rt_device_error* out_error)
{
    ID3D12Resource* const resource = dxr_buffer_resource(native_state_, buffer);
    const bool uploaded = device_.native_state == &native_state_ &&
        (size == 0 || (resource != nullptr &&
            upload_buffer_data(native_state_, resource, offset, data, size)));
    if (!uploaded && out_error != nullptr) {
        out_error->detail = "DXR buffer upload failed";
    }
    return uploaded;
}

bool d3d12_dxr_rhi_device::read_buffer(
    rt_buffer_handle buffer,
    std::size_t offset,
    void* data,
    std::size_t size,
    rt_device_error* out_error)
{
    ID3D12Resource* const resource = dxr_buffer_resource(native_state_, buffer);
    const bool valid = device_.native_state == &native_state_ &&
        resource != nullptr && data != nullptr && size > 0 &&
        offset <= resource->GetDesc().Width &&
        size <= resource->GetDesc().Width - offset;
    if (!valid) {
        if (out_error != nullptr) {
            *out_error = {rt_device_operation::readback, 0, "DXR buffer read request is invalid"};
        }
        return false;
    }
    void* mapped = nullptr;
    const D3D12_RANGE read_range{offset, offset + size};
    const HRESULT hr = resource->Map(0, &read_range, &mapped);
    if (FAILED(hr) || mapped == nullptr) {
        if (out_error != nullptr) {
            *out_error = {rt_device_operation::readback, hr, "DXR buffer map failed"};
        }
        return false;
    }
    std::memcpy(data, static_cast<const std::uint8_t*>(mapped) + offset, size);
    resource->Unmap(0, nullptr);
    return true;
}

void d3d12_dxr_rhi_device::destroy_buffer(rt_buffer_handle buffer) {
    if (device_.native_state != &native_state_) {
        return;
    }
    if (native_state_.camera_constant_buffer_handle == buffer) {
        native_state_.camera_constant_buffer_handle = {};
        native_state_.camera_constant_buffer = nullptr;
    }
    dxr_buffer resource{};
    if (native_state_.buffer_registry.erase(buffer, &resource)) {
        defer_resource_release(native_state_, resource.resource);
    }
}

DXGI_FORMAT dxr_texture_format(rt_texture_format format) {
    switch (format) {
    case rt_texture_format::rgba8_unorm:
        return DXGI_FORMAT_R8G8B8A8_UNORM;
    case rt_texture_format::bgra8_unorm:
        return DXGI_FORMAT_B8G8R8A8_UNORM;
    case rt_texture_format::rgba16_float:
        return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case rt_texture_format::rgba32_float:
        return DXGI_FORMAT_R32G32B32A32_FLOAT;
    }
    return DXGI_FORMAT_UNKNOWN;
}

bool d3d12_dxr_rhi_device::create_texture(
    const rt_texture_desc &desc,
    rt_texture_handle* out_texture,
    rt_device_error* out_error)
{
    if (out_texture != nullptr) {
        *out_texture = {};
    }
    const DXGI_FORMAT format = dxr_texture_format(desc.format);
    if (device_.native_state != &native_state_ ||
        out_texture == nullptr || desc.width == 0 || desc.height == 0 ||
        desc.usage == 0 || format == DXGI_FORMAT_UNKNOWN) {
        if (out_error != nullptr) {
            out_error->detail = "DXR texture request is invalid";
        }
        return false;
    }

    D3D12_HEAP_PROPERTIES heap_properties{};
    heap_properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC resource_desc{};
    resource_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    resource_desc.Width = desc.width;
    resource_desc.Height = desc.height;
    resource_desc.DepthOrArraySize = 1;
    resource_desc.MipLevels = 1;
    resource_desc.Format = format;
    resource_desc.SampleDesc.Count = 1;
    resource_desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    resource_desc.Flags = (desc.usage & rt_texture_usage_shader_write) != 0u
        ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
        : D3D12_RESOURCE_FLAG_NONE;

    dxr_texture texture{};
    texture.desc = desc;
    const HRESULT hr = native_state_.device->CreateCommittedResource(
        &heap_properties,
        D3D12_HEAP_FLAG_NONE,
        &resource_desc,
        texture.state,
        nullptr,
        IID_PPV_ARGS(&texture.resource));
    if (FAILED(hr) || !native_state_.texture_registry.insert(texture, out_texture)) {
        safe_release(texture.resource);
        if (out_error != nullptr) {
            out_error->native_code = static_cast<std::int64_t>(hr);
            out_error->detail = "DXR texture creation failed";
        }
        return false;
    }
    return true;
}

void d3d12_dxr_rhi_device::destroy_texture(rt_texture_handle handle) {
    if (device_.native_state != &native_state_) {
        return;
    }
    dxr_texture texture{};
    if (native_state_.texture_registry.erase(handle, &texture)) {
        defer_resource_release(native_state_, texture.resource);
    }
}

bool d3d12_dxr_rhi_device::get_texture_copy_footprint(
    rt_texture_handle handle,
    rt_texture_copy_footprint* out_footprint,
    rt_device_error* out_error)
{
    if (out_footprint != nullptr) {
        *out_footprint = {};
    }
    const dxr_texture* const texture = native_state_.texture_registry.get(handle);
    if (device_.native_state != &native_state_ ||
        texture == nullptr || texture->resource == nullptr || out_footprint == nullptr) {
        if (out_error != nullptr) {
            out_error->detail = "DXR texture copy footprint request is invalid";
        }
        return false;
    }
    const D3D12_RESOURCE_DESC desc = texture->resource->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 total_size = 0;
    native_state_.device->GetCopyableFootprints(
        &desc,
        0,
        1,
        0,
        &footprint,
        nullptr,
        nullptr,
        &total_size);
    if (total_size == 0) {
        if (out_error != nullptr) {
            out_error->detail = "DXR texture copy footprint is empty";
        }
        return false;
    }
    out_footprint->row_pitch = footprint.Footprint.RowPitch;
    out_footprint->total_size = static_cast<std::size_t>(total_size);
    return true;
}

// =============================================================================
// Acceleration structure creation and build recording.
// =============================================================================

void add_prebuild_info_timing(dxr_backend_state &state, double elapsed_ms, double* path_ms, std::uint32_t* path_count) {
    state.diagnostics.acceleration_prebuild_query_ms += elapsed_ms;
    if (path_ms != nullptr) {
        *path_ms += elapsed_ms;
    }
    if (path_count != nullptr) {
        ++*path_count;
    }
}

D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS dxr_acceleration_build_flags(
    std::uint32_t flags)
{
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS native_flags =
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_NONE;
    if ((flags & rt_acceleration_build_prefer_fast_trace) != 0u) {
        native_flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    }
    if ((flags & rt_acceleration_build_allow_update) != 0u) {
        native_flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
    }
    return native_flags;
}

bool warmup_acceleration_structure_prebuild_info(dxr_backend_state &state) {
    constexpr std::array<std::uint32_t, 3> indices{0, 1, 2};
    const std::array<rt_scene_vertex, 3> vertices{{
        {{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 1.0f}},
        {{1.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 1.0f}},
        {{0.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 1.0f}},
    }};
    constexpr std::size_t vertex_bytes = sizeof(rt_scene_vertex) * vertices.size();
    constexpr std::size_t index_bytes = sizeof(std::uint32_t) * indices.size();

    ID3D12Resource* geometry_buffer = nullptr;
    if (!create_native_buffer(
            state,
            D3D12_HEAP_TYPE_UPLOAD,
            vertex_bytes + index_bytes,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            D3D12_RESOURCE_FLAG_NONE,
            &geometry_buffer)) {
        append_rt_startup_log("DXR startup failed: create AS prebuild warmup geometry buffer");
        return false;
    }

    std::array<std::uint8_t, vertex_bytes + index_bytes> geometry_data{};
    std::memcpy(geometry_data.data(), vertices.data(), vertex_bytes);
    std::memcpy(geometry_data.data() + vertex_bytes, indices.data(), index_bytes);
    if (!upload_buffer_data(state, geometry_buffer, 0, geometry_data.data(), geometry_data.size())) {
        safe_release(geometry_buffer);
        append_rt_startup_log("DXR startup failed: upload AS prebuild warmup geometry buffer");
        return false;
    }

    D3D12_RAYTRACING_GEOMETRY_DESC geometry_desc{};
    geometry_desc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
    geometry_desc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
    geometry_desc.Triangles.IndexFormat = DXGI_FORMAT_R32_UINT;
    geometry_desc.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
    geometry_desc.Triangles.IndexCount = static_cast<UINT>(indices.size());
    geometry_desc.Triangles.VertexCount = static_cast<UINT>(vertices.size());
    geometry_desc.Triangles.IndexBuffer = geometry_buffer->GetGPUVirtualAddress() + vertex_bytes;
    geometry_desc.Triangles.VertexBuffer.StartAddress = geometry_buffer->GetGPUVirtualAddress();
    geometry_desc.Triangles.VertexBuffer.StrideInBytes = sizeof(rt_scene_vertex);

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs{};
    inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    inputs.NumDescs = 1;
    inputs.pGeometryDescs = &geometry_desc;
    inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuild_info{};
    const auto start = std::chrono::steady_clock::now();
    state.device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &prebuild_info);
    state.diagnostics.startup_prebuild_warmup_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    safe_release(geometry_buffer);
    if (prebuild_info.ResultDataMaxSizeInBytes == 0) {
        append_rt_startup_log("DXR startup failed: AS prebuild warmup returned zero result size");
        return false;
    }
    return true;
}

void clear_acceleration_cache(dxr_backend_state &state) {
    release_tlas_handle(state);
    state.tlas_registry.clear([](dxr_acceleration_structure &object) {
        release_acceleration_structure(&object);
    });
    state.blas_registry.clear([](dxr_acceleration_structure &object) {
        release_acceleration_structure(&object);
    });
    state.diagnostics.acceleration_host_prepare_ms = 0.0;
    state.diagnostics.acceleration_instance_build_ms = 0.0;
    state.diagnostics.acceleration_procedural_aabb_ms = 0.0;
    state.diagnostics.acceleration_command_record_ms = 0.0;
    state.diagnostics.acceleration_resource_allocate_ms = 0.0;
    state.diagnostics.acceleration_build_call_record_ms = 0.0;
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
    state.diagnostics.tlas_instance_upload_ms = 0.0;
}

bool build_triangle_blas(
    dxr_backend_state &state,
    const rt_blas_build_desc &command,
    dxr_acceleration_structure* entry,
    rt_blas_build_result* out_result)
{
    if (entry == nullptr || out_result == nullptr || command.geometry_count == 0 ||
        command.geometry_count > kRtBlasChunkSetChunkCount) {
        return false;
    }

    std::array<D3D12_RAYTRACING_GEOMETRY_DESC, kRtBlasChunkSetChunkCount> geometry_descs{};
    for (std::size_t geometry_index = 0; geometry_index < command.geometry_count; ++geometry_index) {
        const rt_acceleration_geometry_desc &source = command.geometries[geometry_index];
        const rt_triangle_geometry_desc &triangles = source.triangles;
        ID3D12Resource* const vertex_buffer =
            dxr_buffer_resource(state, triangles.vertex_buffer);
        ID3D12Resource* const index_buffer =
            dxr_buffer_resource(state, triangles.index_buffer);
        if (source.type != rt_acceleration_geometry_type::triangles ||
            triangles.vertex_format != rt_vertex_format::float3 ||
            triangles.index_format != rt_index_format::uint32 ||
            vertex_buffer == nullptr || index_buffer == nullptr ||
            triangles.vertex_count == 0 || triangles.index_count == 0 ||
            triangles.index_count % 3u != 0 ||
            triangles.vertex_count > (std::numeric_limits<UINT>::max)() ||
            triangles.index_count > (std::numeric_limits<UINT>::max)()) {
            return false;
        }

        D3D12_RAYTRACING_GEOMETRY_DESC &geometry_desc = geometry_descs[geometry_index];
        geometry_desc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
        geometry_desc.Flags = (source.flags & rt_acceleration_geometry_opaque) != 0u
            ? D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE
            : D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;
        geometry_desc.Triangles.Transform3x4 = 0;
        geometry_desc.Triangles.IndexFormat = DXGI_FORMAT_R32_UINT;
        geometry_desc.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
        geometry_desc.Triangles.IndexCount = static_cast<UINT>(triangles.index_count);
        geometry_desc.Triangles.VertexCount = static_cast<UINT>(triangles.vertex_count);
        geometry_desc.Triangles.IndexBuffer =
            index_buffer->GetGPUVirtualAddress() + triangles.index_offset;
        geometry_desc.Triangles.VertexBuffer.StartAddress =
            vertex_buffer->GetGPUVirtualAddress() + triangles.vertex_offset;
        geometry_desc.Triangles.VertexBuffer.StrideInBytes = triangles.vertex_stride;
    }

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs{};
    inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    inputs.NumDescs = static_cast<UINT>(command.geometry_count);
    inputs.pGeometryDescs = geometry_descs.data();
    inputs.Flags = dxr_acceleration_build_flags(command.flags);

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuild_info{};
    const auto prebuild_start = std::chrono::steady_clock::now();
    state.device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &prebuild_info);
    double* const prebuild_ms = command.geometry_count == 1
        ? &state.diagnostics.chunk_blas_prebuild_query_ms
        : &state.diagnostics.grouped_blas_prebuild_query_ms;
    std::uint32_t* const prebuild_count = command.geometry_count == 1
        ? &state.diagnostics.chunk_blas_prebuild_query_count
        : &state.diagnostics.grouped_blas_prebuild_query_count;
    add_prebuild_info_timing(
        state,
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - prebuild_start).count(),
        prebuild_ms,
        prebuild_count);
    if (prebuild_info.ResultDataMaxSizeInBytes == 0) {
        return false;
    }

    const auto alloc_start = std::chrono::steady_clock::now();
    if (!ensure_buffer_capacity(
            state,
            D3D12_HEAP_TYPE_DEFAULT,
            static_cast<std::size_t>(prebuild_info.ResultDataMaxSizeInBytes),
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            &entry->result)) {
        return false;
    }
    state.diagnostics.acceleration_resource_allocate_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - alloc_start).count();

    dxr_blas_build_record record{};
    record.geometry_descs = geometry_descs;
    record.inputs = inputs;
    record.inputs.pGeometryDescs = nullptr;
    record.destination = entry->result;
    record.scratch_size = prebuild_info.ScratchDataSizeInBytes;
    dxr_acceleration_build_context &context = state.acceleration_build;
    context.max_scratch_size = (std::max)(context.max_scratch_size, record.scratch_size);
    context.blas_builds.push_back(std::move(record));
    out_result->acceleration = command.destination;
    out_result->reused = false;
    return static_cast<bool>(out_result->acceleration);
}

bool build_procedural_blas(
    dxr_backend_state &state,
    const rt_blas_build_desc &command,
    dxr_acceleration_structure* entry,
    rt_blas_build_result* out_result)
{
    if (entry == nullptr || out_result == nullptr || command.geometry_count == 0 ||
        command.geometry_count > kRtBlasChunkSetChunkCount) {
        return false;
    }

    std::array<D3D12_RAYTRACING_GEOMETRY_DESC, kRtBlasChunkSetChunkCount> geometry_descs{};
    for (std::size_t geometry_index = 0; geometry_index < command.geometry_count; ++geometry_index) {
        const rt_acceleration_geometry_desc &source = command.geometries[geometry_index];
        const rt_aabb_geometry_desc &aabbs = source.aabbs;
        ID3D12Resource* const aabb_buffer = dxr_buffer_resource(state, aabbs.buffer);
        if (source.type != rt_acceleration_geometry_type::aabbs || aabb_buffer == nullptr ||
            aabbs.count == 0 || aabbs.count > (std::numeric_limits<UINT>::max)()) {
            return false;
        }
        D3D12_RAYTRACING_GEOMETRY_DESC &geometry_desc = geometry_descs[geometry_index];
        geometry_desc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS;
        geometry_desc.Flags = (source.flags & rt_acceleration_geometry_opaque) != 0u
            ? D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE
            : D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;
        geometry_desc.AABBs.AABBCount = static_cast<UINT>(aabbs.count);
        geometry_desc.AABBs.AABBs.StartAddress =
            aabb_buffer->GetGPUVirtualAddress() + aabbs.offset;
        geometry_desc.AABBs.AABBs.StrideInBytes = aabbs.stride;
    }

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs{};
    inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    inputs.NumDescs = static_cast<UINT>(command.geometry_count);
    inputs.pGeometryDescs = geometry_descs.data();
    inputs.Flags = dxr_acceleration_build_flags(command.flags);
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuild_info{};
    const auto prebuild_start = std::chrono::steady_clock::now();
    state.device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &prebuild_info);
    const double prebuild_info_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - prebuild_start).count();
    state.diagnostics.acceleration_prebuild_query_ms += prebuild_info_ms;
    out_result->prebuild_info_ms = prebuild_info_ms;
    if (prebuild_info.ResultDataMaxSizeInBytes == 0) {
        return false;
    }

    const auto alloc_start = std::chrono::steady_clock::now();
    if (!ensure_buffer_capacity(
            state,
            D3D12_HEAP_TYPE_DEFAULT,
            static_cast<std::size_t>(prebuild_info.ResultDataMaxSizeInBytes),
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            &entry->result)) {
        return false;
    }
    state.diagnostics.acceleration_resource_allocate_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - alloc_start).count();

    dxr_blas_build_record record{};
    record.geometry_descs = geometry_descs;
    record.inputs = inputs;
    record.inputs.pGeometryDescs = nullptr;
    record.destination = entry->result;
    record.scratch_size = prebuild_info.ScratchDataSizeInBytes;
    dxr_acceleration_build_context &context = state.acceleration_build;
    context.max_scratch_size = (std::max)(context.max_scratch_size, record.scratch_size);
    context.blas_builds.push_back(std::move(record));
    out_result->acceleration = command.destination;
    out_result->reused = false;
    return static_cast<bool>(out_result->acceleration);
}

bool ensure_acceleration_scratch_buffer(dxr_backend_state &state, UINT64 size) {
    if (size == 0) {
        return true;
    }
    const auto alloc_start = std::chrono::steady_clock::now();
    const bool allocated = ensure_buffer_capacity(
        state,
        D3D12_HEAP_TYPE_DEFAULT,
        static_cast<std::size_t>(size),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        &state.scratch_buffer);
    state.diagnostics.acceleration_resource_allocate_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - alloc_start).count();
    return allocated;
}

bool record_blas_builds(dxr_backend_state &state) {
    dxr_acceleration_build_context &context = state.acceleration_build;
    if (context.blas_builds.empty()) {
        return true;
    }
    if (state.scratch_buffer == nullptr) {
        return false;
    }

    const D3D12_GPU_VIRTUAL_ADDRESS scratch_address = state.scratch_buffer->GetGPUVirtualAddress();
    const auto build_record_start = std::chrono::steady_clock::now();
    for (dxr_blas_build_record &record : context.blas_builds) {
        if (record.destination == nullptr || record.scratch_size > context.max_scratch_size) {
            return false;
        }
        record.inputs.pGeometryDescs = record.geometry_descs.data();
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build_desc{};
        build_desc.Inputs = record.inputs;
        build_desc.ScratchAccelerationStructureData = scratch_address;
        build_desc.DestAccelerationStructureData = record.destination->GetGPUVirtualAddress();
        state.command_list->BuildRaytracingAccelerationStructure(&build_desc, 0, nullptr);

        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        state.command_list->ResourceBarrier(1, &barrier);
    }
    state.diagnostics.acceleration_build_call_record_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - build_record_start).count();
    return true;
}

bool build_top_level_as(dxr_backend_state &state, const rt_tlas_build_desc &request) {
    const std::size_t instance_count = request.instance_count;
    dxr_acceleration_structure* const tlas = state.tlas_registry.get(request.destination);
    if (tlas == nullptr) {
        return false;
    }
    if (instance_count == 0) {
        if (!ensure_acceleration_scratch_buffer(state, state.acceleration_build.max_scratch_size) ||
            !record_blas_builds(state)) {
            return false;
        }
        defer_acceleration_structure(state, tlas);
        return true;
    }
    if (request.instances == nullptr) {
        return false;
    }
    std::vector<D3D12_RAYTRACING_INSTANCE_DESC> instances;
    instances.reserve(instance_count);
    for (std::size_t instance_index = 0; instance_index < instance_count; ++instance_index) {
        const rt_tlas_instance_desc &source = request.instances[instance_index];
        const dxr_acceleration_structure* const acceleration =
            state.blas_registry.get(source.acceleration);
        if (acceleration == nullptr || acceleration->result == nullptr) {
            return false;
        }
        D3D12_RAYTRACING_INSTANCE_DESC desc{};
        std::memcpy(desc.Transform, source.transform.data(), sizeof(desc.Transform));
        desc.InstanceID = source.instance_id;
        desc.InstanceContributionToHitGroupIndex = source.hit_group_contribution;
        desc.InstanceMask = source.mask;
        desc.Flags = D3D12_RAYTRACING_INSTANCE_FLAG_NONE;
        if ((source.flags & rt_acceleration_instance_triangle_cull_disable) != 0u) {
            desc.Flags |= D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_CULL_DISABLE;
        }
        if ((source.flags & rt_acceleration_instance_triangle_front_counterclockwise) != 0u) {
            desc.Flags |= D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_FRONT_COUNTERCLOCKWISE;
        }
        if ((source.flags & rt_acceleration_instance_force_opaque) != 0u) {
            desc.Flags |= D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_OPAQUE;
        }
        if ((source.flags & rt_acceleration_instance_force_non_opaque) != 0u) {
            desc.Flags |= D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_NON_OPAQUE;
        }
        desc.AccelerationStructure = acceleration->result->GetGPUVirtualAddress();
        instances.push_back(desc);
    }

    const auto alloc_start = std::chrono::steady_clock::now();
    dxr_command_slot &slot = state.command_slots[state.active_command_slot_index];
    if (!ensure_buffer_capacity(
            state,
            D3D12_HEAP_TYPE_UPLOAD,
            instances.size() * sizeof(D3D12_RAYTRACING_INSTANCE_DESC),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            D3D12_RESOURCE_FLAG_NONE,
            &slot.tlas_instance_buffer)) {
        return false;
    }
    const auto upload_start = std::chrono::steady_clock::now();
    if (!upload_buffer_data(
            state,
            slot.tlas_instance_buffer,
            0,
            instances.data(),
            instances.size() * sizeof(D3D12_RAYTRACING_INSTANCE_DESC))) {
        return false;
    }
    state.diagnostics.tlas_instance_upload_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - upload_start).count();

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs{};
    inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    inputs.NumDescs = static_cast<UINT>(instance_count);
    inputs.InstanceDescs = slot.tlas_instance_buffer->GetGPUVirtualAddress();
    inputs.Flags = dxr_acceleration_build_flags(request.flags);

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuild_info{};
    const auto prebuild_start = std::chrono::steady_clock::now();
    state.device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &prebuild_info);
    add_prebuild_info_timing(
        state,
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - prebuild_start).count(),
        &state.diagnostics.tlas_prebuild_query_ms,
        &state.diagnostics.tlas_prebuild_query_count);
    if (prebuild_info.ResultDataMaxSizeInBytes == 0) {
        return false;
    }

    const UINT64 required_scratch_size = prebuild_info.ScratchDataSizeInBytes;
    dxr_acceleration_build_context &context = state.acceleration_build;
    context.max_scratch_size = (std::max)(context.max_scratch_size, required_scratch_size);
    if (!ensure_acceleration_scratch_buffer(state, context.max_scratch_size)) {
        return false;
    }
    defer_resource_release(state, tlas->result);
    if (tlas->result == nullptr && !create_native_buffer(
            state,
            D3D12_HEAP_TYPE_DEFAULT,
            static_cast<std::size_t>(prebuild_info.ResultDataMaxSizeInBytes),
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            &tlas->result)) {
        return false;
    }
    state.diagnostics.acceleration_resource_allocate_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - alloc_start).count();

    if (!record_blas_builds(state)) {
        return false;
    }

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build_desc{};
    build_desc.Inputs = inputs;
    build_desc.ScratchAccelerationStructureData =
        state.scratch_buffer->GetGPUVirtualAddress();
    build_desc.DestAccelerationStructureData = tlas->result->GetGPUVirtualAddress();
    const auto build_record_start = std::chrono::steady_clock::now();
    state.command_list->BuildRaytracingAccelerationStructure(&build_desc, 0, nullptr);

    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = nullptr;
    state.command_list->ResourceBarrier(1, &barrier);
    state.diagnostics.acceleration_build_call_record_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - build_record_start).count();
    tlas->instance_count = instance_count;
    return true;
}

bool d3d12_dxr_rhi_device::create_blas(
    rt_blas_handle* out_blas,
    rt_device_error* out_error)
{
    if (out_blas != nullptr) {
        *out_blas = {};
    }
    if (device_.native_state != &native_state_ ||
        out_blas == nullptr || !native_state_.blas_registry.insert({}, out_blas)) {
        if (out_error != nullptr) {
            out_error->detail = "DXR BLAS object creation failed";
        }
        return false;
    }
    return true;
}

void d3d12_dxr_rhi_device::destroy_blas(rt_blas_handle handle) {
    if (device_.native_state != &native_state_ || !handle) {
        return;
    }
    dxr_acceleration_structure object{};
    if (native_state_.blas_registry.erase(handle, &object)) {
        defer_acceleration_structure(native_state_, &object);
    }
}

bool d3d12_dxr_rhi_device::create_tlas(
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
            out_error->detail = "DXR TLAS object creation failed";
        }
        return false;
    }
    native_state_.tlas = *out_tlas;
    return true;
}

void d3d12_dxr_rhi_device::destroy_tlas(rt_tlas_handle handle) {
    if (device_.native_state != &native_state_ || !handle) {
        return;
    }
    dxr_acceleration_structure object{};
    if (native_state_.tlas_registry.erase(handle, &object)) {
        defer_acceleration_structure(native_state_, &object);
    }
    if (native_state_.tlas == handle) {
        native_state_.tlas = {};
    }
}

bool begin_acceleration_recording(dxr_backend_state &state) {
    dxr_acceleration_build_context &context = state.acceleration_build;
    if (context.active) {
        return context.recording;
    }
    context = {};
    context.active = true;
    context.total_start = std::chrono::steady_clock::now();
    state.diagnostics.acceleration_host_prepare_ms = 0.0;
    state.diagnostics.acceleration_instance_build_ms = 0.0;
    state.diagnostics.acceleration_procedural_aabb_ms = 0.0;
    state.diagnostics.acceleration_command_record_ms = 0.0;
    state.diagnostics.acceleration_resource_allocate_ms = 0.0;
    state.diagnostics.acceleration_build_call_record_ms = 0.0;
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
    state.diagnostics.tlas_instance_upload_ms = 0.0;
    if (!ensure_timestamp_query_resources(state)) {
        context.active = false;
        return false;
    }
    context.recording = true;
    context.timestamp_slot = current_timestamp_query_slot(state);
    context.command_start = std::chrono::steady_clock::now();
    write_timestamp_query_begin(state, context.timestamp_slot, timestamp_query_region::accel);
    return true;
}

bool d3d12_dxr_rhi_device::build_blas(
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
            out_error->detail = "DXR BLAS build request is invalid";
        }
        return false;
    }

    dxr_acceleration_structure* const acceleration =
        native_state_.blas_registry.get(command.destination);
    if (acceleration == nullptr) {
        if (out_error != nullptr) {
            out_error->detail = "DXR BLAS build failed";
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
    const bool completed = built && acceleration->result != nullptr;
    if (!completed && out_error != nullptr) {
        out_error->detail = "DXR BLAS build failed";
    }
    return completed;
}

bool d3d12_dxr_rhi_device::build_tlas(
    rt_command_encoder encoder,
    const rt_tlas_build_desc &request,
    rt_device_error* out_error)
{
    if (device_.native_state != &native_state_ ||
        !encoder || encoder.id != native_state_.active_encoder_id ||
        !begin_acceleration_recording(native_state_)) {
        if (out_error != nullptr) {
            out_error->detail = "DXR TLAS build request is invalid";
        }
        return false;
    }
    const bool built = build_top_level_as(native_state_, request);
    if (!built && out_error != nullptr) {
        out_error->detail = "DXR TLAS build failed";
    }
    if (!built) {
        return false;
    }

    dxr_acceleration_build_context &context = native_state_.acceleration_build;
    write_timestamp_query_end_and_resolve(
        native_state_,
        context.timestamp_slot,
        timestamp_query_region::accel);
    native_state_.diagnostics.acceleration_command_record_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - context.command_start).count();
    context.recording = false;
    context.recorded = true;
    return true;
}

// =============================================================================
// Pipeline and binding state.
// =============================================================================

void release_shader_modules(dxr_backend_state &state) {
    state.shader_module_registry.clear([](std::vector<std::uint8_t> &) {});
}

void release_raytracing_runtime_state(dxr_backend_state &state) {
    state.shader_table_registry.clear([](ID3D12Resource* &) {});
    state.pipeline_registry.clear([](ID3D12StateObject* &) {});
    state.shader_table = {};
    state.pipeline = {};
    safe_release(state.raygen_shader_table);
    safe_release(state.miss_shader_table);
    safe_release(state.hitgroup_shader_table);
    safe_release(state.callable_shader_table);
    state.raygen_shader_record_count = 0;
    state.miss_shader_record_count = 0;
    state.hitgroup_shader_record_count = 0;
    state.callable_shader_record_count = 0;
    state.shader_group_export_names.clear();
    state.camera_constant_buffer_handle = {};
    state.camera_constant_buffer = nullptr;
    safe_release(state.raytracing_state_props);
    safe_release(state.raytracing_state_object);
    safe_release(state.global_root_signature);
    safe_release(state.srv_uav_cbv_heap);
    state.diagnostics.acceleration_gpu_ms = 0.0;
    state.diagnostics.dispatch_gpu_ms = 0.0;
}

D3D12_CPU_DESCRIPTOR_HANDLE descriptor_cpu_handle(
    dxr_backend_state &state,
    UINT slot_index,
    UINT index)
{
    D3D12_CPU_DESCRIPTOR_HANDLE handle = state.srv_uav_cbv_heap->GetCPUDescriptorHandleForHeapStart();
    handle.ptr += static_cast<SIZE_T>(slot_index * kDescriptorSlotStride + index) *
        state.srv_uav_cbv_descriptor_size;
    return handle;
}

D3D12_GPU_DESCRIPTOR_HANDLE descriptor_gpu_handle(
    dxr_backend_state &state,
    UINT slot_index,
    UINT index)
{
    D3D12_GPU_DESCRIPTOR_HANDLE handle = state.srv_uav_cbv_heap->GetGPUDescriptorHandleForHeapStart();
    handle.ptr += static_cast<UINT64>(slot_index * kDescriptorSlotStride + index) *
        state.srv_uav_cbv_descriptor_size;
    return handle;
}

bool ensure_descriptor_heap(dxr_backend_state &state) {
    if (state.srv_uav_cbv_heap != nullptr) {
        return true;
    }
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
    heap_desc.NumDescriptors = kDescriptorHeapCount;
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(state.device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&state.srv_uav_cbv_heap)))) {
        record_dxr_failure(state, "ensure_descriptor_heap.CreateDescriptorHeap", E_FAIL);
        return false;
    }
    state.srv_uav_cbv_descriptor_size =
        state.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    return true;
}

bool update_dxr_resource_bindings(
    dxr_backend_state &state,
    const rt_binding_update_request &request,
    UINT slot_index,
    std::string* out_detail)
{
    const auto fail = [out_detail](const rt_binding_write* write, std::string_view reason) {
        if (out_detail != nullptr) {
            *out_detail = "DXR descriptor binding update failed";
            if (write != nullptr) {
                *out_detail += " at binding " + std::to_string(write->location.binding);
            }
            *out_detail += ": ";
            *out_detail += reason;
        }
        return false;
    };
    if (request.writes == nullptr || request.write_count == 0 || !ensure_descriptor_heap(state)) {
        return fail(nullptr, "binding writes or descriptor heap are unavailable");
    }
    for (std::size_t index = 0; index < request.write_count; ++index) {
        const rt_binding_write &write = request.writes[index];
        if (write.location.group != 0 || write.location.binding >= kDescriptorBindingCapacity) {
            return fail(&write, "binding location is out of range");
        }
        if (write.type == rt_descriptor_type::acceleration_structure) {
            const dxr_acceleration_structure* const tlas = state.tlas_registry.get(state.tlas);
            if (tlas == nullptr || tlas->result == nullptr) {
                return fail(&write, "acceleration structure is unavailable");
            }
            D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
            srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
            srv.RaytracingAccelerationStructure.Location = tlas->result->GetGPUVirtualAddress();
            state.device->CreateShaderResourceView(
                nullptr,
                &srv,
                descriptor_cpu_handle(state, slot_index, kSrvDescriptorBase + write.location.binding));
        } else if (write.type == rt_descriptor_type::structured_buffer) {
            ID3D12Resource* const resource = dxr_buffer_resource(state, write.resource);
            if ((resource == nullptr && write.element_count != 0) || write.element_stride == 0) {
                return fail(&write, "structured buffer is invalid");
            }
            D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
            srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            srv.Format = DXGI_FORMAT_UNKNOWN;
            srv.Buffer.NumElements = static_cast<UINT>(write.element_count);
            srv.Buffer.StructureByteStride = static_cast<UINT>(write.element_stride);
            state.device->CreateShaderResourceView(
                resource,
                &srv,
                descriptor_cpu_handle(state, slot_index, kSrvDescriptorBase + write.location.binding));
        } else if (write.type == rt_descriptor_type::storage_buffer) {
            ID3D12Resource* const resource = dxr_buffer_resource(state, write.resource);
            if (resource == nullptr || write.element_count == 0 || write.element_stride == 0) {
                return fail(&write, "storage buffer is invalid");
            }
            D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
            uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            uav.Format = DXGI_FORMAT_UNKNOWN;
            uav.Buffer.NumElements = static_cast<UINT>(write.element_count);
            uav.Buffer.StructureByteStride = static_cast<UINT>(write.element_stride);
            state.device->CreateUnorderedAccessView(
                resource,
                nullptr,
                &uav,
                descriptor_cpu_handle(state, slot_index, kUavDescriptorBase + write.location.binding));
        } else if (write.type == rt_descriptor_type::storage_texture) {
            dxr_texture* const texture = state.texture_registry.get(write.texture);
            if (texture == nullptr || texture->resource == nullptr) {
                return fail(&write, "storage texture is unavailable");
            }
            D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
            uav.Format = dxr_texture_format(texture->desc.format);
            uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            state.device->CreateUnorderedAccessView(
                texture->resource,
                nullptr,
                &uav,
                descriptor_cpu_handle(state, slot_index, kUavDescriptorBase + write.location.binding));
        } else if (write.type == rt_descriptor_type::uniform_buffer) {
            ID3D12Resource* const resource = dxr_buffer_resource(state, write.resource);
            if (resource == nullptr) {
                return fail(&write, "uniform buffer is unavailable");
            }
            state.camera_constant_buffer_handle = write.resource;
            state.camera_constant_buffer = resource;
        } else {
            return fail(&write, "descriptor type is unsupported");
        }
    }
    return true;
}

UINT descriptor_target_slot(const dxr_backend_state &state) {
    return state.active_encoder_id != 0
        ? state.active_command_slot_index
        : state.command_slot_index % kRtCommandSlotCount;
}

bool refresh_command_slot_bindings(dxr_backend_state &state, UINT slot_index) {
    if (state.pending_binding_writes.empty() ||
        state.command_slot_binding_generations[slot_index] == state.binding_generation) {
        return true;
    }
    const rt_binding_update_request request{
        state.pending_binding_writes.data(),
        state.pending_binding_writes.size()};
    if (!update_dxr_resource_bindings(state, request, slot_index, nullptr)) {
        return false;
    }
    state.command_slot_binding_generations[slot_index] = state.binding_generation;
    return true;
}

bool ensure_global_root_signature(
    dxr_backend_state &state,
    const rt_pipeline_desc &desc)
{
    if (state.global_root_signature != nullptr) {
        return true;
    }
    std::vector<D3D12_DESCRIPTOR_RANGE1> uav_ranges;
    std::vector<D3D12_DESCRIPTOR_RANGE1> srv_ranges;
    UINT uniform_binding = UINT_MAX;
    for (std::size_t index = 0; index < desc.binding_count; ++index) {
        const rt_binding_layout_desc &binding = desc.bindings[index];
        if (binding.location.group != 0 ||
            binding.location.binding >= kDescriptorBindingCapacity ||
            binding.count != 1) {
            return false;
        }
        D3D12_DESCRIPTOR_RANGE_TYPE range_type{};
        std::vector<D3D12_DESCRIPTOR_RANGE1>* destination = nullptr;
        if (binding.type == rt_descriptor_type::storage_texture ||
            binding.type == rt_descriptor_type::storage_buffer) {
            range_type = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
            destination = &uav_ranges;
        } else if (binding.type == rt_descriptor_type::acceleration_structure ||
            binding.type == rt_descriptor_type::structured_buffer) {
            range_type = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            destination = &srv_ranges;
        } else if (binding.type == rt_descriptor_type::uniform_buffer) {
            uniform_binding = binding.location.binding;
            continue;
        } else {
            return false;
        }
        D3D12_DESCRIPTOR_RANGE1 range{};
        range.RangeType = range_type;
        range.NumDescriptors = 1;
        range.BaseShaderRegister = binding.location.binding;
        range.OffsetInDescriptorsFromTableStart = binding.location.binding;
        destination->push_back(range);
    }
    if (uav_ranges.empty() || srv_ranges.empty() || uniform_binding == UINT_MAX) {
        return false;
    }

    D3D12_ROOT_PARAMETER1 parameters[3]{};
    parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameters[0].DescriptorTable.NumDescriptorRanges = static_cast<UINT>(uav_ranges.size());
    parameters[0].DescriptorTable.pDescriptorRanges = uav_ranges.data();
    parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameters[1].DescriptorTable.NumDescriptorRanges = static_cast<UINT>(srv_ranges.size());
    parameters[1].DescriptorTable.pDescriptorRanges = srv_ranges.data();
    parameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    parameters[2].Descriptor.ShaderRegister = uniform_binding;

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    root_desc.Desc_1_1.NumParameters = static_cast<UINT>(std::size(parameters));
    root_desc.Desc_1_1.pParameters = parameters;
    root_desc.Desc_1_1.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ID3DBlob* serialized = nullptr;
    ID3DBlob* errors = nullptr;
    const HRESULT serialize_hr = D3D12SerializeVersionedRootSignature(&root_desc, &serialized, &errors);
    if (FAILED(serialize_hr)) {
        record_dxr_failure(state, "ensure_global_root_signature.D3D12SerializeVersionedRootSignature", serialize_hr);
        safe_release(errors);
        safe_release(serialized);
        return false;
    }
    safe_release(errors);
    const HRESULT hr = state.device->CreateRootSignature(
        0,
        serialized->GetBufferPointer(),
        serialized->GetBufferSize(),
        IID_PPV_ARGS(&state.global_root_signature));
    safe_release(serialized);
    if (FAILED(hr)) {
        record_dxr_failure(state, "ensure_global_root_signature.CreateRootSignature", hr);
    }
    return SUCCEEDED(hr);
}

std::wstring dxr_shader_name(const char* name) {
    std::wstring result;
    if (name == nullptr) {
        return result;
    }
    while (*name != '\0') {
        const unsigned char character = static_cast<unsigned char>(*name++);
        if (character > 0x7f) {
            return {};
        }
        result.push_back(static_cast<wchar_t>(character));
    }
    return result;
}

bool ensure_state_object(dxr_backend_state &state, const rt_pipeline_desc &desc) {
    if (state.raytracing_state_object != nullptr && state.raytracing_state_props != nullptr) {
        return true;
    }
    if (!ensure_global_root_signature(state, desc)) {
        return false;
    }

    std::vector<std::wstring> shader_names;
    shader_names.reserve(desc.shader_count);
    for (std::size_t index = 0; index < desc.shader_count; ++index) {
        shader_names.push_back(dxr_shader_name(desc.shaders[index].entry_point));
        if (shader_names.back().empty()) {
            return false;
        }
    }
    std::vector<rt_shader_module_handle> module_handles;
    std::vector<const std::vector<std::uint8_t>*> module_binaries;
    std::vector<std::vector<D3D12_EXPORT_DESC>> module_exports;
    for (std::size_t shader_index = 0; shader_index < desc.shader_count; ++shader_index) {
        const rt_shader_module_handle module_handle = desc.shaders[shader_index].module;
        const std::vector<std::uint8_t>* const module_binary =
            state.shader_module_registry.get(module_handle);
        if (module_binary == nullptr || module_binary->empty()) {
            return false;
        }
        std::size_t module_index = 0;
        while (module_index < module_handles.size() &&
            module_handles[module_index] != module_handle) {
            ++module_index;
        }
        if (module_index == module_handles.size()) {
            module_handles.push_back(module_handle);
            module_binaries.push_back(module_binary);
            module_exports.emplace_back();
        }
        D3D12_EXPORT_DESC shader_export{};
        shader_export.Name = shader_names[shader_index].c_str();
        module_exports[module_index].push_back(shader_export);
    }
    std::vector<D3D12_DXIL_LIBRARY_DESC> dxil_library_descs(module_handles.size());
    for (std::size_t index = 0; index < dxil_library_descs.size(); ++index) {
        D3D12_DXIL_LIBRARY_DESC &library_desc = dxil_library_descs[index];
        library_desc.DXILLibrary.pShaderBytecode = module_binaries[index]->data();
        library_desc.DXILLibrary.BytecodeLength = module_binaries[index]->size();
        library_desc.NumExports = static_cast<UINT>(module_exports[index].size());
        library_desc.pExports = module_exports[index].data();
    }

    std::vector<std::wstring> hit_group_names;
    hit_group_names.reserve(desc.group_count);
    std::vector<std::wstring> group_export_names(desc.group_count);
    std::vector<D3D12_HIT_GROUP_DESC> hit_group_descs;
    hit_group_descs.reserve(desc.group_count);
    for (std::size_t index = 0; index < desc.group_count; ++index) {
        const rt_shader_group_desc &group = desc.groups[index];
        if (group.type == rt_shader_group_type::general) {
            if (group.general_shader >= shader_names.size()) {
                return false;
            }
            group_export_names[index] = shader_names[group.general_shader];
            continue;
        }
        hit_group_names.push_back(dxr_shader_name(group.export_name));
        if (hit_group_names.back().empty() ||
            group.closest_hit_shader >= shader_names.size() ||
            (group.any_hit_shader != kRtUnusedShaderIndex &&
                group.any_hit_shader >= shader_names.size()) ||
            (group.intersection_shader != kRtUnusedShaderIndex &&
                group.intersection_shader >= shader_names.size())) {
            return false;
        }
        group_export_names[index] = hit_group_names.back();
        D3D12_HIT_GROUP_DESC native_group{};
        native_group.HitGroupExport = hit_group_names.back().c_str();
        native_group.ClosestHitShaderImport = shader_names[group.closest_hit_shader].c_str();
        if (group.any_hit_shader != kRtUnusedShaderIndex) {
            native_group.AnyHitShaderImport = shader_names[group.any_hit_shader].c_str();
        }
        if (group.intersection_shader != kRtUnusedShaderIndex) {
            native_group.IntersectionShaderImport =
                shader_names[group.intersection_shader].c_str();
        }
        native_group.Type = group.type == rt_shader_group_type::triangles_hit_group
            ? D3D12_HIT_GROUP_TYPE_TRIANGLES
            : D3D12_HIT_GROUP_TYPE_PROCEDURAL_PRIMITIVE;
        hit_group_descs.push_back(native_group);
    }

    D3D12_RAYTRACING_SHADER_CONFIG shader_config{};
    shader_config.MaxPayloadSizeInBytes = desc.max_payload_size;
    shader_config.MaxAttributeSizeInBytes = desc.max_attribute_size;

    D3D12_GLOBAL_ROOT_SIGNATURE global_root_signature{};
    global_root_signature.pGlobalRootSignature = state.global_root_signature;

    D3D12_RAYTRACING_PIPELINE_CONFIG pipeline_config{};
    pipeline_config.MaxTraceRecursionDepth = desc.max_recursion_depth;

    std::vector<D3D12_STATE_SUBOBJECT> subobjects;
    subobjects.reserve(dxil_library_descs.size() + hit_group_descs.size() + 3);
    for (const D3D12_DXIL_LIBRARY_DESC &library_desc : dxil_library_descs) {
        subobjects.push_back({D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY, &library_desc});
    }
    for (const D3D12_HIT_GROUP_DESC &hit_group : hit_group_descs) {
        subobjects.push_back({D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP, &hit_group});
    }
    subobjects.push_back({D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG, &shader_config});
    subobjects.push_back({
        D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE,
        &global_root_signature});
    subobjects.push_back({
        D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG,
        &pipeline_config});

    D3D12_STATE_OBJECT_DESC state_object_desc{};
    state_object_desc.Type = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE;
    state_object_desc.NumSubobjects = static_cast<UINT>(subobjects.size());
    state_object_desc.pSubobjects = subobjects.data();
    const HRESULT create_hr = state.device->CreateStateObject(
        &state_object_desc,
        IID_PPV_ARGS(&state.raytracing_state_object));
    if (FAILED(create_hr)) {
        record_dxr_failure(state, "ensure_state_object.CreateStateObject", create_hr);
        return false;
    }
    const HRESULT query_hr =
        state.raytracing_state_object->QueryInterface(IID_PPV_ARGS(&state.raytracing_state_props));
    if (FAILED(query_hr)) {
        record_dxr_failure(state, "ensure_state_object.QueryInterface", query_hr);
    } else {
        state.shader_group_export_names = std::move(group_export_names);
    }
    return SUCCEEDED(query_hr);
}

bool create_shader_table_resource(
    dxr_backend_state &state,
    ID3D12Resource** out_resource,
    const void* const* shader_identifiers,
    std::size_t shader_identifier_count)
{
    if (out_resource == nullptr || shader_identifiers == nullptr || shader_identifier_count == 0) {
        return false;
    }
    for (std::size_t i = 0; i < shader_identifier_count; ++i) {
        if (shader_identifiers[i] == nullptr) {
            append_rt_startup_log("DXR shader table creation failed: shader identifier is unavailable");
            return false;
        }
    }

    safe_release(*out_resource);
    const std::size_t table_bytes = align_up(
        kShaderRecordSize * shader_identifier_count,
        D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);
    if (!create_native_buffer(
            state,
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
    return upload_buffer_data(state, *out_resource, 0, table_data.data(), table_data.size());
}

bool create_shader_table_section(
    dxr_backend_state &state,
    const rt_shader_table_section_desc &section,
    ID3D12Resource** out_resource,
    std::size_t* out_record_count)
{
    if (out_resource == nullptr || out_record_count == nullptr ||
        (section.group_count != 0 && section.groups == nullptr)) {
        return false;
    }
    if (section.group_count == 0) {
        safe_release(*out_resource);
        *out_record_count = 0;
        return true;
    }
    std::vector<const void*> identifiers(section.group_count);
    for (std::size_t index = 0; index < section.group_count; ++index) {
        const std::uint32_t group_index = section.groups[index];
        if (group_index >= state.shader_group_export_names.size()) {
            return false;
        }
        identifiers[index] = state.raytracing_state_props->GetShaderIdentifier(
            state.shader_group_export_names[group_index].c_str());
    }
    if (!create_shader_table_resource(
            state,
            out_resource,
            identifiers.data(),
            identifiers.size())) {
        return false;
    }
    *out_record_count = identifiers.size();
    return true;
}

bool ensure_shader_tables(
    dxr_backend_state &state,
    const rt_shader_table_desc &desc)
{
    if (state.raytracing_state_props == nullptr || state.shader_group_export_names.empty()) {
        return false;
    }
    return create_shader_table_section(
            state,
            desc.ray_generation,
            &state.raygen_shader_table,
            &state.raygen_shader_record_count) &&
        create_shader_table_section(
            state,
            desc.miss,
            &state.miss_shader_table,
            &state.miss_shader_record_count) &&
        create_shader_table_section(
            state,
            desc.hit,
            &state.hitgroup_shader_table,
            &state.hitgroup_shader_record_count) &&
        create_shader_table_section(
            state,
            desc.callable,
            &state.callable_shader_table,
            &state.callable_shader_record_count);
}

// =============================================================================
// Command submission, native presentation, and trace dispatch.
// =============================================================================

bool reset_command_list(dxr_backend_state &state) {
    const UINT slot_index = state.command_slot_index % kRtCommandSlotCount;
    dxr_command_slot &slot = state.command_slots[slot_index];
    double reuse_wait_ms = 0.0;
    if (!wait_for_fence_value_timed(state, slot.fence_value, &reuse_wait_ms)) {
        record_dxr_failure(state, "reset_command_list.wait_for_fence");
        return false;
    }
    state.diagnostics.command_slot_reuse_wait_ms += reuse_wait_ms;
    complete_command_slot(state, slot_index);

    ID3D12CommandAllocator* allocator = slot.allocator;
    const HRESULT allocator_hr = allocator->Reset();
    if (FAILED(allocator_hr)) {
        record_dxr_failure(state, "reset_command_list.allocator_reset", allocator_hr);
        return false;
    }
    const HRESULT list_hr = state.command_list->Reset(allocator, nullptr);
    if (FAILED(list_hr)) {
        record_dxr_failure(state, "reset_command_list.command_list_reset", list_hr);
        return false;
    }
    state.active_command_slot_index = slot_index;
    return true;
}

bool close_and_execute_command_list(
    dxr_backend_state &state,
    double* out_submit_cpu_ms,
    rt_submission_token* out_submission)
{
    if (out_submission != nullptr) {
        *out_submission = {};
    }
    const HRESULT close_hr = state.command_list->Close();
    if (FAILED(close_hr)) {
        record_dxr_failure(state, "close_and_execute_command_list.Close", close_hr);
        return false;
    }
    ID3D12CommandList* command_lists[] = {state.command_list};
    const auto submit_start = std::chrono::steady_clock::now();
    state.queue->ExecuteCommandLists(1, command_lists);
    const std::uint64_t fence_value = state.submitted_fence_value + 1u;
    const HRESULT signal_hr = state.queue->Signal(state.fence, fence_value);
    if (FAILED(signal_hr)) {
        record_dxr_failure(state, "close_and_execute_command_list.Signal", signal_hr);
        return false;
    }
    if (out_submit_cpu_ms != nullptr) {
        *out_submit_cpu_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - submit_start).count();
    }
    const UINT slot_index = state.command_slot_index % kRtCommandSlotCount;
    const rt_submission_token submission{state.next_submission_serial++};
    state.command_slots[slot_index].fence_value = fence_value;
    state.command_slots[slot_index].submission = submission;
    state.submitted_fence_value = fence_value;
    state.command_slot_index = (slot_index + 1u) % kRtCommandSlotCount;
    if (out_submission != nullptr) {
        *out_submission = submission;
    }
    return true;
}

bool copy_output_texture_to_native_display_target(
    dxr_backend_state &state,
    rt_device* device,
    rt_texture_handle source_texture,
    ID3D12Resource* source_resource,
    ID3D12Resource* texture_resource,
    double* out_submit_cpu_ms = nullptr)
{
    if (source_resource == nullptr || texture_resource == nullptr) {
        return false;
    }
    if (state.native_display_target != texture_resource) {
        state.native_display_target = texture_resource;
        state.native_display_target_state = D3D12_RESOURCE_STATE_COPY_DEST;
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
        append_rt_diagnostics_log_line("dxr_failure.log", detail);
    }
    rt_command_encoder encoder{};
    if (!begin_rt_commands(device, rt_queue_class::graphics, &encoder, nullptr)) {
        return false;
    }

    D3D12_RESOURCE_BARRIER barriers[4]{};
    UINT barrier_count = 0;
    if (state.native_display_target_state != D3D12_RESOURCE_STATE_COPY_DEST) {
        barriers[barrier_count].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barriers[barrier_count].Transition.pResource = texture_resource;
        barriers[barrier_count].Transition.StateBefore = state.native_display_target_state;
        barriers[barrier_count].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
        barriers[barrier_count].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        ++barrier_count;
    }
    if (barrier_count != 0) {
        state.command_list->ResourceBarrier(barrier_count, barriers);
    }
    if (!transition_rt_texture(
            device,
            encoder,
            source_texture,
            rt_resource_usage::copy_source,
            nullptr)) {
        discard_rt_commands(device, encoder);
        return false;
    }

    state.command_list->CopyResource(texture_resource, source_resource);

    D3D12_RESOURCE_BARRIER end_barrier{};
    end_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    end_barrier.Transition.pResource = texture_resource;
    end_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    end_barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    end_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    state.command_list->ResourceBarrier(1, &end_barrier);
    if (!transition_rt_texture(
            device,
            encoder,
            source_texture,
            rt_resource_usage::shader_write,
            nullptr)) {
        discard_rt_commands(device, encoder);
        return false;
    }

    rt_submission_token submission{};
    rt_device_timing timing{};
    if (!submit_rt_commands(device, encoder, &submission, &timing, nullptr)) {
        return false;
    }
    if (out_submit_cpu_ms != nullptr) {
        *out_submit_cpu_ms = timing.submit_cpu_ms;
    }
    state.native_display_target_state = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    return true;
}

// =============================================================================
// Device lifecycle and operation entry points.
// =============================================================================

bool initialize_dxr_native(dxr_backend_state &state, const rt_rhi_device_desc &desc) {
    const bool uses_shared_shell_device =
        desc.native_device != nullptr && desc.native_graphics_queue != nullptr;
    append_rt_startup_log(
        uses_shared_shell_device
            ? "DXR initialization begin: shared shell device"
            : "DXR initialization begin: independent device");
    if (desc.initial_width == 0 || desc.initial_height == 0) {
        append_rt_startup_log("DXR startup failed: invalid capture size");
        return false;
    }

    HRESULT hr = CreateDXGIFactory2(0, IID_PPV_ARGS(&state.factory));
    if (FAILED(hr)) {
        append_startup_hresult_log("DXR startup failed: CreateDXGIFactory2", hr);
        return false;
    }

    if (uses_shared_shell_device) {
        IUnknown* device_unknown = static_cast<IUnknown*>(desc.native_device);
        IUnknown* queue_unknown = static_cast<IUnknown*>(desc.native_graphics_queue);
        append_rt_startup_log("DXR shared shell device QueryInterface begin");
        hr = device_unknown->QueryInterface(IID_PPV_ARGS(&state.device));
        if (FAILED(hr)) {
            append_startup_hresult_log("DXR startup failed: external device QueryInterface", hr);
            return false;
        }
        append_rt_startup_log("DXR shared shell device QueryInterface complete");
        append_rt_startup_log("DXR shared shell command queue QueryInterface begin");
        hr = queue_unknown->QueryInterface(IID_PPV_ARGS(&state.queue));
        if (FAILED(hr)) {
            append_startup_hresult_log("DXR startup failed: external command queue QueryInterface", hr);
            return false;
        }
        append_rt_startup_log("DXR shared shell command queue QueryInterface complete");
        append_rt_startup_log("DXR shared shell CheckFeatureSupport D3D12_OPTIONS5 begin");
        state.raytracing_supported = device_supports_raytracing(state.device, &hr);
        append_rt_startup_log("DXR shared shell CheckFeatureSupport D3D12_OPTIONS5 complete");
        append_device_removed_reason_log(
            "DXR shared shell device status before allocator creation",
            state.device);
    } else {
        for (UINT index = 0;; ++index) {
            IDXGIAdapter1* candidate = nullptr;
            if (state.factory->EnumAdapters1(index, &candidate) == DXGI_ERROR_NOT_FOUND) {
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
                state.device = candidate_device;
                state.adapter = candidate;
                state.raytracing_supported = true;
                break;
            }
            safe_release(candidate_device);
            candidate->Release();
        }

        if (state.device == nullptr) {
            IDXGIAdapter* warp_adapter = nullptr;
            hr = state.factory->EnumWarpAdapter(IID_PPV_ARGS(&warp_adapter));
            if (FAILED(hr)) {
                append_startup_hresult_log("DXR startup failed: EnumWarpAdapter", hr);
                return false;
            }
            hr = D3D12CreateDevice(warp_adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&state.device));
            warp_adapter->Release();
            if (FAILED(hr)) {
                append_startup_hresult_log("DXR startup failed: D3D12CreateDevice", hr);
                return false;
            }
            state.raytracing_supported = device_supports_raytracing(state.device, &hr);
        }
    }

    if (FAILED(hr)) {
        append_startup_hresult_log("DXR startup failed: CheckFeatureSupport D3D12_OPTIONS5", hr);
    }
    if (!state.raytracing_supported) {
        append_rt_startup_log("DXR startup failed: no D3D12 adapter with raytracing tier support was found");
        return false;
    }
    if (!ensure_required_uav_format_support(state, &hr)) {
        if (FAILED(hr)) {
            append_startup_hresult_log("DXR startup failed: CheckFeatureSupport D3D12_FORMAT_SUPPORT", hr);
        } else {
            char detail[512]{};
            std::snprintf(
                detail,
                sizeof(detail),
                "DXR startup failed: required UAV format support missing "
                "(bgra8_store=%u rgba16f_load=%u rgba16f_store=%u)",
                static_cast<unsigned>(state.output_typed_uav_store_supported),
                static_cast<unsigned>(state.accumulation_typed_uav_load_supported),
                static_cast<unsigned>(state.accumulation_typed_uav_store_supported));
            append_rt_startup_log(detail);
        }
        return false;
    }
    state.native_d3d12_texture_present_supported =
        desc.native_device != nullptr &&
        desc.native_graphics_queue != nullptr &&
        state.device != nullptr &&
        state.queue != nullptr;

    if (state.queue == nullptr) {
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        hr = state.device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&state.queue));
        if (FAILED(hr)) {
            append_startup_hresult_log("DXR startup failed: CreateCommandQueue", hr);
            return false;
        }
        state.owns_device_queue = true;
        state.native_d3d12_texture_present_supported = false;
    }

    for (UINT i = 0; i < kRtCommandSlotCount; ++i) {
        hr = state.device->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(&state.command_slots[i].allocator));
        if (FAILED(hr)) {
            append_startup_hresult_log("DXR startup failed: CreateCommandAllocator", hr);
            append_device_removed_reason_log(
                "DXR shared shell device removal reason after CreateCommandAllocator",
                state.device);
            return false;
        }
    }

    hr = state.device->CreateCommandList(
        0,
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        state.command_slots[0].allocator,
        nullptr,
        IID_PPV_ARGS(&state.command_list));
    if (FAILED(hr)) {
        append_startup_hresult_log("DXR startup failed: CreateCommandList", hr);
        return false;
    }
    state.command_list->Close();

    hr = state.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&state.fence));
    if (FAILED(hr)) {
        append_startup_hresult_log("DXR startup failed: CreateFence", hr);
        return false;
    }

    state.fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (state.fence_event == nullptr) {
        append_rt_startup_log("DXR startup failed: CreateEventW");
        return false;
    }
    if (!ensure_timestamp_query_resources(state)) {
        append_rt_startup_log("DXR startup failed: ensure_timestamp_query_resources");
        return false;
    }
    if (!warmup_acceleration_structure_prebuild_info(state)) {
        return false;
    }

    state.initialized = true;
    append_rt_startup_log("DXR initialization complete");
    return true;
}

void shutdown_dxr_native(dxr_backend_state &state) {
    if (!state.initialized &&
        state.factory == nullptr &&
        state.device == nullptr &&
        state.queue == nullptr &&
        state.command_list == nullptr &&
        state.command_slots[0].allocator == nullptr &&
        state.fence == nullptr) {
        return;
    }

    clear_acceleration_cache(state);
    release_raytracing_runtime_state(state);
    state.texture_registry.clear([](dxr_texture &texture) {
        safe_release(texture.resource);
    });
    state.buffer_registry.clear([](dxr_buffer &resource) {
        safe_release(resource.resource);
    });
    collect_deferred_resource_releases(state, true);
    release_shader_modules(state);
    safe_release(state.command_list);
    safe_release(state.scratch_buffer);
    for (dxr_command_slot &slot : state.command_slots) {
        safe_release(slot.tlas_instance_buffer);
        safe_release(slot.allocator);
    }
    safe_release(state.timestamp_query_readback);
    safe_release(state.timestamp_query_heap);
    safe_release(state.fence);
    safe_release(state.queue);
    safe_release(state.device);
    safe_release(state.adapter);
    safe_release(state.factory);
    if (state.fence_event != nullptr) {
        CloseHandle(state.fence_event);
        state.fence_event = nullptr;
    }
    state.timestamp_frequency = 0;
    for (dxr_command_slot &slot : state.command_slots) {
        slot.dispatch_timestamp_submission = {};
        slot.dispatch_timestamp_pending = false;
        slot.accel_timestamp_submission = {};
        slot.accel_timestamp_pending = false;
    }
    *&state = {};
}

d3d12_dxr_rhi_device::d3d12_dxr_rhi_device() {
    device_.kind = rt_device_kind::d3d12_dxr;
    device_.api = this;
    device_.native_state = &native_state_;
}

d3d12_dxr_rhi_device::~d3d12_dxr_rhi_device() {
    if (device_.api == this) {
        device_.api = nullptr;
    }
}

rt_rhi_device_info d3d12_dxr_rhi_device::info() const {
    std::scoped_lock lock(device_.access_mutex);
    return dxr_backend_info(native_state_);
}

rt_device* d3d12_dxr_rhi_device::device() {
    return &device_;
}

rt_native_texture_extension* d3d12_dxr_rhi_device::native_texture_extension() {
    return this;
}

rt_vulkan_interop_extension* d3d12_dxr_rhi_device::vulkan_interop_extension() {
    return nullptr;
}

bool d3d12_dxr_rhi_device::initialize(
    const rt_rhi_device_desc &desc,
    rt_device_error* out_error)
{
    if (out_error != nullptr) {
        *out_error = {rt_device_operation::initialize, 0, {}};
    }
    if (device_.native_state != &native_state_) {
        if (out_error != nullptr) {
            out_error->detail = "DXR device state is unavailable";
        }
        return false;
    }
    std::scoped_lock lock(device_.access_mutex);
    if (!initialize_dxr_native(native_state_, desc)) {
        if (out_error != nullptr) {
            out_error->detail = "DXR native initialization failed";
        }
        return false;
    }

    device_.capabilities.hardware_ray_tracing = native_state_.raytracing_supported;
    device_.capabilities.timestamp_queries = timestamp_queries_enabled(native_state_);
    device_.capabilities.native_d3d12_target =
        native_state_.native_d3d12_texture_present_supported;
    device_.capabilities.bgra_capture = true;
    device_.capabilities.bgra_readback = true;
    device_.capabilities.shader_binary_format = rt_shader_binary_format::dxil_library;
    device_.capabilities.output_format = rt_texture_format::bgra8_unorm;
    device_.capabilities.accumulation_format = rt_texture_format::rgba16_float;
    return true;
}

bool d3d12_dxr_rhi_device::wait_idle(
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
            out_error->detail = "DXR device state is unavailable";
        }
        return false;
    }

    std::scoped_lock lock(device_.access_mutex);
    if (native_state_.queue == nullptr ||
        native_state_.fence == nullptr ||
        native_state_.fence_event == nullptr) {
        return true;
    }

    double wait_ms = 0.0;
    const bool waited = wait_for_fence_value_timed(native_state_, native_state_.submitted_fence_value, &wait_ms);
    if (waited) {
        native_state_.completed_submission_serial =
            native_state_.next_submission_serial > 1
                ? native_state_.next_submission_serial - 1u
                : 0;
        collect_deferred_resource_releases(native_state_);
    }
    if (out_timing != nullptr) {
        out_timing->gpu_wait_ms = wait_ms;
    }
    if (!waited && out_error != nullptr) {
        out_error->native_code = static_cast<std::int64_t>(GetLastError());
        out_error->detail = "DXR fence wait failed";
    }
    return waited;
}

bool d3d12_dxr_rhi_device::begin_commands(
    rt_queue_class queue,
    rt_command_encoder* out_encoder,
    rt_device_error* out_error)
{
    collect_completed_dispatch_timestamp_queries(native_state_);
    if (out_encoder != nullptr) {
        *out_encoder = {};
    }
    if (device_.native_state != &native_state_) {
        if (out_error != nullptr) {
            *out_error = {rt_device_operation::begin_commands, 0, "DXR command device state is invalid"};
        }
        return false;
    }
    if (out_encoder == nullptr || queue != rt_queue_class::graphics) {
        if (out_error != nullptr) {
            *out_error = {rt_device_operation::begin_commands, 0, "DXR command queue request is invalid"};
        }
        return false;
    }
    if (native_state_.active_encoder_id != 0) {
        if (out_error != nullptr) {
            *out_error = {rt_device_operation::begin_commands, 0, "DXR command encoder is already active"};
        }
        return false;
    }
    if (!reset_command_list(native_state_)) {
        if (out_error != nullptr) {
            *out_error = {rt_device_operation::begin_commands, 0, "DXR command list reset failed"};
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

bool d3d12_dxr_rhi_device::submit_commands(
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
            *out_error = {rt_device_operation::submit_commands, 0, "DXR command submit is invalid"};
        }
        return false;
    }
    double submit_cpu_ms = 0.0;
    const bool submitted = close_and_execute_command_list(
        native_state_,
        &submit_cpu_ms,
        out_submission);
    dxr_acceleration_build_context &acceleration = native_state_.acceleration_build;
    if (acceleration.active) {
        native_state_.diagnostics.acceleration_submit_cpu_ms = submit_cpu_ms;
        native_state_.diagnostics.acceleration_gpu_wait_ms = 0.0;
        native_state_.diagnostics.acceleration_cpu_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - acceleration.total_start).count();
        if (submitted && acceleration.recorded) {
            native_state_.command_slots[acceleration.timestamp_slot].accel_timestamp_submission =
                *out_submission;
            native_state_.command_slots[acceleration.timestamp_slot].accel_timestamp_pending = true;
        }
        acceleration = {};
    }
    if (submitted && native_state_.active_dispatch_timestamp_recorded) {
        const UINT timestamp_slot = native_state_.active_command_slot_index;
        native_state_.command_slots[timestamp_slot].dispatch_timestamp_submission = *out_submission;
        native_state_.command_slots[timestamp_slot].dispatch_timestamp_pending = true;
    }
    native_state_.active_dispatch_timestamp_recorded = false;
    native_state_.active_encoder_id = 0;
    if (out_timing != nullptr) {
        out_timing->submit_cpu_ms = submit_cpu_ms;
        out_timing->gpu_ms = native_state_.diagnostics.dispatch_gpu_ms;
    }
    if (!submitted && out_error != nullptr) {
        *out_error = {rt_device_operation::submit_commands, 0, "DXR command submit failed"};
    }
    return submitted;
}

void d3d12_dxr_rhi_device::discard_commands(
    rt_command_encoder encoder)
{
    if (device_.native_state != &native_state_ ||
        !encoder || encoder.id != native_state_.active_encoder_id) {
        return;
    }
    native_state_.acceleration_build = {};
    native_state_.active_dispatch_timestamp_recorded = false;
    native_state_.active_encoder_id = 0;
}

bool d3d12_dxr_rhi_device::is_complete(
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
            *out_error = {rt_device_operation::query_submission, 0, "DXR submission query is invalid"};
        }
        return false;
    }
    *out_complete = is_submission_complete(native_state_, submission);
    return true;
}

bool d3d12_dxr_rhi_device::wait(
    rt_submission_token submission,
    rt_device_timing* out_timing,
    rt_device_error* out_error)
{
    if (out_timing != nullptr) {
        *out_timing = {};
    }
    if (device_.native_state != &native_state_ || !submission) {
        if (out_error != nullptr) {
            *out_error = {rt_device_operation::wait_submission, 0, "DXR submission wait is invalid"};
        }
        return false;
    }
    double wait_ms = 0.0;
    const bool completed = wait_for_submission(native_state_, submission, &wait_ms);
    if (out_timing != nullptr) {
        out_timing->gpu_wait_ms = wait_ms;
    }
    if (!completed && out_error != nullptr) {
        *out_error = {rt_device_operation::wait_submission, 0, "DXR submission wait failed"};
    }
    return completed;
}

D3D12_RESOURCE_STATES dxr_resource_state(rt_resource_usage usage) {
    switch (usage) {
    case rt_resource_usage::undefined:
        return D3D12_RESOURCE_STATE_COMMON;
    case rt_resource_usage::shader_write:
    case rt_resource_usage::clear_destination:
        return D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    case rt_resource_usage::shader_read:
        return D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    case rt_resource_usage::acceleration_build_input:
        return D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    case rt_resource_usage::acceleration_storage:
        return D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
    case rt_resource_usage::copy_source:
        return D3D12_RESOURCE_STATE_COPY_SOURCE;
    case rt_resource_usage::copy_destination:
    case rt_resource_usage::host_read:
        return D3D12_RESOURCE_STATE_COPY_DEST;
    }
    return D3D12_RESOURCE_STATE_COMMON;
}

bool d3d12_dxr_rhi_device::barrier(
    rt_command_encoder encoder,
    const rt_resource_barrier* barriers,
    std::size_t barrier_count,
    rt_device_error* out_error)
{
    if (device_.native_state != &native_state_ ||
        !encoder || encoder.id != native_state_.active_encoder_id ||
        (barrier_count != 0 && barriers == nullptr)) {
        if (out_error != nullptr) {
            *out_error = {rt_device_operation::transition_resource, 0, "DXR barrier request is invalid"};
        }
        return false;
    }

    std::vector<D3D12_RESOURCE_BARRIER> native_barriers;
    native_barriers.reserve(barrier_count);
    for (std::size_t index = 0; index < barrier_count; ++index) {
        const rt_resource_barrier &source = barriers[index];
        ID3D12Resource* resource = nullptr;
        D3D12_RESOURCE_STATES before = dxr_resource_state(source.before);
        const D3D12_RESOURCE_STATES after = dxr_resource_state(source.after);
        dxr_texture* texture = nullptr;
        if (source.kind == rt_resource_kind::texture) {
            texture = native_state_.texture_registry.get(source.texture);
            if (texture != nullptr) {
                resource = texture->resource;
                before = texture->state;
            }
        } else {
            dxr_buffer* const buffer =
                native_state_.buffer_registry.get(source.buffer);
            if (buffer != nullptr) {
                resource = buffer->resource;
                before = buffer->state;
            }
        }
        if (resource == nullptr) {
            if (out_error != nullptr) {
                *out_error = {rt_device_operation::transition_resource, 0, "DXR barrier resource is invalid"};
            }
            return false;
        }
        if (before == after && after == D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
            D3D12_RESOURCE_BARRIER barrier{};
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            barrier.UAV.pResource = resource;
            native_barriers.push_back(barrier);
        } else if (before != after) {
            D3D12_RESOURCE_BARRIER barrier{};
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Transition.pResource = resource;
            barrier.Transition.StateBefore = before;
            barrier.Transition.StateAfter = after;
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            native_barriers.push_back(barrier);
        }
        if (texture != nullptr) {
            texture->state = after;
        } else if (dxr_buffer* const buffer =
                native_state_.buffer_registry.get(source.buffer)) {
            buffer->state = after;
        }
    }
    if (!native_barriers.empty()) {
        native_state_.command_list->ResourceBarrier(
            static_cast<UINT>(native_barriers.size()),
            native_barriers.data());
    }
    return true;
}

bool d3d12_dxr_rhi_device::copy_buffer(
    rt_command_encoder encoder,
    rt_buffer_handle source,
    rt_buffer_handle destination,
    const rt_buffer_copy_region &region,
    rt_device_error* out_error)
{
    ID3D12Resource* const source_resource = dxr_buffer_resource(native_state_, source);
    ID3D12Resource* const destination_resource = dxr_buffer_resource(native_state_, destination);
    const bool valid = device_.native_state == &native_state_ &&
        encoder && encoder.id == native_state_.active_encoder_id &&
        source_resource != nullptr && destination_resource != nullptr &&
        region.size > 0 &&
        region.source_offset <= source_resource->GetDesc().Width &&
        region.size <= source_resource->GetDesc().Width - region.source_offset &&
        region.destination_offset <= destination_resource->GetDesc().Width &&
        region.size <= destination_resource->GetDesc().Width - region.destination_offset;
    if (!valid) {
        if (out_error != nullptr) {
            *out_error = {rt_device_operation::copy_resource, 0, "DXR buffer copy request is invalid"};
        }
        return false;
    }
    native_state_.command_list->CopyBufferRegion(
        destination_resource,
        region.destination_offset,
        source_resource,
        region.source_offset,
        region.size);
    return true;
}

bool d3d12_dxr_rhi_device::copy_texture_to_buffer(
    rt_command_encoder encoder,
    rt_texture_handle source,
    rt_buffer_handle destination,
    const rt_texture_buffer_copy_region &region,
    rt_device_error* out_error)
{
    const dxr_texture* const source_texture =
        native_state_.texture_registry.get(source);
    ID3D12Resource* const destination_buffer =
        dxr_buffer_resource(native_state_, destination);
    const std::size_t bytes_per_pixel = source_texture != nullptr
        ? rt_texture_format_bytes_per_pixel(source_texture->desc.format)
        : 0;
    const std::size_t required_size = region.buffer_offset +
        region.buffer_row_pitch * static_cast<std::size_t>(region.height);
    const bool valid = device_.native_state == &native_state_ &&
        encoder && encoder.id == native_state_.active_encoder_id &&
        source_texture != nullptr && source_texture->resource != nullptr &&
        destination_buffer != nullptr &&
        region.width > 0 && region.height > 0 && bytes_per_pixel > 0 &&
        region.width <= source_texture->desc.width &&
        region.height <= source_texture->desc.height &&
        region.buffer_row_pitch >= static_cast<std::size_t>(region.width) * bytes_per_pixel &&
        region.buffer_row_pitch % D3D12_TEXTURE_DATA_PITCH_ALIGNMENT == 0 &&
        required_size <= destination_buffer->GetDesc().Width;
    if (!valid) {
        if (out_error != nullptr) {
            *out_error = {
                rt_device_operation::copy_resource,
                0,
                "DXR texture-to-buffer copy request is invalid"};
        }
        return false;
    }

    D3D12_TEXTURE_COPY_LOCATION source_location{};
    source_location.pResource = source_texture->resource;
    source_location.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_TEXTURE_COPY_LOCATION destination_location{};
    destination_location.pResource = destination_buffer;
    destination_location.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    destination_location.PlacedFootprint.Offset = region.buffer_offset;
    destination_location.PlacedFootprint.Footprint.Format =
        dxr_texture_format(source_texture->desc.format);
    destination_location.PlacedFootprint.Footprint.Width = region.width;
    destination_location.PlacedFootprint.Footprint.Height = region.height;
    destination_location.PlacedFootprint.Footprint.Depth = 1;
    destination_location.PlacedFootprint.Footprint.RowPitch =
        static_cast<UINT>(region.buffer_row_pitch);
    native_state_.command_list->CopyTextureRegion(
        &destination_location,
        0,
        0,
        0,
        &source_location,
        nullptr);
    return true;
}

bool d3d12_dxr_rhi_device::clear_texture(
    rt_command_encoder encoder,
    rt_texture_handle texture,
    const float color[4],
    rt_device_error* out_error)
{
    const dxr_texture* const texture_object =
        native_state_.texture_registry.get(texture);
    const bool valid = device_.native_state == &native_state_ &&
        encoder && encoder.id == native_state_.active_encoder_id &&
        texture_object != nullptr && texture_object->resource != nullptr &&
        color != nullptr &&
        (texture_object->desc.usage & rt_texture_usage_shader_write) != 0u &&
        ensure_descriptor_heap(native_state_);
    if (!valid) {
        if (out_error != nullptr) {
            *out_error = {rt_device_operation::clear_texture, 0, "DXR texture clear request is invalid"};
        }
        return false;
    }

    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.Format = dxr_texture_format(texture_object->desc.format);
    uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    native_state_.device->CreateUnorderedAccessView(
        texture_object->resource,
        nullptr,
        &uav,
        descriptor_cpu_handle(
            native_state_,
            native_state_.active_command_slot_index,
            kScratchUavDescriptorIndex));
    ID3D12DescriptorHeap* heaps[] = {native_state_.srv_uav_cbv_heap};
    native_state_.command_list->SetDescriptorHeaps(1, heaps);
    native_state_.command_list->ClearUnorderedAccessViewFloat(
        descriptor_gpu_handle(
            native_state_,
            native_state_.active_command_slot_index,
            kScratchUavDescriptorIndex),
        descriptor_cpu_handle(
            native_state_,
            native_state_.active_command_slot_index,
            kScratchUavDescriptorIndex),
        texture_object->resource,
        color,
        0,
        nullptr);
    return true;
}

bool d3d12_dxr_rhi_device::trace_rays(
    rt_command_encoder encoder,
    const rt_trace_rays_desc &desc,
    rt_device_error* out_error)
{
    ID3D12StateObject* const* const pipeline =
        native_state_.pipeline_registry.get(desc.pipeline);
    ID3D12Resource* const* const shader_table =
        native_state_.shader_table_registry.get(desc.shader_table);
    const bool valid = device_.native_state == &native_state_ &&
        encoder && encoder.id == native_state_.active_encoder_id &&
        pipeline != nullptr && *pipeline == native_state_.raytracing_state_object &&
        shader_table != nullptr && *shader_table == native_state_.raygen_shader_table &&
        native_state_.raygen_shader_table != nullptr &&
        desc.ray_generation_record < native_state_.raygen_shader_record_count &&
        native_state_.miss_shader_table != nullptr &&
        native_state_.hitgroup_shader_table != nullptr &&
        native_state_.camera_constant_buffer != nullptr &&
        native_state_.global_root_signature != nullptr &&
        native_state_.srv_uav_cbv_heap != nullptr &&
        desc.width > 0u && desc.height > 0u && desc.depth > 0u;
    if (!valid) {
        if (out_error != nullptr) {
            *out_error = {rt_device_operation::trace_rays, 0, "DXR ray trace request is invalid"};
        }
        return false;
    }
    if (!refresh_command_slot_bindings(
            native_state_,
            native_state_.active_command_slot_index)) {
        if (out_error != nullptr) {
            *out_error = {
                rt_device_operation::update_bindings,
                0,
                "DXR descriptor snapshot refresh failed"};
        }
        return false;
    }

    ID3D12DescriptorHeap* heaps[] = {native_state_.srv_uav_cbv_heap};
    native_state_.command_list->SetDescriptorHeaps(1, heaps);
    native_state_.command_list->SetComputeRootSignature(native_state_.global_root_signature);
    native_state_.command_list->SetComputeRootDescriptorTable(
        0,
        descriptor_gpu_handle(
            native_state_,
            native_state_.active_command_slot_index,
            kUavDescriptorBase));
    native_state_.command_list->SetComputeRootDescriptorTable(
        1,
        descriptor_gpu_handle(
            native_state_,
            native_state_.active_command_slot_index,
            kSrvDescriptorBase));
    native_state_.command_list->SetComputeRootConstantBufferView(
        2,
        native_state_.camera_constant_buffer->GetGPUVirtualAddress() +
            static_cast<D3D12_GPU_VIRTUAL_ADDRESS>(
                native_state_.active_command_slot_index * kRtViewerConstantSlotStride));
    native_state_.command_list->SetPipelineState1(*pipeline);

    D3D12_DISPATCH_RAYS_DESC native_desc{};
    native_desc.RayGenerationShaderRecord.StartAddress =
        native_state_.raygen_shader_table->GetGPUVirtualAddress() +
        kShaderRecordSize * desc.ray_generation_record;
    native_desc.RayGenerationShaderRecord.SizeInBytes = kShaderRecordSize;
    native_desc.MissShaderTable.StartAddress =
        native_state_.miss_shader_table->GetGPUVirtualAddress();
    native_desc.MissShaderTable.SizeInBytes =
        kShaderRecordSize * native_state_.miss_shader_record_count;
    native_desc.MissShaderTable.StrideInBytes = kShaderRecordSize;
    native_desc.HitGroupTable.StartAddress =
        native_state_.hitgroup_shader_table->GetGPUVirtualAddress();
    native_desc.HitGroupTable.SizeInBytes =
        kShaderRecordSize * native_state_.hitgroup_shader_record_count;
    native_desc.HitGroupTable.StrideInBytes = kShaderRecordSize;
    if (native_state_.callable_shader_record_count != 0) {
        native_desc.CallableShaderTable.StartAddress =
            native_state_.callable_shader_table->GetGPUVirtualAddress();
        native_desc.CallableShaderTable.SizeInBytes =
            kShaderRecordSize * native_state_.callable_shader_record_count;
        native_desc.CallableShaderTable.StrideInBytes = kShaderRecordSize;
    }
    native_desc.Width = desc.width;
    native_desc.Height = desc.height;
    native_desc.Depth = desc.depth;
    if (desc.measure_gpu_time) {
        if (!ensure_timestamp_query_resources(native_state_)) {
            return false;
        }
        write_timestamp_query_begin(
            native_state_,
            native_state_.active_command_slot_index,
            timestamp_query_region::dispatch);
    }
    native_state_.command_list->DispatchRays(&native_desc);
    if (desc.measure_gpu_time) {
        write_timestamp_query_end_and_resolve(
            native_state_,
            native_state_.active_command_slot_index,
            timestamp_query_region::dispatch);
        native_state_.active_dispatch_timestamp_recorded = true;
    }
    return true;
}

bool d3d12_dxr_rhi_device::shutdown(rt_device_error* out_error) {
    if (out_error != nullptr) {
        *out_error = {rt_device_operation::shutdown, 0, {}};
    }

    std::scoped_lock lock(device_.access_mutex);
    shutdown_dxr_native(native_state_);
    device_.capabilities = {};
    return true;
}

bool d3d12_dxr_rhi_device::update_bindings(
    const rt_binding_update_request &request,
    rt_device_error* out_error)
{
    std::string detail;
    if (device_.native_state != &native_state_ ||
        request.writes == nullptr || request.write_count == 0) {
        if (out_error != nullptr) {
            out_error->detail = "DXR descriptor binding update request is invalid";
        }
        return false;
    }

    native_state_.pending_binding_writes.assign(
        request.writes,
        request.writes + request.write_count);
    ++native_state_.binding_generation;
    const UINT slot_index = descriptor_target_slot(native_state_);
    if (native_state_.active_encoder_id == 0) {
        dxr_command_slot &slot = native_state_.command_slots[slot_index];
        if (slot.submission && !wait_for_fence_value(native_state_, slot.fence_value)) {
            if (out_error != nullptr) {
                out_error->detail = "DXR descriptor binding slot wait failed";
            }
            return false;
        }
        if (slot.submission) {
            complete_command_slot(native_state_, slot_index);
        }
    }
    const rt_binding_update_request pending_request{
        native_state_.pending_binding_writes.data(),
        native_state_.pending_binding_writes.size()};
    if (!update_dxr_resource_bindings(native_state_, pending_request, slot_index, &detail)) {
        if (out_error != nullptr) {
            out_error->detail = detail.empty() ? "DXR descriptor binding update failed" : std::move(detail);
        }
        return false;
    }
    native_state_.command_slot_binding_generations[slot_index] = native_state_.binding_generation;
    return true;
}

bool d3d12_dxr_rhi_device::create_shader_module(
    const rt_shader_module_desc &desc,
    rt_shader_module_handle* out_module,
    rt_device_error* out_error)
{
    if (out_module != nullptr) {
        *out_module = {};
    }
    if (device_.native_state != &native_state_ ||
        desc.format != rt_shader_binary_format::dxil_library ||
        desc.data == nullptr || desc.size == 0 || out_module == nullptr) {
        if (out_error != nullptr) {
            out_error->detail = "DXR shader module descriptor is invalid";
        }
        return false;
    }
    const auto* const begin = static_cast<const std::uint8_t*>(desc.data);
    std::vector<std::uint8_t> binary(begin, begin + desc.size);
    if (!native_state_.shader_module_registry.insert(std::move(binary), out_module)) {
        if (out_error != nullptr) {
            out_error->detail = "DXR shader module handle allocation failed";
        }
        return false;
    }
    return true;
}

void d3d12_dxr_rhi_device::destroy_shader_module(
    rt_shader_module_handle module)
{
    if (device_.native_state == &native_state_) {
        native_state_.shader_module_registry.erase(module);
    }
}

bool d3d12_dxr_rhi_device::create_pipeline(
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
            out_error->detail = "DXR pipeline build request is invalid";
        }
        return false;
    }
    if (!ensure_state_object(native_state_, desc)) {
        if (out_error != nullptr) {
            out_error->detail = "DXR pipeline creation failed";
        }
        return false;
    }
    ID3D12StateObject** registered_pipeline =
        native_state_.pipeline_registry.get(native_state_.pipeline);
    if (registered_pipeline == nullptr ||
        *registered_pipeline != native_state_.raytracing_state_object) {
        native_state_.pipeline_registry.clear([](ID3D12StateObject* &) {});
        native_state_.pipeline = {};
        if (!native_state_.pipeline_registry.insert(
                native_state_.raytracing_state_object,
                &native_state_.pipeline)) {
            if (out_error != nullptr) {
                out_error->detail = "DXR pipeline handle allocation failed";
            }
            return false;
        }
    }
    *out_pipeline = native_state_.pipeline;
    return true;
}

bool d3d12_dxr_rhi_device::create_shader_table(
    const rt_shader_table_desc &desc,
    rt_shader_table_handle* out_shader_table,
    rt_device_error* out_error)
{
    if (out_shader_table != nullptr) {
        *out_shader_table = {};
    }
    ID3D12StateObject* const* const registered_pipeline =
        native_state_.pipeline_registry.get(desc.pipeline);
    if (device_.native_state != &native_state_ ||
        registered_pipeline == nullptr ||
        *registered_pipeline != native_state_.raytracing_state_object ||
        out_shader_table == nullptr ||
        !ensure_shader_tables(native_state_, desc)) {
        if (out_error != nullptr) {
            out_error->detail = "DXR shader table creation failed";
        }
        return false;
    }
    ID3D12Resource** registered_shader_table =
        native_state_.shader_table_registry.get(native_state_.shader_table);
    if (registered_shader_table == nullptr ||
        *registered_shader_table != native_state_.raygen_shader_table) {
        native_state_.shader_table_registry.clear([](ID3D12Resource* &) {});
        native_state_.shader_table = {};
        if (!native_state_.shader_table_registry.insert(
                native_state_.raygen_shader_table,
                &native_state_.shader_table)) {
            if (out_error != nullptr) {
                out_error->detail = "DXR shader table handle allocation failed";
            }
            return false;
        }
    }
    *out_shader_table = native_state_.shader_table;
    return true;
}

void d3d12_dxr_rhi_device::get_diagnostics(rt_rhi_diagnostics* out_diagnostics) const {
    if (out_diagnostics != nullptr) {
        *out_diagnostics = native_state_.diagnostics;
    }
}

bool d3d12_dxr_rhi_device::publish_texture(
    rt_texture_handle texture,
    const rt_native_texture_publish_desc &desc,
    rt_device_timing* out_timing,
    rt_device_error* out_error)
{
    if (out_timing != nullptr) {
        *out_timing = {};
    }
    if (device_.native_state != &native_state_ ||
        texture != device_.output_texture || desc.target == nullptr) {
        if (out_error != nullptr) {
            out_error->operation = rt_device_operation::native_texture;
            out_error->detail = "DXR native output request is invalid";
        }
        return false;
    }
    dxr_texture* const source = native_state_.texture_registry.get(texture);
    if (source == nullptr || source->resource == nullptr) {
        if (out_error != nullptr) {
            out_error->operation = rt_device_operation::native_texture;
            out_error->detail = "DXR native output texture is invalid";
        }
        return false;
    }

    double submit_cpu_ms = 0.0;
    bool succeeded = copy_output_texture_to_native_display_target(
        native_state_,
        &device_,
        texture,
        source->resource,
        static_cast<ID3D12Resource*>(desc.target),
        &submit_cpu_ms);
    native_state_.diagnostics.dispatch_submit_cpu_ms += submit_cpu_ms;
    native_state_.diagnostics.dispatch_cpu_ms =
        native_state_.diagnostics.dispatch_submit_cpu_ms + native_state_.diagnostics.dispatch_gpu_wait_ms;
    if (out_timing != nullptr) {
        out_timing->submit_cpu_ms = submit_cpu_ms;
        out_timing->gpu_wait_ms = 0.0;
    }
    if (!succeeded && out_error != nullptr) {
        out_error->operation = rt_device_operation::native_texture;
        out_error->detail = "DXR native output publication failed";
    }
    return succeeded;
}
} // namespace

std::unique_ptr<rt_rhi_device> create_d3d12_dxr_rhi_device() {
    return std::make_unique<d3d12_dxr_rhi_device>();
}

} // namespace rtvdb::viewer_backend
