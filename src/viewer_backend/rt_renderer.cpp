#include "viewer_backend/rt_renderer.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <limits>

namespace rtvdb::viewer_backend {

bool resolve_layer_tlas_instance_source(
    const rt_scene_build &build,
    const rt_acceleration_build_item &item,
    std::string* out_layer,
    bool* out_fallback_visible,
    bool* out_layer_visibility_exempt);

namespace {

constexpr std::size_t kRtPickResultBufferBytes = sizeof(rt_pick_gpu_result);

void set_missing_lifecycle_error(
    rt_rhi_error* out_error,
    rt_rhi_operation operation,
    const char* detail)
{
    if (out_error == nullptr) {
        return;
    }
    out_error->operation = operation;
    out_error->native_code = 0;
    out_error->detail = detail != nullptr ? detail : "RT renderer lifecycle operation is unavailable";
}

void set_frame_error(
    rt_rhi_error* out_error,
    rt_rhi_operation operation,
    const char* detail)
{
    if (out_error == nullptr) {
        return;
    }
    *out_error = {operation, 0, detail != nullptr ? detail : "RT renderer frame operation failed"};
}

void latch_rt_renderer_async_error(rt_renderer* renderer, const rt_rhi_error &error) {
    if (renderer == nullptr || renderer->async_failed) {
        return;
    }
    renderer->async_error = error;
    if (renderer->async_error.detail.empty()) {
        renderer->async_error.detail = "RT renderer asynchronous submission failed";
    }
    renderer->async_failed = true;
    renderer->frame_state.scene_valid = false;
    renderer->frame_state.output_valid = false;
    renderer->last_acceleration_revision = 0;
    renderer->last_acceleration_summary = {};
    renderer->accumulation_state.sample_count = 0;
    renderer->accumulation_state.submitted_sample_count = 0;
    ++renderer->accumulation_state.generation;
    renderer->accumulation_state.active = false;
}

bool environment_flag_enabled(const char* name) {
    if (name == nullptr || *name == '\0') {
        return false;
    }
    const char* const value = std::getenv(name);
    return value != nullptr &&
        (value[0] == '1' || value[0] == 't' || value[0] == 'T' || value[0] == 'y' || value[0] == 'Y');
}

bool consume_rt_renderer_test_query_failure() {
    static bool consumed = false;
    if (consumed || !environment_flag_enabled("RTVDB_TEST_FAIL_RENDER_SUBMISSION_QUERY_ONCE")) {
        return false;
    }
    consumed = true;
    return true;
}

bool consume_rt_renderer_test_delivery_query_failure() {
    static bool consumed = false;
    if (consumed || !environment_flag_enabled("RTVDB_TEST_FAIL_DELIVERY_SUBMISSION_QUERY_ONCE")) {
        return false;
    }
    consumed = true;
    return true;
}

void destroy_viewer_rt_shader_modules(rt_renderer* renderer) {
    if (renderer == nullptr || renderer->rhi == nullptr) {
        return;
    }
    for (rt_shader_module_handle module : renderer->shader_modules) {
        renderer->rhi->destroy_shader_module(module);
    }
    renderer->shader_modules.clear();
    renderer->shader_entry_modules = {};
}

bool create_viewer_rt_shader_module(
    rt_renderer* renderer,
    const rt_shader_module_desc &desc,
    rt_rhi_error* out_error)
{
    rt_shader_module_handle module{};
    if (!renderer->rhi->create_shader_module(desc, &module, out_error) || !module) {
        return false;
    }
    renderer->shader_modules.push_back(module);
    return true;
}

bool ensure_viewer_rt_shader_modules(rt_renderer* renderer, rt_rhi_error* out_error) {
    if (renderer == nullptr || renderer->rhi == nullptr) {
        set_frame_error(
            out_error,
            rt_rhi_operation::create_shader_module,
            "RT shader module creation is unavailable");
        return false;
    }
    if (!renderer->shader_modules.empty()) {
        return true;
    }
    const rt_shader_package_desc &package = renderer->shader_package;
    if (!validate_rt_shader_package_desc(package)) {
        set_frame_error(
            out_error,
            rt_rhi_operation::create_shader_module,
            "RT shader package descriptor is invalid");
        return false;
    }
    bool created = true;
    for (std::size_t module_index = 0; module_index < package.module_count; ++module_index) {
        if (!create_viewer_rt_shader_module(renderer, package.modules[module_index], out_error)) {
            created = false;
            break;
        }
    }
    if (!created) {
        destroy_viewer_rt_shader_modules(renderer);
        if (out_error != nullptr && out_error->detail.empty()) {
            set_frame_error(
                out_error,
                rt_rhi_operation::create_shader_module,
                "RT shader package module creation failed");
        }
        return false;
    }
    renderer->shader_entry_modules = {};
    for (std::size_t entry_index = 0; entry_index < package.entry_count; ++entry_index) {
        const std::size_t logical_index =
            static_cast<std::size_t>(package.logical_entries[entry_index]);
        renderer->shader_entry_modules[logical_index] =
            renderer->shader_modules[package.entry_module_indices[entry_index]];
    }
    return true;
}

void destroy_scene_buffers(rt_renderer* renderer, rt_scene_buffer_resources* resources) {
    if (renderer == nullptr || renderer->rhi == nullptr || resources == nullptr) {
        return;
    }
    renderer->rhi->destroy_buffer(resources->positions);
    renderer->rhi->destroy_buffer(resources->indices);
    renderer->rhi->destroy_buffer(resources->triangle_colors);
    renderer->rhi->destroy_buffer(resources->instance_metadata);
    renderer->rhi->destroy_buffer(resources->points);
    renderer->rhi->destroy_buffer(resources->lines);
    renderer->rhi->destroy_buffer(resources->point_aabbs);
    renderer->rhi->destroy_buffer(resources->line_aabbs);
    for (const rt_scene_buffer_upload &upload : resources->uploads) {
        renderer->rhi->destroy_buffer(upload.staging);
    }
    *resources = {};
}

void destroy_scene_upload_buffers(rt_renderer* renderer, rt_scene_buffer_resources* resources) {
    if (renderer == nullptr || renderer->rhi == nullptr || resources == nullptr) {
        return;
    }
    for (const rt_scene_buffer_upload &upload : resources->uploads) {
        renderer->rhi->destroy_buffer(upload.staging);
    }
    resources->uploads.clear();
}

constexpr std::size_t kRtSceneBufferPoolMaxEntries = kRtCommandSlotCount * 8u;
constexpr std::size_t kRtSceneBufferPoolMaxBytes = std::size_t{512} * 1024u * 1024u;
constexpr std::size_t kRtBlasStoragePoolMaxEntries =
    kRtCommandSlotCount * kRtBlasCachePoolCount * kRtBlasChunkSetChunkCount;
constexpr std::size_t kRtBlasStoragePoolMaxBytes = std::size_t{512} * 1024u * 1024u;

std::size_t add_diagnostic_capacity(std::size_t total, std::size_t value) {
    const std::size_t max_size = (std::numeric_limits<std::size_t>::max)();
    return value > max_size - total ? max_size : total + value;
}

std::size_t scene_buffer_capacity_bytes(const rt_scene_buffer_resources &resources) {
    std::size_t total = 0;
    total = add_diagnostic_capacity(total, resources.positions_capacity_bytes);
    total = add_diagnostic_capacity(total, resources.indices_capacity_bytes);
    total = add_diagnostic_capacity(total, resources.triangle_colors_capacity_bytes);
    total = add_diagnostic_capacity(total, resources.instance_metadata_capacity_bytes);
    total = add_diagnostic_capacity(total, resources.points_capacity_bytes);
    total = add_diagnostic_capacity(total, resources.lines_capacity_bytes);
    total = add_diagnostic_capacity(total, resources.point_aabbs_capacity_bytes);
    total = add_diagnostic_capacity(total, resources.line_aabbs_capacity_bytes);
    return total;
}

std::size_t blas_cache_capacity_bytes(const rt_blas_cache_state &state) {
    std::size_t total = 0;
    for (const std::vector<rt_blas_cache_slot> &pool : state.pools) {
        for (const rt_blas_cache_slot &slot : pool) {
            if (slot.acceleration) {
                total = add_diagnostic_capacity(total, slot.storage_capacity_bytes);
            }
        }
    }
    return total;
}

std::size_t scene_buffer_pool_capacity_bytes(const rt_renderer &renderer) {
    std::size_t total = 0;
    for (const rt_scene_buffer_pool_entry &entry : renderer.scene_buffer_pool) {
        total = add_diagnostic_capacity(total, entry.capacity_bytes);
    }
    return total;
}

std::size_t blas_storage_pool_capacity_bytes(const rt_renderer &renderer) {
    std::size_t total = 0;
    for (const rt_blas_storage_pool_entry &entry : renderer.blas_storage_pool) {
        total = add_diagnostic_capacity(total, entry.capacity_bytes);
    }
    return total;
}

bool rt_submission_is_complete(rt_renderer* renderer, rt_submission_token submission) {
    if (renderer == nullptr || renderer->rhi == nullptr || !submission) {
        return true;
    }
    bool complete = false;
    rt_rhi_error error{};
    if (!renderer->rhi->is_complete(submission, &complete, &error)) {
        latch_rt_renderer_async_error(renderer, error);
        return false;
    }
    return complete;
}

void collect_rt_resource_pools(rt_renderer* renderer) {
    if (renderer == nullptr || renderer->rhi == nullptr) {
        return;
    }
    for (rt_scene_buffer_pool_entry &entry : renderer->scene_buffer_pool) {
        if (entry.retirement_submission &&
            rt_submission_is_complete(renderer, entry.retirement_submission)) {
            entry.retirement_submission = {};
        }
    }
    for (rt_blas_storage_pool_entry &entry : renderer->blas_storage_pool) {
        if (entry.retirement_submission &&
            rt_submission_is_complete(renderer, entry.retirement_submission)) {
            entry.retirement_submission = {};
        }
    }
}

void trim_scene_buffer_pool(rt_renderer* renderer) {
    if (renderer == nullptr || renderer->rhi == nullptr) {
        return;
    }
    collect_rt_resource_pools(renderer);
    while (renderer->scene_buffer_pool.size() > kRtSceneBufferPoolMaxEntries ||
           scene_buffer_pool_capacity_bytes(*renderer) > kRtSceneBufferPoolMaxBytes) {
        std::size_t oldest_index = renderer->scene_buffer_pool.size();
        for (std::size_t index = 0; index < renderer->scene_buffer_pool.size(); ++index) {
            const rt_scene_buffer_pool_entry &entry = renderer->scene_buffer_pool[index];
            if (!entry.retirement_submission &&
                (oldest_index == renderer->scene_buffer_pool.size() ||
                    entry.sequence < renderer->scene_buffer_pool[oldest_index].sequence)) {
                oldest_index = index;
            }
        }
        if (oldest_index == renderer->scene_buffer_pool.size()) {
            return;
        }
        renderer->rhi->destroy_buffer(renderer->scene_buffer_pool[oldest_index].buffer);
        ++renderer->resource_pool_eviction_count;
        renderer->scene_buffer_pool[oldest_index] = renderer->scene_buffer_pool.back();
        renderer->scene_buffer_pool.pop_back();
    }
}

void trim_blas_storage_pool(rt_renderer* renderer) {
    if (renderer == nullptr || renderer->rhi == nullptr) {
        return;
    }
    collect_rt_resource_pools(renderer);
    while (renderer->blas_storage_pool.size() > kRtBlasStoragePoolMaxEntries ||
           blas_storage_pool_capacity_bytes(*renderer) > kRtBlasStoragePoolMaxBytes) {
        std::size_t oldest_index = renderer->blas_storage_pool.size();
        for (std::size_t index = 0; index < renderer->blas_storage_pool.size(); ++index) {
            const rt_blas_storage_pool_entry &entry = renderer->blas_storage_pool[index];
            if (!entry.retirement_submission &&
                (oldest_index == renderer->blas_storage_pool.size() ||
                    entry.sequence < renderer->blas_storage_pool[oldest_index].sequence)) {
                oldest_index = index;
            }
        }
        if (oldest_index == renderer->blas_storage_pool.size()) {
            return;
        }
        renderer->rhi->destroy_blas(renderer->blas_storage_pool[oldest_index].acceleration);
        ++renderer->resource_pool_eviction_count;
        renderer->blas_storage_pool[oldest_index] = renderer->blas_storage_pool.back();
        renderer->blas_storage_pool.pop_back();
    }
}

void enqueue_scene_buffer_pool_entry(
    rt_renderer* renderer,
    rt_buffer_handle buffer,
    std::size_t capacity_bytes,
    rt_scene_buffer_role role,
    std::size_t format_stride,
    std::uint32_t usage,
    rt_submission_token retirement_submission)
{
    if (renderer == nullptr || renderer->rhi == nullptr || !buffer || capacity_bytes == 0) {
        return;
    }
    renderer->scene_buffer_pool.push_back({
        buffer,
        capacity_bytes,
        role,
        format_stride,
        usage,
        rt_memory_domain::device,
        retirement_submission,
        renderer->resource_pool_sequence++});
    trim_scene_buffer_pool(renderer);
}

void enqueue_blas_storage_pool_entry(
    rt_renderer* renderer,
    rt_blas_storage_pool_entry entry,
    rt_submission_token retirement_submission)
{
    if (renderer == nullptr || renderer->rhi == nullptr || !entry.acceleration) {
        return;
    }
    entry.retirement_submission = retirement_submission;
    entry.sequence = renderer->resource_pool_sequence++;
    renderer->blas_storage_pool.push_back(std::move(entry));
    trim_blas_storage_pool(renderer);
}

void retire_blas_cache_for_connection_change(rt_renderer* renderer) {
    if (renderer == nullptr || renderer->rhi == nullptr) {
        return;
    }
    const rt_submission_token retirement_submission = renderer->last_submission;
    for (const std::vector<rt_blas_cache_slot> &pool : renderer->blas_cache_state.pools) {
        for (const rt_blas_cache_slot &slot : pool) {
            if (!slot.acceleration) {
                continue;
            }
            enqueue_blas_storage_pool_entry(
                renderer,
                {
                    slot.storage_key,
                    slot.acceleration,
                    slot.storage_capacity_bytes,
                    {},
                    0},
                retirement_submission);
        }
    }
    renderer->blas_cache_state = {};
}

void destroy_rt_resource_pools(rt_renderer* renderer) {
    if (renderer == nullptr || renderer->rhi == nullptr) {
        return;
    }
    for (rt_scene_buffer_pool_entry &entry : renderer->scene_buffer_pool) {
        renderer->rhi->destroy_buffer(entry.buffer);
    }
    renderer->scene_buffer_pool.clear();
    for (rt_blas_storage_pool_entry &entry : renderer->blas_storage_pool) {
        renderer->rhi->destroy_blas(entry.acceleration);
    }
    renderer->blas_storage_pool.clear();
}

bool acquire_scene_buffer(
    rt_renderer* renderer,
    const rt_buffer_desc &desc,
    std::size_t capacity_hint,
    std::size_t capacity_floor,
    rt_scene_buffer_role role,
    std::size_t format_stride,
    rt_buffer_handle* out_buffer,
    std::size_t* out_capacity,
    rt_rhi_error* out_error)
{
    if (out_buffer != nullptr) {
        *out_buffer = {};
    }
    if (out_capacity != nullptr) {
        *out_capacity = 0;
    }
    if (renderer == nullptr || renderer->rhi == nullptr || out_buffer == nullptr || out_capacity == nullptr) {
        return false;
    }
    const std::size_t required_size = (std::max)(
        (std::max)(desc.size, capacity_floor),
        std::size_t{1});

    collect_rt_resource_pools(renderer);
    std::size_t best_index = renderer->scene_buffer_pool.size();
    for (std::size_t index = 0; index < renderer->scene_buffer_pool.size(); ++index) {
        const rt_scene_buffer_pool_entry &entry = renderer->scene_buffer_pool[index];
        if (entry.retirement_submission || entry.role != role ||
            entry.format_stride != format_stride || entry.usage != desc.usage ||
            entry.memory_domain != desc.memory_domain || entry.capacity_bytes < required_size) {
            continue;
        }
        if (best_index == renderer->scene_buffer_pool.size() ||
            entry.capacity_bytes < renderer->scene_buffer_pool[best_index].capacity_bytes) {
            best_index = index;
        }
    }
    if (best_index != renderer->scene_buffer_pool.size()) {
        const rt_scene_buffer_pool_entry entry = renderer->scene_buffer_pool[best_index];
        renderer->scene_buffer_pool[best_index] = renderer->scene_buffer_pool.back();
        renderer->scene_buffer_pool.pop_back();
        *out_buffer = entry.buffer;
        *out_capacity = entry.capacity_bytes;
        ++renderer->scene_buffer_pool_hit_count;
        return true;
    }

    std::size_t capacity = 0;
    if (!grow_rt_capacity(required_size, capacity_hint, 1, &capacity)) {
        set_frame_error(out_error, rt_rhi_operation::create_resource, "RT scene buffer capacity overflow");
        return false;
    }
    rt_buffer_desc allocation_desc = desc;
    allocation_desc.size = capacity;
    if (!renderer->rhi->create_buffer(allocation_desc, out_buffer, out_error) || !*out_buffer) {
        return false;
    }
    *out_capacity = capacity;
    ++renderer->scene_buffer_allocation_count;
    if (capacity_hint != 0 && capacity > capacity_hint) {
        ++renderer->scene_buffer_growth_count;
    }
    ++renderer->scene_buffer_pool_miss_count;
    return true;
}

void retire_scene_buffers(rt_renderer* renderer, rt_scene_buffer_resources* resources) {
    if (renderer == nullptr || renderer->rhi == nullptr || resources == nullptr) {
        return;
    }
    destroy_scene_upload_buffers(renderer, resources);
    const rt_submission_token retirement_submission = renderer->last_submission;
    const auto retire = [renderer, retirement_submission](
                            rt_buffer_handle buffer,
                            std::size_t capacity_bytes,
                            rt_scene_buffer_role role,
                            std::size_t format_stride,
                            std::uint32_t usage) {
        if (buffer && capacity_bytes == 0) {
            renderer->rhi->destroy_buffer(buffer);
            return;
        }
        enqueue_scene_buffer_pool_entry(
            renderer,
            buffer,
            capacity_bytes,
            role,
            format_stride,
            usage,
            retirement_submission);
    };
    const std::uint32_t geometry_usage = rt_buffer_usage_shader_read |
        rt_buffer_usage_acceleration_build_input | rt_buffer_usage_device_address;
    const std::uint32_t acceleration_input_usage =
        rt_buffer_usage_acceleration_build_input | rt_buffer_usage_device_address;
    const std::uint32_t copy_destination = rt_buffer_usage_copy_destination;
    retire(
        resources->positions,
        resources->positions_capacity_bytes,
        rt_scene_buffer_role::positions,
        sizeof(rt_scene_gpu_position),
        geometry_usage | copy_destination);
    retire(
        resources->indices,
        resources->indices_capacity_bytes,
        rt_scene_buffer_role::indices,
        sizeof(std::uint32_t),
        geometry_usage | copy_destination);
    retire(resources->triangle_colors, resources->triangle_colors_capacity_bytes,
        rt_scene_buffer_role::triangle_colors,
        sizeof(rtvdb::rgba),
        rt_buffer_usage_shader_read | copy_destination);
    retire(resources->instance_metadata, resources->instance_metadata_capacity_bytes,
        rt_scene_buffer_role::instance_metadata,
        sizeof(rt_scene_geometry_metadata),
        rt_buffer_usage_shader_read | copy_destination);
    retire(
        resources->points,
        resources->points_capacity_bytes,
        rt_scene_buffer_role::points,
        sizeof(rt_scene_gpu_point),
        rt_buffer_usage_shader_read | copy_destination);
    retire(
        resources->lines,
        resources->lines_capacity_bytes,
        rt_scene_buffer_role::lines,
        sizeof(rt_scene_gpu_line),
        rt_buffer_usage_shader_read | copy_destination);
    retire(resources->point_aabbs, resources->point_aabbs_capacity_bytes,
        rt_scene_buffer_role::point_aabbs,
        sizeof(rt_scene_gpu_aabb),
        acceleration_input_usage | copy_destination);
    retire(resources->line_aabbs, resources->line_aabbs_capacity_bytes,
        rt_scene_buffer_role::line_aabbs,
        sizeof(rt_scene_gpu_aabb),
        acceleration_input_usage | copy_destination);
    *resources = {};
}

bool acquire_blas_storage(
    rt_renderer* renderer,
    const rt_blas_storage_key &key,
    rt_blas_storage_pool_entry* out_entry)
{
    if (out_entry != nullptr) {
        *out_entry = {};
    }
    if (renderer == nullptr || renderer->rhi == nullptr || out_entry == nullptr) {
        return false;
    }
    collect_rt_resource_pools(renderer);
    for (std::size_t index = 0; index < renderer->blas_storage_pool.size(); ++index) {
        const rt_blas_storage_pool_entry &entry = renderer->blas_storage_pool[index];
        if (entry.retirement_submission || !rt_blas_storage_key_equals(entry.key, key)) {
            continue;
        }
        *out_entry = entry;
        renderer->blas_storage_pool[index] = renderer->blas_storage_pool.back();
        renderer->blas_storage_pool.pop_back();
        ++renderer->blas_storage_pool_hit_count;
        return true;
    }
    ++renderer->blas_storage_pool_miss_count;
    return false;
}

void destroy_output_resources(rt_renderer* renderer) {
    if (renderer == nullptr || renderer->rhi == nullptr) {
        return;
    }
    renderer->rhi->destroy_texture(renderer->output_texture);
    renderer->rhi->destroy_texture(renderer->accumulation_texture);
    renderer->rhi->destroy_buffer(renderer->output_readback_buffer);
    renderer->output_texture = {};
    renderer->accumulation_texture = {};
    renderer->output_readback_buffer = {};
    renderer->output_readback_footprint = {};
    renderer->output_readback_submission = {};
    renderer->output_readback_width = 0;
    renderer->output_readback_height = 0;
    renderer->output_readback_pending = false;
}

void destroy_pick_buffers(rt_renderer* renderer) {
    if (renderer == nullptr || renderer->rhi == nullptr) {
        return;
    }
    renderer->rhi->destroy_buffer(renderer->pick_output_buffer);
    for (rt_pick_slot &slot : renderer->pick_slots) {
        renderer->rhi->destroy_buffer(slot.readback_buffer);
    }
    renderer->pick_output_buffer = {};
    renderer->pick_slots = {};
}

void destroy_viewer_resources(rt_renderer* renderer) {
    if (renderer == nullptr || renderer->rhi == nullptr) {
        return;
    }
    renderer->rhi->destroy_buffer(renderer->viewer_constant_buffer);
    renderer->viewer_constant_buffer = {};
}

bool ensure_viewer_constant_buffer(
    rt_renderer* renderer,
    bool* out_changed,
    rt_rhi_error* out_error)
{
    if (out_changed != nullptr) {
        *out_changed = false;
    }
    if (renderer == nullptr || renderer->rhi == nullptr) {
        return false;
    }
    if (renderer->viewer_constant_buffer) {
        return true;
    }

    rt_buffer_handle constants{};
    if (!renderer->rhi->create_buffer(
            {
                kRtViewerConstantBufferBytes,
                rt_buffer_usage_uniform,
                rt_memory_domain::upload},
            &constants,
            out_error)) {
        return false;
    }
    destroy_viewer_resources(renderer);
    renderer->viewer_constant_buffer = constants;
    if (out_changed != nullptr) {
        *out_changed = true;
    }
    return true;
}

bool ensure_pick_buffers(
    rt_renderer* renderer,
    bool* out_changed,
    rt_rhi_error* out_error)
{
    if (out_changed != nullptr) {
        *out_changed = false;
    }
    if (renderer == nullptr || renderer->rhi == nullptr) {
        return false;
    }
    bool has_readback_buffers = true;
    for (const rt_pick_slot &slot : renderer->pick_slots) {
        has_readback_buffers = has_readback_buffers && static_cast<bool>(slot.readback_buffer);
    }
    if (renderer->pick_output_buffer && has_readback_buffers) {
        return true;
    }

    rt_buffer_handle output{};
    std::array<rt_buffer_handle, kRtCommandSlotCount> readbacks{};
    const rt_buffer_desc output_desc{
        kRtPickResultBufferBytes,
        rt_buffer_usage_shader_write |
            rt_buffer_usage_copy_source |
            rt_buffer_usage_copy_destination,
        rt_memory_domain::device};
    const rt_buffer_desc readback_desc{
        kRtPickResultBufferBytes,
        rt_buffer_usage_copy_destination,
        rt_memory_domain::readback};
    if (!renderer->rhi->create_buffer(output_desc, &output, out_error)) {
        renderer->rhi->destroy_buffer(output);
        return false;
    }
    for (rt_buffer_handle &readback : readbacks) {
        if (renderer->rhi->create_buffer(readback_desc, &readback, out_error)) {
            continue;
        }
        renderer->rhi->destroy_buffer(output);
        for (rt_buffer_handle allocated_readback : readbacks) {
            renderer->rhi->destroy_buffer(allocated_readback);
        }
        return false;
    }
    destroy_pick_buffers(renderer);
    renderer->pick_output_buffer = output;
    for (std::size_t slot_index = 0; slot_index < kRtCommandSlotCount; ++slot_index) {
        renderer->pick_slots[slot_index].readback_buffer = readbacks[slot_index];
    }
    if (out_changed != nullptr) {
        *out_changed = true;
    }
    return true;
}

bool create_output_resources(
    rt_renderer* renderer,
    int width,
    int height,
    rt_rhi_error* out_error)
{
    if (renderer == nullptr || renderer->rhi == nullptr || width <= 0 || height <= 0) {
        return false;
    }
    const std::uint32_t output_usage =
        rt_texture_usage_shader_read |
        rt_texture_usage_shader_write |
        rt_texture_usage_copy_source |
        rt_texture_usage_copy_destination;
    rt_texture_handle output{};
    rt_texture_handle accumulation{};
    rt_buffer_handle readback{};
    if (!renderer->rhi->create_texture(
            {
                static_cast<std::uint32_t>(width),
                static_cast<std::uint32_t>(height),
                renderer->capabilities.output_format,
                output_usage},
            &output,
            out_error) ||
        !renderer->rhi->create_texture(
            {
                static_cast<std::uint32_t>(width),
                static_cast<std::uint32_t>(height),
                renderer->capabilities.accumulation_format,
                rt_texture_usage_shader_write},
            &accumulation,
            out_error)) {
        renderer->rhi->destroy_texture(output);
        renderer->rhi->destroy_texture(accumulation);
        return false;
    }
    rt_texture_copy_footprint footprint{};
    if (!renderer->rhi->get_texture_copy_footprint(
            output,
            &footprint,
            out_error) ||
        footprint.total_size == 0 ||
        !renderer->rhi->create_buffer(
            {
                footprint.total_size,
                rt_buffer_usage_copy_destination,
                rt_memory_domain::readback},
            &readback,
            out_error)) {
        renderer->rhi->destroy_texture(output);
        renderer->rhi->destroy_texture(accumulation);
        renderer->rhi->destroy_buffer(readback);
        return false;
    }
    destroy_output_resources(renderer);
    renderer->output_texture = output;
    renderer->accumulation_texture = accumulation;
    renderer->output_readback_buffer = readback;
    renderer->output_readback_footprint = footprint;
    return true;
}

bool create_uploaded_buffer(
    rt_renderer* renderer,
    const rt_buffer_desc &desc,
    const void* data,
    rt_buffer_handle* out_destination,
    rt_buffer_handle* out_staging,
    rt_rhi_error* out_error)
{
    if (out_destination == nullptr || out_staging == nullptr) {
        return false;
    }
    *out_destination = {};
    *out_staging = {};
    rt_buffer_desc destination_desc = desc;
    destination_desc.usage |= rt_buffer_usage_copy_destination;
    destination_desc.memory_domain = rt_memory_domain::device;
    const rt_buffer_desc staging_desc{
        desc.size,
        rt_buffer_usage_copy_source,
        rt_memory_domain::upload};
    if (!renderer->rhi->create_buffer(destination_desc, out_destination, out_error)) {
        return false;
    }
    if (desc.size == 0) {
        return true;
    }
    if (!renderer->rhi->create_buffer(staging_desc, out_staging, out_error) ||
        !renderer->rhi->upload_buffer(*out_staging, 0, data, desc.size, out_error)) {
        renderer->rhi->destroy_buffer(*out_destination);
        renderer->rhi->destroy_buffer(*out_staging);
        *out_destination = {};
        *out_staging = {};
        return false;
    }
    return true;
}

bool create_scene_uploaded_buffer(
    rt_renderer* renderer,
    rt_scene_buffer_resources* resources,
    const char* name,
    rt_scene_buffer_role role,
    std::size_t format_stride,
    const rt_buffer_desc &desc,
    const void* data,
    rt_resource_usage destination_usage,
    std::size_t capacity_hint,
    std::size_t capacity_floor,
    rt_buffer_handle* out_destination,
    std::size_t* out_capacity,
    rt_rhi_error* out_error)
{
    if (resources == nullptr || out_destination == nullptr) {
        return false;
    }
    *out_destination = {};
    if (out_capacity != nullptr) {
        *out_capacity = 0;
    }
    rt_buffer_desc destination_desc = desc;
    destination_desc.usage |= rt_buffer_usage_copy_destination;
    destination_desc.memory_domain = rt_memory_domain::device;
    if (!acquire_scene_buffer(
            renderer,
            destination_desc,
            capacity_hint,
            capacity_floor,
            role,
            format_stride,
            out_destination,
            out_capacity,
            out_error)) {
        if (out_error != nullptr) {
            char detail[160]{};
            std::snprintf(
                detail,
                sizeof(detail),
                "RT scene buffer creation failed: %s (%zu bytes)",
                name != nullptr ? name : "unnamed",
                desc.size);
            out_error->detail = detail;
        }
        return false;
    }
    if (desc.size == 0) {
        return true;
    }
    const rt_buffer_desc staging_desc{
        desc.size,
        rt_buffer_usage_copy_source,
        rt_memory_domain::upload};
    rt_buffer_handle staging{};
    if (!renderer->rhi->create_buffer(staging_desc, &staging, out_error) ||
        !renderer->rhi->upload_buffer(staging, 0, data, desc.size, out_error)) {
        renderer->rhi->destroy_buffer(*out_destination);
        *out_destination = {};
        if (out_capacity != nullptr) {
            *out_capacity = 0;
        }
        return false;
    }
    if (staging) {
        resources->uploads.push_back({staging, *out_destination, desc.size, destination_usage});
    }
    return true;
}

bool upload_scene_buffers(
    rt_renderer* renderer,
    const rt_scene_resource_data &resources,
    const rt_acceleration_build_plan &build_plan,
    rt_rhi_error* out_error)
{
    if (renderer == nullptr || renderer->rhi == nullptr) {
        set_frame_error(out_error, rt_rhi_operation::upload_scene_buffers, "RT scene buffer API is unavailable");
        return false;
    }
    if (build_plan.revision != resources.revision) {
        set_frame_error(
            out_error,
            rt_rhi_operation::upload_scene_buffers,
            "RT scene resource and acceleration build revisions do not match");
        return false;
    }

    rt_scene_buffer_resources next{};
    const rt_scene_buffer_resources &current = renderer->scene_buffers;
    const std::uint32_t geometry_usage = rt_buffer_usage_shader_read |
        rt_buffer_usage_acceleration_build_input |
        rt_buffer_usage_device_address;
    const std::uint32_t acceleration_input_usage =
        rt_buffer_usage_acceleration_build_input | rt_buffer_usage_device_address;
    const std::uint32_t shader_read_usage = rt_buffer_usage_shader_read;
    std::size_t indices_capacity_bytes = resources.indices.size() * sizeof(std::uint32_t);
    std::size_t point_aabbs_capacity_bytes = resources.point_aabbs.size() * sizeof(rt_scene_gpu_aabb);
    std::size_t line_aabbs_capacity_bytes = resources.line_aabbs.size() * sizeof(rt_scene_gpu_aabb);
    const auto include_capacity_range = [](
        std::size_t element_offset,
        std::size_t element_count,
        std::size_t element_size,
        std::size_t* out_bytes) {
        if (out_bytes == nullptr ||
            element_count > (std::numeric_limits<std::size_t>::max)() - element_offset) {
            return false;
        }
        const std::size_t element_end = element_offset + element_count;
        if (element_size != 0 && element_end > (std::numeric_limits<std::size_t>::max)() / element_size) {
            return false;
        }
        *out_bytes = (std::max)(*out_bytes, element_end * element_size);
        return true;
    };
    for (std::size_t item_index = 0; item_index < build_plan.items.size(); ++item_index) {
        const rt_acceleration_build_item &item = build_plan.items[item_index];
        const std::size_t default_capacity = item.kind == rt_acceleration_geometry_kind::triangle
            ? kDefaultRtSceneTriangleChunkPrimitives
            : kDefaultRtSceneProceduralChunkPrimitives;
        for (std::size_t geometry_index = 0; geometry_index < item.group.chunk_count; ++geometry_index) {
            const std::size_t primitive_count = (std::max)(
                default_capacity,
                item.primitive_counts[geometry_index]);
            if (item.kind == rt_acceleration_geometry_kind::triangle) {
                const std::size_t metadata_index =
                    item_index * kRtBlasChunkSetChunkCount + geometry_index;
                if (primitive_count > (std::numeric_limits<std::size_t>::max)() / 3u) {
                    set_frame_error(
                        out_error,
                        rt_rhi_operation::upload_scene_buffers,
                        "RT scene triangle capacity overflows");
                    return false;
                }
                if (metadata_index >= resources.instance_geometry.size() ||
                    !include_capacity_range(
                        resources.instance_geometry[metadata_index].index_offset,
                        primitive_count * 3u,
                        sizeof(std::uint32_t),
                        &indices_capacity_bytes)) {
                    set_frame_error(
                        out_error,
                        rt_rhi_operation::upload_scene_buffers,
                        "RT scene index capacity overflows");
                    return false;
                }
            } else {
                std::size_t* const capacity_bytes = item.kind == rt_acceleration_geometry_kind::point
                    ? &point_aabbs_capacity_bytes
                    : &line_aabbs_capacity_bytes;
                if (!include_capacity_range(
                        item.first_primitives[geometry_index],
                        primitive_count,
                        sizeof(rt_scene_gpu_aabb),
                        capacity_bytes)) {
                    set_frame_error(
                        out_error,
                        rt_rhi_operation::upload_scene_buffers,
                        "RT scene AABB capacity overflows");
                    return false;
                }
            }
        }
    }
    std::vector<std::uint32_t> padded_indices = resources.indices;
    if (indices_capacity_bytes > padded_indices.size() * sizeof(std::uint32_t)) {
        padded_indices.resize(indices_capacity_bytes / sizeof(std::uint32_t), 0u);
    }
    std::vector<rt_scene_gpu_aabb> padded_point_aabbs = resources.point_aabbs;
    if (point_aabbs_capacity_bytes > padded_point_aabbs.size() * sizeof(rt_scene_gpu_aabb)) {
        padded_point_aabbs.resize(point_aabbs_capacity_bytes / sizeof(rt_scene_gpu_aabb));
    }
    std::vector<rt_scene_gpu_aabb> padded_line_aabbs = resources.line_aabbs;
    if (line_aabbs_capacity_bytes > padded_line_aabbs.size() * sizeof(rt_scene_gpu_aabb)) {
        padded_line_aabbs.resize(line_aabbs_capacity_bytes / sizeof(rt_scene_gpu_aabb));
    }
    if (!create_scene_uploaded_buffer(
            renderer,
            &next,
            "positions",
            rt_scene_buffer_role::positions,
            sizeof(rt_scene_gpu_position),
            {resources.positions.size() * sizeof(rt_scene_gpu_position), geometry_usage},
            resources.positions.data(),
            rt_resource_usage::acceleration_build_input,
            current.positions_capacity_bytes,
            0,
            &next.positions,
            &next.positions_capacity_bytes,
            out_error) ||
        !create_scene_uploaded_buffer(
            renderer,
            &next,
            "indices",
            rt_scene_buffer_role::indices,
            sizeof(std::uint32_t),
            {padded_indices.size() * sizeof(std::uint32_t), geometry_usage},
            padded_indices.data(),
            rt_resource_usage::acceleration_build_input,
            current.indices_capacity_bytes,
            indices_capacity_bytes,
            &next.indices,
            &next.indices_capacity_bytes,
            out_error) ||
        !create_scene_uploaded_buffer(
            renderer,
            &next,
            "triangle_colors",
            rt_scene_buffer_role::triangle_colors,
            sizeof(rtvdb::rgba),
            {resources.triangle_colors.size() * sizeof(rtvdb::rgba), shader_read_usage},
            resources.triangle_colors.data(),
            rt_resource_usage::shader_read,
            current.triangle_colors_capacity_bytes,
            0,
            &next.triangle_colors,
            &next.triangle_colors_capacity_bytes,
            out_error) ||
        !create_scene_uploaded_buffer(
            renderer,
            &next,
            "instance_metadata",
            rt_scene_buffer_role::instance_metadata,
            sizeof(rt_scene_geometry_metadata),
            {
                resources.instance_geometry.size() * sizeof(rt_scene_geometry_metadata),
                shader_read_usage},
            resources.instance_geometry.data(),
            rt_resource_usage::shader_read,
            current.instance_metadata_capacity_bytes,
            0,
            &next.instance_metadata,
            &next.instance_metadata_capacity_bytes,
            out_error) ||
        !create_scene_uploaded_buffer(
            renderer,
            &next,
            "points",
            rt_scene_buffer_role::points,
            sizeof(rt_scene_gpu_point),
            {resources.points.size() * sizeof(rt_scene_gpu_point), shader_read_usage},
            resources.points.data(),
            rt_resource_usage::shader_read,
            current.points_capacity_bytes,
            0,
            &next.points,
            &next.points_capacity_bytes,
            out_error) ||
        !create_scene_uploaded_buffer(
            renderer,
            &next,
            "lines",
            rt_scene_buffer_role::lines,
            sizeof(rt_scene_gpu_line),
            {resources.lines.size() * sizeof(rt_scene_gpu_line), shader_read_usage},
            resources.lines.data(),
            rt_resource_usage::shader_read,
            current.lines_capacity_bytes,
            0,
            &next.lines,
            &next.lines_capacity_bytes,
            out_error)) {
        destroy_scene_buffers(renderer, &next);
        return false;
    }

    const bool same_connection = current.connection_serial == resources.connection_serial;
    const bool reuse_point_aabbs =
        same_connection &&
        current.point_aabbs &&
        current.point_geometry_fingerprint == build_plan.point_geometry_fingerprint &&
        current.point_aabb_count == resources.point_aabbs.size();
    const bool reuse_line_aabbs =
        same_connection &&
        current.line_aabbs &&
        current.line_geometry_fingerprint == build_plan.line_geometry_fingerprint &&
        current.line_aabb_count == resources.line_aabbs.size();
    const auto aabb_upload_start = std::chrono::steady_clock::now();
    if ((!reuse_point_aabbs && !resources.point_aabbs.empty() &&
            !create_scene_uploaded_buffer(
                renderer,
                &next,
                "point_aabbs",
                rt_scene_buffer_role::point_aabbs,
                sizeof(rt_scene_gpu_aabb),
                {padded_point_aabbs.size() * sizeof(rt_scene_gpu_aabb), acceleration_input_usage},
                padded_point_aabbs.data(),
                rt_resource_usage::acceleration_build_input,
                current.point_aabbs_capacity_bytes,
                point_aabbs_capacity_bytes,
                &next.point_aabbs,
                &next.point_aabbs_capacity_bytes,
                out_error)) ||
        (!reuse_line_aabbs && !resources.line_aabbs.empty() &&
            !create_scene_uploaded_buffer(
                renderer,
                &next,
                "line_aabbs",
                rt_scene_buffer_role::line_aabbs,
                sizeof(rt_scene_gpu_aabb),
                {padded_line_aabbs.size() * sizeof(rt_scene_gpu_aabb), acceleration_input_usage},
                padded_line_aabbs.data(),
                rt_resource_usage::acceleration_build_input,
                current.line_aabbs_capacity_bytes,
                line_aabbs_capacity_bytes,
                &next.line_aabbs,
                &next.line_aabbs_capacity_bytes,
                out_error))) {
        destroy_scene_buffers(renderer, &next);
        return false;
    }
    next.procedural_aabb_upload_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - aabb_upload_start).count();
    if (reuse_point_aabbs) {
        next.point_aabbs = current.point_aabbs;
        next.point_aabbs_capacity_bytes = current.point_aabbs_capacity_bytes;
        renderer->scene_buffers.point_aabbs = {};
        renderer->scene_buffers.point_aabbs_capacity_bytes = 0;
    }
    if (reuse_line_aabbs) {
        next.line_aabbs = current.line_aabbs;
        next.line_aabbs_capacity_bytes = current.line_aabbs_capacity_bytes;
        renderer->scene_buffers.line_aabbs = {};
        renderer->scene_buffers.line_aabbs_capacity_bytes = 0;
    }

    next.position_count = resources.positions.size();
    next.index_count = resources.indices.size();
    next.triangle_color_count = resources.triangle_colors.size();
    next.point_count = resources.points.size();
    next.line_count = resources.lines.size();
    next.point_aabb_count = resources.point_aabbs.size();
    next.line_aabb_count = resources.line_aabbs.size();
    next.instance_metadata_count = resources.instance_geometry.size();
    next.point_geometry_fingerprint = build_plan.point_geometry_fingerprint;
    next.line_geometry_fingerprint = build_plan.line_geometry_fingerprint;
    next.revision = resources.revision;
    next.connection_serial = resources.connection_serial;
    retire_scene_buffers(renderer, &renderer->scene_buffers);
    renderer->scene_buffers = next;
    renderer->scene_buffer_peak_capacity_bytes = (std::max)(
        renderer->scene_buffer_peak_capacity_bytes,
        scene_buffer_capacity_bytes(renderer->scene_buffers));
    return true;
}

bool resolve_rt_blas_geometry_resources(
    const rt_scene_buffer_resources &resources,
    rt_blas_build_command* command)
{
    if (command == nullptr || command->geometry_count == 0 ||
        command->geometry_count > command->geometries.size()) {
        return false;
    }
    for (std::size_t geometry_index = 0; geometry_index < command->geometry_count; ++geometry_index) {
        rt_acceleration_geometry_desc &geometry = command->geometries[geometry_index];
        if (geometry.type == rt_acceleration_geometry_type::triangles) {
            geometry.triangles.vertex_buffer = resources.positions;
            geometry.triangles.index_buffer = resources.indices;
            if (!geometry.triangles.vertex_buffer || !geometry.triangles.index_buffer) {
                return false;
            }
        } else if (geometry.type == rt_acceleration_geometry_type::aabbs) {
            geometry.aabbs.buffer = command->kind == rt_acceleration_geometry_kind::point
                ? resources.point_aabbs
                : command->kind == rt_acceleration_geometry_kind::line
                    ? resources.line_aabbs
                    : rt_buffer_handle{};
            if (!geometry.aabbs.buffer) {
                return false;
            }
        } else {
            return false;
        }
    }
    return true;
}

bool rt_blas_cache_contains(
    const rt_blas_cache_state &state,
    rt_blas_handle acceleration)
{
    if (!acceleration) {
        return false;
    }
    for (const std::vector<rt_blas_cache_slot> &pool : state.pools) {
        for (const rt_blas_cache_slot &slot : pool) {
            if (slot.valid && slot.acceleration == acceleration) {
                return true;
            }
        }
    }
    return false;
}

bool record_scene_buffer_uploads(
    rt_renderer* renderer,
    rt_command_encoder encoder,
    const rt_scene_buffer_resources &resources,
    rt_rhi_error* out_error)
{
    for (const rt_scene_buffer_upload &upload : resources.uploads) {
        if (!upload.staging || !upload.destination || upload.size == 0 ||
            !transition_rt_buffer(
                renderer,
                encoder,
                upload.destination,
                rt_resource_usage::copy_destination,
                out_error) ||
            !record_rt_buffer_copy(
                renderer,
                encoder,
                upload.staging,
                upload.destination,
                {0, 0, upload.size},
                out_error) ||
            !transition_rt_buffer(
                renderer,
                encoder,
                upload.destination,
                upload.destination_usage,
                out_error)) {
            return false;
        }
    }
    return true;
}

bool transition_triangle_geometry_to_shader_read(
    rt_renderer* renderer,
    rt_command_encoder encoder,
    const rt_scene_buffer_resources &resources,
    rt_rhi_error* out_error)
{
    return (!resources.positions || transition_rt_buffer(
                renderer,
                encoder,
                resources.positions,
                rt_resource_usage::shader_read,
                out_error)) &&
        (!resources.indices || transition_rt_buffer(
                renderer,
                encoder,
                resources.indices,
                rt_resource_usage::shader_read,
                out_error));
}

void destroy_rt_blas_cache(rt_renderer* renderer, rt_blas_cache_state* state) {
    if (renderer == nullptr || renderer->rhi == nullptr || state == nullptr) {
        return;
    }
    for (std::vector<rt_blas_cache_slot> &pool : state->pools) {
        for (rt_blas_cache_slot &slot : pool) {
            renderer->rhi->destroy_blas(slot.acceleration);
            slot.acceleration = {};
        }
    }
    *state = {};
}

bool sync_rt_renderer_acceleration(
    rt_renderer* renderer,
    const rt_renderer_frame_request &request,
    rt_renderer_frame_result* out_result,
    rt_rhi_error* out_error,
    rt_deferred_acceleration_submission* deferred_acceleration)
{
    if (renderer == nullptr || renderer->rhi == nullptr || request.acceleration_plan == nullptr ||
        request.blas_cache_plan == nullptr || request.resources == nullptr ||
        request.acceleration_commands == nullptr || out_result == nullptr) {
        set_frame_error(
            out_error,
            rt_rhi_operation::begin_commands,
            "RT acceleration build request is invalid");
        return false;
    }

    const rt_acceleration_build_plan &build_plan = *request.acceleration_plan;
    rt_blas_cache_update_plan &cache_plan = *request.blas_cache_plan;
    const rt_scene_resource_data &resources = *request.resources;
    const rt_acceleration_command_plan &commands = *request.acceleration_commands;
    if (build_plan.revision != resources.revision || cache_plan.revision != resources.revision ||
        commands.revision != resources.revision ||
        commands.blas_commands.size() != cache_plan.assignments.size()) {
        set_frame_error(
            out_error,
            rt_rhi_operation::begin_commands,
            "RT acceleration build contracts do not match");
        return false;
    }

    rt_command_encoder encoder{};
    rt_rhi_error stage_error{rt_rhi_operation::begin_commands, 0, {}};
    if (!begin_rt_commands(
            renderer,
            rt_queue_class::graphics,
            &encoder,
            &stage_error,
            &out_result->acceleration_timing)) {
        if (out_error != nullptr) {
            *out_error = stage_error;
        }
        return false;
    }
    const auto sync_start = std::chrono::steady_clock::now();
    std::vector<rt_blas_handle> created_accelerations;
    std::vector<rt_blas_storage_pool_entry> acquired_accelerations;
    if (!renderer->tlas) {
        stage_error = {rt_rhi_operation::build_tlas, 0, {}};
        if (!renderer->rhi->create_tlas(&renderer->tlas, &stage_error) ||
            !renderer->tlas) {
            discard_rt_commands(renderer, encoder);
            destroy_scene_upload_buffers(renderer, &renderer->scene_buffers);
            if (out_error != nullptr) {
                *out_error = stage_error;
            }
            return false;
        }
    }
    const auto fail_build = [&]() {
        discard_rt_commands(renderer, encoder);
        destroy_scene_upload_buffers(renderer, &renderer->scene_buffers);
        for (rt_blas_handle acceleration : created_accelerations) {
            renderer->rhi->destroy_blas(acceleration);
        }
        for (rt_blas_storage_pool_entry &entry : acquired_accelerations) {
            enqueue_blas_storage_pool_entry(renderer, std::move(entry), {});
        }
        if (out_error != nullptr) {
            *out_error = stage_error;
        }
        return false;
    };

    if (!record_scene_buffer_uploads(renderer, encoder, renderer->scene_buffers, &stage_error)) {
        return fail_build();
    }

    rt_acceleration_build_summary summary{};
    renderer->last_point_blas_prebuild_info_ms = 0.0;
    renderer->last_point_blas_prebuild_info_count = 0;
    renderer->last_line_blas_prebuild_info_ms = 0.0;
    renderer->last_line_blas_prebuild_info_count = 0;
    std::vector<rt_tlas_instance_desc> tlas_instances;
    tlas_instances.reserve(commands.blas_commands.size());
    std::vector<rt_layer_tlas_instance> layer_tlas_instances;
    layer_tlas_instances.reserve(commands.blas_commands.size());
    for (std::size_t command_index = 0;
        command_index < commands.blas_commands.size();
        ++command_index) {
        const rt_blas_build_command &command = commands.blas_commands[command_index];
        const rt_blas_cache_assignment &assignment = cache_plan.assignments[command_index];
        const std::size_t pool_index = static_cast<std::size_t>(command.kind);
        if (pool_index >= cache_plan.next_state.pools.size() ||
            assignment.cache_index >= cache_plan.next_state.pools[pool_index].size()) {
            stage_error = {
                rt_rhi_operation::build_blas,
                0,
                "RT BLAS cache assignment is invalid"};
            return fail_build();
        }
        rt_blas_cache_slot &slot =
            cache_plan.next_state.pools[pool_index][assignment.cache_index];
        const bool reuse = assignment.reuse_candidate && slot.acceleration;
        rt_blas_build_command resolved_command = command;
        resolved_command.destination = slot.acceleration;
        if (!resolve_rt_blas_geometry_resources(renderer->scene_buffers, &resolved_command)) {
            stage_error = {
                rt_rhi_operation::build_blas,
                0,
                "RT BLAS geometry resources could not be resolved"};
            return fail_build();
        }
        if (reuse) {
            ++summary.blas_reused_count;
            if (command.kind == rt_acceleration_geometry_kind::triangle) {
                summary.blas_reused_triangle_chunk_count += command.geometry_count;
            }
        } else {
            rt_blas_handle acceleration{};
            const rt_blas_storage_key storage_key = make_rt_blas_storage_key(
                resolved_command,
                rt_acceleration_build_prefer_fast_trace);
            rt_blas_storage_pool_entry acquired_entry{};
            stage_error = {rt_rhi_operation::build_blas, 0, {}};
            if (acquire_blas_storage(renderer, storage_key, &acquired_entry)) {
                acceleration = acquired_entry.acceleration;
                acquired_accelerations.push_back(acquired_entry);
            } else {
                if (!renderer->rhi->create_blas(&acceleration, &stage_error) ||
                    !acceleration) {
                    return fail_build();
                }
                created_accelerations.push_back(acceleration);
            }
            resolved_command.destination = acceleration;
            const rt_blas_build_desc build_desc{
                acceleration,
                resolved_command.geometries.data(),
                resolved_command.geometry_count,
                rt_acceleration_build_prefer_fast_trace,
                resolved_command.allocation_primitive_counts.data()};
            rt_blas_build_result blas_result{};
            if (!renderer->rhi->build_blas(
                    encoder,
                    build_desc,
                    &blas_result,
                    &stage_error) ||
                !(blas_result.acceleration == acceleration)) {
                return fail_build();
            }
            slot.acceleration = acceleration;
            slot.storage_key = storage_key;
            slot.storage_capacity_bytes = blas_result.allocation_size_bytes;
            if (command.kind == rt_acceleration_geometry_kind::point) {
                renderer->last_point_blas_prebuild_info_ms +=
                    blas_result.prebuild_info_ms;
                ++renderer->last_point_blas_prebuild_info_count;
            } else if (command.kind == rt_acceleration_geometry_kind::line) {
                renderer->last_line_blas_prebuild_info_ms +=
                    blas_result.prebuild_info_ms;
                ++renderer->last_line_blas_prebuild_info_count;
            }
            ++summary.blas_rebuilt_count;
            if (command.kind == rt_acceleration_geometry_kind::triangle) {
                summary.blas_rebuilt_triangle_chunk_count += command.geometry_count;
            }
        }
        rt_tlas_instance_desc instance{};
        instance.acceleration = slot.acceleration;
        instance.instance_id = command.instance_index;
        instance.mask = command.visible ? 0xff : 0x00;
        instance.hit_group_contribution = command.hit_group_contribution;
        tlas_instances.push_back(instance);

        rt_layer_tlas_instance layer_instance{};
        layer_instance.instance = instance;
        layer_instance.kind = command.kind;
        layer_instance.geometry_count = command.geometry_count;
        if (!resolve_layer_tlas_instance_source(
                *request.build,
                build_plan.items[command_index],
                &layer_instance.layer,
                &layer_instance.fallback_visible,
                &layer_instance.layer_visibility_exempt)) {
            stage_error = {
                rt_rhi_operation::build_tlas,
                0,
                "RT layer TLAS instance source is invalid"};
            return fail_build();
        }
        layer_tlas_instances.push_back(std::move(layer_instance));
    }

    const rt_tlas_build_desc tlas_desc{
        renderer->tlas,
        tlas_instances.data(),
        tlas_instances.size(),
        rt_acceleration_build_prefer_fast_trace};
    stage_error = {rt_rhi_operation::build_tlas, 0, {}};
    if (!renderer->rhi->build_tlas(encoder, tlas_desc, &stage_error)) {
        return fail_build();
    }
    if (!transition_triangle_geometry_to_shader_read(
            renderer,
            encoder,
            renderer->scene_buffers,
            &stage_error)) {
        return fail_build();
    }
    summary.tlas_rebuild_count = tlas_instances.empty() ? 0 : 1;
    renderer->acceleration_peak_capacity_bytes = (std::max)(
        renderer->acceleration_peak_capacity_bytes,
        blas_cache_capacity_bytes(cache_plan.next_state));

    std::vector<rt_blas_storage_pool_entry> retired_accelerations;
    for (const std::vector<rt_blas_cache_slot> &pool : renderer->blas_cache_state.pools) {
        for (const rt_blas_cache_slot &slot : pool) {
            if (slot.acceleration &&
                !rt_blas_cache_contains(cache_plan.next_state, slot.acceleration)) {
                retired_accelerations.push_back({
                    slot.storage_key,
                    slot.acceleration,
                    slot.storage_capacity_bytes,
                    {},
                    0});
            }
        }
    }
    out_result->blas_reused_count = summary.blas_reused_count;
    out_result->blas_rebuilt_count = summary.blas_rebuilt_count;
    out_result->blas_reused_triangle_chunk_count = summary.blas_reused_triangle_chunk_count;
    out_result->blas_rebuilt_triangle_chunk_count = summary.blas_rebuilt_triangle_chunk_count;
    out_result->tlas_rebuild_count = summary.tlas_rebuild_count;
    const std::uint64_t submitted_revision = request.scene_revision != 0
        ? request.scene_revision
        : resources.revision;
    const double command_record_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - sync_start).count();
    out_result->acceleration_timing.command_record_ms += command_record_ms;
    renderer->last_acceleration_cpu_ms = command_record_ms;
    if (deferred_acceleration != nullptr) {
        deferred_acceleration->encoder = encoder;
        deferred_acceleration->next_blas_cache_state = std::move(cache_plan.next_state);
        deferred_acceleration->created_accelerations = std::move(created_accelerations);
        deferred_acceleration->acquired_accelerations = std::move(acquired_accelerations);
        deferred_acceleration->retired_accelerations = std::move(retired_accelerations);
        deferred_acceleration->next_layer_tlas_instances = std::move(layer_tlas_instances);
        deferred_acceleration->scene_revision = resources.revision;
        deferred_acceleration->presentation_revision = submitted_revision;
        deferred_acceleration->connection_serial = resources.connection_serial;
        deferred_acceleration->acceleration_summary = summary;
        deferred_acceleration->scene_changed = true;
        return true;
    }

    rt_submission_token submission{};
    stage_error = {rt_rhi_operation::submit_commands, 0, {}};
    if (!submit_rt_commands(
            renderer,
            encoder,
            &submission,
            &out_result->acceleration_timing,
            &stage_error)) {
        return fail_build();
    }
    for (rt_blas_storage_pool_entry &entry : retired_accelerations) {
        enqueue_blas_storage_pool_entry(renderer, std::move(entry), submission);
    }
    renderer->submitted_acceleration_revision = submitted_revision;
    renderer->submitted_acceleration_summary = summary;
    renderer->layer_tlas_instances = std::move(layer_tlas_instances);
    destroy_scene_upload_buffers(renderer, &renderer->scene_buffers);
    return true;
}

void commit_deferred_acceleration_submission(
    rt_renderer* renderer,
    rt_deferred_acceleration_submission* deferred_acceleration)
{
    if (renderer == nullptr || renderer->rhi == nullptr || deferred_acceleration == nullptr ||
        !deferred_acceleration->encoder) {
        return;
    }
    if (deferred_acceleration->scene_changed) {
        renderer->blas_cache_state = std::move(deferred_acceleration->next_blas_cache_state);
        for (rt_blas_storage_pool_entry &entry : deferred_acceleration->retired_accelerations) {
            enqueue_blas_storage_pool_entry(renderer, std::move(entry), renderer->last_submission);
        }
        renderer->layer_tlas_instances = std::move(deferred_acceleration->next_layer_tlas_instances);
        destroy_scene_upload_buffers(renderer, &renderer->scene_buffers);
        renderer->frame_state.scene_revision = deferred_acceleration->scene_revision;
        renderer->frame_state.scene_valid = true;
        renderer->current_connection_serial = deferred_acceleration->connection_serial;
    }
    renderer->frame_state.presentation_revision = deferred_acceleration->presentation_revision;
    renderer->submitted_acceleration_revision = deferred_acceleration->presentation_revision;
    renderer->submitted_acceleration_summary = deferred_acceleration->acceleration_summary;
    *deferred_acceleration = {};
}

void discard_deferred_acceleration_submission(
    rt_renderer* renderer,
    rt_deferred_acceleration_submission* deferred_acceleration)
{
    if (renderer == nullptr || renderer->rhi == nullptr || deferred_acceleration == nullptr ||
        !deferred_acceleration->encoder) {
        return;
    }
    discard_rt_commands(renderer, deferred_acceleration->encoder);
    for (rt_blas_handle acceleration : deferred_acceleration->created_accelerations) {
        renderer->rhi->destroy_blas(acceleration);
    }
    for (rt_blas_storage_pool_entry &entry : deferred_acceleration->acquired_accelerations) {
        enqueue_blas_storage_pool_entry(renderer, std::move(entry), {});
    }
    destroy_scene_upload_buffers(renderer, &renderer->scene_buffers);
    *deferred_acceleration = {};
}

bool update_rt_renderer_bindings(
    rt_renderer* renderer,
    const rt_renderer_frame_request &request,
    const rt_renderer_frame_result &changes,
    rt_rhi_error* out_error)
{
    if (!rt_scene_has_renderable_primitives(*request.build) ||
        (!changes.output_changed &&
            !changes.viewer_resources_changed &&
            !changes.pick_resources_changed &&
            !changes.scene_changed &&
            !changes.acceleration_changed)) {
        return true;
    }
    const rt_scene_buffer_resources &resources = renderer->scene_buffers;
    const std::array<rt_binding_write, kViewerRtBindingCount> writes{{
        {{0, viewer_rt_binding_index(viewer_rt_binding::scene)},
            rt_descriptor_type::acceleration_structure, {}, 1, 0, {}, renderer->tlas},
        {{0, viewer_rt_binding_index(viewer_rt_binding::output)},
            rt_descriptor_type::storage_texture, {}, 1, 0, renderer->output_texture},
        {{0, viewer_rt_binding_index(viewer_rt_binding::accumulation)},
            rt_descriptor_type::storage_texture, {}, 1, 0, renderer->accumulation_texture},
        {{0, viewer_rt_binding_index(viewer_rt_binding::pick_output)},
            rt_descriptor_type::storage_buffer, renderer->pick_output_buffer, 1, kRtPickResultBufferBytes},
        {{0, viewer_rt_binding_index(viewer_rt_binding::triangle_colors)},
            rt_descriptor_type::structured_buffer, resources.triangle_colors,
            resources.triangle_color_count, sizeof(rtvdb::rgba)},
        {{0, viewer_rt_binding_index(viewer_rt_binding::instance_metadata)},
            rt_descriptor_type::structured_buffer, resources.instance_metadata,
            resources.instance_metadata_count, sizeof(rt_scene_geometry_metadata)},
        {{0, viewer_rt_binding_index(viewer_rt_binding::positions)},
            rt_descriptor_type::structured_buffer, resources.positions,
            resources.position_count, sizeof(rt_scene_gpu_position)},
        {{0, viewer_rt_binding_index(viewer_rt_binding::indices)},
            rt_descriptor_type::structured_buffer, resources.indices,
            resources.index_count, sizeof(std::uint32_t)},
        {{0, viewer_rt_binding_index(viewer_rt_binding::points)},
            rt_descriptor_type::structured_buffer, resources.points,
            resources.point_count, sizeof(rt_scene_gpu_point)},
        {{0, viewer_rt_binding_index(viewer_rt_binding::lines)},
            rt_descriptor_type::structured_buffer, resources.lines,
            resources.line_count, sizeof(rt_scene_gpu_line)},
        {
            {0, viewer_rt_binding_index(viewer_rt_binding::viewer_constants)},
            rt_descriptor_type::uniform_buffer,
            renderer->viewer_constant_buffer,
            kRtCommandSlotCount,
            kRtViewerConstantSlotStride},
    }};
    const rt_binding_update_request binding_request{writes.data(), writes.size()};
    rt_rhi_error stage_error{rt_rhi_operation::update_bindings, 0, {}};
    if (!renderer->rhi->update_bindings(binding_request, &stage_error)) {
        if (out_error != nullptr) {
            *out_error = stage_error;
        }
        return false;
    }
    return true;
}
} // namespace

bool initialize_rt_renderer(
    rt_renderer* renderer,
    const backend_config &config,
    rt_rhi_error* out_error)
{
    if (renderer == nullptr || renderer->rhi == nullptr) {
        set_missing_lifecycle_error(out_error, rt_rhi_operation::initialize, "RT renderer initialize is unavailable");
        return false;
    }
    if (config.capture_width <= 0 || config.capture_height <= 0) {
        if (out_error != nullptr) {
            *out_error = {
                rt_rhi_operation::initialize,
                0,
                "RT renderer initial output size must be positive"};
        }
        return false;
    }
    renderer->capabilities = {};
    renderer->frame_state = {};
    renderer->scene_buffers = {};
    renderer->output_texture = {};
    renderer->accumulation_texture = {};
    renderer->output_readback_buffer = {};
    renderer->output_readback_footprint = {};
    renderer->output_readback_submission = {};
    renderer->output_readback_width = 0;
    renderer->output_readback_height = 0;
    renderer->output_readback_pending = false;
    renderer->pick_output_buffer = {};
    renderer->pick_slots = {};
    renderer->viewer_constant_buffer = {};
    renderer->blas_cache_state = {};
    renderer->layer_tlas_instances.clear();
    renderer->current_connection_serial = 0;
    renderer->resource_pool_sequence = 1;
    renderer->last_submission = {};
    renderer->blas_storage_pool_hit_count = 0;
    renderer->blas_storage_pool_miss_count = 0;
    renderer->scene_buffer_pool_hit_count = 0;
    renderer->scene_buffer_pool_miss_count = 0;
    renderer->scene_buffer_allocation_count = 0;
    renderer->scene_buffer_growth_count = 0;
    renderer->resource_pool_eviction_count = 0;
    renderer->acceleration_peak_capacity_bytes = 0;
    renderer->scene_buffer_peak_capacity_bytes = 0;
    renderer->tlas = {};
    renderer->submitted_acceleration_revision = 0;
    renderer->submitted_acceleration_summary = {};
    renderer->last_acceleration_revision = 0;
    renderer->last_acceleration_summary = {};
    renderer->last_acceleration_cpu_ms = 0.0;
    renderer->last_point_blas_prebuild_info_ms = 0.0;
    renderer->last_point_blas_prebuild_info_count = 0;
    renderer->last_line_blas_prebuild_info_ms = 0.0;
    renderer->last_line_blas_prebuild_info_count = 0;
    renderer->shader_modules.clear();
    renderer->shader_entry_modules = {};
    renderer->pipeline = {};
    renderer->accumulation_state = {};
    renderer->pending_render_submissions.clear();
    renderer->pending_delivery_submissions.clear();
    renderer->async_error = {};
    renderer->async_failed = false;
    renderer->last_present_result = {};
    renderer->blas_reuse_enabled = !environment_flag_enabled("RTVDB_DISABLE_BLAS_REUSE");
    renderer->continuous_render = config.continuous_render;
    const rt_rhi_device_desc desc{
        static_cast<std::uint32_t>(config.capture_width),
        static_cast<std::uint32_t>(config.capture_height),
        config.d3d12.device,
        config.d3d12.command_queue};
    if (!renderer->rhi->initialize(desc, out_error)) {
        return false;
    }
    renderer->capabilities = renderer->rhi->capabilities();
    return true;
}

bool shutdown_rt_renderer(rt_renderer* renderer, rt_rhi_error* out_error) {
    if (renderer == nullptr || renderer->rhi == nullptr) {
        set_missing_lifecycle_error(out_error, rt_rhi_operation::shutdown, "RT renderer shutdown is unavailable");
        return false;
    }
    rt_rhi_timing idle_timing{};
    rt_rhi_error idle_error{};
    const bool waited = renderer->rhi->wait_idle(&idle_timing, &idle_error);
    destroy_scene_buffers(renderer, &renderer->scene_buffers);
    destroy_output_resources(renderer);
    destroy_pick_buffers(renderer);
    destroy_viewer_resources(renderer);
    destroy_rt_blas_cache(renderer, &renderer->blas_cache_state);
    destroy_rt_resource_pools(renderer);
    renderer->rhi->destroy_tlas(renderer->tlas);
    renderer->tlas = {};
    destroy_viewer_rt_shader_modules(renderer);
    const bool shutdown = renderer->rhi->shutdown(out_error);
    renderer->capabilities = {};
    renderer->frame_state = {};
    renderer->current_connection_serial = 0;
    renderer->layer_tlas_instances.clear();
    renderer->last_submission = {};
    renderer->resource_pool_sequence = 1;
    renderer->blas_storage_pool_hit_count = 0;
    renderer->blas_storage_pool_miss_count = 0;
    renderer->scene_buffer_pool_hit_count = 0;
    renderer->scene_buffer_pool_miss_count = 0;
    renderer->scene_buffer_allocation_count = 0;
    renderer->scene_buffer_growth_count = 0;
    renderer->resource_pool_eviction_count = 0;
    renderer->acceleration_peak_capacity_bytes = 0;
    renderer->scene_buffer_peak_capacity_bytes = 0;
    renderer->submitted_acceleration_revision = 0;
    renderer->submitted_acceleration_summary = {};
    renderer->last_acceleration_revision = 0;
    renderer->last_acceleration_summary = {};
    renderer->last_acceleration_cpu_ms = 0.0;
    renderer->last_point_blas_prebuild_info_ms = 0.0;
    renderer->last_point_blas_prebuild_info_count = 0;
    renderer->last_line_blas_prebuild_info_ms = 0.0;
    renderer->last_line_blas_prebuild_info_count = 0;
    renderer->pipeline = {};
    renderer->accumulation_state = {};
    renderer->pending_render_submissions.clear();
    renderer->pending_delivery_submissions.clear();
    renderer->async_error = {};
    renderer->async_failed = false;
    renderer->last_present_result = {};
    renderer->blas_reuse_enabled = true;
    renderer->continuous_render = false;
    if (!waited && shutdown && out_error != nullptr) {
        *out_error = idle_error;
        out_error->operation = rt_rhi_operation::shutdown;
    }
    return waited && shutdown;
}

bool wait_for_rt_renderer_idle(
    rt_renderer* renderer,
    rt_rhi_timing* out_timing,
    rt_rhi_error* out_error)
{
    if (renderer == nullptr || renderer->rhi == nullptr) {
        set_missing_lifecycle_error(out_error, rt_rhi_operation::wait_idle, "RT renderer wait_idle is unavailable");
        return false;
    }
    if (!renderer->rhi->wait_idle(out_timing, out_error)) {
        return false;
    }
    return collect_rt_renderer_render_submissions(renderer, out_error);
}

bool rt_pick_dispatch_request_matches(
    const rt_pick_dispatch_request &left,
    const rt_pick_dispatch_request &right)
{
    return
        left.pixel_x == right.pixel_x &&
        left.pixel_y == right.pixel_y &&
        rt_pick_dispatch_request_matches_scene_and_view(left, right);
}

bool rt_pick_dispatch_request_matches_scene_and_view(
    const rt_pick_dispatch_request &left,
    const rt_pick_dispatch_request &right)
{
    if (left.scene_revision != right.scene_revision ||
        left.width != right.width ||
        left.height != right.height) {
        return false;
    }

    rt_viewer_constants left_constants = left.constants;
    rt_viewer_constants right_constants = right.constants;
    left_constants.accumulation_sample_index = 0;
    right_constants.accumulation_sample_index = 0;
    left_constants.accumulation_jitter[0] = 0.0f;
    left_constants.accumulation_jitter[1] = 0.0f;
    right_constants.accumulation_jitter[0] = 0.0f;
    right_constants.accumulation_jitter[1] = 0.0f;
    left_constants.hover_highlight_kind = 0;
    right_constants.hover_highlight_kind = 0;
    left_constants.hover_primitive_index = 0;
    right_constants.hover_primitive_index = 0;
    left_constants.selection_highlight_kind = 0;
    right_constants.selection_highlight_kind = 0;
    left_constants.selection_primitive_index = 0;
    right_constants.selection_primitive_index = 0;
    left_constants.pick_pixel_x = 0;
    left_constants.pick_pixel_y = 0;
    right_constants.pick_pixel_x = 0;
    right_constants.pick_pixel_y = 0;
    return std::memcmp(&left_constants, &right_constants, sizeof(left_constants)) == 0;
}

bool begin_rt_commands(
    rt_renderer* renderer,
    rt_queue_class queue,
    rt_command_encoder* out_encoder,
    rt_rhi_error* out_error,
    rt_rhi_timing* out_timing)
{
    if (renderer == nullptr || renderer->rhi == nullptr) {
        set_missing_lifecycle_error(
            out_error,
            rt_rhi_operation::begin_commands,
            "RT command begin is unavailable");
        return false;
    }
    if (renderer->async_failed) {
        if (out_error != nullptr) {
            *out_error = renderer->async_error;
        }
        return false;
    }
    const auto begin_start = std::chrono::steady_clock::now();
    const bool begun = renderer->rhi->begin_commands(queue, out_encoder, out_error);
    if (out_timing != nullptr) {
        out_timing->command_slot_wait_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin_start).count();
    }
    return begun;
}

bool submit_rt_commands(
    rt_renderer* renderer,
    rt_command_encoder encoder,
    rt_submission_token* out_submission,
    rt_rhi_timing* out_timing,
    rt_rhi_error* out_error)
{
    if (renderer == nullptr || renderer->rhi == nullptr) {
        set_missing_lifecycle_error(
            out_error,
            rt_rhi_operation::submit_commands,
            "RT command submit is unavailable");
        return false;
    }
    const bool submitted = renderer->rhi->submit_commands(
        encoder,
        out_submission,
        out_timing,
        out_error);
    if (submitted && out_submission != nullptr) {
        renderer->last_submission = *out_submission;
    }
    return submitted;
}

void discard_rt_commands(
    rt_renderer* renderer,
    rt_command_encoder encoder)
{
    if (renderer != nullptr && renderer->rhi != nullptr && encoder) {
        renderer->rhi->discard_commands(encoder);
    }
}

bool is_rt_submission_complete(
    rt_renderer* renderer,
    rt_submission_token submission,
    bool* out_complete,
    rt_rhi_error* out_error)
{
    if (renderer == nullptr || renderer->rhi == nullptr) {
        set_missing_lifecycle_error(
            out_error,
            rt_rhi_operation::query_submission,
            "RT submission query is unavailable");
        return false;
    }
    return renderer->rhi->is_complete(submission, out_complete, out_error);
}

bool wait_for_rt_submission(
    rt_renderer* renderer,
    rt_submission_token submission,
    rt_rhi_timing* out_timing,
    rt_rhi_error* out_error)
{
    if (renderer == nullptr || renderer->rhi == nullptr) {
        set_missing_lifecycle_error(
            out_error,
            rt_rhi_operation::wait_submission,
            "RT submission wait is unavailable");
        return false;
    }
    return renderer->rhi->wait(submission, out_timing, out_error);
}

bool record_rt_barriers(
    rt_renderer* renderer,
    rt_command_encoder encoder,
    const rt_resource_barrier* barriers,
    std::size_t barrier_count,
    rt_rhi_error* out_error)
{
    if (renderer == nullptr || renderer->rhi == nullptr) {
        set_frame_error(
            out_error,
            rt_rhi_operation::transition_resource,
            "RT resource barrier API is unavailable");
        return false;
    }
    return renderer->rhi->barrier(
        encoder,
        barriers,
        barrier_count,
        out_error);
}

bool record_rt_buffer_copy(
    rt_renderer* renderer,
    rt_command_encoder encoder,
    rt_buffer_handle source,
    rt_buffer_handle destination,
    const rt_buffer_copy_region &region,
    rt_rhi_error* out_error)
{
    if (renderer == nullptr || renderer->rhi == nullptr) {
        set_frame_error(
            out_error,
            rt_rhi_operation::copy_resource,
            "RT buffer copy API is unavailable");
        return false;
    }
    return renderer->rhi->copy_buffer(
        encoder,
        source,
        destination,
        region,
        out_error);
}

bool record_rt_texture_to_buffer_copy(
    rt_renderer* renderer,
    rt_command_encoder encoder,
    rt_texture_handle source,
    rt_buffer_handle destination,
    const rt_texture_buffer_copy_region &region,
    rt_rhi_error* out_error)
{
    if (renderer == nullptr || renderer->rhi == nullptr) {
        set_frame_error(
            out_error,
            rt_rhi_operation::copy_resource,
            "RT texture-to-buffer copy API is unavailable");
        return false;
    }
    return renderer->rhi->copy_texture_to_buffer(
        encoder,
        source,
        destination,
        region,
        out_error);
}

bool record_rt_texture_clear(
    rt_renderer* renderer,
    rt_command_encoder encoder,
    rt_texture_handle texture,
    const float color[4],
    rt_rhi_error* out_error)
{
    if (renderer == nullptr || renderer->rhi == nullptr) {
        set_frame_error(
            out_error,
            rt_rhi_operation::copy_resource,
            "RT texture clear API is unavailable");
        return false;
    }
    return renderer->rhi->clear_texture(
        encoder,
        texture,
        color,
        out_error);
}

bool record_rt_trace_rays(
    rt_renderer* renderer,
    rt_command_encoder encoder,
    const rt_trace_rays_desc &desc,
    rt_rhi_error* out_error)
{
    if (renderer == nullptr || renderer->rhi == nullptr) {
        set_frame_error(
            out_error,
            rt_rhi_operation::trace_rays,
            "RT ray trace API is unavailable");
        return false;
    }
    return renderer->rhi->trace_rays(
        encoder,
        desc,
        out_error);
}

bool write_rt_trace_constants(
    rt_renderer* renderer,
    rt_command_encoder encoder,
    const rt_viewer_constants &constants,
    rt_rhi_error* out_error)
{
    if (renderer == nullptr || renderer->rhi == nullptr || !renderer->viewer_constant_buffer ||
        !encoder || encoder.slot >= kRtCommandSlotCount) {
        set_frame_error(
            out_error,
            rt_rhi_operation::trace_rays,
            "RT trace constant upload request is invalid");
        return false;
    }
    const rt_viewer_constants_gpu packed = pack_rt_viewer_constants(constants);
    return renderer->rhi->upload_buffer(
        renderer->viewer_constant_buffer,
        static_cast<std::size_t>(encoder.slot) * kRtViewerConstantSlotStride,
        &packed,
        sizeof(packed),
        out_error);
}

bool read_rt_buffer(
    rt_renderer* renderer,
    rt_buffer_handle buffer,
    std::size_t offset,
    void* data,
    std::size_t size,
    rt_rhi_error* out_error)
{
    if (renderer == nullptr || renderer->rhi == nullptr) {
        set_frame_error(
            out_error,
            rt_rhi_operation::readback,
            "RT buffer read API is unavailable");
        return false;
    }
    return renderer->rhi->read_buffer(
        buffer,
        offset,
        data,
        size,
        out_error);
}

bool transition_rt_texture(
    rt_renderer* renderer,
    rt_command_encoder encoder,
    rt_texture_handle texture,
    rt_resource_usage usage,
    rt_rhi_error* out_error)
{
    if (!texture) {
        set_frame_error(
            out_error,
            rt_rhi_operation::transition_resource,
            "RT texture transition resource is invalid");
        return false;
    }
    const rt_resource_barrier barrier{
        rt_resource_kind::texture,
        {},
        texture,
        rt_resource_usage::undefined,
        usage};
    return record_rt_barriers(renderer, encoder, &barrier, 1, out_error);
}

bool transition_rt_buffer(
    rt_renderer* renderer,
    rt_command_encoder encoder,
    rt_buffer_handle buffer,
    rt_resource_usage usage,
    rt_rhi_error* out_error)
{
    if (!buffer) {
        set_frame_error(
            out_error,
            rt_rhi_operation::transition_resource,
            "RT buffer transition resource is invalid");
        return false;
    }
    const rt_resource_barrier barrier{
        rt_resource_kind::buffer,
        buffer,
        {},
        rt_resource_usage::undefined,
        usage};
    return record_rt_barriers(renderer, encoder, &barrier, 1, out_error);
}

bool prepare_rt_renderer_pipeline(
    rt_renderer* renderer,
    const rt_pipeline_desc &pipeline_desc,
    rt_renderer_frame_result* in_out_result,
    rt_rhi_error* out_error)
{
    if (renderer == nullptr || renderer->rhi == nullptr || in_out_result == nullptr ||
        !validate_rt_pipeline_desc(pipeline_desc)) {
        set_frame_error(
            out_error,
            rt_rhi_operation::prepare_pipeline,
            "RT renderer pipeline preparation is unavailable");
        return false;
    }
    if (!renderer->pipeline) {
        rt_pipeline_handle pipeline{};
        if (!renderer->rhi->create_pipeline(pipeline_desc, &pipeline, out_error) || !pipeline) {
            return false;
        }
        renderer->pipeline = pipeline;
        in_out_result->pipeline_changed = true;
    }
    return true;
}

void begin_rt_renderer_access(rt_renderer* renderer) {
    if (renderer != nullptr) {
        renderer->access_mutex.lock();
    }
}

void end_rt_renderer_access(rt_renderer* renderer) {
    if (renderer != nullptr) {
        renderer->access_mutex.unlock();
    }
}

bool submit_rt_output_render(
    rt_renderer* renderer,
    const rt_native_frame_request &request,
    rt_command_encoder encoder,
    rt_submission_token* out_submission,
    rt_rhi_timing* out_timing,
    rt_rhi_error* out_error)
{
    if (out_submission != nullptr) {
        *out_submission = {};
    }
    if (!encoder &&
        !begin_rt_commands(renderer, rt_queue_class::graphics, &encoder, out_error, out_timing)) {
        return false;
    }
    const auto record_start = std::chrono::steady_clock::now();
    const bool recorded =
        write_rt_trace_constants(renderer, encoder, request.constants, out_error) &&
        transition_rt_texture(
            renderer,
            encoder,
            renderer->accumulation_texture,
            rt_resource_usage::shader_write,
            out_error) &&
        transition_rt_texture(
            renderer,
            encoder,
            renderer->output_texture,
            rt_resource_usage::shader_write,
            out_error) &&
        record_rt_trace_rays(
            renderer,
            encoder,
            {
                renderer->pipeline,
                rt_logical_dispatch_entry::render,
                static_cast<std::uint32_t>(request.width),
                static_cast<std::uint32_t>(request.height),
                1,
                renderer->capabilities.timestamp_queries},
            out_error) &&
        transition_rt_texture(
            renderer,
            encoder,
            renderer->output_texture,
            rt_resource_usage::shader_read,
            out_error);
    if (!recorded) {
        discard_rt_commands(renderer, encoder);
        return false;
    }
    if (out_timing != nullptr) {
        out_timing->command_record_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - record_start).count();
    }
    return submit_rt_commands(
        renderer,
        encoder,
        out_submission,
        out_timing,
        out_error);
}

bool submit_rt_output_clear(
    rt_renderer* renderer,
    const rt_native_frame_request &request,
    rt_command_encoder encoder,
    rt_submission_token* out_submission,
    rt_rhi_timing* out_timing,
    rt_rhi_error* out_error)
{
    if (out_submission != nullptr) {
        *out_submission = {};
    }
    if (!encoder &&
        !begin_rt_commands(renderer, rt_queue_class::graphics, &encoder, out_error, out_timing)) {
        return false;
    }
    const auto record_start = std::chrono::steady_clock::now();
    const bool recorded =
        transition_rt_texture(
            renderer,
            encoder,
            renderer->output_texture,
            rt_resource_usage::clear_destination,
            out_error) &&
        record_rt_texture_clear(
            renderer,
            encoder,
            renderer->output_texture,
            request.clear_color,
            out_error) &&
        transition_rt_texture(
            renderer,
            encoder,
            renderer->output_texture,
            rt_resource_usage::shader_read,
            out_error);
    if (!recorded) {
        discard_rt_commands(renderer, encoder);
        return false;
    }
    if (out_timing != nullptr) {
        out_timing->command_record_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - record_start).count();
    }
    return submit_rt_commands(
        renderer,
        encoder,
        out_submission,
        out_timing,
        out_error);
}

bool submit_rt_reused_output(
    rt_renderer* renderer,
    rt_command_encoder encoder,
    rt_submission_token* out_submission,
    rt_rhi_timing* out_timing,
    rt_rhi_error* out_error)
{
    if (out_submission != nullptr) {
        *out_submission = {};
    }
    if (!encoder &&
        !begin_rt_commands(renderer, rt_queue_class::graphics, &encoder, out_error, out_timing)) {
        return false;
    }
    const auto record_start = std::chrono::steady_clock::now();
    if (!transition_rt_texture(
            renderer,
            encoder,
            renderer->output_texture,
            rt_resource_usage::shader_read,
            out_error)) {
        discard_rt_commands(renderer, encoder);
        return false;
    }
    if (out_timing != nullptr) {
        out_timing->command_record_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - record_start).count();
    }
    return submit_rt_commands(
        renderer,
        encoder,
        out_submission,
        out_timing,
        out_error);
}

bool collect_rt_output_readback(
    rt_renderer* renderer,
    std::vector<std::uint8_t>* out_pixels,
    rt_rhi_error* out_error)
{
    bool complete = false;
    if (!is_rt_submission_complete(
            renderer,
            renderer->output_readback_submission,
            &complete,
            out_error)) {
        return false;
    }
    if (!complete) {
        return true;
    }
    const rt_texture_copy_footprint footprint = renderer->output_readback_footprint;
    std::vector<std::uint8_t> readback(footprint.total_size);
    if (!renderer->rhi->read_buffer(
            renderer->output_readback_buffer,
            0,
            readback.data(),
            readback.size(),
            out_error)) {
        return false;
    }
    const std::size_t row_bytes =
        static_cast<std::size_t>(renderer->output_readback_width) * 4u;
    out_pixels->resize(
        row_bytes *
        static_cast<std::size_t>(renderer->output_readback_height));
    for (int y = 0; y < renderer->output_readback_height; ++y) {
        std::memcpy(
            out_pixels->data() + static_cast<std::size_t>(y) * row_bytes,
            readback.data() + static_cast<std::size_t>(y) * footprint.row_pitch,
            row_bytes);
    }
    if (renderer->capabilities.output_format == rt_texture_format::rgba8_unorm) {
        for (std::size_t offset = 0; offset + 3u < out_pixels->size(); offset += 4u) {
            std::swap((*out_pixels)[offset], (*out_pixels)[offset + 2u]);
        }
    }
    renderer->output_readback_submission = {};
    renderer->output_readback_pending = false;
    return true;
}

void discard_rt_output_readback(rt_renderer* renderer) {
    if (renderer == nullptr) {
        return;
    }
    renderer->output_readback_submission = {};
    renderer->output_readback_pending = false;
}

bool readback_rt_output(
    rt_renderer* renderer,
    int width,
    int height,
    std::vector<std::uint8_t>* out_pixels,
    rt_rhi_error* out_error)
{
    if (renderer == nullptr || renderer->rhi == nullptr || out_pixels == nullptr ||
        width <= 0 || height <= 0 ||
        width != renderer->frame_state.output_width ||
        height != renderer->frame_state.output_height ||
        !renderer->output_texture || !renderer->output_readback_buffer ||
        renderer->output_readback_footprint.total_size == 0) {
        set_frame_error(
            out_error,
            rt_rhi_operation::readback,
            "RT output readback request is invalid");
        return false;
    }
    out_pixels->clear();
    if (renderer->output_readback_pending) {
        if (!collect_rt_output_readback(renderer, out_pixels, out_error)) {
            discard_rt_output_readback(renderer);
            return false;
        }
        if (renderer->output_readback_pending || !out_pixels->empty()) {
            return true;
        }
    }

    rt_command_encoder encoder{};
    if (!begin_rt_commands(renderer, rt_queue_class::graphics, &encoder, out_error)) {
        return false;
    }
    const rt_texture_copy_footprint footprint = renderer->output_readback_footprint;
    const bool recorded =
        transition_rt_texture(
            renderer,
            encoder,
            renderer->output_texture,
            rt_resource_usage::copy_source,
            out_error) &&
        record_rt_texture_to_buffer_copy(
            renderer,
            encoder,
            renderer->output_texture,
            renderer->output_readback_buffer,
            {
                0,
                footprint.row_pitch,
                static_cast<std::uint32_t>(width),
                static_cast<std::uint32_t>(height)},
            out_error) &&
        transition_rt_texture(
            renderer,
            encoder,
            renderer->output_texture,
            rt_resource_usage::shader_read,
            out_error);
    if (!recorded) {
        discard_rt_commands(renderer, encoder);
        return false;
    }
    rt_rhi_timing timing{};
    if (!submit_rt_commands(
            renderer,
            encoder,
            &renderer->output_readback_submission,
            &timing,
            out_error)) {
        return false;
    }
    renderer->output_readback_width = width;
    renderer->output_readback_height = height;
    renderer->output_readback_pending = true;
    return true;
}

bool execute_rt_renderer_native_frame(
    rt_renderer* renderer,
    const rt_native_frame_request &request,
    rt_present_result* out_result,
    rt_rhi_error* out_error,
    rt_deferred_acceleration_submission* deferred_acceleration)
{
    if (out_result != nullptr) {
        *out_result = {};
    }
    if (renderer == nullptr || renderer->rhi == nullptr) {
        set_frame_error(out_error, rt_rhi_operation::present, "RT renderer frame operation is unavailable");
        return false;
    }
    rt_present_result result{};
    bool succeeded = false;
    if (request.operation == rt_present_operation::readback_bgra) {
        const auto readback_start = std::chrono::steady_clock::now();
        succeeded = readback_rt_output(
            renderer,
            request.width,
            request.height,
            request.out_pixels,
            out_error);
        result.readback_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - readback_start).count();
        result.captured = succeeded && request.out_pixels != nullptr && !request.out_pixels->empty();
        result.readback_submission = renderer->output_readback_submission;
    } else if (request.operation == rt_present_operation::post_present) {
        succeeded = true;
    } else {
        if (request.build == nullptr || request.width <= 0 || request.height <= 0) {
            set_frame_error(out_error, rt_rhi_operation::trace_rays, "RT output dispatch request is invalid");
            discard_deferred_acceleration_submission(renderer, deferred_acceleration);
            return false;
        }
        rt_rhi_timing dispatch_timing{};
        const rt_command_encoder acceleration_encoder = deferred_acceleration != nullptr
            ? deferred_acceleration->encoder
            : rt_command_encoder{};
        if (request.reuse_output) {
            succeeded = submit_rt_reused_output(
                renderer,
                acceleration_encoder,
                &result.render_submission,
                &dispatch_timing,
                out_error);
        } else if (request.dispatch == rt_dispatch_kind::render) {
            succeeded = submit_rt_output_render(
                renderer,
                request,
                acceleration_encoder,
                &result.render_submission,
                &dispatch_timing,
                out_error);
        } else if (request.dispatch == rt_dispatch_kind::clear) {
            succeeded = submit_rt_output_clear(
                renderer,
                request,
                acceleration_encoder,
                &result.render_submission,
                &dispatch_timing,
                out_error);
        } else {
            set_frame_error(out_error, rt_rhi_operation::trace_rays, "RT output operation is invalid");
            discard_deferred_acceleration_submission(renderer, deferred_acceleration);
            return false;
        }
        if (!succeeded) {
            discard_deferred_acceleration_submission(renderer, deferred_acceleration);
            return false;
        }
        if (deferred_acceleration != nullptr && deferred_acceleration->encoder) {
            const auto acceleration_finalize_start = std::chrono::steady_clock::now();
            commit_deferred_acceleration_submission(renderer, deferred_acceleration);
            result.acceleration_finalize_cpu_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - acceleration_finalize_start).count();
        }
        result.output_timing = dispatch_timing;
        result.timing = dispatch_timing;
        result.reused_output = request.reuse_output;
        if (out_result != nullptr) {
            *out_result = result;
        }

        if (request.operation == rt_present_operation::native_texture) {
            const rt_native_texture_publish_desc native_desc{
                request.native_target,
                request.out_native_target,
                &result.native_publish_submission};
            rt_native_texture_extension* const extension =
                renderer->rhi->native_texture_extension();
            rt_rhi_timing delivery_timing{};
            const auto native_publish_start = std::chrono::steady_clock::now();
            succeeded = extension != nullptr && extension->publish_texture(
                renderer->output_texture,
                native_desc,
                &delivery_timing,
                out_error);
            result.native_publish_cpu_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - native_publish_start).count();
            result.native_publish_timing = delivery_timing;
            result.timing.submit_cpu_ms += delivery_timing.submit_cpu_ms;
            result.timing.gpu_wait_ms += delivery_timing.gpu_wait_ms;
            result.timing.gpu_ms += delivery_timing.gpu_ms;
            result.rendered = succeeded;
            result.reused_output = succeeded && request.reuse_output;
        } else if (request.operation == rt_present_operation::capture_bgra) {
            const auto readback_start = std::chrono::steady_clock::now();
            succeeded = readback_rt_output(
                renderer,
                request.width,
                request.height,
                request.out_pixels,
                out_error);
            result.readback_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - readback_start).count();
            result.captured = succeeded && request.out_pixels != nullptr && !request.out_pixels->empty();
            result.readback_submission = renderer->output_readback_submission;
        } else {
            set_frame_error(out_error, rt_rhi_operation::present, "RT output delivery operation is invalid");
            return false;
        }
    }
    if (out_result != nullptr) {
        *out_result = result;
    }
    if (!succeeded) {
        return false;
    }
    renderer->last_present_result = result;
    return true;
}
bool dispatch_rt_renderer_pick(
    rt_renderer* renderer,
    const rt_pick_dispatch_request &request,
    pick_result* out_result,
    rt_pick_dispatch_request* out_completed_request,
    rt_rhi_error* out_error,
    bool* out_pending)
{
    if (out_pending != nullptr) {
        *out_pending = false;
    }
    if (out_result != nullptr) {
        *out_result = {};
    }
    if (out_completed_request != nullptr) {
        *out_completed_request = {};
    }
    if (renderer == nullptr || renderer->rhi == nullptr || out_result == nullptr) {
        set_frame_error(out_error, rt_rhi_operation::dispatch_pick, "RT renderer pick dispatch is unavailable");
        return false;
    }
    if (request.width <= 0 || request.height <= 0 ||
        request.pixel_x < 0 || request.pixel_y < 0 ||
        !renderer->pipeline || !renderer->pick_output_buffer) {
        set_frame_error(out_error, rt_rhi_operation::dispatch_pick, "RT renderer pick request is invalid");
        return false;
    }
    for (const rt_pick_slot &slot : renderer->pick_slots) {
        if (!slot.readback_buffer) {
            set_frame_error(out_error, rt_rhi_operation::dispatch_pick, "RT renderer pick readback slot is invalid");
            return false;
        }
    }
    if (!renderer->tlas) {
        if (out_completed_request != nullptr) {
            *out_completed_request = request;
        }
        return true;
    }

    bool completed = false;
    for (rt_pick_slot &slot : renderer->pick_slots) {
        if (!slot.pending) {
            continue;
        }
        bool complete = false;
        if (!is_rt_submission_complete(renderer, slot.submission, &complete, out_error)) {
            return false;
        }
        if (!complete) {
            if (out_pending != nullptr) {
                *out_pending = true;
            }
            return false;
        }
        rt_pick_gpu_result result{};
        if (!read_rt_buffer(
                renderer,
                slot.readback_buffer,
                0,
                &result,
                sizeof(result),
                out_error)) {
            return false;
        }
        const rt_pick_dispatch_request completed_request = slot.request;
        out_result->kind = static_cast<hover_highlight_kind>(result.primitive_kind);
        out_result->primitive_index = result.primitive_index;
        out_result->distance = result.hit != 0u ? result.distance : 0.0f;
        if (out_completed_request != nullptr) {
            *out_completed_request = completed_request;
        }
        slot.pending = false;
        slot.submission = {};
        if (rt_pick_dispatch_request_matches(completed_request, request)) {
            return true;
        }
        completed = true;
        break;
    }

    rt_command_encoder encoder{};
    if (!begin_rt_commands(
            renderer,
            rt_queue_class::graphics,
            &encoder,
            out_error)) {
        return false;
    }
    if (encoder.slot >= kRtCommandSlotCount) {
        discard_rt_commands(renderer, encoder);
        set_frame_error(out_error, rt_rhi_operation::dispatch_pick, "RT renderer pick command slot is invalid");
        return false;
    }
    rt_pick_slot &slot = renderer->pick_slots[encoder.slot];
    if (slot.pending) {
        discard_rt_commands(renderer, encoder);
        set_frame_error(out_error, rt_rhi_operation::dispatch_pick, "RT renderer pick command slot is still pending");
        return false;
    }

    const bool recorded =
        write_rt_trace_constants(
            renderer,
            encoder,
            request.constants,
            out_error) &&
        transition_rt_buffer(
            renderer,
            encoder,
            renderer->pick_output_buffer,
            rt_resource_usage::shader_write,
            out_error) &&
        record_rt_trace_rays(
            renderer,
            encoder,
            {
                renderer->pipeline,
                rt_logical_dispatch_entry::pick,
                1,
                1,
                1,
            },
            out_error) &&
        transition_rt_buffer(
            renderer,
            encoder,
            renderer->pick_output_buffer,
            rt_resource_usage::copy_source,
            out_error) &&
        record_rt_buffer_copy(
            renderer,
            encoder,
            renderer->pick_output_buffer,
            slot.readback_buffer,
            {0, 0, sizeof(rt_pick_gpu_result)},
            out_error) &&
        transition_rt_buffer(
            renderer,
            encoder,
            slot.readback_buffer,
            rt_resource_usage::host_read,
            out_error) &&
        transition_rt_buffer(
            renderer,
            encoder,
            renderer->pick_output_buffer,
            rt_resource_usage::shader_write,
            out_error);
    if (!recorded) {
        discard_rt_commands(renderer, encoder);
        return false;
    }

    rt_rhi_timing timing{};
    if (!submit_rt_commands(
            renderer,
            encoder,
            &slot.submission,
            &timing,
            out_error)) {
        return false;
    }
    slot.request = request;
    slot.pending = true;
    if (out_pending != nullptr) {
        *out_pending = true;
    }
    return completed;
}

void copy_rt_renderer_diagnostics(scene_build_info* out_info, const rt_renderer &renderer) {
    if (out_info == nullptr) {
        return;
    }
    out_info->revision = renderer.last_acceleration_revision;
    out_info->blas_reused_count = renderer.last_acceleration_summary.blas_reused_count;
    out_info->blas_rebuilt_count = renderer.last_acceleration_summary.blas_rebuilt_count;
    out_info->blas_reused_triangle_chunk_count = renderer.last_acceleration_summary.blas_reused_triangle_chunk_count;
    out_info->blas_rebuilt_triangle_chunk_count = renderer.last_acceleration_summary.blas_rebuilt_triangle_chunk_count;
    out_info->tlas_rebuild_count = renderer.last_acceleration_summary.tlas_rebuild_count;
    out_info->accel_build_ms = renderer.last_acceleration_cpu_ms;
    out_info->accel_procedural_aabb_ms = renderer.scene_buffers.procedural_aabb_upload_ms;
    out_info->accel_point_blas_prebuild_info_ms =
        renderer.last_point_blas_prebuild_info_ms;
    out_info->accel_point_blas_prebuild_info_count =
        renderer.last_point_blas_prebuild_info_count;
    out_info->accel_line_blas_prebuild_info_ms =
        renderer.last_line_blas_prebuild_info_ms;
    out_info->accel_line_blas_prebuild_info_count =
        renderer.last_line_blas_prebuild_info_count;
    out_info->paint_rt_scene_snapshot_ms =
        renderer.last_present_result.scene_snapshot_cpu_ms;
    out_info->paint_rt_pre_acceleration_prepare_ms =
        renderer.last_present_result.frame_pre_acceleration_prepare_cpu_ms;
    out_info->paint_as_command_slot_wait_ms =
        renderer.last_present_result.acceleration_timing.command_slot_wait_ms;
    out_info->paint_accel_command_record_ms =
        renderer.last_present_result.acceleration_timing.command_record_ms;
    out_info->paint_rt_post_acceleration_prepare_ms =
        renderer.last_present_result.frame_post_acceleration_prepare_cpu_ms;
    out_info->paint_rt_output_prepare_ms =
        renderer.last_present_result.rt_output_prepare_cpu_ms;
    out_info->paint_rt_output_command_slot_wait_ms =
        renderer.last_present_result.output_timing.command_slot_wait_ms;
    out_info->paint_rt_command_record_ms =
        renderer.last_present_result.output_timing.command_record_ms;
    out_info->paint_rt_submit_ms = renderer.last_present_result.output_timing.submit_cpu_ms;
    out_info->paint_as_finalize_ms = renderer.last_present_result.acceleration_finalize_cpu_ms;
    out_info->paint_native_target_publish_ms =
        renderer.last_present_result.native_publish_cpu_ms;
    out_info->paint_rt_accumulation_finalize_ms =
        renderer.last_present_result.accumulation_finalize_cpu_ms;
    out_info->dispatch_submit_cpu_ms = renderer.last_present_result.timing.submit_cpu_ms;
    out_info->dispatch_gpu_wait_ms = renderer.last_present_result.timing.gpu_wait_ms;
    out_info->dispatch_gpu_ms = renderer.last_present_result.timing.gpu_ms;
    out_info->dispatch_ms =
        renderer.last_present_result.timing.submit_cpu_ms +
        renderer.last_present_result.timing.gpu_wait_ms;
    out_info->readback_ms = renderer.last_present_result.readback_ms;
    out_info->accumulation_sample_count = renderer.accumulation_state.sample_count;
    out_info->accumulation_target_sample_count = kRtMaxAccumulationSamples;
    out_info->accumulation_in_progress = renderer.accumulation_state.active;
    out_info->blas_storage_pool_hit_count = renderer.blas_storage_pool_hit_count;
    out_info->blas_storage_pool_miss_count = renderer.blas_storage_pool_miss_count;
    out_info->scene_buffer_pool_hit_count = renderer.scene_buffer_pool_hit_count;
    out_info->scene_buffer_pool_miss_count = renderer.scene_buffer_pool_miss_count;
    out_info->scene_buffer_allocation_count = renderer.scene_buffer_allocation_count;
    out_info->scene_buffer_growth_count = renderer.scene_buffer_growth_count;
    out_info->acceleration_capacity_bytes = blas_cache_capacity_bytes(renderer.blas_cache_state);
    out_info->scene_buffer_capacity_bytes = scene_buffer_capacity_bytes(renderer.scene_buffers);
    out_info->acceleration_peak_capacity_bytes = renderer.acceleration_peak_capacity_bytes;
    out_info->scene_buffer_peak_capacity_bytes = renderer.scene_buffer_peak_capacity_bytes;
    out_info->resource_pool_eviction_count = renderer.resource_pool_eviction_count;
    for (const rt_blas_storage_pool_entry &entry : renderer.blas_storage_pool) {
        out_info->blas_storage_pool_bytes += entry.capacity_bytes;
        if (entry.retirement_submission) {
            out_info->retired_resource_bytes += entry.capacity_bytes;
        }
    }
    for (const rt_scene_buffer_pool_entry &entry : renderer.scene_buffer_pool) {
        out_info->scene_buffer_pool_bytes += entry.capacity_bytes;
        if (entry.retirement_submission) {
            out_info->retired_resource_bytes += entry.capacity_bytes;
        }
    }
}

void copy_rt_rhi_diagnostics(
    scene_build_info* out_info,
    const rt_rhi_diagnostics &diagnostics)
{
    if (out_info == nullptr) {
        return;
    }
    out_info->accel_build_ms = diagnostics.acceleration_cpu_ms;
    out_info->accel_host_prep_ms = diagnostics.acceleration_host_prepare_ms;
    out_info->accel_instance_build_ms = diagnostics.acceleration_instance_build_ms;
    out_info->accel_procedural_aabb_ms = diagnostics.acceleration_procedural_aabb_ms;
    out_info->accel_command_record_ms = diagnostics.acceleration_command_record_ms;
    out_info->accel_resource_alloc_ms = diagnostics.acceleration_resource_allocate_ms;
    out_info->accel_build_call_record_ms = diagnostics.acceleration_build_call_record_ms;
    out_info->accel_prebuild_info_ms = diagnostics.acceleration_prebuild_query_ms;
    out_info->accel_chunk_blas_prebuild_info_ms = diagnostics.chunk_blas_prebuild_query_ms;
    out_info->accel_chunk_blas_prebuild_info_count = diagnostics.chunk_blas_prebuild_query_count;
    out_info->accel_group_blas_prebuild_info_ms = diagnostics.grouped_blas_prebuild_query_ms;
    out_info->accel_group_blas_prebuild_info_count = diagnostics.grouped_blas_prebuild_query_count;
    out_info->accel_point_blas_prebuild_info_ms = diagnostics.point_blas_prebuild_query_ms;
    out_info->accel_point_blas_prebuild_info_count = diagnostics.point_blas_prebuild_query_count;
    out_info->accel_line_blas_prebuild_info_ms = diagnostics.line_blas_prebuild_query_ms;
    out_info->accel_line_blas_prebuild_info_count = diagnostics.line_blas_prebuild_query_count;
    out_info->accel_tlas_prebuild_info_ms = diagnostics.tlas_prebuild_query_ms;
    out_info->accel_tlas_prebuild_info_count = diagnostics.tlas_prebuild_query_count;
    out_info->accel_startup_prebuild_warmup_ms = diagnostics.startup_prebuild_warmup_ms;
    out_info->accel_tlas_instance_upload_ms = diagnostics.tlas_instance_upload_ms;
    out_info->accel_submit_cpu_ms = diagnostics.acceleration_submit_cpu_ms;
    out_info->accel_gpu_wait_ms = diagnostics.acceleration_gpu_wait_ms;
    out_info->accel_gpu_ms = diagnostics.acceleration_gpu_ms;
    out_info->dispatch_ms = diagnostics.dispatch_cpu_ms;
    out_info->dispatch_submit_cpu_ms = diagnostics.dispatch_submit_cpu_ms;
    out_info->dispatch_gpu_wait_ms = diagnostics.dispatch_gpu_wait_ms;
    out_info->dispatch_gpu_ms = diagnostics.dispatch_gpu_ms;
    out_info->command_slot_reuse_wait_ms = diagnostics.command_slot_reuse_wait_ms;
    out_info->readback_ms = diagnostics.readback_cpu_ms;
    out_info->scratch_growth_count = diagnostics.scratch_growth_count;
    out_info->scratch_capacity_bytes = diagnostics.scratch_capacity_bytes;
    out_info->scratch_peak_capacity_bytes = diagnostics.scratch_peak_capacity_bytes;
    out_info->acceleration_resource_allocation_count =
        diagnostics.acceleration_resource_allocation_count;
    out_info->acceleration_resource_reallocation_count =
        diagnostics.acceleration_resource_reallocation_count;
    out_info->acceleration_capacity_bytes = diagnostics.acceleration_capacity_bytes;
    out_info->acceleration_peak_capacity_bytes = diagnostics.acceleration_peak_capacity_bytes;
    out_info->scene_buffer_allocation_count = diagnostics.scene_buffer_allocation_count;
    out_info->scene_buffer_growth_count = diagnostics.scene_buffer_growth_count;
    out_info->scene_buffer_capacity_bytes = diagnostics.scene_buffer_capacity_bytes;
    out_info->scene_buffer_peak_capacity_bytes = diagnostics.scene_buffer_peak_capacity_bytes;
}

void reset_rt_renderer_accumulation(rt_renderer* renderer) {
    if (renderer != nullptr) {
        reset_rt_accumulation_state(&renderer->accumulation_state);
    }
}

bool begin_rt_renderer_accumulation(
    rt_renderer* renderer,
    const rt_accumulation_key &next_key,
    bool continuous_render)
{
    return renderer != nullptr &&
        begin_rt_accumulation(&renderer->accumulation_state, next_key, continuous_render);
}

bool track_rt_renderer_render_submission(
    rt_renderer* renderer,
    rt_submission_token submission,
    std::uint64_t accumulation_generation,
    std::uint32_t sample_index,
    bool contributes_sample,
    bool confirms_scene,
    std::uint64_t scene_revision,
    const rt_acceleration_build_summary &acceleration_summary,
    rt_rhi_error* out_error)
{
    if (renderer == nullptr || renderer->rhi == nullptr || !submission) {
        set_frame_error(
            out_error,
            rt_rhi_operation::submit_commands,
            "RT render submission tracking request is invalid");
        return false;
    }
    if (renderer->async_failed) {
        if (out_error != nullptr) {
            *out_error = renderer->async_error;
        }
        return false;
    }
    if (contributes_sample && !submit_rt_accumulation_sample(
            &renderer->accumulation_state,
            accumulation_generation,
            sample_index,
            renderer->continuous_render)) {
        const rt_rhi_error error{
            rt_rhi_operation::submit_commands,
            0,
            "RT accumulation submission sequence is invalid"};
        latch_rt_renderer_async_error(renderer, error);
        if (out_error != nullptr) {
            *out_error = renderer->async_error;
        }
        return false;
    }
    renderer->pending_render_submissions.push_back({
        submission,
        accumulation_generation,
        sample_index,
        contributes_sample,
        confirms_scene,
        scene_revision,
        acceleration_summary});
    return true;
}

bool collect_rt_renderer_render_submissions(
    rt_renderer* renderer,
    rt_rhi_error* out_error)
{
    if (renderer == nullptr || renderer->rhi == nullptr) {
        set_frame_error(
            out_error,
            rt_rhi_operation::query_submission,
            "RT render submission collection is unavailable");
        return false;
    }
    if (renderer->async_failed) {
        if (out_error != nullptr) {
            *out_error = renderer->async_error;
        }
        return false;
    }

    std::size_t completed_count = 0;
    for (const rt_pending_render_submission &pending : renderer->pending_render_submissions) {
        bool complete = false;
        rt_rhi_error query_error{};
        if (consume_rt_renderer_test_query_failure()) {
            query_error = {
                rt_rhi_operation::query_submission,
                -1,
                "Injected RT render submission query failure"};
            latch_rt_renderer_async_error(renderer, query_error);
            if (out_error != nullptr) {
                *out_error = renderer->async_error;
            }
            return false;
        }
        if (!renderer->rhi->is_complete(pending.submission, &complete, &query_error)) {
            latch_rt_renderer_async_error(renderer, query_error);
            if (out_error != nullptr) {
                *out_error = renderer->async_error;
            }
            return false;
        }
        if (!complete) {
            break;
        }
        if (pending.contributes_sample && !complete_rt_accumulation_sample(
                &renderer->accumulation_state,
                pending.accumulation_generation,
                pending.sample_index,
                renderer->continuous_render)) {
            const rt_rhi_error error{
                rt_rhi_operation::query_submission,
                0,
                "RT accumulation completion sequence is invalid"};
            latch_rt_renderer_async_error(renderer, error);
            if (out_error != nullptr) {
                *out_error = renderer->async_error;
            }
            return false;
        }
        if (pending.confirms_scene &&
            pending.scene_revision == renderer->frame_state.presentation_revision) {
            renderer->last_acceleration_revision = pending.scene_revision;
            renderer->last_acceleration_summary = pending.acceleration_summary;
        }
        ++completed_count;
    }
    if (completed_count != 0) {
        renderer->pending_render_submissions.erase(
            renderer->pending_render_submissions.begin(),
            renderer->pending_render_submissions.begin() + static_cast<std::ptrdiff_t>(completed_count));
    }
    completed_count = 0;
    for (rt_submission_token submission : renderer->pending_delivery_submissions) {
        bool complete = false;
        rt_rhi_error query_error{};
        if (consume_rt_renderer_test_delivery_query_failure()) {
            query_error = {
                rt_rhi_operation::query_submission,
                -1,
                "Injected RT delivery submission query failure"};
            latch_rt_renderer_async_error(renderer, query_error);
            if (out_error != nullptr) {
                *out_error = renderer->async_error;
            }
            return false;
        }
        if (!renderer->rhi->is_complete(submission, &complete, &query_error)) {
            latch_rt_renderer_async_error(renderer, query_error);
            if (out_error != nullptr) {
                *out_error = renderer->async_error;
            }
            return false;
        }
        if (!complete) {
            break;
        }
        ++completed_count;
    }
    if (completed_count != 0) {
        renderer->pending_delivery_submissions.erase(
            renderer->pending_delivery_submissions.begin(),
            renderer->pending_delivery_submissions.begin() + static_cast<std::ptrdiff_t>(completed_count));
    }
    return true;
}

bool track_rt_renderer_delivery_submission(
    rt_renderer* renderer,
    rt_submission_token submission,
    rt_rhi_error* out_error)
{
    if (renderer == nullptr || renderer->rhi == nullptr || !submission) {
        set_frame_error(
            out_error,
            rt_rhi_operation::submit_commands,
            "RT delivery submission tracking request is invalid");
        return false;
    }
    if (renderer->async_failed) {
        if (out_error != nullptr) {
            *out_error = renderer->async_error;
        }
        return false;
    }
    if (std::find(
            renderer->pending_delivery_submissions.begin(),
            renderer->pending_delivery_submissions.end(),
            submission) != renderer->pending_delivery_submissions.end()) {
        return true;
    }
    renderer->pending_delivery_submissions.push_back(submission);
    return true;
}

bool effective_layer_visibility(
    const layer_visibility_map* visibility,
    const std::string &path,
    bool fallback)
{
    if (visibility == nullptr || visibility->empty()) {
        return fallback;
    }
    bool found_visibility = false;
    std::size_t separator = 0;
    for (;;) {
        separator = path.find('/', separator);
        const auto found = visibility->find(path.substr(0, separator));
        if (found != visibility->end()) {
            found_visibility = true;
            if (!found->second) {
                return false;
            }
        }
        if (separator == std::string::npos) {
            break;
        }
        ++separator;
    }
    return found_visibility ? true : fallback;
}

bool resolve_layer_tlas_instance_source(
    const rt_scene_build &build,
    const rt_acceleration_build_item &item,
    std::string* out_layer,
    bool* out_fallback_visible,
    bool* out_layer_visibility_exempt)
{
    if (out_layer == nullptr || out_fallback_visible == nullptr ||
        out_layer_visibility_exempt == nullptr || item.group.chunk_count == 0 ||
        item.group.chunk_count > kRtBlasChunkSetChunkCount) {
        return false;
    }

    const auto resolve_chunk = [&](std::size_t chunk_index,
                                   const std::string** out_chunk_layer,
                                   bool* out_chunk_visible,
                                   bool* out_chunk_exempt) {
        if (out_chunk_layer == nullptr || out_chunk_visible == nullptr || out_chunk_exempt == nullptr) {
            return false;
        }
        switch (item.kind) {
        case rt_acceleration_geometry_kind::triangle:
            if (chunk_index >= build.triangle_chunks.size()) {
                return false;
            }
            *out_chunk_layer = &build.triangle_chunks[chunk_index].layer;
            *out_chunk_visible = build.triangle_chunks[chunk_index].visible;
            *out_chunk_exempt = build.triangle_chunks[chunk_index].layer_visibility_exempt;
            return true;
        case rt_acceleration_geometry_kind::point:
            if (chunk_index >= build.point_chunks.size()) {
                return false;
            }
            *out_chunk_layer = &build.point_chunks[chunk_index].layer;
            *out_chunk_visible = build.point_chunks[chunk_index].visible;
            *out_chunk_exempt = build.point_chunks[chunk_index].layer_visibility_exempt;
            return true;
        case rt_acceleration_geometry_kind::line:
            if (chunk_index >= build.line_chunks.size()) {
                return false;
            }
            *out_chunk_layer = &build.line_chunks[chunk_index].layer;
            *out_chunk_visible = build.line_chunks[chunk_index].visible;
            *out_chunk_exempt = build.line_chunks[chunk_index].layer_visibility_exempt;
            return true;
        default:
            return false;
        }
    };

    const std::string* first_layer = nullptr;
    bool first_visible = true;
    bool first_exempt = false;
    if (!resolve_chunk(
            item.group.chunk_indices[0],
            &first_layer,
            &first_visible,
            &first_exempt) ||
        first_layer == nullptr) {
        return false;
    }
    for (std::size_t geometry_index = 1; geometry_index < item.group.chunk_count; ++geometry_index) {
        const std::string* layer = nullptr;
        bool visible = true;
        bool exempt = false;
        if (!resolve_chunk(
                item.group.chunk_indices[geometry_index],
                &layer,
                &visible,
                &exempt) ||
            layer == nullptr || *layer != *first_layer || visible != first_visible || exempt != first_exempt) {
            return false;
        }
    }
    *out_layer = *first_layer;
    *out_fallback_visible = first_visible;
    *out_layer_visibility_exempt = first_exempt;
    return true;
}

bool apply_layer_visibility(
    const rt_scene_build &build,
    const rt_acceleration_build_plan &build_plan,
    const layer_visibility_map* visibility,
    rt_acceleration_command_plan* commands)
{
    if (commands == nullptr || commands->blas_commands.size() != build_plan.items.size()) {
        return false;
    }
    for (std::size_t item_index = 0; item_index < build_plan.items.size(); ++item_index) {
        const rt_acceleration_build_item &item = build_plan.items[item_index];
        std::string layer;
        bool fallback = true;
        bool visibility_exempt = false;
        if (!resolve_layer_tlas_instance_source(
                build,
                item,
                &layer,
                &fallback,
                &visibility_exempt)) {
            return false;
        }
        commands->blas_commands[item_index].visible =
            visibility_exempt ? fallback : effective_layer_visibility(visibility, layer, fallback);
    }
    return true;
}

bool sync_rt_renderer_layer_visibility(
    rt_renderer* renderer,
    const rt_renderer_frame_request &request,
    std::uint64_t presentation_revision,
    rt_renderer_frame_result* out_result,
    rt_rhi_error* out_error,
    rt_deferred_acceleration_submission* deferred_acceleration)
{
    if (renderer == nullptr || renderer->rhi == nullptr || request.build == nullptr ||
        out_result == nullptr || !renderer->frame_state.scene_valid || !renderer->tlas ||
        renderer->current_connection_serial != request.build->connection_serial) {
        set_frame_error(
            out_error,
            rt_rhi_operation::begin_commands,
            "RT layer visibility update request is invalid");
        return false;
    }

    std::vector<rt_tlas_instance_desc> instances;
    instances.reserve(renderer->layer_tlas_instances.size());
    rt_acceleration_build_summary summary{};
    for (const rt_layer_tlas_instance &source : renderer->layer_tlas_instances) {
        if (!source.instance.acceleration || source.geometry_count == 0) {
            set_frame_error(
                out_error,
                rt_rhi_operation::build_tlas,
                "RT layer visibility instance is invalid");
            return false;
        }
        rt_tlas_instance_desc instance = source.instance;
        const bool visible = source.layer_visibility_exempt
            ? source.fallback_visible
            : effective_layer_visibility(
                  request.layer_visibility,
                  source.layer,
                  source.fallback_visible);
        instance.mask = visible ? 0xff : 0x00;
        instances.push_back(instance);
        ++summary.blas_reused_count;
        if (source.kind == rt_acceleration_geometry_kind::triangle) {
            summary.blas_reused_triangle_chunk_count += source.geometry_count;
        }
    }

    rt_command_encoder encoder{};
    rt_rhi_error stage_error{rt_rhi_operation::begin_commands, 0, {}};
    if (!begin_rt_commands(
            renderer,
            rt_queue_class::graphics,
            &encoder,
            &stage_error,
            &out_result->acceleration_timing)) {
        if (out_error != nullptr) {
            *out_error = stage_error;
        }
        return false;
    }
    const auto sync_start = std::chrono::steady_clock::now();
    const rt_tlas_build_desc tlas_desc{
        renderer->tlas,
        instances.data(),
        instances.size(),
        rt_acceleration_build_prefer_fast_trace};
    stage_error = {rt_rhi_operation::build_tlas, 0, {}};
    if (!renderer->rhi->build_tlas(encoder, tlas_desc, &stage_error)) {
        discard_rt_commands(renderer, encoder);
        if (out_error != nullptr) {
            *out_error = stage_error;
        }
        return false;
    }
    summary.tlas_rebuild_count = instances.empty() ? 0 : 1;
    const double command_record_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - sync_start).count();
    out_result->blas_reused_count = summary.blas_reused_count;
    out_result->blas_reused_triangle_chunk_count = summary.blas_reused_triangle_chunk_count;
    out_result->tlas_rebuild_count = summary.tlas_rebuild_count;
    out_result->acceleration_timing.command_record_ms += command_record_ms;
    renderer->last_acceleration_cpu_ms = command_record_ms;
    if (deferred_acceleration != nullptr) {
        deferred_acceleration->encoder = encoder;
        deferred_acceleration->presentation_revision = presentation_revision;
        deferred_acceleration->acceleration_summary = summary;
        deferred_acceleration->scene_changed = false;
        return true;
    }

    rt_submission_token submission{};
    stage_error = {rt_rhi_operation::submit_commands, 0, {}};
    if (!submit_rt_commands(
            renderer,
            encoder,
            &submission,
            &out_result->acceleration_timing,
            &stage_error)) {
        discard_rt_commands(renderer, encoder);
        if (out_error != nullptr) {
            *out_error = stage_error;
        }
        return false;
    }
    renderer->submitted_acceleration_revision = presentation_revision;
    renderer->submitted_acceleration_summary = summary;
    return true;
}

bool prepare_rt_renderer_frame(
    rt_renderer* renderer,
    const rt_renderer_frame_request &request,
    rt_renderer_frame_result* out_result,
    rt_rhi_error* out_error,
    rt_deferred_acceleration_submission* deferred_acceleration)
{
    if (out_result != nullptr) {
        *out_result = {};
    }
    if (deferred_acceleration != nullptr) {
        *deferred_acceleration = {};
    }
    if (renderer != nullptr && renderer->async_failed) {
        if (out_error != nullptr) {
            *out_error = renderer->async_error;
        }
        return false;
    }
    if (renderer == nullptr || renderer->rhi == nullptr) {
        set_frame_error(out_error, rt_rhi_operation::begin_frame, "RT renderer frame operations are unavailable");
        return false;
    }
    collect_rt_resource_pools(renderer);
    if (renderer->async_failed) {
        if (out_error != nullptr) {
            *out_error = renderer->async_error;
        }
        return false;
    }
    if (request.build == nullptr || request.width <= 0 || request.height <= 0) {
        set_frame_error(out_error, rt_rhi_operation::begin_frame, "RT renderer frame request is invalid");
        return false;
    }
    if (renderer->frame_state.active) {
        set_frame_error(out_error, rt_rhi_operation::begin_frame, "RT renderer frame is already active");
        return false;
    }

    const auto preparation_start = std::chrono::steady_clock::now();
    renderer->frame_state.active = true;
    rt_renderer_frame_result result{};
    const std::uint64_t presentation_revision = request.scene_revision != 0
        ? request.scene_revision
        : request.build->revision;
    const std::uint64_t geometry_revision = request.build->revision;
    result.output_changed = request.require_output &&
        (!renderer->frame_state.output_valid ||
            renderer->frame_state.output_width != request.width ||
            renderer->frame_state.output_height != request.height);
    result.scene_changed = !renderer->frame_state.scene_valid ||
        renderer->frame_state.scene_revision != geometry_revision;
    const bool visibility_changed = !result.scene_changed &&
        renderer->frame_state.presentation_revision != presentation_revision;
    const bool connection_changed = renderer->current_connection_serial != 0 &&
        renderer->current_connection_serial != request.build->connection_serial;

    rt_acceleration_build_plan acceleration_plan{};
    rt_blas_cache_update_plan blas_cache_plan{};
    rt_scene_resource_data resources{};
    rt_acceleration_command_plan acceleration_commands{};
    rt_renderer_frame_request resolved_request = request;
    if (result.scene_changed) {
        if (connection_changed) {
            retire_blas_cache_for_connection_change(renderer);
        }
        if (!make_rt_acceleration_build_plan(*request.build, &acceleration_plan)) {
            renderer->frame_state.active = false;
            set_frame_error(out_error, rt_rhi_operation::begin_frame, "RT acceleration build plan is invalid");
            return false;
        }
        if (!make_rt_blas_cache_update_plan(acceleration_plan, renderer->blas_cache_state, &blas_cache_plan)) {
            renderer->frame_state.active = false;
            set_frame_error(out_error, rt_rhi_operation::begin_frame, "RT BLAS cache update plan is invalid");
            return false;
        }
        if (!renderer->blas_reuse_enabled) {
            for (rt_blas_cache_assignment &assignment : blas_cache_plan.assignments) {
                assignment.reuse_candidate = false;
            }
        }
        if (!make_rt_scene_resource_data(*request.build, acceleration_plan, &resources)) {
            renderer->frame_state.active = false;
            set_frame_error(out_error, rt_rhi_operation::begin_frame, "RT scene resource data is invalid");
            return false;
        }
        if (!make_rt_acceleration_command_plan(
                *request.build,
                acceleration_plan,
                blas_cache_plan,
                resources,
                &acceleration_commands)) {
            renderer->frame_state.active = false;
            set_frame_error(
                out_error,
                rt_rhi_operation::begin_frame,
                "RT acceleration command plan is invalid");
            return false;
        }
        if (!apply_layer_visibility(
                *request.build,
                acceleration_plan,
                request.layer_visibility,
                &acceleration_commands)) {
            renderer->frame_state.active = false;
            set_frame_error(out_error, rt_rhi_operation::begin_frame, "RT layer visibility is invalid");
            return false;
        }
        acceleration_plan.revision = geometry_revision;
        blas_cache_plan.revision = geometry_revision;
        resources.revision = geometry_revision;
        acceleration_commands.revision = geometry_revision;
        resolved_request.acceleration_plan = &acceleration_plan;
        resolved_request.blas_cache_plan = &blas_cache_plan;
        resolved_request.resources = &resources;
        resolved_request.acceleration_commands = &acceleration_commands;
    }

    rt_rhi_error stage_error{rt_rhi_operation::create_resource, 0, {}};
    if (!ensure_viewer_constant_buffer(
            renderer,
            &result.viewer_resources_changed,
            &stage_error)) {
        renderer->frame_state.active = false;
        if (out_error != nullptr) {
            *out_error = stage_error;
        }
        return false;
    }
    if (result.output_changed) {
        stage_error = {rt_rhi_operation::create_resource, 0, {}};
        if (!create_output_resources(
                renderer,
                request.width,
                request.height,
                &stage_error)) {
            renderer->frame_state.active = false;
            if (out_error != nullptr) {
                *out_error = stage_error;
            }
            return false;
        }
    }
    if (request.require_pick || rt_scene_has_renderable_primitives(*request.build)) {
        stage_error = {rt_rhi_operation::create_resource, 0, {}};
        if (!ensure_pick_buffers(
                renderer,
                &result.pick_resources_changed,
                &stage_error)) {
            renderer->frame_state.active = false;
            if (out_error != nullptr) {
                *out_error = stage_error;
            }
            return false;
        }
    }
    if (result.scene_changed) {
        stage_error = {rt_rhi_operation::upload_scene_buffers, 0, {}};
        if (resolved_request.resources == nullptr ||
            resolved_request.acceleration_plan == nullptr ||
            !upload_scene_buffers(
                renderer,
                *resolved_request.resources,
                *resolved_request.acceleration_plan,
                &stage_error)) {
            renderer->frame_state.active = false;
            if (out_error != nullptr) {
                *out_error = stage_error;
            }
            return false;
        }
    }

    result.pre_acceleration_prepare_cpu_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - preparation_start).count();

    if (result.scene_changed &&
        !sync_rt_renderer_acceleration(
            renderer,
            resolved_request,
            &result,
            out_error,
            deferred_acceleration)) {
        renderer->frame_state.active = false;
        return false;
    }
    if (visibility_changed &&
        !sync_rt_renderer_layer_visibility(
            renderer,
            resolved_request,
            presentation_revision,
            &result,
            out_error,
            deferred_acceleration)) {
        renderer->frame_state.active = false;
        return false;
    }
    result.acceleration_changed = result.scene_changed || visibility_changed;
    if (result.scene_changed &&
        (deferred_acceleration == nullptr || !deferred_acceleration->encoder)) {
        renderer->blas_cache_state = std::move(blas_cache_plan.next_state);
    }

    const auto post_acceleration_prepare_start = std::chrono::steady_clock::now();

    if (!update_rt_renderer_bindings(renderer, resolved_request, result, out_error)) {
        discard_deferred_acceleration_submission(renderer, deferred_acceleration);
        renderer->frame_state.active = false;
        return false;
    }

    if (rt_scene_has_renderable_primitives(*request.build)) {
        stage_error = {rt_rhi_operation::create_shader_module, 0, {}};
        if (!ensure_viewer_rt_shader_modules(renderer, &stage_error)) {
            discard_deferred_acceleration_submission(renderer, deferred_acceleration);
            renderer->frame_state.active = false;
            if (out_error != nullptr) {
                *out_error = stage_error;
            }
            return false;
        }
        const viewer_rt_pipeline_desc pipeline = make_viewer_rt_pipeline_desc(
            renderer->shader_package.pipeline_model,
            renderer->shader_entry_modules);
        const rt_pipeline_desc pipeline_desc = pipeline.pipeline();
        stage_error = {rt_rhi_operation::prepare_pipeline, 0, {}};
        if (!prepare_rt_renderer_pipeline(
                renderer,
                pipeline_desc,
                &result,
                &stage_error)) {
            discard_deferred_acceleration_submission(renderer, deferred_acceleration);
            renderer->frame_state.active = false;
            if (out_error != nullptr) {
                *out_error = stage_error;
            }
            return false;
        }
    }

    if (deferred_acceleration == nullptr || !deferred_acceleration->encoder) {
        renderer->frame_state.scene_revision = geometry_revision;
        renderer->frame_state.presentation_revision = presentation_revision;
        renderer->frame_state.scene_valid = true;
        if (result.scene_changed) {
            renderer->current_connection_serial = request.build->connection_serial;
        }
    }
    if (request.require_output) {
        renderer->frame_state.output_width = request.width;
        renderer->frame_state.output_height = request.height;
        renderer->frame_state.output_valid = true;
    }
    ++renderer->frame_state.serial;
    renderer->frame_state.active = false;
    result.post_acceleration_prepare_cpu_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - post_acceleration_prepare_start).count();
    if (out_result != nullptr) {
        *out_result = result;
    }
    if (out_error != nullptr) {
        *out_error = {rt_rhi_operation::end_frame, 0, {}};
    }
    return true;
}

} // namespace rtvdb::viewer_backend
