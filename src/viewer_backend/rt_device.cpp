#include "viewer_backend/rt_device.h"

#include "viewer_backend/rt_backend_common.h"

#if defined(RTVDB_ENABLE_D3D12_DXR)
#include "rtvdb_d3d12_raytracing_dxil.h"
#endif
#if defined(RTVDB_ENABLE_VULKAN_RT)
#include "rtvdb_vulkan_rt_line_rchit_spv.h"
#include "rtvdb_vulkan_rt_line_rint_spv.h"
#include "rtvdb_vulkan_rt_pick_rgen_spv.h"
#include "rtvdb_vulkan_rt_point_rchit_spv.h"
#include "rtvdb_vulkan_rt_point_rint_spv.h"
#include "rtvdb_vulkan_rt_rchit_spv.h"
#include "rtvdb_vulkan_rt_rgen_spv.h"
#include "rtvdb_vulkan_rt_rmiss_spv.h"
#endif

#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>

namespace rtvdb::viewer_backend {

namespace {

constexpr std::size_t kRtPickResultBufferBytes = sizeof(rt_pick_gpu_result);

void set_missing_lifecycle_error(
    rt_device_error* out_error,
    rt_device_operation operation,
    const char* detail)
{
    if (out_error == nullptr) {
        return;
    }
    out_error->operation = operation;
    out_error->native_code = 0;
    out_error->detail = detail != nullptr ? detail : "RT device lifecycle operation is unavailable";
}

void set_frame_error(
    rt_device_error* out_error,
    rt_device_operation operation,
    const char* detail)
{
    if (out_error == nullptr) {
        return;
    }
    *out_error = {operation, 0, detail != nullptr ? detail : "RT device frame operation failed"};
}

bool environment_flag_enabled(const char* name) {
    if (name == nullptr || *name == '\0') {
        return false;
    }
    const char* const value = std::getenv(name);
    return value != nullptr &&
        (value[0] == '1' || value[0] == 't' || value[0] == 'T' || value[0] == 'y' || value[0] == 'Y');
}

void destroy_viewer_rt_shader_modules(rt_device* device) {
    if (device == nullptr || device->api == nullptr) {
        return;
    }
    for (rt_shader_module_handle module : device->shader_modules) {
        device->api->destroy_shader_module(module);
    }
    device->shader_modules.clear();
}

bool create_viewer_rt_shader_module(
    rt_device* device,
    const rt_shader_module_desc &desc,
    rt_device_error* out_error)
{
    rt_shader_module_handle module{};
    if (!device->api->create_shader_module(desc, &module, out_error) || !module) {
        return false;
    }
    device->shader_modules.push_back(module);
    return true;
}

bool ensure_viewer_rt_shader_modules(rt_device* device, rt_device_error* out_error) {
    if (device == nullptr || device->api == nullptr) {
        set_frame_error(
            out_error,
            rt_device_operation::create_shader_module,
            "RT shader module creation is unavailable");
        return false;
    }
    if (!device->shader_modules.empty()) {
        return true;
    }
    bool created = false;
    if (device->capabilities.shader_binary_format == rt_shader_binary_format::dxil_library) {
#if defined(RTVDB_ENABLE_D3D12_DXR)
        created = create_viewer_rt_shader_module(
            device,
            {
                rt_shader_binary_format::dxil_library,
                kD3d12RaytracingDxil,
                kD3d12RaytracingDxilSize},
            out_error);
#endif
    } else if (device->capabilities.shader_binary_format == rt_shader_binary_format::spirv) {
#if defined(RTVDB_ENABLE_VULKAN_RT)
        const std::array<rt_shader_module_desc, 8> modules = {{
            {rt_shader_binary_format::spirv, kVulkanRtRaygenSpirv, kVulkanRtRaygenSpirvSize},
            {
                rt_shader_binary_format::spirv,
                kVulkanRtPickRaygenSpirv,
                kVulkanRtPickRaygenSpirvSize},
            {rt_shader_binary_format::spirv, kVulkanRtMissSpirv, kVulkanRtMissSpirvSize},
            {
                rt_shader_binary_format::spirv,
                kVulkanRtClosestHitSpirv,
                kVulkanRtClosestHitSpirvSize},
            {
                rt_shader_binary_format::spirv,
                kVulkanRtPointClosestHitSpirv,
                kVulkanRtPointClosestHitSpirvSize},
            {
                rt_shader_binary_format::spirv,
                kVulkanRtPointIntersectionSpirv,
                kVulkanRtPointIntersectionSpirvSize},
            {
                rt_shader_binary_format::spirv,
                kVulkanRtLineClosestHitSpirv,
                kVulkanRtLineClosestHitSpirvSize},
            {
                rt_shader_binary_format::spirv,
                kVulkanRtLineIntersectionSpirv,
                kVulkanRtLineIntersectionSpirvSize},
        }};
        created = true;
        for (const rt_shader_module_desc &module : modules) {
            if (!create_viewer_rt_shader_module(device, module, out_error)) {
                created = false;
                break;
            }
        }
#endif
    }
    if (!created) {
        destroy_viewer_rt_shader_modules(device);
        if (out_error != nullptr && out_error->detail.empty()) {
            set_frame_error(
                out_error,
                rt_device_operation::create_shader_module,
                "RT shader binary format is unavailable in this build");
        }
        return false;
    }
    return true;
}

bool rt_shader_index_is_valid(const rt_pipeline_desc &desc, std::uint32_t index, rt_shader_stage stage) {
    return index < desc.shader_count &&
        desc.shaders[index].module &&
        desc.shaders[index].stage == stage &&
        desc.shaders[index].entry_point != nullptr &&
        desc.shaders[index].entry_point[0] != '\0';
}

bool rt_pipeline_desc_is_valid(const rt_pipeline_desc &desc) {
    if (desc.bindings == nullptr || desc.binding_count == 0 ||
        desc.shaders == nullptr || desc.shader_count == 0 ||
        desc.groups == nullptr || desc.group_count == 0 ||
        desc.max_recursion_depth == 0 || desc.max_payload_size == 0 ||
        desc.max_attribute_size == 0) {
        return false;
    }
    for (std::size_t index = 0; index < desc.binding_count; ++index) {
        const rt_binding_layout_desc &binding = desc.bindings[index];
        if (binding.count == 0) {
            return false;
        }
        for (std::size_t previous = 0; previous < index; ++previous) {
            if (desc.bindings[previous].location == binding.location) {
                return false;
            }
        }
    }
    for (std::size_t index = 0; index < desc.shader_count; ++index) {
        if (!desc.shaders[index].module ||
            desc.shaders[index].entry_point == nullptr ||
            desc.shaders[index].entry_point[0] == '\0') {
            return false;
        }
    }
    for (std::size_t index = 0; index < desc.group_count; ++index) {
        const rt_shader_group_desc &group = desc.groups[index];
        if (group.type == rt_shader_group_type::general) {
            if (group.general_shader >= desc.shader_count ||
                (desc.shaders[group.general_shader].stage != rt_shader_stage::ray_generation &&
                    desc.shaders[group.general_shader].stage != rt_shader_stage::miss &&
                    desc.shaders[group.general_shader].stage != rt_shader_stage::callable)) {
                return false;
            }
        } else if (group.export_name == nullptr || group.export_name[0] == '\0' ||
            !rt_shader_index_is_valid(desc, group.closest_hit_shader, rt_shader_stage::closest_hit)) {
            return false;
        } else if (group.any_hit_shader != kRtUnusedShaderIndex &&
            !rt_shader_index_is_valid(desc, group.any_hit_shader, rt_shader_stage::any_hit)) {
            return false;
        } else if (group.type == rt_shader_group_type::triangles_hit_group &&
            group.intersection_shader != kRtUnusedShaderIndex) {
            return false;
        } else if (group.type == rt_shader_group_type::procedural_hit_group &&
            !rt_shader_index_is_valid(desc, group.intersection_shader, rt_shader_stage::intersection)) {
            return false;
        }
    }
    return true;
}

bool rt_shader_table_section_is_valid(
    const rt_pipeline_desc &pipeline,
    const rt_shader_table_section_desc &section,
    rt_shader_stage expected_general_stage,
    bool expect_hit_group)
{
    if (section.group_count != 0 && section.groups == nullptr) {
        return false;
    }
    for (std::size_t index = 0; index < section.group_count; ++index) {
        const std::uint32_t group_index = section.groups[index];
        if (group_index >= pipeline.group_count) {
            return false;
        }
        const rt_shader_group_desc &group = pipeline.groups[group_index];
        if (expect_hit_group) {
            if (group.type == rt_shader_group_type::general) {
                return false;
            }
        } else if (group.type != rt_shader_group_type::general ||
            group.general_shader >= pipeline.shader_count ||
            pipeline.shaders[group.general_shader].stage != expected_general_stage) {
            return false;
        }
    }
    return true;
}

void destroy_scene_buffers(rt_device* device, rt_scene_buffer_resources* resources) {
    if (device == nullptr || device->api == nullptr || resources == nullptr) {
        return;
    }
    device->api->destroy_buffer(resources->positions);
    device->api->destroy_buffer(resources->indices);
    device->api->destroy_buffer(resources->triangle_colors);
    device->api->destroy_buffer(resources->instance_metadata);
    device->api->destroy_buffer(resources->points);
    device->api->destroy_buffer(resources->lines);
    device->api->destroy_buffer(resources->point_aabbs);
    device->api->destroy_buffer(resources->line_aabbs);
    for (const rt_scene_buffer_upload &upload : resources->uploads) {
        device->api->destroy_buffer(upload.staging);
    }
    *resources = {};
}

void destroy_scene_upload_buffers(rt_device* device, rt_scene_buffer_resources* resources) {
    if (device == nullptr || device->api == nullptr || resources == nullptr) {
        return;
    }
    for (const rt_scene_buffer_upload &upload : resources->uploads) {
        device->api->destroy_buffer(upload.staging);
    }
    resources->uploads.clear();
}

void destroy_output_resources(rt_device* device) {
    if (device == nullptr || device->api == nullptr) {
        return;
    }
    device->api->destroy_texture(device->output_texture);
    device->api->destroy_texture(device->accumulation_texture);
    device->api->destroy_buffer(device->output_readback_buffer);
    device->output_texture = {};
    device->accumulation_texture = {};
    device->output_readback_buffer = {};
    device->output_readback_footprint = {};
    device->output_readback_submission = {};
    device->output_readback_width = 0;
    device->output_readback_height = 0;
    device->output_readback_pending = false;
}

void destroy_pick_buffers(rt_device* device) {
    if (device == nullptr || device->api == nullptr) {
        return;
    }
    device->api->destroy_buffer(device->pick_output_buffer);
    for (rt_pick_slot &slot : device->pick_slots) {
        device->api->destroy_buffer(slot.readback_buffer);
    }
    device->pick_output_buffer = {};
    device->pick_slots = {};
}

void destroy_viewer_resources(rt_device* device) {
    if (device == nullptr || device->api == nullptr) {
        return;
    }
    device->api->destroy_buffer(device->viewer_constant_buffer);
    device->viewer_constant_buffer = {};
}

bool ensure_viewer_constant_buffer(
    rt_device* device,
    bool* out_changed,
    rt_device_error* out_error)
{
    if (out_changed != nullptr) {
        *out_changed = false;
    }
    if (device == nullptr || device->api == nullptr) {
        return false;
    }
    if (device->viewer_constant_buffer) {
        return true;
    }

    rt_buffer_handle constants{};
    if (!device->api->create_buffer(
            {
                kRtViewerConstantBufferBytes,
                rt_buffer_usage_uniform,
                rt_memory_domain::upload},
            &constants,
            out_error)) {
        return false;
    }
    destroy_viewer_resources(device);
    device->viewer_constant_buffer = constants;
    if (out_changed != nullptr) {
        *out_changed = true;
    }
    return true;
}

bool ensure_pick_buffers(
    rt_device* device,
    bool* out_changed,
    rt_device_error* out_error)
{
    if (out_changed != nullptr) {
        *out_changed = false;
    }
    if (device == nullptr || device->api == nullptr) {
        return false;
    }
    bool has_readback_buffers = true;
    for (const rt_pick_slot &slot : device->pick_slots) {
        has_readback_buffers = has_readback_buffers && static_cast<bool>(slot.readback_buffer);
    }
    if (device->pick_output_buffer && has_readback_buffers) {
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
    if (!device->api->create_buffer(output_desc, &output, out_error)) {
        device->api->destroy_buffer(output);
        return false;
    }
    for (rt_buffer_handle &readback : readbacks) {
        if (device->api->create_buffer(readback_desc, &readback, out_error)) {
            continue;
        }
        device->api->destroy_buffer(output);
        for (rt_buffer_handle allocated_readback : readbacks) {
            device->api->destroy_buffer(allocated_readback);
        }
        return false;
    }
    destroy_pick_buffers(device);
    device->pick_output_buffer = output;
    for (std::size_t slot_index = 0; slot_index < kRtCommandSlotCount; ++slot_index) {
        device->pick_slots[slot_index].readback_buffer = readbacks[slot_index];
    }
    if (out_changed != nullptr) {
        *out_changed = true;
    }
    return true;
}

bool create_output_resources(
    rt_device* device,
    int width,
    int height,
    rt_device_error* out_error)
{
    if (device == nullptr || device->api == nullptr || width <= 0 || height <= 0) {
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
    if (!device->api->create_texture(
            {
                static_cast<std::uint32_t>(width),
                static_cast<std::uint32_t>(height),
                device->capabilities.output_format,
                output_usage},
            &output,
            out_error) ||
        !device->api->create_texture(
            {
                static_cast<std::uint32_t>(width),
                static_cast<std::uint32_t>(height),
                device->capabilities.accumulation_format,
                rt_texture_usage_shader_write},
            &accumulation,
            out_error)) {
        device->api->destroy_texture(output);
        device->api->destroy_texture(accumulation);
        return false;
    }
    rt_texture_copy_footprint footprint{};
    if (!device->api->get_texture_copy_footprint(
            output,
            &footprint,
            out_error) ||
        footprint.total_size == 0 ||
        !device->api->create_buffer(
            {
                footprint.total_size,
                rt_buffer_usage_copy_destination,
                rt_memory_domain::readback},
            &readback,
            out_error)) {
        device->api->destroy_texture(output);
        device->api->destroy_texture(accumulation);
        device->api->destroy_buffer(readback);
        return false;
    }
    destroy_output_resources(device);
    device->output_texture = output;
    device->accumulation_texture = accumulation;
    device->output_readback_buffer = readback;
    device->output_readback_footprint = footprint;
    return true;
}

bool create_uploaded_buffer(
    rt_device* device,
    const rt_buffer_desc &desc,
    const void* data,
    rt_buffer_handle* out_destination,
    rt_buffer_handle* out_staging,
    rt_device_error* out_error)
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
    if (!device->api->create_buffer(destination_desc, out_destination, out_error)) {
        return false;
    }
    if (desc.size == 0) {
        return true;
    }
    if (!device->api->create_buffer(staging_desc, out_staging, out_error) ||
        !device->api->upload_buffer(*out_staging, 0, data, desc.size, out_error)) {
        device->api->destroy_buffer(*out_destination);
        device->api->destroy_buffer(*out_staging);
        *out_destination = {};
        *out_staging = {};
        return false;
    }
    return true;
}

bool create_scene_uploaded_buffer(
    rt_device* device,
    rt_scene_buffer_resources* resources,
    const char* name,
    const rt_buffer_desc &desc,
    const void* data,
    rt_resource_usage destination_usage,
    rt_buffer_handle* out_destination,
    rt_device_error* out_error)
{
    if (resources == nullptr || out_destination == nullptr) {
        return false;
    }
    rt_buffer_handle staging{};
    if (!create_uploaded_buffer(
            device,
            desc,
            data,
            out_destination,
            &staging,
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
    if (staging && desc.size != 0) {
        resources->uploads.push_back({staging, *out_destination, desc.size, destination_usage});
    }
    return true;
}

bool upload_scene_buffers(
    rt_device* device,
    const rt_scene_resource_data &resources,
    const rt_acceleration_build_plan &build_plan,
    rt_device_error* out_error)
{
    if (device == nullptr || device->api == nullptr) {
        set_frame_error(out_error, rt_device_operation::upload_scene_buffers, "RT scene buffer API is unavailable");
        return false;
    }
    if (build_plan.revision != resources.revision) {
        set_frame_error(
            out_error,
            rt_device_operation::upload_scene_buffers,
            "RT scene resource and acceleration build revisions do not match");
        return false;
    }

    rt_scene_buffer_resources next{};
    const std::uint32_t geometry_usage = rt_buffer_usage_shader_read |
        rt_buffer_usage_acceleration_build_input |
        rt_buffer_usage_device_address;
    const std::uint32_t acceleration_input_usage =
        rt_buffer_usage_acceleration_build_input | rt_buffer_usage_device_address;
    const std::uint32_t shader_read_usage = rt_buffer_usage_shader_read;
    if (!create_scene_uploaded_buffer(
            device,
            &next,
            "positions",
            {resources.positions.size() * sizeof(rt_scene_gpu_position), geometry_usage},
            resources.positions.data(),
            rt_resource_usage::acceleration_build_input,
            &next.positions,
            out_error) ||
        !create_scene_uploaded_buffer(
            device,
            &next,
            "indices",
            {resources.indices.size() * sizeof(std::uint32_t), geometry_usage},
            resources.indices.data(),
            rt_resource_usage::acceleration_build_input,
            &next.indices,
            out_error) ||
        !create_scene_uploaded_buffer(
            device,
            &next,
            "triangle_colors",
            {resources.triangle_colors.size() * sizeof(rtvdb::rgba), shader_read_usage},
            resources.triangle_colors.data(),
            rt_resource_usage::shader_read,
            &next.triangle_colors,
            out_error) ||
        !create_scene_uploaded_buffer(
            device,
            &next,
            "instance_metadata",
            {
                resources.instance_geometry.size() * sizeof(rt_scene_geometry_metadata),
                shader_read_usage},
            resources.instance_geometry.data(),
            rt_resource_usage::shader_read,
            &next.instance_metadata,
            out_error) ||
        !create_scene_uploaded_buffer(
            device,
            &next,
            "points",
            {resources.points.size() * sizeof(rt_scene_gpu_point), shader_read_usage},
            resources.points.data(),
            rt_resource_usage::shader_read,
            &next.points,
            out_error) ||
        !create_scene_uploaded_buffer(
            device,
            &next,
            "lines",
            {resources.lines.size() * sizeof(rt_scene_gpu_line), shader_read_usage},
            resources.lines.data(),
            rt_resource_usage::shader_read,
            &next.lines,
            out_error)) {
        destroy_scene_buffers(device, &next);
        return false;
    }

    const rt_scene_buffer_resources &current = device->scene_buffers;
    const bool reuse_point_aabbs =
        current.point_aabbs &&
        current.point_geometry_fingerprint == build_plan.point_geometry_fingerprint &&
        current.point_aabb_count == resources.point_aabbs.size();
    const bool reuse_line_aabbs =
        current.line_aabbs &&
        current.line_geometry_fingerprint == build_plan.line_geometry_fingerprint &&
        current.line_aabb_count == resources.line_aabbs.size();
    const auto aabb_upload_start = std::chrono::steady_clock::now();
    if ((!reuse_point_aabbs && !resources.point_aabbs.empty() &&
            !create_scene_uploaded_buffer(
                device,
                &next,
                "point_aabbs",
                {resources.point_aabbs.size() * sizeof(rt_scene_gpu_aabb), acceleration_input_usage},
                resources.point_aabbs.data(),
                rt_resource_usage::acceleration_build_input,
                &next.point_aabbs,
                out_error)) ||
        (!reuse_line_aabbs && !resources.line_aabbs.empty() &&
            !create_scene_uploaded_buffer(
                device,
                &next,
                "line_aabbs",
                {resources.line_aabbs.size() * sizeof(rt_scene_gpu_aabb), acceleration_input_usage},
                resources.line_aabbs.data(),
                rt_resource_usage::acceleration_build_input,
                &next.line_aabbs,
                out_error))) {
        destroy_scene_buffers(device, &next);
        return false;
    }
    next.procedural_aabb_upload_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - aabb_upload_start).count();
    if (reuse_point_aabbs) {
        next.point_aabbs = current.point_aabbs;
        device->scene_buffers.point_aabbs = {};
    }
    if (reuse_line_aabbs) {
        next.line_aabbs = current.line_aabbs;
        device->scene_buffers.line_aabbs = {};
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
    destroy_scene_buffers(device, &device->scene_buffers);
    device->scene_buffers = next;
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
    rt_device* device,
    rt_command_encoder encoder,
    const rt_scene_buffer_resources &resources,
    rt_device_error* out_error)
{
    for (const rt_scene_buffer_upload &upload : resources.uploads) {
        if (!upload.staging || !upload.destination || upload.size == 0 ||
            !transition_rt_buffer(
                device,
                encoder,
                upload.destination,
                rt_resource_usage::copy_destination,
                out_error) ||
            !record_rt_buffer_copy(
                device,
                encoder,
                upload.staging,
                upload.destination,
                {0, 0, upload.size},
                out_error) ||
            !transition_rt_buffer(
                device,
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
    rt_device* device,
    rt_command_encoder encoder,
    const rt_scene_buffer_resources &resources,
    rt_device_error* out_error)
{
    return (!resources.positions || transition_rt_buffer(
                device,
                encoder,
                resources.positions,
                rt_resource_usage::shader_read,
                out_error)) &&
        (!resources.indices || transition_rt_buffer(
                device,
                encoder,
                resources.indices,
                rt_resource_usage::shader_read,
                out_error));
}

void destroy_rt_blas_cache(rt_device* device, rt_blas_cache_state* state) {
    if (device == nullptr || device->api == nullptr || state == nullptr) {
        return;
    }
    for (std::vector<rt_blas_cache_slot> &pool : state->pools) {
        for (rt_blas_cache_slot &slot : pool) {
            device->api->destroy_blas(slot.acceleration);
            slot.acceleration = {};
        }
    }
    *state = {};
}

bool sync_rt_device_acceleration(
    rt_device* device,
    const rt_device_frame_request &request,
    rt_device_frame_result* out_result,
    rt_device_error* out_error,
    rt_deferred_acceleration_submission* deferred_acceleration)
{
    if (device == nullptr || device->api == nullptr || request.acceleration_plan == nullptr ||
        request.blas_cache_plan == nullptr || request.resources == nullptr ||
        request.acceleration_commands == nullptr || out_result == nullptr) {
        set_frame_error(
            out_error,
            rt_device_operation::begin_commands,
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
            rt_device_operation::begin_commands,
            "RT acceleration build contracts do not match");
        return false;
    }

    rt_command_encoder encoder{};
    rt_device_error stage_error{rt_device_operation::begin_commands, 0, {}};
    if (!begin_rt_commands(
            device,
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
    if (!device->tlas) {
        stage_error = {rt_device_operation::build_tlas, 0, {}};
        if (!device->api->create_tlas(&device->tlas, &stage_error) ||
            !device->tlas) {
            discard_rt_commands(device, encoder);
            destroy_scene_upload_buffers(device, &device->scene_buffers);
            if (out_error != nullptr) {
                *out_error = stage_error;
            }
            return false;
        }
    }
    const auto fail_build = [&]() {
        discard_rt_commands(device, encoder);
        destroy_scene_upload_buffers(device, &device->scene_buffers);
        for (rt_blas_handle acceleration : created_accelerations) {
            device->api->destroy_blas(acceleration);
        }
        if (out_error != nullptr) {
            *out_error = stage_error;
        }
        return false;
    };

    if (!record_scene_buffer_uploads(device, encoder, device->scene_buffers, &stage_error)) {
        return fail_build();
    }

    rt_acceleration_build_summary summary{};
    device->last_point_blas_prebuild_info_ms = 0.0;
    device->last_point_blas_prebuild_info_count = 0;
    device->last_line_blas_prebuild_info_ms = 0.0;
    device->last_line_blas_prebuild_info_count = 0;
    std::vector<rt_tlas_instance_desc> tlas_instances;
    tlas_instances.reserve(commands.blas_commands.size());
    for (std::size_t command_index = 0;
        command_index < commands.blas_commands.size();
        ++command_index) {
        const rt_blas_build_command &command = commands.blas_commands[command_index];
        const rt_blas_cache_assignment &assignment = cache_plan.assignments[command_index];
        const std::size_t pool_index = static_cast<std::size_t>(command.kind);
        if (pool_index >= cache_plan.next_state.pools.size() ||
            assignment.cache_index >= cache_plan.next_state.pools[pool_index].size()) {
            stage_error = {
                rt_device_operation::build_blas,
                0,
                "RT BLAS cache assignment is invalid"};
            return fail_build();
        }
        rt_blas_cache_slot &slot =
            cache_plan.next_state.pools[pool_index][assignment.cache_index];
        const bool reuse = assignment.reuse_candidate && slot.acceleration;
        rt_blas_build_command resolved_command = command;
        resolved_command.destination = slot.acceleration;
        if (!resolve_rt_blas_geometry_resources(device->scene_buffers, &resolved_command)) {
            stage_error = {
                rt_device_operation::build_blas,
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
            stage_error = {rt_device_operation::build_blas, 0, {}};
            if (!device->api->create_blas(&acceleration, &stage_error) ||
                !acceleration) {
                return fail_build();
            }
            created_accelerations.push_back(acceleration);
            resolved_command.destination = acceleration;
            const rt_blas_build_desc build_desc{
                acceleration,
                resolved_command.geometries.data(),
                resolved_command.geometry_count};
            rt_blas_build_result blas_result{};
            if (!device->api->build_blas(
                    encoder,
                    build_desc,
                    &blas_result,
                    &stage_error) ||
                !(blas_result.acceleration == acceleration)) {
                return fail_build();
            }
            slot.acceleration = acceleration;
            if (command.kind == rt_acceleration_geometry_kind::point) {
                device->last_point_blas_prebuild_info_ms +=
                    blas_result.prebuild_info_ms;
                ++device->last_point_blas_prebuild_info_count;
            } else if (command.kind == rt_acceleration_geometry_kind::line) {
                device->last_line_blas_prebuild_info_ms +=
                    blas_result.prebuild_info_ms;
                ++device->last_line_blas_prebuild_info_count;
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
    }

    const rt_tlas_build_desc tlas_desc{
        device->tlas,
        tlas_instances.data(),
        tlas_instances.size()};
    stage_error = {rt_device_operation::build_tlas, 0, {}};
    if (!device->api->build_tlas(encoder, tlas_desc, &stage_error)) {
        return fail_build();
    }
    if (!transition_triangle_geometry_to_shader_read(
            device,
            encoder,
            device->scene_buffers,
            &stage_error)) {
        return fail_build();
    }
    summary.tlas_rebuild_count = tlas_instances.empty() ? 0 : 1;

    std::vector<rt_blas_handle> retired_accelerations;
    for (const std::vector<rt_blas_cache_slot> &pool : device->blas_cache_state.pools) {
        for (const rt_blas_cache_slot &slot : pool) {
            if (slot.acceleration &&
                !rt_blas_cache_contains(cache_plan.next_state, slot.acceleration)) {
                retired_accelerations.push_back(slot.acceleration);
            }
        }
    }
    out_result->blas_reused_count = summary.blas_reused_count;
    out_result->blas_rebuilt_count = summary.blas_rebuilt_count;
    out_result->blas_reused_triangle_chunk_count = summary.blas_reused_triangle_chunk_count;
    out_result->blas_rebuilt_triangle_chunk_count = summary.blas_rebuilt_triangle_chunk_count;
    out_result->tlas_rebuild_count = summary.tlas_rebuild_count;
    device->last_acceleration_revision = resources.revision;
    device->last_acceleration_summary = summary;
    const double command_record_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - sync_start).count();
    out_result->acceleration_timing.command_record_ms += command_record_ms;
    device->last_acceleration_cpu_ms = command_record_ms;
    if (deferred_acceleration != nullptr) {
        deferred_acceleration->encoder = encoder;
        deferred_acceleration->next_blas_cache_state = std::move(cache_plan.next_state);
        deferred_acceleration->created_accelerations = std::move(created_accelerations);
        deferred_acceleration->retired_accelerations = std::move(retired_accelerations);
        deferred_acceleration->scene_revision = resources.revision;
        return true;
    }

    rt_submission_token submission{};
    stage_error = {rt_device_operation::submit_commands, 0, {}};
    if (!submit_rt_commands(
            device,
            encoder,
            &submission,
            &out_result->acceleration_timing,
            &stage_error)) {
        return fail_build();
    }
    for (rt_blas_handle acceleration : retired_accelerations) {
        device->api->destroy_blas(acceleration);
    }
    destroy_scene_upload_buffers(device, &device->scene_buffers);
    return true;
}

void commit_deferred_acceleration_submission(
    rt_device* device,
    rt_deferred_acceleration_submission* deferred_acceleration)
{
    if (device == nullptr || device->api == nullptr || deferred_acceleration == nullptr ||
        !deferred_acceleration->encoder) {
        return;
    }
    device->blas_cache_state = std::move(deferred_acceleration->next_blas_cache_state);
    for (rt_blas_handle acceleration : deferred_acceleration->retired_accelerations) {
        device->api->destroy_blas(acceleration);
    }
    destroy_scene_upload_buffers(device, &device->scene_buffers);
    device->frame_state.scene_revision = deferred_acceleration->scene_revision;
    device->frame_state.scene_valid = true;
    *deferred_acceleration = {};
}

void discard_deferred_acceleration_submission(
    rt_device* device,
    rt_deferred_acceleration_submission* deferred_acceleration)
{
    if (device == nullptr || device->api == nullptr || deferred_acceleration == nullptr ||
        !deferred_acceleration->encoder) {
        return;
    }
    discard_rt_commands(device, deferred_acceleration->encoder);
    for (rt_blas_handle acceleration : deferred_acceleration->created_accelerations) {
        device->api->destroy_blas(acceleration);
    }
    destroy_scene_upload_buffers(device, &device->scene_buffers);
    *deferred_acceleration = {};
}

bool update_rt_device_bindings(
    rt_device* device,
    const rt_device_frame_request &request,
    const rt_device_frame_result &changes,
    rt_device_error* out_error)
{
    if (!rt_scene_has_renderable_primitives(*request.build) ||
        (!changes.output_changed &&
            !changes.viewer_resources_changed &&
            !changes.pick_resources_changed &&
            !changes.scene_changed &&
            !changes.acceleration_changed)) {
        return true;
    }
    const rt_scene_buffer_resources &resources = device->scene_buffers;
    const std::array<rt_binding_write, kViewerRtBindingCount> writes{{
        {{0, viewer_rt_binding_index(viewer_rt_binding::scene)},
            rt_descriptor_type::acceleration_structure, {}, 1, 0},
        {{0, viewer_rt_binding_index(viewer_rt_binding::output)},
            rt_descriptor_type::storage_texture, {}, 1, 0, device->output_texture},
        {{0, viewer_rt_binding_index(viewer_rt_binding::accumulation)},
            rt_descriptor_type::storage_texture, {}, 1, 0, device->accumulation_texture},
        {{0, viewer_rt_binding_index(viewer_rt_binding::pick_output)},
            rt_descriptor_type::storage_buffer, device->pick_output_buffer, 1, kRtPickResultBufferBytes},
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
            device->viewer_constant_buffer,
            kRtCommandSlotCount,
            kRtViewerConstantSlotStride},
    }};
    const rt_binding_update_request binding_request{writes.data(), writes.size()};
    rt_device_error stage_error{rt_device_operation::update_bindings, 0, {}};
    if (!device->api->update_bindings(binding_request, &stage_error)) {
        if (out_error != nullptr) {
            *out_error = stage_error;
        }
        return false;
    }
    return true;
}
} // namespace

bool initialize_rt_device(
    rt_device* device,
    const backend_config &config,
    rt_device_error* out_error)
{
    if (device == nullptr || device->api == nullptr) {
        set_missing_lifecycle_error(out_error, rt_device_operation::initialize, "RT device initialize is unavailable");
        return false;
    }
    if (config.capture_width <= 0 || config.capture_height <= 0) {
        if (out_error != nullptr) {
            *out_error = {
                rt_device_operation::initialize,
                0,
                "RT device initial output size must be positive"};
        }
        return false;
    }
    device->capabilities = {};
    device->frame_state = {};
    device->scene_buffers = {};
    device->output_texture = {};
    device->accumulation_texture = {};
    device->output_readback_buffer = {};
    device->output_readback_footprint = {};
    device->output_readback_submission = {};
    device->output_readback_width = 0;
    device->output_readback_height = 0;
    device->output_readback_pending = false;
    device->pick_output_buffer = {};
    device->pick_slots = {};
    device->viewer_constant_buffer = {};
    device->blas_cache_state = {};
    device->tlas = {};
    device->last_acceleration_revision = 0;
    device->last_acceleration_summary = {};
    device->last_acceleration_cpu_ms = 0.0;
    device->last_point_blas_prebuild_info_ms = 0.0;
    device->last_point_blas_prebuild_info_count = 0;
    device->last_line_blas_prebuild_info_ms = 0.0;
    device->last_line_blas_prebuild_info_count = 0;
    device->shader_modules.clear();
    device->pipeline = {};
    device->shader_table = {};
    device->accumulation_state = {};
    device->last_present_result = {};
    device->blas_reuse_enabled = !environment_flag_enabled("RTVDB_DISABLE_BLAS_REUSE");
    device->continuous_render = config.continuous_render;
    const rt_rhi_device_desc desc{
        static_cast<std::uint32_t>(config.capture_width),
        static_cast<std::uint32_t>(config.capture_height),
        config.d3d12.device,
        config.d3d12.command_queue};
    return device->api->initialize(desc, out_error);
}

bool shutdown_rt_device(rt_device* device, rt_device_error* out_error) {
    if (device == nullptr || device->api == nullptr) {
        set_missing_lifecycle_error(out_error, rt_device_operation::shutdown, "RT device shutdown is unavailable");
        return false;
    }
    rt_device_timing idle_timing{};
    rt_device_error idle_error{};
    const bool waited = device->api->wait_idle(&idle_timing, &idle_error);
    destroy_scene_buffers(device, &device->scene_buffers);
    destroy_output_resources(device);
    destroy_pick_buffers(device);
    destroy_viewer_resources(device);
    destroy_rt_blas_cache(device, &device->blas_cache_state);
    device->api->destroy_tlas(device->tlas);
    device->tlas = {};
    destroy_viewer_rt_shader_modules(device);
    const bool shutdown = device->api->shutdown(out_error);
    device->frame_state = {};
    device->last_acceleration_revision = 0;
    device->last_acceleration_summary = {};
    device->last_acceleration_cpu_ms = 0.0;
    device->last_point_blas_prebuild_info_ms = 0.0;
    device->last_point_blas_prebuild_info_count = 0;
    device->last_line_blas_prebuild_info_ms = 0.0;
    device->last_line_blas_prebuild_info_count = 0;
    device->pipeline = {};
    device->shader_table = {};
    device->accumulation_state = {};
    device->last_present_result = {};
    device->blas_reuse_enabled = true;
    device->continuous_render = false;
    if (!waited && shutdown && out_error != nullptr) {
        *out_error = idle_error;
        out_error->operation = rt_device_operation::shutdown;
    }
    return waited && shutdown;
}

bool wait_for_rt_device_idle(
    rt_device* device,
    rt_device_timing* out_timing,
    rt_device_error* out_error)
{
    if (device == nullptr || device->api == nullptr) {
        set_missing_lifecycle_error(out_error, rt_device_operation::wait_idle, "RT device wait_idle is unavailable");
        return false;
    }
    return device->api->wait_idle(out_timing, out_error);
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
    left_constants.pick_pixel_x = 0;
    left_constants.pick_pixel_y = 0;
    right_constants.pick_pixel_x = 0;
    right_constants.pick_pixel_y = 0;
    return std::memcmp(&left_constants, &right_constants, sizeof(left_constants)) == 0;
}

bool begin_rt_commands(
    rt_device* device,
    rt_queue_class queue,
    rt_command_encoder* out_encoder,
    rt_device_error* out_error,
    rt_device_timing* out_timing)
{
    if (device == nullptr || device->api == nullptr) {
        set_missing_lifecycle_error(
            out_error,
            rt_device_operation::begin_commands,
            "RT command begin is unavailable");
        return false;
    }
    const auto begin_start = std::chrono::steady_clock::now();
    const bool begun = device->api->begin_commands(queue, out_encoder, out_error);
    if (out_timing != nullptr) {
        out_timing->command_slot_wait_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin_start).count();
    }
    return begun;
}

bool submit_rt_commands(
    rt_device* device,
    rt_command_encoder encoder,
    rt_submission_token* out_submission,
    rt_device_timing* out_timing,
    rt_device_error* out_error)
{
    if (device == nullptr || device->api == nullptr) {
        set_missing_lifecycle_error(
            out_error,
            rt_device_operation::submit_commands,
            "RT command submit is unavailable");
        return false;
    }
    return device->api->submit_commands(
        encoder,
        out_submission,
        out_timing,
        out_error);
}

void discard_rt_commands(
    rt_device* device,
    rt_command_encoder encoder)
{
    if (device != nullptr && device->api != nullptr && encoder) {
        device->api->discard_commands(encoder);
    }
}

bool is_rt_submission_complete(
    rt_device* device,
    rt_submission_token submission,
    bool* out_complete,
    rt_device_error* out_error)
{
    if (device == nullptr || device->api == nullptr) {
        set_missing_lifecycle_error(
            out_error,
            rt_device_operation::query_submission,
            "RT submission query is unavailable");
        return false;
    }
    return device->api->is_complete(submission, out_complete, out_error);
}

bool wait_for_rt_submission(
    rt_device* device,
    rt_submission_token submission,
    rt_device_timing* out_timing,
    rt_device_error* out_error)
{
    if (device == nullptr || device->api == nullptr) {
        set_missing_lifecycle_error(
            out_error,
            rt_device_operation::wait_submission,
            "RT submission wait is unavailable");
        return false;
    }
    return device->api->wait(submission, out_timing, out_error);
}

bool record_rt_barriers(
    rt_device* device,
    rt_command_encoder encoder,
    const rt_resource_barrier* barriers,
    std::size_t barrier_count,
    rt_device_error* out_error)
{
    if (device == nullptr || device->api == nullptr) {
        set_frame_error(
            out_error,
            rt_device_operation::transition_resource,
            "RT resource barrier API is unavailable");
        return false;
    }
    return device->api->barrier(
        encoder,
        barriers,
        barrier_count,
        out_error);
}

bool record_rt_buffer_copy(
    rt_device* device,
    rt_command_encoder encoder,
    rt_buffer_handle source,
    rt_buffer_handle destination,
    const rt_buffer_copy_region &region,
    rt_device_error* out_error)
{
    if (device == nullptr || device->api == nullptr) {
        set_frame_error(
            out_error,
            rt_device_operation::copy_resource,
            "RT buffer copy API is unavailable");
        return false;
    }
    return device->api->copy_buffer(
        encoder,
        source,
        destination,
        region,
        out_error);
}

bool record_rt_texture_to_buffer_copy(
    rt_device* device,
    rt_command_encoder encoder,
    rt_texture_handle source,
    rt_buffer_handle destination,
    const rt_texture_buffer_copy_region &region,
    rt_device_error* out_error)
{
    if (device == nullptr || device->api == nullptr) {
        set_frame_error(
            out_error,
            rt_device_operation::copy_resource,
            "RT texture-to-buffer copy API is unavailable");
        return false;
    }
    return device->api->copy_texture_to_buffer(
        encoder,
        source,
        destination,
        region,
        out_error);
}

bool record_rt_texture_clear(
    rt_device* device,
    rt_command_encoder encoder,
    rt_texture_handle texture,
    const float color[4],
    rt_device_error* out_error)
{
    if (device == nullptr || device->api == nullptr) {
        set_frame_error(
            out_error,
            rt_device_operation::copy_resource,
            "RT texture clear API is unavailable");
        return false;
    }
    return device->api->clear_texture(
        encoder,
        texture,
        color,
        out_error);
}

bool record_rt_trace_rays(
    rt_device* device,
    rt_command_encoder encoder,
    const rt_trace_rays_desc &desc,
    rt_device_error* out_error)
{
    if (device == nullptr || device->api == nullptr) {
        set_frame_error(
            out_error,
            rt_device_operation::trace_rays,
            "RT ray trace API is unavailable");
        return false;
    }
    return device->api->trace_rays(
        encoder,
        desc,
        out_error);
}

bool write_rt_trace_constants(
    rt_device* device,
    rt_command_encoder encoder,
    const rt_viewer_constants &constants,
    rt_device_error* out_error)
{
    if (device == nullptr || device->api == nullptr || !device->viewer_constant_buffer ||
        !encoder || encoder.slot >= kRtCommandSlotCount) {
        set_frame_error(
            out_error,
            rt_device_operation::trace_rays,
            "RT trace constant upload request is invalid");
        return false;
    }
    const rt_viewer_constants_gpu packed = pack_rt_viewer_constants(constants);
    return device->api->upload_buffer(
        device->viewer_constant_buffer,
        static_cast<std::size_t>(encoder.slot) * kRtViewerConstantSlotStride,
        &packed,
        sizeof(packed),
        out_error);
}

bool read_rt_buffer(
    rt_device* device,
    rt_buffer_handle buffer,
    std::size_t offset,
    void* data,
    std::size_t size,
    rt_device_error* out_error)
{
    if (device == nullptr || device->api == nullptr) {
        set_frame_error(
            out_error,
            rt_device_operation::readback,
            "RT buffer read API is unavailable");
        return false;
    }
    return device->api->read_buffer(
        buffer,
        offset,
        data,
        size,
        out_error);
}

bool transition_rt_texture(
    rt_device* device,
    rt_command_encoder encoder,
    rt_texture_handle texture,
    rt_resource_usage usage,
    rt_device_error* out_error)
{
    if (!texture) {
        set_frame_error(
            out_error,
            rt_device_operation::transition_resource,
            "RT texture transition resource is invalid");
        return false;
    }
    const rt_resource_barrier barrier{
        rt_resource_kind::texture,
        {},
        texture,
        rt_resource_usage::undefined,
        usage};
    return record_rt_barriers(device, encoder, &barrier, 1, out_error);
}

bool transition_rt_buffer(
    rt_device* device,
    rt_command_encoder encoder,
    rt_buffer_handle buffer,
    rt_resource_usage usage,
    rt_device_error* out_error)
{
    if (!buffer) {
        set_frame_error(
            out_error,
            rt_device_operation::transition_resource,
            "RT buffer transition resource is invalid");
        return false;
    }
    const rt_resource_barrier barrier{
        rt_resource_kind::buffer,
        buffer,
        {},
        rt_resource_usage::undefined,
        usage};
    return record_rt_barriers(device, encoder, &barrier, 1, out_error);
}

bool prepare_rt_device_pipeline(
    rt_device* device,
    const rt_pipeline_desc &pipeline_desc,
    const rt_shader_table_desc &shader_table_desc,
    rt_device_frame_result* in_out_result,
    rt_device_error* out_error)
{
    if (device == nullptr || device->api == nullptr || in_out_result == nullptr ||
        !rt_pipeline_desc_is_valid(pipeline_desc) ||
        !rt_shader_table_section_is_valid(
            pipeline_desc,
            shader_table_desc.ray_generation,
            rt_shader_stage::ray_generation,
            false) ||
        !rt_shader_table_section_is_valid(
            pipeline_desc,
            shader_table_desc.miss,
            rt_shader_stage::miss,
            false) ||
        !rt_shader_table_section_is_valid(
            pipeline_desc,
            shader_table_desc.hit,
            rt_shader_stage::closest_hit,
            true) ||
        !rt_shader_table_section_is_valid(
            pipeline_desc,
            shader_table_desc.callable,
            rt_shader_stage::callable,
            false)) {
        set_frame_error(
            out_error,
            rt_device_operation::prepare_pipeline,
            "RT device pipeline preparation is unavailable");
        return false;
    }
    if (!device->pipeline) {
        rt_pipeline_handle pipeline{};
        if (!device->api->create_pipeline(pipeline_desc, &pipeline, out_error) || !pipeline) {
            return false;
        }
        device->pipeline = pipeline;
        in_out_result->pipeline_changed = true;
    }
    if (!device->shader_table) {
        rt_shader_table_desc resolved_shader_table_desc = shader_table_desc;
        resolved_shader_table_desc.pipeline = device->pipeline;
        rt_shader_table_handle shader_table{};
        if (!device->api->create_shader_table(
                resolved_shader_table_desc,
                &shader_table,
                out_error) ||
            !shader_table) {
            return false;
        }
        device->shader_table = shader_table;
        in_out_result->shader_table_changed = true;
    }
    return true;
}

void begin_rt_device_access(rt_device* device) {
    if (device != nullptr) {
        device->access_mutex.lock();
    }
}

void end_rt_device_access(rt_device* device) {
    if (device != nullptr) {
        device->access_mutex.unlock();
    }
}

bool submit_rt_output_render(
    rt_device* device,
    const rt_native_frame_request &request,
    rt_command_encoder encoder,
    rt_device_timing* out_timing,
    rt_device_error* out_error)
{
    if (!encoder &&
        !begin_rt_commands(device, rt_queue_class::graphics, &encoder, out_error, out_timing)) {
        return false;
    }
    const auto record_start = std::chrono::steady_clock::now();
    const bool recorded =
        write_rt_trace_constants(device, encoder, request.constants, out_error) &&
        transition_rt_texture(
            device,
            encoder,
            device->accumulation_texture,
            rt_resource_usage::shader_write,
            out_error) &&
        transition_rt_texture(
            device,
            encoder,
            device->output_texture,
            rt_resource_usage::shader_write,
            out_error) &&
        record_rt_trace_rays(
            device,
            encoder,
            {
                device->pipeline,
                device->shader_table,
                0,
                static_cast<std::uint32_t>(request.width),
                static_cast<std::uint32_t>(request.height),
                1,
                true},
            out_error) &&
        transition_rt_texture(
            device,
            encoder,
            device->output_texture,
            rt_resource_usage::shader_read,
            out_error);
    if (!recorded) {
        discard_rt_commands(device, encoder);
        return false;
    }
    if (out_timing != nullptr) {
        out_timing->command_record_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - record_start).count();
    }
    rt_submission_token submission{};
    return submit_rt_commands(
        device,
        encoder,
        &submission,
        out_timing,
        out_error);
}

bool submit_rt_output_clear(
    rt_device* device,
    const rt_native_frame_request &request,
    rt_command_encoder encoder,
    rt_device_timing* out_timing,
    rt_device_error* out_error)
{
    if (!encoder &&
        !begin_rt_commands(device, rt_queue_class::graphics, &encoder, out_error, out_timing)) {
        return false;
    }
    const auto record_start = std::chrono::steady_clock::now();
    const bool recorded =
        transition_rt_texture(
            device,
            encoder,
            device->output_texture,
            rt_resource_usage::clear_destination,
            out_error) &&
        record_rt_texture_clear(
            device,
            encoder,
            device->output_texture,
            request.clear_color,
            out_error) &&
        transition_rt_texture(
            device,
            encoder,
            device->output_texture,
            rt_resource_usage::shader_read,
            out_error);
    if (!recorded) {
        discard_rt_commands(device, encoder);
        return false;
    }
    if (out_timing != nullptr) {
        out_timing->command_record_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - record_start).count();
    }
    rt_submission_token submission{};
    return submit_rt_commands(
        device,
        encoder,
        &submission,
        out_timing,
        out_error);
}

bool submit_rt_reused_output(
    rt_device* device,
    rt_command_encoder encoder,
    rt_device_timing* out_timing,
    rt_device_error* out_error)
{
    if (!encoder &&
        !begin_rt_commands(device, rt_queue_class::graphics, &encoder, out_error, out_timing)) {
        return false;
    }
    const auto record_start = std::chrono::steady_clock::now();
    if (!transition_rt_texture(
            device,
            encoder,
            device->output_texture,
            rt_resource_usage::shader_read,
            out_error)) {
        discard_rt_commands(device, encoder);
        return false;
    }
    if (out_timing != nullptr) {
        out_timing->command_record_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - record_start).count();
    }
    rt_submission_token submission{};
    return submit_rt_commands(
        device,
        encoder,
        &submission,
        out_timing,
        out_error);
}

bool collect_rt_output_readback(
    rt_device* device,
    std::vector<std::uint8_t>* out_pixels,
    rt_device_error* out_error)
{
    bool complete = false;
    if (!is_rt_submission_complete(
            device,
            device->output_readback_submission,
            &complete,
            out_error)) {
        return false;
    }
    if (!complete) {
        return true;
    }
    const rt_texture_copy_footprint footprint = device->output_readback_footprint;
    std::vector<std::uint8_t> readback(footprint.total_size);
    if (!device->api->read_buffer(
            device->output_readback_buffer,
            0,
            readback.data(),
            readback.size(),
            out_error)) {
        return false;
    }
    const std::size_t row_bytes =
        static_cast<std::size_t>(device->output_readback_width) * 4u;
    out_pixels->resize(
        row_bytes *
        static_cast<std::size_t>(device->output_readback_height));
    for (int y = 0; y < device->output_readback_height; ++y) {
        std::memcpy(
            out_pixels->data() + static_cast<std::size_t>(y) * row_bytes,
            readback.data() + static_cast<std::size_t>(y) * footprint.row_pitch,
            row_bytes);
    }
    if (device->capabilities.output_format == rt_texture_format::rgba8_unorm) {
        for (std::size_t offset = 0; offset + 3u < out_pixels->size(); offset += 4u) {
            std::swap((*out_pixels)[offset], (*out_pixels)[offset + 2u]);
        }
    }
    device->output_readback_submission = {};
    device->output_readback_pending = false;
    return true;
}

bool readback_rt_output(
    rt_device* device,
    int width,
    int height,
    std::vector<std::uint8_t>* out_pixels,
    rt_device_error* out_error)
{
    if (device == nullptr || device->api == nullptr || out_pixels == nullptr ||
        width <= 0 || height <= 0 ||
        width != device->frame_state.output_width ||
        height != device->frame_state.output_height ||
        !device->output_texture || !device->output_readback_buffer ||
        device->output_readback_footprint.total_size == 0) {
        set_frame_error(
            out_error,
            rt_device_operation::readback,
            "RT output readback request is invalid");
        return false;
    }
    out_pixels->clear();
    if (device->output_readback_pending) {
        if (!collect_rt_output_readback(device, out_pixels, out_error)) {
            return false;
        }
        if (device->output_readback_pending || !out_pixels->empty()) {
            return true;
        }
    }

    rt_command_encoder encoder{};
    if (!begin_rt_commands(device, rt_queue_class::graphics, &encoder, out_error)) {
        return false;
    }
    const rt_texture_copy_footprint footprint = device->output_readback_footprint;
    const bool recorded =
        transition_rt_texture(
            device,
            encoder,
            device->output_texture,
            rt_resource_usage::copy_source,
            out_error) &&
        record_rt_texture_to_buffer_copy(
            device,
            encoder,
            device->output_texture,
            device->output_readback_buffer,
            {
                0,
                footprint.row_pitch,
                static_cast<std::uint32_t>(width),
                static_cast<std::uint32_t>(height)},
            out_error) &&
        transition_rt_texture(
            device,
            encoder,
            device->output_texture,
            rt_resource_usage::shader_read,
            out_error);
    if (!recorded) {
        discard_rt_commands(device, encoder);
        return false;
    }
    rt_device_timing timing{};
    if (!submit_rt_commands(
            device,
            encoder,
            &device->output_readback_submission,
            &timing,
            out_error)) {
        return false;
    }
    device->output_readback_width = width;
    device->output_readback_height = height;
    device->output_readback_pending = true;
    return true;
}

bool execute_rt_device_native_frame(
    rt_device* device,
    const rt_native_frame_request &request,
    rt_present_result* out_result,
    rt_device_error* out_error,
    rt_deferred_acceleration_submission* deferred_acceleration)
{
    if (out_result != nullptr) {
        *out_result = {};
    }
    if (device == nullptr || device->api == nullptr) {
        set_frame_error(out_error, rt_device_operation::present, "RT device frame operation is unavailable");
        return false;
    }
    rt_present_result result{};
    bool succeeded = false;
    if (request.operation == rt_present_operation::readback_bgra) {
        const auto readback_start = std::chrono::steady_clock::now();
        succeeded = readback_rt_output(
            device,
            request.width,
            request.height,
            request.out_pixels,
            out_error);
        result.readback_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - readback_start).count();
        result.captured = succeeded && request.out_pixels != nullptr && !request.out_pixels->empty();
    } else if (request.operation == rt_present_operation::post_present) {
        succeeded = true;
    } else {
        if (request.build == nullptr || request.width <= 0 || request.height <= 0) {
            set_frame_error(out_error, rt_device_operation::trace_rays, "RT output dispatch request is invalid");
            discard_deferred_acceleration_submission(device, deferred_acceleration);
            return false;
        }
        rt_device_timing dispatch_timing{};
        const rt_command_encoder acceleration_encoder = deferred_acceleration != nullptr
            ? deferred_acceleration->encoder
            : rt_command_encoder{};
        if (request.reuse_output) {
            succeeded = submit_rt_reused_output(
                device,
                acceleration_encoder,
                &dispatch_timing,
                out_error);
        } else if (request.dispatch == rt_dispatch_kind::render) {
            succeeded = submit_rt_output_render(
                device,
                request,
                acceleration_encoder,
                &dispatch_timing,
                out_error);
        } else if (request.dispatch == rt_dispatch_kind::clear) {
            succeeded = submit_rt_output_clear(
                device,
                request,
                acceleration_encoder,
                &dispatch_timing,
                out_error);
        } else {
            set_frame_error(out_error, rt_device_operation::trace_rays, "RT output operation is invalid");
            discard_deferred_acceleration_submission(device, deferred_acceleration);
            return false;
        }
        if (!succeeded) {
            discard_deferred_acceleration_submission(device, deferred_acceleration);
            return false;
        }
        if (deferred_acceleration != nullptr && deferred_acceleration->encoder) {
            const auto acceleration_finalize_start = std::chrono::steady_clock::now();
            commit_deferred_acceleration_submission(device, deferred_acceleration);
            result.acceleration_finalize_cpu_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - acceleration_finalize_start).count();
        }
        result.output_timing = dispatch_timing;
        result.timing = dispatch_timing;
        result.reused_output = request.reuse_output;

        if (request.operation == rt_present_operation::native_d3d12 ||
            request.operation == rt_present_operation::native_vulkan) {
            const rt_native_texture_publish_desc native_desc{
                request.native_target,
                request.out_native_target};
            rt_native_texture_extension* const extension =
                device->api->native_texture_extension();
            rt_device_timing delivery_timing{};
            const auto native_publish_start = std::chrono::steady_clock::now();
            succeeded = extension != nullptr && extension->publish_texture(
                device->output_texture,
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
                device,
                request.width,
                request.height,
                request.out_pixels,
                out_error);
            result.readback_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - readback_start).count();
            result.captured = succeeded && request.out_pixels != nullptr && !request.out_pixels->empty();
        } else {
            set_frame_error(out_error, rt_device_operation::present, "RT output delivery operation is invalid");
            return false;
        }
    }
    if (!succeeded) {
        return false;
    }
    device->last_present_result = result;
    if (out_result != nullptr) {
        *out_result = result;
    }
    return true;
}
bool dispatch_rt_device_pick(
    rt_device* device,
    const rt_pick_dispatch_request &request,
    pick_result* out_result,
    rt_pick_dispatch_request* out_completed_request,
    rt_device_error* out_error,
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
    if (device == nullptr || device->api == nullptr || out_result == nullptr) {
        set_frame_error(out_error, rt_device_operation::dispatch_pick, "RT device pick dispatch is unavailable");
        return false;
    }
    if (request.width <= 0 || request.height <= 0 ||
        request.pixel_x < 0 || request.pixel_y < 0 ||
        !device->pipeline || !device->shader_table || !device->pick_output_buffer) {
        set_frame_error(out_error, rt_device_operation::dispatch_pick, "RT device pick request is invalid");
        return false;
    }
    for (const rt_pick_slot &slot : device->pick_slots) {
        if (!slot.readback_buffer) {
            set_frame_error(out_error, rt_device_operation::dispatch_pick, "RT device pick readback slot is invalid");
            return false;
        }
    }
    if (!device->tlas) {
        if (out_completed_request != nullptr) {
            *out_completed_request = request;
        }
        return true;
    }

    bool completed = false;
    for (rt_pick_slot &slot : device->pick_slots) {
        if (!slot.pending) {
            continue;
        }
        bool complete = false;
        if (!is_rt_submission_complete(device, slot.submission, &complete, out_error)) {
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
                device,
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
            device,
            rt_queue_class::graphics,
            &encoder,
            out_error)) {
        return false;
    }
    if (encoder.slot >= kRtCommandSlotCount) {
        discard_rt_commands(device, encoder);
        set_frame_error(out_error, rt_device_operation::dispatch_pick, "RT device pick command slot is invalid");
        return false;
    }
    rt_pick_slot &slot = device->pick_slots[encoder.slot];
    if (slot.pending) {
        discard_rt_commands(device, encoder);
        set_frame_error(out_error, rt_device_operation::dispatch_pick, "RT device pick command slot is still pending");
        return false;
    }

    const bool recorded =
        write_rt_trace_constants(
            device,
            encoder,
            request.constants,
            out_error) &&
        transition_rt_buffer(
            device,
            encoder,
            device->pick_output_buffer,
            rt_resource_usage::shader_write,
            out_error) &&
        record_rt_trace_rays(
            device,
            encoder,
            {
                device->pipeline,
                device->shader_table,
                1,
                1,
                1,
                1,
            },
            out_error) &&
        transition_rt_buffer(
            device,
            encoder,
            device->pick_output_buffer,
            rt_resource_usage::copy_source,
            out_error) &&
        record_rt_buffer_copy(
            device,
            encoder,
            device->pick_output_buffer,
            slot.readback_buffer,
            {0, 0, sizeof(rt_pick_gpu_result)},
            out_error) &&
        transition_rt_buffer(
            device,
            encoder,
            slot.readback_buffer,
            rt_resource_usage::host_read,
            out_error) &&
        transition_rt_buffer(
            device,
            encoder,
            device->pick_output_buffer,
            rt_resource_usage::shader_write,
            out_error);
    if (!recorded) {
        discard_rt_commands(device, encoder);
        return false;
    }

    rt_device_timing timing{};
    if (!submit_rt_commands(
            device,
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

void copy_rt_device_diagnostics(scene_build_info* out_info, const rt_device &device) {
    if (out_info == nullptr) {
        return;
    }
    out_info->revision = device.last_acceleration_revision;
    out_info->blas_reused_count = device.last_acceleration_summary.blas_reused_count;
    out_info->blas_rebuilt_count = device.last_acceleration_summary.blas_rebuilt_count;
    out_info->blas_reused_triangle_chunk_count = device.last_acceleration_summary.blas_reused_triangle_chunk_count;
    out_info->blas_rebuilt_triangle_chunk_count = device.last_acceleration_summary.blas_rebuilt_triangle_chunk_count;
    out_info->tlas_rebuild_count = device.last_acceleration_summary.tlas_rebuild_count;
    out_info->accel_build_ms = device.last_acceleration_cpu_ms;
    out_info->accel_procedural_aabb_ms = device.scene_buffers.procedural_aabb_upload_ms;
    out_info->accel_point_blas_prebuild_info_ms =
        device.last_point_blas_prebuild_info_ms;
    out_info->accel_point_blas_prebuild_info_count =
        device.last_point_blas_prebuild_info_count;
    out_info->accel_line_blas_prebuild_info_ms =
        device.last_line_blas_prebuild_info_ms;
    out_info->accel_line_blas_prebuild_info_count =
        device.last_line_blas_prebuild_info_count;
    out_info->paint_rt_scene_snapshot_ms =
        device.last_present_result.scene_snapshot_cpu_ms;
    out_info->paint_rt_pre_acceleration_prepare_ms =
        device.last_present_result.frame_pre_acceleration_prepare_cpu_ms;
    out_info->paint_as_command_slot_wait_ms =
        device.last_present_result.acceleration_timing.command_slot_wait_ms;
    out_info->paint_accel_command_record_ms =
        device.last_present_result.acceleration_timing.command_record_ms;
    out_info->paint_rt_post_acceleration_prepare_ms =
        device.last_present_result.frame_post_acceleration_prepare_cpu_ms;
    out_info->paint_rt_output_prepare_ms =
        device.last_present_result.rt_output_prepare_cpu_ms;
    out_info->paint_rt_output_command_slot_wait_ms =
        device.last_present_result.output_timing.command_slot_wait_ms;
    out_info->paint_rt_command_record_ms =
        device.last_present_result.output_timing.command_record_ms;
    out_info->paint_rt_submit_ms = device.last_present_result.output_timing.submit_cpu_ms;
    out_info->paint_as_finalize_ms = device.last_present_result.acceleration_finalize_cpu_ms;
    out_info->paint_native_target_publish_ms =
        device.last_present_result.native_publish_cpu_ms;
    out_info->paint_rt_accumulation_finalize_ms =
        device.last_present_result.accumulation_finalize_cpu_ms;
    out_info->dispatch_submit_cpu_ms = device.last_present_result.timing.submit_cpu_ms;
    out_info->dispatch_gpu_wait_ms = device.last_present_result.timing.gpu_wait_ms;
    out_info->dispatch_gpu_ms = device.last_present_result.timing.gpu_ms;
    out_info->dispatch_ms =
        device.last_present_result.timing.submit_cpu_ms +
        device.last_present_result.timing.gpu_wait_ms;
    out_info->readback_ms = device.last_present_result.readback_ms;
    out_info->accumulation_sample_count = device.accumulation_state.sample_count;
    out_info->accumulation_target_sample_count = kRtMaxAccumulationSamples;
    out_info->accumulation_in_progress = device.accumulation_state.active;
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
}

void reset_rt_device_accumulation(rt_device* device) {
    if (device != nullptr) {
        reset_rt_accumulation_state(&device->accumulation_state);
    }
}

bool begin_rt_device_accumulation(
    rt_device* device,
    const rt_accumulation_key &next_key,
    bool continuous_render)
{
    return device != nullptr &&
        begin_rt_accumulation(&device->accumulation_state, next_key, continuous_render);
}

void complete_rt_device_accumulation(rt_device* device, bool continuous_render) {
    if (device != nullptr) {
        complete_rt_accumulation(&device->accumulation_state, continuous_render);
    }
}

bool prepare_rt_device_frame(
    rt_device* device,
    const rt_device_frame_request &request,
    rt_device_frame_result* out_result,
    rt_device_error* out_error,
    rt_deferred_acceleration_submission* deferred_acceleration)
{
    if (out_result != nullptr) {
        *out_result = {};
    }
    if (deferred_acceleration != nullptr) {
        *deferred_acceleration = {};
    }
    if (device == nullptr || device->api == nullptr) {
        set_frame_error(out_error, rt_device_operation::begin_frame, "RT device frame operations are unavailable");
        return false;
    }
    if (request.build == nullptr || request.width <= 0 || request.height <= 0) {
        set_frame_error(out_error, rt_device_operation::begin_frame, "RT device frame request is invalid");
        return false;
    }
    if (device->frame_state.active) {
        set_frame_error(out_error, rt_device_operation::begin_frame, "RT device frame is already active");
        return false;
    }

    const auto preparation_start = std::chrono::steady_clock::now();
    device->frame_state.active = true;
    rt_device_frame_result result{};
    result.output_changed = request.require_output &&
        (!device->frame_state.output_valid ||
            device->frame_state.output_width != request.width ||
            device->frame_state.output_height != request.height);
    result.scene_changed = !device->frame_state.scene_valid ||
        device->frame_state.scene_revision != request.build->revision;

    rt_acceleration_build_plan acceleration_plan{};
    rt_blas_cache_update_plan blas_cache_plan{};
    rt_scene_resource_data resources{};
    rt_acceleration_command_plan acceleration_commands{};
    rt_device_frame_request resolved_request = request;
    if (result.scene_changed) {
        if (!make_rt_acceleration_build_plan(*request.build, &acceleration_plan)) {
            device->frame_state.active = false;
            set_frame_error(out_error, rt_device_operation::begin_frame, "RT acceleration build plan is invalid");
            return false;
        }
        if (!make_rt_blas_cache_update_plan(acceleration_plan, device->blas_cache_state, &blas_cache_plan)) {
            device->frame_state.active = false;
            set_frame_error(out_error, rt_device_operation::begin_frame, "RT BLAS cache update plan is invalid");
            return false;
        }
        if (!device->blas_reuse_enabled || result.scene_changed) {
            for (rt_blas_cache_assignment &assignment : blas_cache_plan.assignments) {
                assignment.reuse_candidate = false;
            }
        }
        if (!make_rt_scene_resource_data(*request.build, acceleration_plan, &resources)) {
            device->frame_state.active = false;
            set_frame_error(out_error, rt_device_operation::begin_frame, "RT scene resource data is invalid");
            return false;
        }
        if (!make_rt_acceleration_command_plan(
                *request.build,
                acceleration_plan,
                blas_cache_plan,
                resources,
                &acceleration_commands)) {
            device->frame_state.active = false;
            set_frame_error(
                out_error,
                rt_device_operation::begin_frame,
                "RT acceleration command plan is invalid");
            return false;
        }
        resolved_request.acceleration_plan = &acceleration_plan;
        resolved_request.blas_cache_plan = &blas_cache_plan;
        resolved_request.resources = &resources;
        resolved_request.acceleration_commands = &acceleration_commands;
    }

    rt_device_error stage_error{rt_device_operation::create_resource, 0, {}};
    if (!ensure_viewer_constant_buffer(
            device,
            &result.viewer_resources_changed,
            &stage_error)) {
        device->frame_state.active = false;
        if (out_error != nullptr) {
            *out_error = stage_error;
        }
        return false;
    }
    if (result.output_changed) {
        stage_error = {rt_device_operation::create_resource, 0, {}};
        if (!create_output_resources(
                device,
                request.width,
                request.height,
                &stage_error)) {
            device->frame_state.active = false;
            if (out_error != nullptr) {
                *out_error = stage_error;
            }
            return false;
        }
    }
    if (request.require_pick || rt_scene_has_renderable_primitives(*request.build)) {
        stage_error = {rt_device_operation::create_resource, 0, {}};
        if (!ensure_pick_buffers(
                device,
                &result.pick_resources_changed,
                &stage_error)) {
            device->frame_state.active = false;
            if (out_error != nullptr) {
                *out_error = stage_error;
            }
            return false;
        }
    }
    if (result.scene_changed) {
        stage_error = {rt_device_operation::upload_scene_buffers, 0, {}};
        if (resolved_request.resources == nullptr ||
            resolved_request.acceleration_plan == nullptr ||
            !upload_scene_buffers(
                device,
                *resolved_request.resources,
                *resolved_request.acceleration_plan,
                &stage_error)) {
            device->frame_state.active = false;
            if (out_error != nullptr) {
                *out_error = stage_error;
            }
            return false;
        }
    }

    result.pre_acceleration_prepare_cpu_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - preparation_start).count();

    if (result.scene_changed &&
        !sync_rt_device_acceleration(
            device,
            resolved_request,
            &result,
            out_error,
            deferred_acceleration)) {
        device->frame_state.active = false;
        return false;
    }
    result.acceleration_changed = result.scene_changed;
    if (result.scene_changed &&
        (deferred_acceleration == nullptr || !deferred_acceleration->encoder)) {
        device->blas_cache_state = std::move(blas_cache_plan.next_state);
    }

    const auto post_acceleration_prepare_start = std::chrono::steady_clock::now();

    if (!update_rt_device_bindings(device, resolved_request, result, out_error)) {
        discard_deferred_acceleration_submission(device, deferred_acceleration);
        device->frame_state.active = false;
        return false;
    }

    if (rt_scene_has_renderable_primitives(*request.build)) {
        stage_error = {rt_device_operation::create_shader_module, 0, {}};
        if (!ensure_viewer_rt_shader_modules(device, &stage_error)) {
            discard_deferred_acceleration_submission(device, deferred_acceleration);
            device->frame_state.active = false;
            if (out_error != nullptr) {
                *out_error = stage_error;
            }
            return false;
        }
        const viewer_rt_pipeline_desc pipeline = make_viewer_rt_pipeline_desc(
            device->shader_modules.data(),
            device->shader_modules.size());
        const rt_pipeline_desc pipeline_desc = pipeline.pipeline();
        const rt_shader_table_desc shader_table_desc = pipeline.shader_table({});
        stage_error = {rt_device_operation::prepare_pipeline, 0, {}};
        if (!prepare_rt_device_pipeline(
                device,
                pipeline_desc,
                shader_table_desc,
                &result,
                &stage_error)) {
            discard_deferred_acceleration_submission(device, deferred_acceleration);
            device->frame_state.active = false;
            if (out_error != nullptr) {
                *out_error = stage_error;
            }
            return false;
        }
    }

    if (deferred_acceleration == nullptr || !deferred_acceleration->encoder) {
        device->frame_state.scene_revision = request.build->revision;
        device->frame_state.scene_valid = true;
    }
    if (request.require_output) {
        device->frame_state.output_width = request.width;
        device->frame_state.output_height = request.height;
        device->frame_state.output_valid = true;
    }
    ++device->frame_state.serial;
    device->frame_state.active = false;
    result.post_acceleration_prepare_cpu_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - post_acceleration_prepare_start).count();
    if (out_result != nullptr) {
        *out_result = result;
    }
    if (out_error != nullptr) {
        *out_error = {rt_device_operation::end_frame, 0, {}};
    }
    return true;
}

} // namespace rtvdb::viewer_backend
