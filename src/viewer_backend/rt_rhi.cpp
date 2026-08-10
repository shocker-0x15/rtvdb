#include "viewer_backend/rt_rhi.h"

#include <algorithm>
#include <array>
#include <limits>

namespace rtvdb::viewer_backend {

bool validate_rt_shader_package_desc(const rt_shader_package_desc &desc) {
    if (desc.modules == nullptr || desc.module_count == 0 ||
        desc.entry_module_indices == nullptr || desc.logical_entries == nullptr ||
        desc.entry_count == 0 ||
        (desc.pipeline_model != rt_pipeline_model::native_ray_tracing &&
            desc.pipeline_model != rt_pipeline_model::compute_intersector)) {
        return false;
    }
    const rt_shader_binary_format package_format = desc.modules[0].format;
    for (std::size_t module_index = 0; module_index < desc.module_count; ++module_index) {
        const rt_shader_module_desc &module = desc.modules[module_index];
        if (module.data == nullptr || module.size == 0 || module.format != package_format) {
            return false;
        }
    }
    if ((desc.pipeline_model == rt_pipeline_model::native_ray_tracing &&
            package_format != rt_shader_binary_format::dxil_library &&
            package_format != rt_shader_binary_format::spirv) ||
        (desc.pipeline_model == rt_pipeline_model::compute_intersector &&
            package_format != rt_shader_binary_format::metallib)) {
        return false;
    }
    std::array<bool, static_cast<std::size_t>(rt_logical_shader_entry::count)> logical_entries{};
    for (std::size_t entry_index = 0; entry_index < desc.entry_count; ++entry_index) {
        const std::size_t logical_index =
            static_cast<std::size_t>(desc.logical_entries[entry_index]);
        if (desc.entry_module_indices[entry_index] >= desc.module_count ||
            logical_index >= logical_entries.size() || logical_entries[logical_index]) {
            return false;
        }
        logical_entries[logical_index] = true;
        if (desc.pipeline_model == rt_pipeline_model::compute_intersector &&
            desc.logical_entries[entry_index] != rt_logical_shader_entry::render &&
            desc.logical_entries[entry_index] != rt_logical_shader_entry::pick) {
            return false;
        }
    }
    if (desc.pipeline_model == rt_pipeline_model::compute_intersector &&
        (!logical_entries[static_cast<std::size_t>(rt_logical_shader_entry::render)] ||
            !logical_entries[static_cast<std::size_t>(rt_logical_shader_entry::pick)])) {
        return false;
    }
    return true;
}

namespace {

bool rt_shader_index_is_valid(
    const rt_pipeline_desc &desc,
    std::uint32_t index,
    rt_shader_stage stage)
{
    return index < desc.shader_count &&
        desc.shaders[index].module &&
        desc.shaders[index].stage == stage &&
        desc.shaders[index].entry_point != nullptr &&
        desc.shaders[index].entry_point[0] != '\0';
}

bool rt_binding_layout_is_valid(const rt_pipeline_desc &desc) {
    if (desc.bindings == nullptr || desc.binding_count == 0) {
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
    return true;
}

bool rt_pipeline_dispatch_entries_are_valid(const rt_pipeline_desc &desc) {
    if (desc.dispatch_entries == nullptr || desc.dispatch_entry_count == 0) {
        return false;
    }
    std::array<bool, static_cast<std::size_t>(rt_logical_dispatch_entry::count)> logical_entries{};
    for (std::size_t index = 0; index < desc.dispatch_entry_count; ++index) {
        const rt_pipeline_dispatch_entry_desc &entry = desc.dispatch_entries[index];
        const std::size_t logical_index = static_cast<std::size_t>(entry.logical_entry);
        if (logical_index >= logical_entries.size() || logical_entries[logical_index]) {
            return false;
        }
        logical_entries[logical_index] = true;
        if (desc.model == rt_pipeline_model::native_ray_tracing) {
            if (entry.shader_or_group_index >= desc.group_count) {
                return false;
            }
            const rt_shader_group_desc &group = desc.groups[entry.shader_or_group_index];
            if (group.type != rt_shader_group_type::general ||
                !rt_shader_index_is_valid(desc, group.general_shader, rt_shader_stage::ray_generation)) {
                return false;
            }
        } else if (!rt_shader_index_is_valid(
                desc,
                entry.shader_or_group_index,
                rt_shader_stage::compute)) {
            return false;
        }
    }
    return logical_entries[static_cast<std::size_t>(rt_logical_dispatch_entry::render)] &&
        logical_entries[static_cast<std::size_t>(rt_logical_dispatch_entry::pick)];
}

} // namespace

bool validate_rt_pipeline_desc(const rt_pipeline_desc &desc) {
    if (!rt_binding_layout_is_valid(desc) ||
        desc.shaders == nullptr || desc.shader_count == 0 ||
        (desc.model != rt_pipeline_model::native_ray_tracing &&
            desc.model != rt_pipeline_model::compute_intersector)) {
        return false;
    }
    for (std::size_t index = 0; index < desc.shader_count; ++index) {
        const rt_shader_entry_desc &shader = desc.shaders[index];
        if (!shader.module || shader.entry_point == nullptr || shader.entry_point[0] == '\0') {
            return false;
        }
    }
    if (desc.model == rt_pipeline_model::compute_intersector) {
        if (desc.groups != nullptr || desc.group_count != 0 ||
            desc.max_recursion_depth != 0 || desc.max_payload_size != 0 ||
            desc.max_attribute_size != 0) {
            return false;
        }
        for (std::size_t index = 0; index < desc.shader_count; ++index) {
            if (desc.shaders[index].stage != rt_shader_stage::compute) {
                return false;
            }
        }
        return rt_pipeline_dispatch_entries_are_valid(desc);
    }
    if (desc.groups == nullptr || desc.group_count == 0 ||
        desc.max_recursion_depth == 0 || desc.max_payload_size == 0 ||
        desc.max_attribute_size == 0) {
        return false;
    }
    for (std::size_t index = 0; index < desc.shader_count; ++index) {
        if (desc.shaders[index].stage == rt_shader_stage::compute) {
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
    return rt_pipeline_dispatch_entries_are_valid(desc);
}

bool get_rt_pipeline_dispatch_entry_index(
    const rt_pipeline_desc &desc,
    rt_logical_dispatch_entry logical_entry,
    std::uint32_t* out_index)
{
    if (out_index != nullptr) {
        *out_index = kRtUnusedShaderIndex;
    }
    if (out_index == nullptr || desc.dispatch_entries == nullptr ||
        static_cast<std::size_t>(logical_entry) >=
            static_cast<std::size_t>(rt_logical_dispatch_entry::count)) {
        return false;
    }
    for (std::size_t index = 0; index < desc.dispatch_entry_count; ++index) {
        if (desc.dispatch_entries[index].logical_entry == logical_entry) {
            *out_index = desc.dispatch_entries[index].shader_or_group_index;
            return true;
        }
    }
    return false;
}

bool get_rt_blas_geometry_counts(
    const rt_blas_build_desc &desc,
    std::size_t geometry_index,
    rt_blas_geometry_counts* out_counts)
{
    if (out_counts != nullptr) {
        *out_counts = {};
    }
    if (out_counts == nullptr || desc.geometries == nullptr ||
        geometry_index >= desc.geometry_count || desc.geometry_count > kRtMaxBlasGeometryCount) {
        return false;
    }

    const std::size_t max_native_count = (std::numeric_limits<std::uint32_t>::max)();
    const rt_acceleration_geometry_desc &geometry = desc.geometries[geometry_index];
    std::size_t actual = 0;
    switch (geometry.type) {
    case rt_acceleration_geometry_type::triangles:
        if (!geometry.triangles.vertex_buffer || !geometry.triangles.index_buffer ||
            geometry.triangles.vertex_format != rt_vertex_format::float3 ||
            geometry.triangles.index_format != rt_index_format::uint32 ||
            geometry.triangles.vertex_count == 0 || geometry.triangles.index_count == 0 ||
            geometry.triangles.index_count % 3u != 0 ||
            geometry.triangles.vertex_count > max_native_count ||
            geometry.triangles.index_count > max_native_count ||
            geometry.triangles.vertex_stride < sizeof(float) * 3u ||
            geometry.triangles.vertex_stride > max_native_count) {
            return false;
        }
        actual = geometry.triangles.index_count / 3u;
        break;
    case rt_acceleration_geometry_type::aabbs:
        if (!geometry.aabbs.buffer || geometry.aabbs.count == 0 ||
            geometry.aabbs.count > max_native_count || geometry.aabbs.offset > max_native_count ||
            geometry.aabbs.stride < sizeof(float) * 6u ||
            geometry.aabbs.stride > max_native_count || geometry.aabbs.stride % sizeof(float) != 0) {
            return false;
        }
        actual = geometry.aabbs.count;
        break;
    default:
        return false;
    }

    const std::size_t allocation = (std::max)(
        actual,
        desc.allocation_primitive_counts != nullptr
            ? desc.allocation_primitive_counts[geometry_index]
            : actual);
    const std::size_t max_allocation = geometry.type == rt_acceleration_geometry_type::triangles
        ? max_native_count / 3u
        : max_native_count;
    if (allocation > max_allocation) {
        return false;
    }
    *out_counts = {actual, allocation};
    return true;
}

bool validate_rt_blas_build_desc(const rt_blas_build_desc &desc) {
    constexpr std::uint32_t kKnownBuildFlags =
        rt_acceleration_build_prefer_fast_trace | rt_acceleration_build_allow_update;
    constexpr std::uint32_t kKnownGeometryFlags = rt_acceleration_geometry_opaque;
    if (!desc.destination || desc.geometries == nullptr || desc.geometry_count == 0 ||
        desc.geometry_count > kRtMaxBlasGeometryCount || (desc.flags & ~kKnownBuildFlags) != 0u) {
        return false;
    }

    const rt_acceleration_geometry_type type = desc.geometries[0].type;
    for (std::size_t geometry_index = 0; geometry_index < desc.geometry_count; ++geometry_index) {
        rt_blas_geometry_counts counts{};
        if (desc.geometries[geometry_index].type != type ||
            (desc.geometries[geometry_index].flags & ~kKnownGeometryFlags) != 0u ||
            !get_rt_blas_geometry_counts(desc, geometry_index, &counts)) {
            return false;
        }
    }
    return true;
}

bool validate_rt_tlas_build_desc(const rt_tlas_build_desc &desc) {
    constexpr std::uint32_t kKnownBuildFlags =
        rt_acceleration_build_prefer_fast_trace | rt_acceleration_build_allow_update;
    constexpr std::uint32_t kKnownInstanceFlags =
        rt_acceleration_instance_triangle_cull_disable |
        rt_acceleration_instance_triangle_front_counterclockwise |
        rt_acceleration_instance_force_opaque |
        rt_acceleration_instance_force_non_opaque;
    constexpr std::uint32_t kMaxInstanceFieldValue = (1u << 24u) - 1u;
    if (!desc.destination || desc.instance_count > (std::numeric_limits<std::uint32_t>::max)() ||
        (desc.instance_count != 0 && desc.instances == nullptr) ||
        (desc.flags & ~kKnownBuildFlags) != 0u) {
        return false;
    }

    for (std::size_t instance_index = 0; instance_index < desc.instance_count; ++instance_index) {
        const rt_tlas_instance_desc &instance = desc.instances[instance_index];
        const bool conflicting_opacity =
            (instance.flags & rt_acceleration_instance_force_opaque) != 0u &&
            (instance.flags & rt_acceleration_instance_force_non_opaque) != 0u;
        if (!instance.acceleration || instance.instance_id > kMaxInstanceFieldValue ||
            instance.hit_group_contribution > kMaxInstanceFieldValue ||
            (instance.flags & ~kKnownInstanceFlags) != 0u || conflicting_opacity) {
            return false;
        }
    }
    return true;
}

} // namespace rtvdb::viewer_backend
