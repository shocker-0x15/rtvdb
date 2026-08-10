#include "viewer_backend/rt_acceleration_plan.h"

#include "viewer_backend/rt_render_plan.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace rtvdb::viewer_backend {
namespace {

constexpr std::uint64_t kFnvOffsetBasis = 14695981039346656037ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

void append_geometry_hash(std::uint64_t* hash, const void* data, std::size_t size) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    for (std::size_t index = 0; index < size; ++index) {
        *hash ^= bytes[index];
        *hash *= kFnvPrime;
    }
}

} // namespace

bool make_rt_acceleration_build_plan(
    const rt_scene_build &build,
    rt_acceleration_build_plan* out_plan)
{
    if (out_plan == nullptr) {
        return false;
    }

    rt_acceleration_build_plan plan{};
    plan.revision = build.revision;
    plan.items.reserve(
        build.triangle_blas_chunk_sets.size() +
        build.point_blas_chunk_sets.size() +
        build.line_blas_chunk_sets.size());

    const auto append_triangle_items = [&]() {
        for (std::size_t group_index = 0; group_index < build.triangle_blas_chunk_sets.size(); ++group_index) {
            const rt_blas_chunk_set &group = build.triangle_blas_chunk_sets[group_index];
            if (group.chunk_count == 0 || group.chunk_count > kRtBlasChunkSetChunkCount) {
                return false;
            }
            rt_acceleration_build_item item{};
            item.kind = rt_acceleration_geometry_kind::triangle;
            item.group_index = group_index;
            item.group = group;
            for (std::size_t geometry_index = 0; geometry_index < group.chunk_count; ++geometry_index) {
                const std::size_t chunk_index = group.chunk_indices[geometry_index];
                if (chunk_index >= build.triangle_chunks.size()) {
                    return false;
                }
                const rt_triangle_chunk &chunk = build.triangle_chunks[chunk_index];
                item.geometry_fingerprints[geometry_index] = chunk.fingerprint;
                item.first_primitives[geometry_index] = chunk.first_triangle;
                item.primitive_counts[geometry_index] = chunk.triangle_count;
            }
            plan.items.push_back(item);
            ++plan.triangle_item_count;
        }
        return true;
    };

    const auto append_procedural_items = [&]<typename GroupList>(
                                             rt_acceleration_geometry_kind kind,
                                             const GroupList &groups,
                                             const std::vector<rt_blas_chunk_set> &blas_groups,
                                             bool points) {
        for (std::size_t group_index = 0; group_index < blas_groups.size(); ++group_index) {
            const rt_blas_chunk_set &group = blas_groups[group_index];
            if (group.chunk_count == 0 || group.chunk_count > kRtBlasChunkSetChunkCount) {
                return false;
            }
            rt_acceleration_build_item item{};
            item.kind = kind;
            item.group_index = group_index;
            item.group = group;
            for (std::size_t geometry_index = 0; geometry_index < group.chunk_count; ++geometry_index) {
                const std::size_t chunk_index = group.chunk_indices[geometry_index];
                if (chunk_index >= groups.size()) {
                    return false;
                }
                const rt_procedural_chunk &procedural_group = groups[chunk_index];
                item.geometry_fingerprints[geometry_index] =
                    rt_procedural_chunk_fingerprint(build, procedural_group, points);
                item.first_primitives[geometry_index] = procedural_group.first_primitive;
                item.primitive_counts[geometry_index] = procedural_group.primitive_count;
            }
            plan.items.push_back(item);
            if (kind == rt_acceleration_geometry_kind::point) {
                ++plan.point_item_count;
            } else {
                ++plan.line_item_count;
            }
        }
        return true;
    };

    if (!append_triangle_items() ||
        !append_procedural_items(
            rt_acceleration_geometry_kind::point,
            build.point_chunks,
            build.point_blas_chunk_sets,
            true) ||
        !append_procedural_items(
            rt_acceleration_geometry_kind::line,
            build.line_chunks,
            build.line_blas_chunk_sets,
            false)) {
        return false;
    }

    plan.point_geometry_fingerprint = rt_point_geometry_fingerprint(build);
    plan.line_geometry_fingerprint = rt_line_geometry_fingerprint(build);
    plan.build_tlas = !plan.items.empty();
    *out_plan = std::move(plan);
    return true;
}

bool make_rt_scene_resource_data(
    const rt_scene_build &build,
    const rt_acceleration_build_plan &plan,
    rt_scene_resource_data* out_data)
{
    if (out_data == nullptr || plan.revision != build.revision) {
        return false;
    }

    rt_scene_resource_data data{};
    data.revision = build.revision;
    data.connection_serial = build.connection_serial;
    data.positions.resize(build.vertex_count);
    data.indices = build.indices;
    data.triangle_colors.resize(build.triangle_count);
    data.points.resize(build.point_count);
    data.lines.resize(build.line_count);
    data.point_aabbs.resize(build.point_count);
    data.line_aabbs.resize(build.line_count);
    data.instance_geometry.resize(plan.items.size() * kRtBlasChunkSetChunkCount);

    for (std::size_t vertex_index = 0; vertex_index < build.vertex_count; ++vertex_index) {
        const rtvdb::vec3 &position = build.vertices[vertex_index].position;
        data.positions[vertex_index] = {position.x, position.y, position.z, 0.0f};
    }

    for (std::size_t chunk_index = 0; chunk_index < build.triangle_chunks.size(); ++chunk_index) {
        const rt_triangle_chunk &chunk = build.triangle_chunks[chunk_index];
        if (chunk.vertex_offset + chunk.vertex_count > build.vertices.size() ||
            chunk.index_offset + chunk.index_count > build.indices.size() ||
            chunk.first_triangle + chunk.triangle_count > data.triangle_colors.size()) {
            return false;
        }

        for (std::size_t local_triangle_index = 0; local_triangle_index < chunk.triangle_count;
             ++local_triangle_index) {
            const std::size_t index_base = chunk.index_offset + local_triangle_index * 3u;
            if (index_base + 2u >= build.indices.size()) {
                return false;
            }
            const std::uint32_t vertex_index = build.indices[index_base];
            if (vertex_index >= build.vertices.size()) {
                return false;
            }
            const rtvdb::rgba &color = build.vertices[vertex_index].color;
            data.triangle_colors[chunk.first_triangle + local_triangle_index] = {
                color.r,
                color.g,
                color.b,
                color.a,
            };
        }
    }

    for (std::size_t point_index = 0; point_index < build.point_count; ++point_index) {
        const point &source = build.points[point_index];
        rt_scene_gpu_point &target = data.points[point_index];
        target.position_radius[0] = source.position.x;
        target.position_radius[1] = source.position.y;
        target.position_radius[2] = source.position.z;
        target.position_radius[3] = source.radius;
        target.color[0] = source.color.r;
        target.color[1] = source.color.g;
        target.color[2] = source.color.b;
        target.color[3] = source.color.a;
        const float radius = (std::max)(source.radius, 1.0e-6f);
        data.point_aabbs[point_index] = {
            source.position.x - radius,
            source.position.y - radius,
            source.position.z - radius,
            source.position.x + radius,
            source.position.y + radius,
            source.position.z + radius};
    }
    for (std::size_t line_index = 0; line_index < build.line_count; ++line_index) {
        const line &source = build.lines[line_index];
        rt_scene_gpu_line &target = data.lines[line_index];
        target.a_radius[0] = source.a.x;
        target.a_radius[1] = source.a.y;
        target.a_radius[2] = source.a.z;
        target.a_radius[3] = source.radius;
        target.b_pad[0] = source.b.x;
        target.b_pad[1] = source.b.y;
        target.b_pad[2] = source.b.z;
        target.color[0] = source.color.r;
        target.color[1] = source.color.g;
        target.color[2] = source.color.b;
        target.color[3] = source.color.a;
        target.flags = static_cast<std::uint32_t>(source.flags);
        const float radius = (std::max)(source.radius, 1.0e-6f);
        data.line_aabbs[line_index] = {
            (std::min)(source.a.x, source.b.x) - radius,
            (std::min)(source.a.y, source.b.y) - radius,
            (std::min)(source.a.z, source.b.z) - radius,
            (std::max)(source.a.x, source.b.x) + radius,
            (std::max)(source.a.y, source.b.y) + radius,
            (std::max)(source.a.z, source.b.z) + radius};
    }

    for (std::size_t instance_index = 0; instance_index < plan.items.size(); ++instance_index) {
        const rt_acceleration_build_item &item = plan.items[instance_index];
        if (item.group.chunk_count == 0 || item.group.chunk_count > kRtBlasChunkSetChunkCount) {
            return false;
        }
        std::uint32_t primitive_offset = 0;
        for (std::size_t geometry_index = 0; geometry_index < item.group.chunk_count; ++geometry_index) {
            if (item.first_primitives[geometry_index] > std::numeric_limits<std::uint32_t>::max() ||
                item.primitive_counts[geometry_index] > std::numeric_limits<std::uint32_t>::max()) {
                return false;
            }
            rt_scene_geometry_metadata &metadata =
                data.instance_geometry[instance_index * kRtBlasChunkSetChunkCount + geometry_index];
            metadata.primitive_base = static_cast<std::uint32_t>(item.first_primitives[geometry_index]);
            metadata.primitive_count = static_cast<std::uint32_t>(item.primitive_counts[geometry_index]);
            if (item.kind == rt_acceleration_geometry_kind::triangle) {
                const std::size_t chunk_index = item.group.chunk_indices[geometry_index];
                if (chunk_index >= build.triangle_chunks.size() ||
                    build.triangle_chunks[chunk_index].index_offset > std::numeric_limits<std::uint32_t>::max() ||
                    metadata.primitive_count > std::numeric_limits<std::uint32_t>::max() - primitive_offset) {
                    return false;
                }
                metadata.index_offset = static_cast<std::uint32_t>(build.triangle_chunks[chunk_index].index_offset);
                metadata.primitive_offset = primitive_offset;
                primitive_offset += metadata.primitive_count;
            }
        }
    }

    *out_data = std::move(data);
    return true;
}

bool make_rt_acceleration_command_plan(
    const rt_scene_build &build,
    const rt_acceleration_build_plan &build_plan,
    const rt_blas_cache_update_plan &cache_plan,
    const rt_scene_resource_data &resources,
    rt_acceleration_command_plan* out_plan)
{
    if (out_plan == nullptr || build_plan.revision != build.revision ||
        cache_plan.revision != build_plan.revision || resources.revision != build.revision ||
        cache_plan.assignments.size() != build_plan.items.size() ||
        resources.instance_geometry.size() != build_plan.items.size() * kRtBlasChunkSetChunkCount) {
        return false;
    }

    rt_acceleration_command_plan plan{};
    plan.revision = build.revision;
    plan.blas_commands.reserve(build_plan.items.size());
    for (std::size_t item_index = 0; item_index < build_plan.items.size(); ++item_index) {
        if (item_index > std::numeric_limits<std::uint32_t>::max()) {
            return false;
        }
        const rt_acceleration_build_item &item = build_plan.items[item_index];
        if (item.group.chunk_count == 0 || item.group.chunk_count > kRtBlasChunkSetChunkCount) {
            return false;
        }
        rt_blas_build_command command{};
        command.kind = item.kind;
        command.geometry_count = item.group.chunk_count;
        command.instance_index = static_cast<std::uint32_t>(item_index);
        switch (item.kind) {
        case rt_acceleration_geometry_kind::triangle:
            command.hit_group_contribution =
                viewer_rt_hit_group_contribution(viewer_rt_shader_group::triangle_hit);
            break;
        case rt_acceleration_geometry_kind::point:
            command.hit_group_contribution =
                viewer_rt_hit_group_contribution(viewer_rt_shader_group::point_hit);
            break;
        case rt_acceleration_geometry_kind::line:
            command.hit_group_contribution =
                viewer_rt_hit_group_contribution(viewer_rt_shader_group::line_hit);
            break;
        default:
            return false;
        }
        for (std::size_t geometry_index = 0; geometry_index < item.group.chunk_count; ++geometry_index) {
            const std::size_t source_index = item.group.chunk_indices[geometry_index];
            bool visible = false;
            rt_acceleration_geometry_desc &geometry = command.geometries[geometry_index];
            geometry.flags = rt_acceleration_geometry_opaque;
            if (item.kind == rt_acceleration_geometry_kind::triangle) {
                if (source_index >= build.triangle_chunks.size()) {
                    return false;
                }
                visible = build.triangle_chunks[source_index].visible;
                if (item.primitive_counts[geometry_index] >
                    std::numeric_limits<std::size_t>::max() / 3u) {
                    return false;
                }
                const std::size_t metadata_index =
                    item_index * kRtBlasChunkSetChunkCount + geometry_index;
                const rt_scene_geometry_metadata &metadata =
                    resources.instance_geometry[metadata_index];
                geometry.type = rt_acceleration_geometry_type::triangles;
                geometry.triangles.vertex_count = resources.positions.size();
                geometry.triangles.vertex_stride = sizeof(rt_scene_gpu_position);
                geometry.triangles.index_offset =
                    static_cast<std::size_t>(metadata.index_offset) * sizeof(std::uint32_t);
                geometry.triangles.index_count = item.primitive_counts[geometry_index] * 3u;
            } else if (item.kind == rt_acceleration_geometry_kind::point) {
                if (source_index >= build.point_chunks.size()) {
                    return false;
                }
                if (item.first_primitives[geometry_index] >
                    (std::numeric_limits<std::size_t>::max)() / sizeof(rt_scene_gpu_aabb)) {
                    return false;
                }
                visible = build.point_chunks[source_index].visible;
                geometry.type = rt_acceleration_geometry_type::aabbs;
                geometry.aabbs.offset =
                    item.first_primitives[geometry_index] * sizeof(rt_scene_gpu_aabb);
                geometry.aabbs.count = item.primitive_counts[geometry_index];
                geometry.aabbs.stride = sizeof(rt_scene_gpu_aabb);
            } else {
                if (source_index >= build.line_chunks.size()) {
                    return false;
                }
                if (item.first_primitives[geometry_index] >
                    (std::numeric_limits<std::size_t>::max)() / sizeof(rt_scene_gpu_aabb)) {
                    return false;
                }
                visible = build.line_chunks[source_index].visible;
                geometry.type = rt_acceleration_geometry_type::aabbs;
                geometry.aabbs.offset =
                    item.first_primitives[geometry_index] * sizeof(rt_scene_gpu_aabb);
                geometry.aabbs.count = item.primitive_counts[geometry_index];
                geometry.aabbs.stride = sizeof(rt_scene_gpu_aabb);
            }
            if (geometry_index == 0) {
                command.visible = visible;
            }
            const std::size_t default_capacity = item.kind == rt_acceleration_geometry_kind::triangle
                ? kDefaultRtSceneTriangleChunkPrimitives
                : kDefaultRtSceneProceduralChunkPrimitives;
            command.allocation_primitive_counts[geometry_index] =
                (std::max)(default_capacity, item.primitive_counts[geometry_index]);
        }
        plan.blas_commands.push_back(command);
    }
    plan.build_tlas = !plan.blas_commands.empty();
    *out_plan = std::move(plan);
    return true;
}
std::uint64_t rt_procedural_chunk_fingerprint(
    const rt_scene_build &build,
    const rt_procedural_chunk &group,
    bool points)
{
    std::uint64_t hash = kFnvOffsetBasis;
    for (std::size_t index = 0; index < group.primitive_count; ++index) {
        if (points) {
            const point &value = build.points[group.first_primitive + index];
            append_geometry_hash(&hash, &value.position, sizeof(value.position));
            append_geometry_hash(&hash, &value.radius, sizeof(value.radius));
        } else {
            const line &value = build.lines[group.first_primitive + index];
            append_geometry_hash(&hash, &value.a, sizeof(value.a));
            append_geometry_hash(&hash, &value.radius, sizeof(value.radius));
            append_geometry_hash(&hash, &value.b, sizeof(value.b));
        }
    }
    return hash;
}

std::uint64_t rt_point_geometry_fingerprint(const rt_scene_build &build) {
    std::uint64_t hash = kFnvOffsetBasis;
    append_geometry_hash(&hash, &build.point_count, sizeof(build.point_count));
    for (const point &value : build.points) {
        append_geometry_hash(&hash, &value.position, sizeof(value.position));
        append_geometry_hash(&hash, &value.radius, sizeof(value.radius));
    }
    return hash;
}

std::uint64_t rt_line_geometry_fingerprint(const rt_scene_build &build) {
    std::uint64_t hash = kFnvOffsetBasis;
    append_geometry_hash(&hash, &build.line_count, sizeof(build.line_count));
    for (const line &value : build.lines) {
        append_geometry_hash(&hash, &value.a, sizeof(value.a));
        append_geometry_hash(&hash, &value.radius, sizeof(value.radius));
        append_geometry_hash(&hash, &value.b, sizeof(value.b));
    }
    return hash;
}

rt_procedural_geometry_update_plan make_rt_procedural_geometry_update_plan(
    const rt_scene_build &build,
    std::size_t cached_point_count,
    std::uint64_t cached_point_fingerprint,
    std::size_t cached_line_count,
    std::uint64_t cached_line_fingerprint)
{
    rt_procedural_geometry_update_plan plan{};
    plan.point_fingerprint = rt_point_geometry_fingerprint(build);
    plan.line_fingerprint = rt_line_geometry_fingerprint(build);
    plan.point_geometry_changed = cached_point_count != build.point_count ||
        cached_point_fingerprint != plan.point_fingerprint;
    plan.line_geometry_changed = cached_line_count != build.line_count ||
        cached_line_fingerprint != plan.line_fingerprint;
    return plan;
}

rt_blas_cache_key make_rt_blas_cache_key(const rt_acceleration_build_item &item) {
    rt_blas_cache_key key{};
    key.kind = item.kind;
    key.first_primitives = item.first_primitives;
    key.primitive_counts = item.primitive_counts;
    key.geometry_fingerprints = item.geometry_fingerprints;
    key.geometry_count = item.group.chunk_count;
    return key;
}

bool rt_blas_cache_key_equals(const rt_blas_cache_key &a, const rt_blas_cache_key &b) {
    if (a.kind != b.kind || a.geometry_count != b.geometry_count ||
        a.geometry_count > kRtBlasChunkSetChunkCount) {
        return false;
    }
    for (std::size_t geometry_index = 0; geometry_index < a.geometry_count; ++geometry_index) {
        if (a.first_primitives[geometry_index] != b.first_primitives[geometry_index] ||
            a.primitive_counts[geometry_index] != b.primitive_counts[geometry_index] ||
            a.geometry_fingerprints[geometry_index] != b.geometry_fingerprints[geometry_index]) {
            return false;
        }
    }
    return true;
}

bool rt_blas_storage_key_equals(const rt_blas_storage_key &a, const rt_blas_storage_key &b) {
    if (a.kind != b.kind || a.geometry_count != b.geometry_count ||
        a.build_flags != b.build_flags ||
        a.geometry_count > kRtBlasChunkSetChunkCount) {
        return false;
    }
    for (std::size_t geometry_index = 0; geometry_index < a.geometry_count; ++geometry_index) {
        if (a.geometry_types[geometry_index] != b.geometry_types[geometry_index] ||
            a.geometry_flags[geometry_index] != b.geometry_flags[geometry_index] ||
            a.geometry_formats[geometry_index] != b.geometry_formats[geometry_index] ||
            a.primitive_counts[geometry_index] != b.primitive_counts[geometry_index] ||
            a.strides[geometry_index] != b.strides[geometry_index]) {
            return false;
        }
    }
    return true;
}

rt_blas_storage_key make_rt_blas_storage_key(
    const rt_blas_build_command &command,
    std::uint32_t build_flags)
{
    rt_blas_storage_key key{};
    key.kind = command.kind;
    key.geometry_count = command.geometry_count;
    key.build_flags = build_flags;
    for (std::size_t geometry_index = 0;
         geometry_index < command.geometry_count &&
             geometry_index < kRtBlasChunkSetChunkCount;
         ++geometry_index) {
        const rt_acceleration_geometry_desc &geometry = command.geometries[geometry_index];
        key.geometry_types[geometry_index] = geometry.type;
        key.geometry_flags[geometry_index] = geometry.flags;
        key.primitive_counts[geometry_index] = command.allocation_primitive_counts[geometry_index];
        if (geometry.type == rt_acceleration_geometry_type::triangles) {
            key.geometry_formats[geometry_index] =
                (static_cast<std::uint32_t>(geometry.triangles.vertex_format) << 16u) |
                static_cast<std::uint32_t>(geometry.triangles.index_format);
            key.strides[geometry_index] = geometry.triangles.vertex_stride;
        } else if (geometry.type == rt_acceleration_geometry_type::aabbs) {
            key.strides[geometry_index] = geometry.aabbs.stride;
        }
    }
    return key;
}

bool make_rt_blas_cache_update_plan(
    const rt_acceleration_build_plan &build_plan,
    const rt_blas_cache_state &current_state,
    rt_blas_cache_update_plan* out_plan)
{
    if (out_plan == nullptr) {
        return false;
    }

    rt_blas_cache_update_plan plan{};
    plan.revision = build_plan.revision;
    plan.assignments.resize(build_plan.items.size());
    plan.next_state = current_state;
    std::vector<rt_blas_cache_key> keys(build_plan.items.size());
    std::array<std::vector<bool>, kRtBlasCachePoolCount> claimed;
    for (std::size_t pool_index = 0; pool_index < kRtBlasCachePoolCount; ++pool_index) {
        claimed[pool_index].resize(current_state.pools[pool_index].size(), false);
    }

    for (std::size_t item_index = 0; item_index < build_plan.items.size(); ++item_index) {
        const rt_acceleration_build_item &item = build_plan.items[item_index];
        const std::size_t pool_index = static_cast<std::size_t>(item.kind);
        if (pool_index >= kRtBlasCachePoolCount || item.group.chunk_count == 0 ||
            item.group.chunk_count > kRtBlasChunkSetChunkCount) {
            return false;
        }
        keys[item_index] = make_rt_blas_cache_key(item);
        const std::vector<rt_blas_cache_slot> &slots = current_state.pools[pool_index];
        for (std::size_t slot_index = 0; slot_index < slots.size(); ++slot_index) {
            if (!claimed[pool_index][slot_index] && slots[slot_index].valid &&
                rt_blas_cache_key_equals(slots[slot_index].key, keys[item_index])) {
                claimed[pool_index][slot_index] = true;
                plan.assignments[item_index] = {slot_index, true};
                break;
            }
        }
    }

    for (std::size_t item_index = 0; item_index < build_plan.items.size(); ++item_index) {
        rt_blas_cache_assignment &assignment = plan.assignments[item_index];
        if (assignment.reuse_candidate) {
            continue;
        }
        const std::size_t pool_index = static_cast<std::size_t>(build_plan.items[item_index].kind);
        std::vector<rt_blas_cache_slot> &slots = plan.next_state.pools[pool_index];
        std::vector<bool> &pool_claimed = claimed[pool_index];
        std::size_t cache_index = slots.size();
        for (std::size_t slot_index = 0; slot_index < slots.size(); ++slot_index) {
            const rt_blas_cache_slot &slot = slots[slot_index];
            const bool expires_after_update =
                slot.unused_revision_count + 1u >= kRtBlasCacheRetentionRevisions;
            if (!pool_claimed[slot_index] && (!slot.valid || expires_after_update)) {
                cache_index = slot_index;
                break;
            }
        }
        if (cache_index == slots.size()) {
            slots.push_back({});
            pool_claimed.push_back(false);
        }
        pool_claimed[cache_index] = true;
        assignment.cache_index = cache_index;
    }

    for (std::size_t item_index = 0; item_index < build_plan.items.size(); ++item_index) {
        const std::size_t pool_index = static_cast<std::size_t>(build_plan.items[item_index].kind);
        const std::size_t cache_index = plan.assignments[item_index].cache_index;
        rt_blas_cache_slot &slot = plan.next_state.pools[pool_index][cache_index];
        slot.key = keys[item_index];
        slot.unused_revision_count = 0;
        slot.valid = true;
    }
    for (std::size_t pool_index = 0; pool_index < kRtBlasCachePoolCount; ++pool_index) {
        std::vector<rt_blas_cache_slot> &slots = plan.next_state.pools[pool_index];
        const std::vector<bool> &pool_claimed = claimed[pool_index];
        for (std::size_t slot_index = 0; slot_index < slots.size(); ++slot_index) {
            if (!pool_claimed[slot_index]) {
                rt_blas_cache_slot &slot = slots[slot_index];
                if (slot.valid &&
                    ++slot.unused_revision_count >= kRtBlasCacheRetentionRevisions) {
                    slot.key = {};
                    slot.storage_key = {};
                    slot.storage_capacity_bytes = 0;
                    slot.acceleration = {};
                    slot.unused_revision_count = 0;
                    slot.valid = false;
                }
            }
        }
    }

    *out_plan = std::move(plan);
    return true;
}

bool grow_rt_capacity(
    std::size_t required,
    std::size_t current,
    std::size_t alignment,
    std::size_t* out_capacity)
{
    if (out_capacity == nullptr || alignment == 0) {
        return false;
    }
    if (required == 0) {
        *out_capacity = 0;
        return true;
    }

    const std::size_t max_size = (std::numeric_limits<std::size_t>::max)();
    const auto align_checked = [max_size, alignment](std::size_t value, std::size_t* out_value) {
        const std::size_t remainder = value % alignment;
        if (remainder == 0) {
            *out_value = value;
            return true;
        }
        const std::size_t padding = alignment - remainder;
        if (value > max_size - padding) {
            return false;
        }
        *out_value = value + padding;
        return true;
    };

    std::size_t aligned_required = 0;
    if (!align_checked(required, &aligned_required)) {
        return false;
    }
    std::size_t aligned_current = 0;
    if (!align_checked(current, &aligned_current)) {
        return false;
    }
    if (aligned_current >= aligned_required) {
        *out_capacity = aligned_current;
        return true;
    }

    const std::size_t half = aligned_current / 2u + (aligned_current % 2u);
    if (aligned_current > max_size - half) {
        return false;
    }
    const std::size_t grown = aligned_current + half;
    const std::size_t target = (std::max)(aligned_required, grown);
    return align_checked(target, out_capacity);
}

} // namespace rtvdb::viewer_backend
