#include "viewer_backend/rt_scene_builder.h"

#include <algorithm>
#include <bit>

namespace rtvdb::viewer_backend {
namespace {

constexpr std::uint64_t kFnvOffset = 1469598103934665603ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

void expand_bounds(scene_bounds* bounds, const rtvdb::vec3 &point) {
    if (bounds == nullptr) {
        return;
    }
    if (!bounds->valid) {
        bounds->min = point;
        bounds->max = point;
        bounds->valid = true;
        return;
    }

    bounds->min.x = (std::min)(bounds->min.x, point.x);
    bounds->min.y = (std::min)(bounds->min.y, point.y);
    bounds->min.z = (std::min)(bounds->min.z, point.z);
    bounds->max.x = (std::max)(bounds->max.x, point.x);
    bounds->max.y = (std::max)(bounds->max.y, point.y);
    bounds->max.z = (std::max)(bounds->max.z, point.z);
}

void expand_bounds(scene_bounds* bounds, const triangle &tri) {
    expand_bounds(bounds, tri.a);
    expand_bounds(bounds, tri.b);
    expand_bounds(bounds, tri.c);
}

void expand_bounds(scene_bounds* bounds, const point &value) {
    const rtvdb::vec3 extent{value.radius, value.radius, value.radius};
    expand_bounds(
        bounds,
        rtvdb::vec3{value.position.x - extent.x, value.position.y - extent.y, value.position.z - extent.z});
    expand_bounds(
        bounds,
        rtvdb::vec3{value.position.x + extent.x, value.position.y + extent.y, value.position.z + extent.z});
}

void expand_bounds(scene_bounds* bounds, const line &value) {
    const rtvdb::vec3 extent{value.radius, value.radius, value.radius};
    expand_bounds(bounds, rtvdb::vec3{value.a.x - extent.x, value.a.y - extent.y, value.a.z - extent.z});
    expand_bounds(bounds, rtvdb::vec3{value.a.x + extent.x, value.a.y + extent.y, value.a.z + extent.z});
    expand_bounds(bounds, rtvdb::vec3{value.b.x - extent.x, value.b.y - extent.y, value.b.z - extent.z});
    expand_bounds(bounds, rtvdb::vec3{value.b.x + extent.x, value.b.y + extent.y, value.b.z + extent.z});
}

void hash_u32(std::uint64_t* state, std::uint32_t value) {
    if (state == nullptr) {
        return;
    }
    *state ^= value;
    *state *= kFnvPrime;
}

void hash_f32(std::uint64_t* state, float value) {
    hash_u32(state, std::bit_cast<std::uint32_t>(value));
}

void hash_vec3(std::uint64_t* state, const rtvdb::vec3 &value) {
    hash_f32(state, value.x);
    hash_f32(state, value.y);
    hash_f32(state, value.z);
}

void hash_rgba(std::uint64_t* state, const rtvdb::rgba &value) {
    hash_f32(state, value.r);
    hash_f32(state, value.g);
    hash_f32(state, value.b);
    hash_f32(state, value.a);
}

void compute_chunk_metadata(
    const frame_scene &scene,
    std::size_t first_triangle,
    std::size_t triangle_count,
    rt_scene_chunk* out_chunk,
    scene_bounds* out_build_bounds)
{
    if (out_chunk == nullptr) {
        return;
    }

    rt_scene_chunk chunk{};
    chunk.first_triangle = first_triangle;
    chunk.triangle_count = triangle_count;
    chunk.layer = scene.triangles[first_triangle].layer;
    chunk.visible = scene.triangles[first_triangle].visible;
    chunk.fingerprint = kFnvOffset;
    for (std::size_t i = 0; i < triangle_count; ++i) {
        const triangle &tri = scene.triangles[first_triangle + i];
        expand_bounds(&chunk.bounds, tri);
        if (tri.visible) {
            expand_bounds(out_build_bounds, tri);
        }
        hash_vec3(&chunk.fingerprint, tri.a);
        hash_vec3(&chunk.fingerprint, tri.b);
        hash_vec3(&chunk.fingerprint, tri.c);
        hash_rgba(&chunk.fingerprint, tri.color);
    }
    *out_chunk = chunk;
}

const rt_scene_chunk* find_reusable_chunk(const rt_scene_build* previous_build, const rt_scene_chunk &requested_chunk) {
    if (previous_build == nullptr) {
        return nullptr;
    }

    for (const rt_scene_chunk &candidate : previous_build->chunks) {
        if (candidate.first_triangle != requested_chunk.first_triangle) {
            continue;
        }
        if (candidate.triangle_count != requested_chunk.triangle_count) {
            continue;
        }
        if (candidate.layer != requested_chunk.layer) {
            continue;
        }
        if (candidate.fingerprint != requested_chunk.fingerprint) {
            continue;
        }
        return &candidate;
    }
    return nullptr;
}

void append_reused_chunk_geometry(
    rt_scene_build* build,
    rt_scene_chunk* chunk,
    const rt_scene_build &previous_build,
    const rt_scene_chunk &previous_chunk)
{
    if (build == nullptr || chunk == nullptr) {
        return;
    }

    chunk->vertex_offset = build->vertices.size();
    chunk->index_offset = build->indices.size();

    build->vertices.insert(
        build->vertices.end(),
        previous_build.vertices.begin() + static_cast<std::ptrdiff_t>(previous_chunk.vertex_offset),
        previous_build.vertices.begin() +
            static_cast<std::ptrdiff_t>(previous_chunk.vertex_offset + previous_chunk.vertex_count));

    const std::uint32_t base_vertex = static_cast<std::uint32_t>(chunk->vertex_offset);
    for (std::size_t i = 0; i < previous_chunk.index_count; ++i) {
        const std::uint32_t old_index = previous_build.indices[previous_chunk.index_offset + i];
        build->indices.push_back(base_vertex + (old_index - static_cast<std::uint32_t>(previous_chunk.vertex_offset)));
    }

    chunk->vertex_count = previous_chunk.vertex_count;
    chunk->index_count = previous_chunk.index_count;
}

void append_new_chunk_geometry(const frame_scene &scene, rt_scene_build* build, rt_scene_chunk* chunk) {
    if (build == nullptr || chunk == nullptr) {
        return;
    }

    chunk->vertex_offset = build->vertices.size();
    chunk->index_offset = build->indices.size();
    for (std::size_t i = 0; i < chunk->triangle_count; ++i) {
        const triangle &tri = scene.triangles[chunk->first_triangle + i];
        const std::uint32_t base_vertex = static_cast<std::uint32_t>(build->vertices.size());
        build->vertices.push_back({tri.a, tri.color});
        build->vertices.push_back({tri.b, tri.color});
        build->vertices.push_back({tri.c, tri.color});
        build->indices.push_back(base_vertex + 0);
        build->indices.push_back(base_vertex + 1);
        build->indices.push_back(base_vertex + 2);
    }
    chunk->vertex_count = build->vertices.size() - chunk->vertex_offset;
    chunk->index_count = build->indices.size() - chunk->index_offset;
}

} // namespace

bool build_rt_scene_input(
    const frame_scene &scene,
    const rt_scene_build* previous_build,
    std::uint64_t revision,
    std::size_t chunk_triangle_count,
    rt_scene_build* out_build)
{
    if (out_build == nullptr) {
        return false;
    }

    rt_scene_build build{};
    build.revision = revision;
    build.triangle_count = scene.triangles.size();
    build.point_count = scene.points.size();
    build.line_count = scene.lines.size();
    build.vertices.reserve(scene.triangles.size() * 3);
    build.indices.reserve(scene.triangles.size() * 3);
    build.points = scene.points;
    build.lines = scene.lines;

    const std::size_t effective_chunk_size = (std::max)(std::size_t{1}, chunk_triangle_count);
    for (std::size_t first = 0; first < scene.triangles.size();) {
        std::size_t layer_end = first + 1;
        while (layer_end < scene.triangles.size() &&
               scene.triangles[layer_end].layer == scene.triangles[first].layer &&
               scene.triangles[layer_end].visible == scene.triangles[first].visible) {
            ++layer_end;
        }
        const std::size_t triangle_count = (std::min)(effective_chunk_size, layer_end - first);
        rt_scene_chunk chunk{};
        compute_chunk_metadata(
            scene,
            first,
            triangle_count,
            &chunk,
            &build.bounds);
        chunk.sealed = chunk.triangle_count == effective_chunk_size;

        const rt_scene_chunk* reusable_chunk = find_reusable_chunk(previous_build, chunk);
        if (reusable_chunk != nullptr) {
            append_reused_chunk_geometry(&build, &chunk, *previous_build, *reusable_chunk);
            ++build.reused_chunk_count;
        } else {
            append_new_chunk_geometry(scene, &build, &chunk);
            ++build.rebuilt_chunk_count;
        }

        build.chunks.push_back(chunk);
        first += triangle_count;
    }

    for (std::size_t first = 0; first < scene.points.size();) {
        std::size_t end = first + 1;
        while (end < scene.points.size() && scene.points[end].layer == scene.points[first].layer &&
               scene.points[end].visible == scene.points[first].visible) {
            ++end;
        }
        rt_procedural_group group{};
        group.first_primitive = first;
        group.primitive_count = end - first;
        group.layer = scene.points[first].layer;
        group.visible = scene.points[first].visible;
        build.point_groups.push_back(std::move(group));
        first = end;
    }
    for (const point &value : scene.points) {
        if (value.visible) {
            expand_bounds(&build.bounds, value);
        }
    }
    for (std::size_t first = 0; first < scene.lines.size();) {
        std::size_t end = first + 1;
        while (end < scene.lines.size() && scene.lines[end].layer == scene.lines[first].layer &&
               scene.lines[end].visible == scene.lines[first].visible) {
            ++end;
        }
        rt_procedural_group group{};
        group.first_primitive = first;
        group.primitive_count = end - first;
        group.layer = scene.lines[first].layer;
        group.visible = scene.lines[first].visible;
        build.line_groups.push_back(std::move(group));
        first = end;
    }
    for (const line &value : scene.lines) {
        if (value.visible) {
            expand_bounds(&build.bounds, value);
        }
    }

    build.vertex_count = build.vertices.size();
    build.index_count = build.indices.size();
    *out_build = std::move(build);
    return true;
}

} // namespace rtvdb::viewer_backend
