#include "viewer_backend/rt_scene_builder.h"

#include <algorithm>
#include <bit>
#include <cmath>

namespace rtvdb::viewer_backend {
namespace {

constexpr std::uint64_t kFnvOffset = 1469598103934665603ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;
constexpr std::uint32_t kMortonAxisBits = 10;
constexpr std::uint32_t kMortonAxisMax = (1u << kMortonAxisBits) - 1u;

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

scene_bounds merge_bounds(const scene_bounds &a, const scene_bounds &b) {
    if (!a.valid) return b;
    if (!b.valid) return a;
    scene_bounds result{};
    result.valid = true;
    result.min = {(std::min)(a.min.x, b.min.x), (std::min)(a.min.y, b.min.y), (std::min)(a.min.z, b.min.z)};
    result.max = {(std::max)(a.max.x, b.max.x), (std::max)(a.max.y, b.max.y), (std::max)(a.max.z, b.max.z)};
    return result;
}

float bounds_diagonal_length(const scene_bounds &bounds) {
    if (!bounds.valid) return 0.0f;
    const rtvdb::vec3 d{bounds.max.x - bounds.min.x, bounds.max.y - bounds.min.y, bounds.max.z - bounds.min.z};
    return std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
}

rtvdb::vec3 bounds_center(const scene_bounds &bounds) {
    return {(bounds.min.x + bounds.max.x) * 0.5f, (bounds.min.y + bounds.max.y) * 0.5f, (bounds.min.z + bounds.max.z) * 0.5f};
}

bool can_append_blas_group(const scene_bounds &current, const scene_bounds &candidate) {
    constexpr float kCenterDistanceScale = 1.5f;
    constexpr float kMergedDiagonalScale = 2.5f;
    const scene_bounds merged = merge_bounds(current, candidate);
    const float scale = (std::max)(0.001f, (std::max)(bounds_diagonal_length(current), bounds_diagonal_length(candidate)));
    const rtvdb::vec3 a = bounds_center(current);
    const rtvdb::vec3 b = bounds_center(candidate);
    const float dx = b.x - a.x;
    const float dy = b.y - a.y;
    const float dz = b.z - a.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz) <= scale * kCenterDistanceScale &&
        bounds_diagonal_length(merged) <= scale * kMergedDiagonalScale;
}

std::uint32_t expand_morton_axis(std::uint32_t value) {
    value &= kMortonAxisMax;
    value = (value | (value << 16u)) & 0x030000ffu;
    value = (value | (value << 8u)) & 0x0300f00fu;
    value = (value | (value << 4u)) & 0x030c30c3u;
    value = (value | (value << 2u)) & 0x09249249u;
    return value;
}

std::uint32_t quantize_morton_axis(float value, float min_value, float max_value) {
    const float extent = max_value - min_value;
    if (!(extent > 0.0f)) {
        return kMortonAxisMax / 2u;
    }
    const float normalized = std::clamp((value - min_value) / extent, 0.0f, 1.0f);
    return static_cast<std::uint32_t>(normalized * static_cast<float>(kMortonAxisMax) + 0.5f);
}

std::uint32_t morton_code(const scene_bounds &bounds, const scene_bounds &center_bounds) {
    if (!bounds.valid || !center_bounds.valid) {
        return 0;
    }
    const rtvdb::vec3 center = bounds_center(bounds);
    const std::uint32_t x = quantize_morton_axis(center.x, center_bounds.min.x, center_bounds.max.x);
    const std::uint32_t y = quantize_morton_axis(center.y, center_bounds.min.y, center_bounds.max.y);
    const std::uint32_t z = quantize_morton_axis(center.z, center_bounds.min.z, center_bounds.max.z);
    return expand_morton_axis(x) | (expand_morton_axis(y) << 1u) | (expand_morton_axis(z) << 2u);
}

template <typename Chunk>
std::vector<rt_blas_chunk_set> build_blas_chunk_sets(const std::vector<Chunk> &chunks, bool require_sealed) {
    std::vector<rt_blas_chunk_set> result;
    std::vector<bool> claimed(chunks.size(), false);
    for (std::size_t bucket_start = 0; bucket_start < chunks.size(); ++bucket_start) {
        if (claimed[bucket_start] || (require_sealed && !chunks[bucket_start].sealed)) {
            continue;
        }

        const Chunk &bucket_source = chunks[bucket_start];
        std::vector<std::size_t> sorted_indices;
        scene_bounds center_bounds{};
        for (std::size_t chunk_index = bucket_start; chunk_index < chunks.size(); ++chunk_index) {
            const Chunk &candidate = chunks[chunk_index];
            if (claimed[chunk_index] || (require_sealed && !candidate.sealed) ||
                candidate.layer != bucket_source.layer || candidate.visible != bucket_source.visible) {
                continue;
            }
            claimed[chunk_index] = true;
            sorted_indices.push_back(chunk_index);
            expand_bounds(&center_bounds, bounds_center(candidate.bounds));
        }
        std::stable_sort(
            sorted_indices.begin(),
            sorted_indices.end(),
            [&](std::size_t a, std::size_t b) {
                const std::uint32_t a_code = morton_code(chunks[a].bounds, center_bounds);
                const std::uint32_t b_code = morton_code(chunks[b].bounds, center_bounds);
                return a_code != b_code ? a_code < b_code : a < b;
            });

        rt_blas_chunk_set pending{};
        scene_bounds pending_bounds{};
        const auto flush = [&]() {
            if (pending.chunk_count != 0) {
                result.push_back(pending);
            }
            pending = {};
            pending_bounds = {};
        };
        for (const std::size_t chunk_index : sorted_indices) {
            const Chunk &chunk = chunks[chunk_index];
            if (pending.chunk_count == kRtBlasChunkSetChunkCount ||
                (pending.chunk_count != 0 && !can_append_blas_group(pending_bounds, chunk.bounds))) {
                flush();
            }
            pending.chunk_indices[pending.chunk_count++] = chunk_index;
            pending_bounds = merge_bounds(pending_bounds, chunk.bounds);
        }
        flush();
    }
    for (std::size_t chunk_index = 0; chunk_index < chunks.size(); ++chunk_index) {
        if (!claimed[chunk_index]) {
            rt_blas_chunk_set group{};
            group.chunk_indices[0] = chunk_index;
            group.chunk_count = 1;
            result.push_back(group);
        }
    }
    return result;
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
    rt_triangle_chunk* out_chunk,
    scene_bounds* out_build_bounds)
{
    if (out_chunk == nullptr) {
        return;
    }

    rt_triangle_chunk chunk{};
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

const rt_triangle_chunk* find_reusable_chunk(
    const rt_scene_build* previous_build,
    const rt_triangle_chunk &requested_chunk)
{
    if (previous_build == nullptr) {
        return nullptr;
    }

    for (const rt_triangle_chunk &candidate : previous_build->triangle_chunks) {
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
    rt_triangle_chunk* chunk,
    const rt_scene_build &previous_build,
    const rt_triangle_chunk &previous_chunk)
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

void append_new_chunk_geometry(const frame_scene &scene, rt_scene_build* build, rt_triangle_chunk* chunk) {
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
    std::size_t triangle_chunk_primitive_count,
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

    const std::size_t effective_chunk_size = (std::max)(std::size_t{1}, triangle_chunk_primitive_count);
    for (std::size_t first = 0; first < scene.triangles.size();) {
        std::size_t layer_end = first + 1;
        while (layer_end < scene.triangles.size() &&
               scene.triangles[layer_end].layer == scene.triangles[first].layer &&
               scene.triangles[layer_end].visible == scene.triangles[first].visible) {
            ++layer_end;
        }
        const std::size_t triangle_count = (std::min)(effective_chunk_size, layer_end - first);
        rt_triangle_chunk chunk{};
        compute_chunk_metadata(
            scene,
            first,
            triangle_count,
            &chunk,
            &build.bounds);
        chunk.sealed = chunk.triangle_count == effective_chunk_size;

        const rt_triangle_chunk* reusable_chunk = find_reusable_chunk(previous_build, chunk);
        if (reusable_chunk != nullptr) {
            append_reused_chunk_geometry(&build, &chunk, *previous_build, *reusable_chunk);
            ++build.reused_triangle_chunk_count;
        } else {
            append_new_chunk_geometry(scene, &build, &chunk);
            ++build.rebuilt_triangle_chunk_count;
        }

        build.triangle_chunks.push_back(chunk);
        first += triangle_count;
    }

    const std::size_t effective_procedural_group_size =
        (std::max)(std::size_t{1}, kDefaultRtSceneProceduralChunkPrimitives);
    for (std::size_t first = 0; first < scene.points.size();) {
        std::size_t end = first + 1;
        while (end < scene.points.size() && end - first < effective_procedural_group_size &&
               scene.points[end].layer == scene.points[first].layer &&
               scene.points[end].visible == scene.points[first].visible) {
            ++end;
        }
        rt_procedural_chunk group{};
        group.first_primitive = first;
        group.primitive_count = end - first;
        group.layer = scene.points[first].layer;
        group.visible = scene.points[first].visible;
        for (std::size_t index = first; index < end; ++index) {
            expand_bounds(&group.bounds, scene.points[index]);
        }
        build.point_chunks.push_back(std::move(group));
        first = end;
    }
    for (const point &value : scene.points) {
        if (value.visible) {
            expand_bounds(&build.bounds, value);
        }
    }
    for (std::size_t first = 0; first < scene.lines.size();) {
        std::size_t end = first + 1;
        while (end < scene.lines.size() && end - first < effective_procedural_group_size &&
               scene.lines[end].layer == scene.lines[first].layer &&
               scene.lines[end].visible == scene.lines[first].visible) {
            ++end;
        }
        rt_procedural_chunk group{};
        group.first_primitive = first;
        group.primitive_count = end - first;
        group.layer = scene.lines[first].layer;
        group.visible = scene.lines[first].visible;
        for (std::size_t index = first; index < end; ++index) {
            expand_bounds(&group.bounds, scene.lines[index]);
        }
        build.line_chunks.push_back(std::move(group));
        first = end;
    }
    for (const line &value : scene.lines) {
        if (value.visible) {
            expand_bounds(&build.bounds, value);
        }
    }

    build.triangle_blas_chunk_sets = build_blas_chunk_sets(build.triangle_chunks, true);
    build.point_blas_chunk_sets = build_blas_chunk_sets(build.point_chunks, false);
    build.line_blas_chunk_sets = build_blas_chunk_sets(build.line_chunks, false);

    build.vertex_count = build.vertices.size();
    build.index_count = build.indices.size();
    *out_build = std::move(build);
    return true;
}

} // namespace rtvdb::viewer_backend
