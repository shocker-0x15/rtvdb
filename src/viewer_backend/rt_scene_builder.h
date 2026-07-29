#pragma once

#include "viewer_backend/backend.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace rtvdb::viewer_backend {

constexpr std::size_t kRtBlasChunkSetChunkCount = 4;

struct scene_bounds {
    rtvdb::vec3 min{};
    rtvdb::vec3 max{};
    bool valid = false;
};

struct rt_scene_vertex {
    rtvdb::vec3 position{};
    rtvdb::rgba color{};
};

struct rt_triangle_chunk {
    std::size_t first_triangle = 0;
    std::size_t triangle_count = 0;
    bool sealed = false;
    std::size_t vertex_offset = 0;
    std::size_t vertex_count = 0;
    std::size_t index_offset = 0;
    std::size_t index_count = 0;
    std::uint64_t fingerprint = 0;
    scene_bounds bounds{};
    std::string layer;
    bool visible = true;
};

struct rt_procedural_chunk {
    std::size_t first_primitive = 0;
    std::size_t primitive_count = 0;
    bool sealed = true;
    std::uint64_t fingerprint = 0;
    scene_bounds bounds{};
    std::string layer;
    bool visible = true;
};

struct rt_blas_chunk_set {
    std::array<std::size_t, kRtBlasChunkSetChunkCount> chunk_indices{};
    std::size_t chunk_count = 0;
};

struct rt_scene_build {
    std::uint64_t revision = 0;
    std::size_t triangle_count = 0;
    std::size_t point_count = 0;
    std::size_t line_count = 0;
    std::size_t vertex_count = 0;
    std::size_t index_count = 0;
    std::size_t reused_triangle_chunk_count = 0;
    std::size_t rebuilt_triangle_chunk_count = 0;
    scene_bounds bounds{};
    std::vector<rt_scene_vertex> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<rt_triangle_chunk> triangle_chunks;
    std::vector<point> points;
    std::vector<line> lines;
    std::vector<rt_procedural_chunk> point_chunks;
    std::vector<rt_procedural_chunk> line_chunks;
    std::vector<rt_blas_chunk_set> triangle_blas_chunk_sets;
    std::vector<rt_blas_chunk_set> point_blas_chunk_sets;
    std::vector<rt_blas_chunk_set> line_blas_chunk_sets;
};

constexpr std::size_t kDefaultRtSceneTriangleChunkPrimitives = 8192;
constexpr std::size_t kDefaultRtSceneProceduralChunkPrimitives = 8192;

bool build_rt_scene_input(
    const frame_scene &scene,
    const rt_scene_build* previous_build,
    std::uint64_t revision,
    std::size_t triangle_chunk_primitive_count,
    rt_scene_build* out_build);

} // namespace rtvdb::viewer_backend
