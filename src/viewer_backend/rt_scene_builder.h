#pragma once

#include "viewer_backend/backend.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rtvdb::viewer_backend {

struct scene_bounds {
    rtvdb::vec3 min{};
    rtvdb::vec3 max{};
    bool valid = false;
};

struct rt_scene_vertex {
    rtvdb::vec3 position{};
    rtvdb::rgba color{};
};

struct rt_scene_chunk {
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

struct rt_procedural_group {
    std::size_t first_primitive = 0;
    std::size_t primitive_count = 0;
    std::uint64_t fingerprint = 0;
    std::string layer;
    bool visible = true;
};

struct rt_scene_build {
    std::uint64_t revision = 0;
    std::size_t triangle_count = 0;
    std::size_t point_count = 0;
    std::size_t line_count = 0;
    std::size_t vertex_count = 0;
    std::size_t index_count = 0;
    std::size_t reused_chunk_count = 0;
    std::size_t rebuilt_chunk_count = 0;
    scene_bounds bounds{};
    std::vector<rt_scene_vertex> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<rt_scene_chunk> chunks;
    std::vector<point> points;
    std::vector<line> lines;
    std::vector<rt_procedural_group> point_groups;
    std::vector<rt_procedural_group> line_groups;
};

constexpr std::size_t kDefaultRtSceneChunkTriangles = 8192;

bool build_rt_scene_input(
    const frame_scene &scene,
    const rt_scene_build* previous_build,
    std::uint64_t revision,
    std::size_t chunk_triangle_count,
    rt_scene_build* out_build);

} // namespace rtvdb::viewer_backend
