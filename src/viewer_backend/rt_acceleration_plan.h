#pragma once

#include "viewer_backend/rt_rhi.h"
#include "viewer_backend/rt_scene_builder.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace rtvdb::viewer_backend {

struct rt_procedural_geometry_update_plan {
    std::uint64_t point_fingerprint = 0;
    std::uint64_t line_fingerprint = 0;
    bool point_geometry_changed = false;
    bool line_geometry_changed = false;
};

enum class rt_acceleration_geometry_kind : std::uint32_t {
    triangle,
    point,
    line,
};

struct rt_acceleration_build_item {
    rt_acceleration_geometry_kind kind = rt_acceleration_geometry_kind::triangle;
    std::size_t group_index = 0;
    rt_blas_chunk_set group{};
    std::array<std::uint64_t, kRtBlasChunkSetChunkCount> geometry_fingerprints{};
    std::array<std::size_t, kRtBlasChunkSetChunkCount> first_primitives{};
    std::array<std::size_t, kRtBlasChunkSetChunkCount> primitive_counts{};
};

struct rt_blas_cache_key {
    rt_acceleration_geometry_kind kind = rt_acceleration_geometry_kind::triangle;
    std::array<std::size_t, kRtBlasChunkSetChunkCount> first_primitives{};
    std::array<std::size_t, kRtBlasChunkSetChunkCount> primitive_counts{};
    std::array<std::uint64_t, kRtBlasChunkSetChunkCount> geometry_fingerprints{};
    std::size_t geometry_count = 0;
};

struct rt_blas_storage_key {
    rt_acceleration_geometry_kind kind = rt_acceleration_geometry_kind::triangle;
    std::array<rt_acceleration_geometry_type, kRtBlasChunkSetChunkCount> geometry_types{};
    std::array<std::uint32_t, kRtBlasChunkSetChunkCount> geometry_flags{};
    std::array<std::uint32_t, kRtBlasChunkSetChunkCount> geometry_formats{};
    std::array<std::size_t, kRtBlasChunkSetChunkCount> primitive_counts{};
    std::array<std::size_t, kRtBlasChunkSetChunkCount> strides{};
    std::size_t geometry_count = 0;
    std::uint32_t build_flags = rt_acceleration_build_prefer_fast_trace;
};

constexpr std::size_t kRtBlasCachePoolCount = 3;
constexpr std::uint8_t kRtBlasCacheRetentionRevisions = 4;

struct rt_blas_cache_slot {
    rt_blas_cache_key key{};
    rt_blas_storage_key storage_key{};
    std::size_t storage_capacity_bytes = 0;
    rt_blas_handle acceleration{};
    std::uint8_t unused_revision_count = 0;
    bool valid = false;
};

struct rt_blas_cache_state {
    std::array<std::vector<rt_blas_cache_slot>, kRtBlasCachePoolCount> pools;
};

struct rt_blas_cache_assignment {
    std::size_t cache_index = 0;
    bool reuse_candidate = false;
};

struct rt_blas_cache_update_plan {
    std::uint64_t revision = 0;
    std::vector<rt_blas_cache_assignment> assignments;
    rt_blas_cache_state next_state{};
};

struct rt_acceleration_build_plan {
    std::uint64_t revision = 0;
    std::vector<rt_acceleration_build_item> items;
    std::size_t triangle_item_count = 0;
    std::size_t point_item_count = 0;
    std::size_t line_item_count = 0;
    std::uint64_t point_geometry_fingerprint = 0;
    std::uint64_t line_geometry_fingerprint = 0;
    bool build_tlas = false;
};

struct rt_scene_gpu_position {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 0.0f;
};

struct rt_scene_gpu_point {
    float position_radius[4]{};
    float color[4]{};
};

struct rt_scene_gpu_line {
    float a_radius[4]{};
    float b_pad[4]{};
    float color[4]{};
    std::uint32_t flags = 0;
    float padding[3]{};
};

struct rt_scene_gpu_aabb {
    float min_x = 0.0f;
    float min_y = 0.0f;
    float min_z = 0.0f;
    float max_x = 0.0f;
    float max_y = 0.0f;
    float max_z = 0.0f;
};

struct rt_scene_geometry_metadata {
    std::uint32_t primitive_base = 0;
    std::uint32_t index_offset = 0;
    std::uint32_t primitive_offset = 0;
    std::uint32_t primitive_count = 0;
};


struct rt_scene_resource_data {
    std::uint64_t revision = 0;
    std::uint64_t connection_serial = 0;
    std::vector<rt_scene_gpu_position> positions;
    std::vector<std::uint32_t> indices;
    std::vector<rtvdb::rgba> triangle_colors;
    std::vector<rt_scene_gpu_point> points;
    std::vector<rt_scene_gpu_line> lines;
    std::vector<rt_scene_gpu_aabb> point_aabbs;
    std::vector<rt_scene_gpu_aabb> line_aabbs;
    std::vector<rt_scene_geometry_metadata> instance_geometry;
};

struct rt_blas_build_command {
    rt_acceleration_geometry_kind kind = rt_acceleration_geometry_kind::triangle;
    std::array<rt_acceleration_geometry_desc, kRtBlasChunkSetChunkCount> geometries{};
    std::size_t geometry_count = 0;
    std::array<std::size_t, kRtBlasChunkSetChunkCount> allocation_primitive_counts{};
    rt_blas_handle destination{};
    std::uint32_t instance_index = 0;
    std::uint32_t hit_group_contribution = 0;
    bool visible = false;
};

struct rt_acceleration_command_plan {
    std::uint64_t revision = 0;
    std::vector<rt_blas_build_command> blas_commands;
    bool build_tlas = false;
};

static_assert(sizeof(rt_scene_gpu_position) == 16);
static_assert(sizeof(rt_scene_gpu_point) == 32);
static_assert(sizeof(rt_scene_gpu_line) == 64);
static_assert(sizeof(rt_scene_gpu_aabb) == 24);

bool make_rt_acceleration_build_plan(
    const rt_scene_build &build,
    rt_acceleration_build_plan* out_plan);
bool make_rt_scene_resource_data(
    const rt_scene_build &build,
    const rt_acceleration_build_plan &plan,
    rt_scene_resource_data* out_data);
bool make_rt_acceleration_command_plan(
    const rt_scene_build &build,
    const rt_acceleration_build_plan &build_plan,
    const rt_blas_cache_update_plan &cache_plan,
    const rt_scene_resource_data &resources,
    rt_acceleration_command_plan* out_plan);
std::uint64_t rt_procedural_chunk_fingerprint(
    const rt_scene_build &build,
    const rt_procedural_chunk &group,
    bool points);
std::uint64_t rt_point_geometry_fingerprint(const rt_scene_build &build);
std::uint64_t rt_line_geometry_fingerprint(const rt_scene_build &build);
rt_procedural_geometry_update_plan make_rt_procedural_geometry_update_plan(
    const rt_scene_build &build,
    std::size_t cached_point_count,
    std::uint64_t cached_point_fingerprint,
    std::size_t cached_line_count,
    std::uint64_t cached_line_fingerprint);
rt_blas_cache_key make_rt_blas_cache_key(const rt_acceleration_build_item &item);
bool rt_blas_cache_key_equals(const rt_blas_cache_key &a, const rt_blas_cache_key &b);
bool rt_blas_storage_key_equals(const rt_blas_storage_key &a, const rt_blas_storage_key &b);
rt_blas_storage_key make_rt_blas_storage_key(
    const rt_blas_build_command &command,
    std::uint32_t build_flags);
bool make_rt_blas_cache_update_plan(
    const rt_acceleration_build_plan &build_plan,
    const rt_blas_cache_state &current_state,
    rt_blas_cache_update_plan* out_plan);
bool grow_rt_capacity(
    std::size_t required,
    std::size_t current,
    std::size_t alignment,
    std::size_t* out_capacity);

} // namespace rtvdb::viewer_backend
