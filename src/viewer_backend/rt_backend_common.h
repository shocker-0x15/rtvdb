#pragma once

#include "viewer_backend/rt_rhi.h"
#include "viewer_backend/rt_scene_builder.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace rtvdb::viewer_backend {

enum class viewer_rt_binding : std::uint32_t {
    scene = 0,
    output = 1,
    triangle_colors = 2,
    instance_metadata = 3,
    positions = 4,
    indices = 5,
    points = 6,
    lines = 7,
    viewer_constants = 8,
    pick_output = 9,
    accumulation = 10,
    count,
};

constexpr std::uint32_t viewer_rt_binding_index(viewer_rt_binding binding) {
    return static_cast<std::uint32_t>(binding);
}

constexpr std::size_t kViewerRtBindingCount =
    static_cast<std::size_t>(viewer_rt_binding::count);

enum class viewer_rt_shader_group : std::uint32_t {
    raygen = 0,
    pick_raygen,
    miss,
    triangle_hit,
    point_hit,
    line_hit,
    count,
};

constexpr std::uint32_t viewer_rt_shader_group_index(viewer_rt_shader_group group) {
    return static_cast<std::uint32_t>(group);
}

constexpr std::uint32_t kViewerRtShaderGroupCount =
    viewer_rt_shader_group_index(viewer_rt_shader_group::count);
constexpr std::uint32_t kViewerRtHitGroupCount =
    viewer_rt_shader_group_index(viewer_rt_shader_group::count) -
    viewer_rt_shader_group_index(viewer_rt_shader_group::triangle_hit);
constexpr std::uint32_t viewer_rt_hit_group_contribution(viewer_rt_shader_group group) {
    return viewer_rt_shader_group_index(group) -
        viewer_rt_shader_group_index(viewer_rt_shader_group::triangle_hit);
}

struct viewer_rt_pipeline_desc {
    std::array<rt_binding_layout_desc, kViewerRtBindingCount> bindings{};
    std::array<rt_shader_entry_desc, 8> shaders{};
    std::array<rt_shader_group_desc, kViewerRtShaderGroupCount> groups{};
    std::array<std::uint32_t, 2> ray_generation_groups{};
    std::array<std::uint32_t, 1> miss_groups{};
    std::array<std::uint32_t, kViewerRtHitGroupCount> hit_groups{};

    rt_pipeline_desc pipeline() const;
    rt_shader_table_desc shader_table(rt_pipeline_handle pipeline_handle) const;
};

enum class rt_dispatch_kind : std::uint32_t {
    clear,
    render,
    pick,
};

struct rt_dispatch_plan {
    rt_dispatch_kind kind = rt_dispatch_kind::clear;
    bool uses_accumulation = false;
};

viewer_rt_pipeline_desc make_viewer_rt_pipeline_desc(
    const rt_shader_module_handle* modules,
    std::size_t module_count);
rt_dispatch_plan make_rt_dispatch_plan(const rt_scene_build &build, bool pick_pass);
constexpr std::uint32_t kRtMaxAccumulationSamples = 64;

struct rt_viewer_constants {
    float origin[3]{};
    float forward[3]{};
    float right[3]{};
    float up[3]{};
    float scene_bounds_min[3]{};
    float scene_bounds_max[3]{};
    float projection_from[2]{};
    float projection_to[2]{};
    float accumulation_jitter[2]{};
    float aspect = 1.0f;
    float projection_blend_t = 1.0f;
    float hover_highlight_mix = 0.85f;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t display_mode = 0;
    std::uint32_t accumulation_sample_index = 0;
    std::uint32_t projection = 0;
    std::uint32_t projection_blend_from = 0;
    std::uint32_t projection_blend_to = 0;
    std::uint32_t scene_bounds_valid = 0;
    std::uint32_t hover_highlight_kind = 0;
    std::uint32_t hover_primitive_index = 0;
    std::uint32_t pick_pixel_x = 0;
    std::uint32_t pick_pixel_y = 0;
    std::uint32_t is_pick_pass = 0;
    std::uint32_t has_frame = 0;
    std::uint32_t triangle_count = 0;
    std::uint32_t point_count = 0;
    std::uint32_t line_count = 0;
};

struct rt_viewer_constants_gpu {
    float origin[4]{};
    float forward[4]{};
    float right[4]{};
    float up[4]{};
    float scene_bounds_min[4]{};
    float scene_bounds_max[4]{};
    std::uint32_t size_and_mode[4]{};
    float projection_from[4]{};
    float projection_to[4]{};
    std::uint32_t projection_modes[4]{};
    float blend_and_jitter[4]{};
    std::uint32_t pick_and_flags[4]{};
    std::uint32_t pick_params[4]{};
};

static_assert(sizeof(rt_viewer_constants_gpu) == 208);

rt_viewer_constants make_rt_viewer_constants(
    const frame_scene &scene,
    bool has_frame,
    const rt_scene_build &build,
    int width,
    int height,
    std::uint32_t display_mode,
    std::uint32_t accumulation_sample_index,
    std::uint32_t hover_highlight_kind,
    std::uint32_t hover_primitive_index,
    bool is_pick_pass = false,
    int pick_pixel_x = 0,
    int pick_pixel_y = 0);
rt_viewer_constants_gpu pack_rt_viewer_constants(const rt_viewer_constants &source);

struct rt_accumulation_key {
    std::uint64_t build_revision = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t display_mode = 0;
    std::uint32_t has_frame = 0;
    std::uint32_t hover_highlight_kind = 0;
    std::uint32_t hover_primitive_index = 0;
    float camera_origin[3]{};
    float camera_target[3]{};
    float camera_up[3]{};
    float camera_vertical_fov_degrees = 0.0f;
    std::uint32_t camera_projection = 0;
    float camera_fisheye_theta_degrees = 0.0f;
    float camera_fisheye_phi_degrees = 0.0f;
    float camera_orthographic_height = 0.0f;
    std::uint32_t camera_projection_blend_from = 0;
    std::uint32_t camera_projection_blend_to = 0;
    float camera_projection_blend_t = 1.0f;
};

struct rt_accumulation_state {
    rt_accumulation_key key{};
    std::uint32_t sample_count = 0;
    bool active = false;
};

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

bool rt_accumulation_key_equals(const rt_accumulation_key &a, const rt_accumulation_key &b);
void reset_rt_accumulation_state(rt_accumulation_state* state);
bool begin_rt_accumulation(
    rt_accumulation_state* state,
    const rt_accumulation_key &next_key,
    bool continuous_render);
void complete_rt_accumulation(rt_accumulation_state* state, bool continuous_render);
rt_accumulation_key make_rt_accumulation_key(
    const frame_scene &scene,
    bool has_frame,
    const rt_scene_build &build,
    int width,
    int height,
    std::uint32_t display_mode);
void fill_rt_accumulation_jitter(std::uint32_t sample_index, float out_jitter[2]);
void fill_rt_projection_parameters(
    const rtvdb::camera &camera,
    rtvdb::camera_projection projection,
    float aspect,
    float* out_param0,
    float* out_param1);

bool rt_scene_has_renderable_primitives(const rt_scene_build &build);
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
void copy_rt_diagnostics(scene_build_info* out_info, const scene_build_info &diagnostics);
void append_rt_diagnostics_log_line(std::string_view filename, std::string_view text);
void append_rt_startup_log(std::string_view text);

bool grow_rt_capacity(
    std::size_t required,
    std::size_t current,
    std::size_t alignment,
    std::size_t* out_capacity);

} // namespace rtvdb::viewer_backend
