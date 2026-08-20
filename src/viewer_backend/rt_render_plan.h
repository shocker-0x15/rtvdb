#pragma once

#include "viewer_backend/backend.h"
#include "viewer_backend/rt_rhi.h"
#include "viewer_backend/rt_scene_builder.h"

#include <array>
#include <cstddef>
#include <cstdint>

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
constexpr std::size_t kRtViewerConstantSlotStride = 256;
constexpr std::size_t kRtViewerConstantBufferBytes =
    static_cast<std::size_t>(kRtCommandSlotCount) * kRtViewerConstantSlotStride;
constexpr std::size_t kViewerRtShaderEntryCount = 8;

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
    rt_pipeline_model model = rt_pipeline_model::native_ray_tracing;
    std::array<rt_binding_layout_desc, kViewerRtBindingCount> bindings{};
    std::array<rt_shader_entry_desc, kViewerRtShaderEntryCount> shaders{};
    std::array<rt_shader_group_desc, kViewerRtShaderGroupCount> groups{};
    std::array<rt_pipeline_dispatch_entry_desc, 2> dispatch_entries{};
    std::size_t shader_count = 0;
    std::size_t group_count = 0;

    rt_pipeline_desc pipeline() const;
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
    rt_pipeline_model model,
    const std::array<rt_shader_module_handle, kViewerRtShaderEntryCount> &entry_modules);
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
    float render_scale_x = 1.0f;
    float render_scale_y = 1.0f;
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
    std::uint32_t selection_highlight_kind = 0;
    std::uint32_t selection_primitive_index = 0;
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
    float render_scale[4]{1.0f, 1.0f, 0.0f, 0.0f};
};

static_assert(sizeof(rt_viewer_constants_gpu) == 224);

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
    std::uint32_t selection_highlight_kind,
    std::uint32_t selection_primitive_index,
    bool is_pick_pass = false,
    int pick_pixel_x = 0,
    int pick_pixel_y = 0,
    float render_scale_x = 1.0f,
    float render_scale_y = 1.0f);
rt_viewer_constants_gpu pack_rt_viewer_constants(const rt_viewer_constants &source);

struct rt_accumulation_key {
    std::uint64_t build_revision = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t display_mode = 0;
    std::uint32_t has_frame = 0;
    std::uint32_t hover_highlight_kind = 0;
    std::uint32_t hover_primitive_index = 0;
    std::uint32_t selection_highlight_kind = 0;
    std::uint32_t selection_primitive_index = 0;
    float render_scale_x = 1.0f;
    float render_scale_y = 1.0f;
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
    std::uint32_t submitted_sample_count = 0;
    std::uint64_t generation = 0;
    bool active = false;
};

bool rt_accumulation_key_equals(const rt_accumulation_key &a, const rt_accumulation_key &b);
void reset_rt_accumulation_state(rt_accumulation_state* state);
bool begin_rt_accumulation(
    rt_accumulation_state* state,
    const rt_accumulation_key &next_key,
    bool continuous_render);
bool get_rt_accumulation_submission(
    const rt_accumulation_state &state,
    std::uint64_t* out_generation,
    std::uint32_t* out_sample_index);
bool submit_rt_accumulation_sample(
    rt_accumulation_state* state,
    std::uint64_t generation,
    std::uint32_t sample_index,
    bool continuous_render);
bool complete_rt_accumulation_sample(
    rt_accumulation_state* state,
    std::uint64_t generation,
    std::uint32_t sample_index,
    bool continuous_render);
rt_accumulation_key make_rt_accumulation_key(
    const frame_scene &scene,
    bool has_frame,
    const rt_scene_build &build,
    int width,
    int height,
    std::uint32_t display_mode,
    float render_scale_x = 1.0f,
    float render_scale_y = 1.0f);
void fill_rt_accumulation_jitter(std::uint32_t sample_index, float out_jitter[2]);
void fill_rt_projection_parameters(
    const rtvdb::camera &camera,
    rtvdb::camera_projection projection,
    float aspect,
    float* out_param0,
    float* out_param1);

bool rt_scene_has_renderable_primitives(const rt_scene_build &build);

} // namespace rtvdb::viewer_backend
