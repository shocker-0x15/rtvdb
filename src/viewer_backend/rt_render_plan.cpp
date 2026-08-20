#include "viewer_backend/rt_render_plan.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace rtvdb::viewer_backend {

rt_pipeline_desc viewer_rt_pipeline_desc::pipeline() const {
    return {
        model,
        bindings.data(),
        bindings.size(),
        shaders.data(),
        shader_count,
        group_count != 0 ? groups.data() : nullptr,
        group_count,
        dispatch_entries.data(),
        dispatch_entries.size(),
        model == rt_pipeline_model::native_ray_tracing ? 1u : 0u,
        model == rt_pipeline_model::native_ray_tracing
            ? static_cast<std::uint32_t>(sizeof(float) * 8u)
            : 0u,
        model == rt_pipeline_model::native_ray_tracing
            ? static_cast<std::uint32_t>(sizeof(float) * 2u)
            : 0u};
}

viewer_rt_pipeline_desc make_viewer_rt_pipeline_desc(
    rt_pipeline_model model,
    const std::array<rt_shader_module_handle, kViewerRtShaderEntryCount> &entry_modules)
{
    viewer_rt_pipeline_desc result{};
    result.model = model;
    result.bindings = {{
        {{0, 0}, rt_descriptor_type::acceleration_structure, 1},
        {{0, 1}, rt_descriptor_type::storage_texture, 1},
        {{0, 2}, rt_descriptor_type::structured_buffer, 1},
        {{0, 3}, rt_descriptor_type::structured_buffer, 1},
        {{0, 4}, rt_descriptor_type::structured_buffer, 1},
        {{0, 5}, rt_descriptor_type::structured_buffer, 1},
        {{0, 6}, rt_descriptor_type::structured_buffer, 1},
        {{0, 7}, rt_descriptor_type::structured_buffer, 1},
        {{0, 8}, rt_descriptor_type::uniform_buffer, 1},
        {{0, 9}, rt_descriptor_type::storage_buffer, 1},
        {{0, 10}, rt_descriptor_type::storage_texture, 1},
    }};
    if (model == rt_pipeline_model::compute_intersector) {
        if (entry_modules[0] && entry_modules[1]) {
            result.shaders[0] = {
                entry_modules[0],
                rt_shader_stage::compute,
                "rtvdb_trace_kernel"};
            result.shaders[1] = {
                entry_modules[1],
                rt_shader_stage::compute,
                "rtvdb_pick_kernel"};
            result.shader_count = 2;
        }
    } else if (std::all_of(
            entry_modules.begin(),
            entry_modules.end(),
            [](rt_shader_module_handle module) { return static_cast<bool>(module); })) {
        result.shaders = {{
            {entry_modules[0], rt_shader_stage::ray_generation, "RayGen"},
            {entry_modules[1], rt_shader_stage::ray_generation, "PickRayGen"},
            {entry_modules[2], rt_shader_stage::miss, "Miss"},
            {entry_modules[3], rt_shader_stage::closest_hit, "ClosestHitTriangle"},
            {entry_modules[4], rt_shader_stage::closest_hit, "ClosestHitPoint"},
            {entry_modules[5], rt_shader_stage::intersection, "IntersectionPoint"},
            {entry_modules[6], rt_shader_stage::closest_hit, "ClosestHitLine"},
            {entry_modules[7], rt_shader_stage::intersection, "IntersectionLine"},
        }};
        result.shader_count = result.shaders.size();
    }
    if (model == rt_pipeline_model::native_ray_tracing) {
        result.groups = {{
            {rt_shader_group_type::general, nullptr, 0},
            {rt_shader_group_type::general, nullptr, 1},
            {rt_shader_group_type::general, nullptr, 2},
            {
                rt_shader_group_type::triangles_hit_group,
                "TriangleHitGroup",
                kRtUnusedShaderIndex,
                3},
            {
                rt_shader_group_type::procedural_hit_group,
                "PointHitGroup",
                kRtUnusedShaderIndex,
                4,
                kRtUnusedShaderIndex,
                5},
            {
                rt_shader_group_type::procedural_hit_group,
                "LineHitGroup",
                kRtUnusedShaderIndex,
                6,
                kRtUnusedShaderIndex,
                7},
        }};
        result.group_count = result.groups.size();
    }
    result.dispatch_entries = {{
        {rt_logical_dispatch_entry::render, 0},
        {rt_logical_dispatch_entry::pick, 1},
    }};
    return result;
}
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
    bool is_pick_pass,
    int pick_pixel_x,
    int pick_pixel_y,
    float render_scale_x,
    float render_scale_y)
{
    rt_viewer_constants constants{};
    constants.width = static_cast<std::uint32_t>((std::max)(width, 0));
    constants.height = static_cast<std::uint32_t>((std::max)(height, 0));
    constants.aspect = height == 0 ? 1.0f : static_cast<float>(width) / static_cast<float>(height);
    constants.display_mode = display_mode;
    constants.accumulation_sample_index = accumulation_sample_index;
    constants.hover_highlight_kind = hover_highlight_kind;
    constants.hover_primitive_index = hover_primitive_index;
    constants.selection_highlight_kind = selection_highlight_kind;
    constants.selection_primitive_index = selection_primitive_index;
    constants.render_scale_x = std::isfinite(render_scale_x) && render_scale_x > 0.0f
        ? render_scale_x
        : 1.0f;
    constants.render_scale_y = std::isfinite(render_scale_y) && render_scale_y > 0.0f
        ? render_scale_y
        : 1.0f;
    constants.is_pick_pass = is_pick_pass ? 1u : 0u;
    constants.pick_pixel_x = static_cast<std::uint32_t>((std::max)(pick_pixel_x, 0));
    constants.pick_pixel_y = static_cast<std::uint32_t>((std::max)(pick_pixel_y, 0));
    constants.has_frame = has_frame ? 1u : 0u;
    constants.triangle_count = static_cast<std::uint32_t>(build.triangle_count);
    constants.point_count = static_cast<std::uint32_t>(build.point_count);
    constants.line_count = static_cast<std::uint32_t>(build.line_count);

    rtvdb::camera camera{};
    if (has_frame) {
        camera = scene.camera;
    }
    const auto subtract = [](const rtvdb::vec3 &a, const rtvdb::vec3 &b) {
        return rtvdb::vec3{a.x - b.x, a.y - b.y, a.z - b.z};
    };
    const auto normalize_or = [](const rtvdb::vec3 &value, const rtvdb::vec3 &fallback) {
        const float length_squared = value.x * value.x + value.y * value.y + value.z * value.z;
        if (length_squared <= 1.0e-12f) {
            return fallback;
        }
        const float inverse_length = 1.0f / std::sqrt(length_squared);
        return rtvdb::vec3{value.x * inverse_length, value.y * inverse_length, value.z * inverse_length};
    };
    const auto cross = [](const rtvdb::vec3 &a, const rtvdb::vec3 &b) {
        return rtvdb::vec3{
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x,
        };
    };
    const rtvdb::vec3 forward = normalize_or(subtract(camera.target, camera.origin), {0.0f, 0.0f, 1.0f});
    rtvdb::vec3 up = normalize_or(camera.up, {0.0f, 1.0f, 0.0f});
    const rtvdb::vec3 right = normalize_or(cross(forward, up), {1.0f, 0.0f, 0.0f});
    up = normalize_or(cross(right, forward), {0.0f, 1.0f, 0.0f});
    const rtvdb::vec3 vectors[] = {camera.origin, forward, right, up};
    float* outputs[] = {constants.origin, constants.forward, constants.right, constants.up};
    for (std::size_t vector_index = 0; vector_index < 4; ++vector_index) {
        outputs[vector_index][0] = vectors[vector_index].x;
        outputs[vector_index][1] = vectors[vector_index].y;
        outputs[vector_index][2] = vectors[vector_index].z;
    }
    if (build.bounds.valid) {
        constants.scene_bounds_min[0] = build.bounds.min.x;
        constants.scene_bounds_min[1] = build.bounds.min.y;
        constants.scene_bounds_min[2] = build.bounds.min.z;
        constants.scene_bounds_max[0] = build.bounds.max.x;
        constants.scene_bounds_max[1] = build.bounds.max.y;
        constants.scene_bounds_max[2] = build.bounds.max.z;
        constants.scene_bounds_valid = 1;
    }
    constants.projection = static_cast<std::uint32_t>(camera.projection);
    constants.projection_blend_from = static_cast<std::uint32_t>(scene.projection_blend_from);
    constants.projection_blend_to = static_cast<std::uint32_t>(scene.projection_blend_to);
    constants.projection_blend_t = scene.projection_blend_t;
    fill_rt_projection_parameters(
        camera, scene.projection_blend_from, constants.aspect,
        &constants.projection_from[0], &constants.projection_from[1]);
    fill_rt_projection_parameters(
        camera, scene.projection_blend_to, constants.aspect,
        &constants.projection_to[0], &constants.projection_to[1]);
    fill_rt_accumulation_jitter(accumulation_sample_index, constants.accumulation_jitter);
    return constants;
}

rt_viewer_constants_gpu pack_rt_viewer_constants(const rt_viewer_constants &source) {
    rt_viewer_constants_gpu constants{};
    const float* source_vectors[] = {source.origin, source.forward, source.right, source.up};
    float* target_vectors[] = {constants.origin, constants.forward, constants.right, constants.up};
    for (std::size_t vector_index = 0; vector_index < 4; ++vector_index) {
        std::memcpy(target_vectors[vector_index], source_vectors[vector_index], sizeof(float) * 3);
    }
    std::memcpy(constants.scene_bounds_min, source.scene_bounds_min, sizeof(float) * 3);
    std::memcpy(constants.scene_bounds_max, source.scene_bounds_max, sizeof(float) * 3);
    constants.scene_bounds_max[3] = source.scene_bounds_valid != 0 ? 1.0f : 0.0f;
    constants.size_and_mode[0] = source.width;
    constants.size_and_mode[1] = source.height;
    constants.size_and_mode[2] = source.display_mode;
    constants.size_and_mode[3] = source.accumulation_sample_index;
    constants.projection_from[0] = source.projection_from[0];
    constants.projection_from[1] = source.projection_from[1];
    constants.projection_from[2] = static_cast<float>(source.projection_blend_from);
    constants.projection_from[3] = source.aspect;
    constants.projection_to[0] = source.projection_to[0];
    constants.projection_to[1] = source.projection_to[1];
    constants.projection_to[2] = static_cast<float>(source.projection_blend_to);
    constants.projection_modes[0] = source.projection_blend_from;
    constants.projection_modes[1] = source.projection_blend_to;
    constants.projection_modes[2] = source.hover_highlight_kind;
    constants.projection_modes[3] = source.hover_primitive_index;
    constants.blend_and_jitter[0] = source.projection_blend_t;
    constants.blend_and_jitter[1] = source.accumulation_jitter[0];
    constants.blend_and_jitter[2] = source.accumulation_jitter[1];
    constants.blend_and_jitter[3] = source.hover_highlight_mix;
    constants.pick_and_flags[0] = source.triangle_count;
    constants.pick_and_flags[1] = source.point_count;
    constants.pick_and_flags[2] = source.line_count;
    constants.pick_and_flags[3] = source.selection_highlight_kind;
    constants.pick_params[0] = source.pick_pixel_x;
    constants.pick_params[1] = source.pick_pixel_y;
    constants.pick_params[2] = source.is_pick_pass;
    constants.pick_params[3] = source.selection_primitive_index;
    constants.render_scale[0] = source.render_scale_x;
    constants.render_scale[1] = source.render_scale_y;
    return constants;
}

rt_dispatch_plan make_rt_dispatch_plan(const rt_scene_build &build, bool pick_pass) {
    if (!rt_scene_has_renderable_primitives(build)) {
        return {};
    }
    return {pick_pass ? rt_dispatch_kind::pick : rt_dispatch_kind::render, !pick_pass};
}

bool rt_accumulation_key_equals(const rt_accumulation_key &a, const rt_accumulation_key &b) {
    const auto float3_equals = [](const float (&left)[3], const float (&right)[3]) {
        return left[0] == right[0] && left[1] == right[1] && left[2] == right[2];
    };

    return a.build_revision == b.build_revision &&
        a.width == b.width &&
        a.height == b.height &&
        a.display_mode == b.display_mode &&
        a.has_frame == b.has_frame &&
        a.hover_highlight_kind == b.hover_highlight_kind &&
        a.hover_primitive_index == b.hover_primitive_index &&
        a.selection_highlight_kind == b.selection_highlight_kind &&
        a.selection_primitive_index == b.selection_primitive_index &&
        a.render_scale_x == b.render_scale_x &&
        a.render_scale_y == b.render_scale_y &&
        float3_equals(a.camera_origin, b.camera_origin) &&
        float3_equals(a.camera_target, b.camera_target) &&
        float3_equals(a.camera_up, b.camera_up) &&
        a.camera_vertical_fov_degrees == b.camera_vertical_fov_degrees &&
        a.camera_projection == b.camera_projection &&
        a.camera_fisheye_theta_degrees == b.camera_fisheye_theta_degrees &&
        a.camera_fisheye_phi_degrees == b.camera_fisheye_phi_degrees &&
        a.camera_orthographic_height == b.camera_orthographic_height &&
        a.camera_projection_blend_from == b.camera_projection_blend_from &&
        a.camera_projection_blend_to == b.camera_projection_blend_to &&
        a.camera_projection_blend_t == b.camera_projection_blend_t;
}

void reset_rt_accumulation_state(rt_accumulation_state* state) {
    if (state != nullptr) {
        *state = {};
    }
}

bool begin_rt_accumulation(
    rt_accumulation_state* state,
    const rt_accumulation_key &next_key,
    bool continuous_render)
{
    if (state == nullptr) {
        return false;
    }

    const bool changed = !rt_accumulation_key_equals(state->key, next_key);
    if (changed) {
        state->key = next_key;
        state->sample_count = 0;
        state->submitted_sample_count = 0;
        ++state->generation;
        state->active = false;
    }
    const bool restarted = continuous_render &&
        state->sample_count >= kRtMaxAccumulationSamples &&
        state->submitted_sample_count == state->sample_count;
    if (restarted) {
        state->sample_count = 0;
        state->submitted_sample_count = 0;
        ++state->generation;
        state->active = false;
    }
    return changed || restarted;
}

bool get_rt_accumulation_submission(
    const rt_accumulation_state &state,
    std::uint64_t* out_generation,
    std::uint32_t* out_sample_index)
{
    if (out_generation != nullptr) {
        *out_generation = state.generation;
    }
    if (out_sample_index != nullptr) {
        *out_sample_index = state.submitted_sample_count;
    }
    return state.submitted_sample_count < kRtMaxAccumulationSamples;
}

bool submit_rt_accumulation_sample(
    rt_accumulation_state* state,
    std::uint64_t generation,
    std::uint32_t sample_index,
    bool continuous_render)
{
    if (state == nullptr || generation != state->generation ||
        sample_index != state->submitted_sample_count ||
        sample_index >= kRtMaxAccumulationSamples) {
        return false;
    }
    ++state->submitted_sample_count;
    state->active = continuous_render || state->sample_count < kRtMaxAccumulationSamples;
    return true;
}

bool complete_rt_accumulation_sample(
    rt_accumulation_state* state,
    std::uint64_t generation,
    std::uint32_t sample_index,
    bool continuous_render)
{
    if (state == nullptr) {
        return false;
    }
    if (generation != state->generation) {
        return true;
    }
    if (sample_index != state->sample_count || sample_index >= state->submitted_sample_count) {
        return false;
    }
    ++state->sample_count;
    state->active = continuous_render || state->sample_count < kRtMaxAccumulationSamples;
    return true;
}

rt_accumulation_key make_rt_accumulation_key(
    const frame_scene &scene,
    bool has_frame,
    const rt_scene_build &build,
    int width,
    int height,
    std::uint32_t display_mode,
    float render_scale_x,
    float render_scale_y)
{
    rt_accumulation_key key{};
    hover_highlight highlight{};
    get_hover_highlight(&highlight);
    selection_highlight selection{};
    get_selection_highlight(&selection);
    key.build_revision = build.revision;
    key.width = static_cast<std::uint32_t>(width);
    key.height = static_cast<std::uint32_t>(height);
    key.display_mode = display_mode;
    key.has_frame = has_frame ? 1u : 0u;
    key.hover_highlight_kind = static_cast<std::uint32_t>(highlight.kind);
    key.hover_primitive_index = highlight.primitive_index;
    key.selection_highlight_kind = static_cast<std::uint32_t>(selection.kind);
    key.selection_primitive_index = selection.primitive_index;
    key.render_scale_x = render_scale_x;
    key.render_scale_y = render_scale_y;
    key.camera_origin[0] = scene.camera.origin.x;
    key.camera_origin[1] = scene.camera.origin.y;
    key.camera_origin[2] = scene.camera.origin.z;
    key.camera_target[0] = scene.camera.target.x;
    key.camera_target[1] = scene.camera.target.y;
    key.camera_target[2] = scene.camera.target.z;
    key.camera_up[0] = scene.camera.up.x;
    key.camera_up[1] = scene.camera.up.y;
    key.camera_up[2] = scene.camera.up.z;
    key.camera_vertical_fov_degrees = scene.camera.vertical_fov_degrees;
    key.camera_projection = static_cast<std::uint32_t>(scene.camera.projection);
    key.camera_fisheye_theta_degrees = scene.camera.fisheye_theta_degrees;
    key.camera_fisheye_phi_degrees = scene.camera.fisheye_phi_degrees;
    key.camera_orthographic_height = scene.camera.orthographic_height;
    key.camera_projection_blend_from = static_cast<std::uint32_t>(scene.projection_blend_from);
    key.camera_projection_blend_to = static_cast<std::uint32_t>(scene.projection_blend_to);
    key.camera_projection_blend_t = scene.projection_blend_t;
    return key;
}

void fill_rt_accumulation_jitter(std::uint32_t sample_index, float out_jitter[2]) {
    if (out_jitter == nullptr) {
        return;
    }
    if (sample_index == 0u) {
        out_jitter[0] = 0.0f;
        out_jitter[1] = 0.0f;
        return;
    }

    const auto halton = [](std::uint32_t index, std::uint32_t base) {
        float result = 0.0f;
        float factor = 1.0f / static_cast<float>(base);
        while (index > 0) {
            result += factor * static_cast<float>(index % base);
            index /= base;
            factor /= static_cast<float>(base);
        }
        return result;
    };
    out_jitter[0] = halton(sample_index + 1u, 2u) - 0.5f;
    out_jitter[1] = halton(sample_index + 1u, 3u) - 0.5f;
}

void fill_rt_projection_parameters(
    const rtvdb::camera &camera,
    rtvdb::camera_projection projection,
    float aspect,
    float* out_param0,
    float* out_param1)
{
    if (out_param0 == nullptr || out_param1 == nullptr) {
        return;
    }

    switch (projection) {
    case rtvdb::camera_projection::fisheye:
        *out_param0 = camera.fisheye_phi_degrees * 3.1415926535f / 180.0f;
        *out_param1 = camera.fisheye_theta_degrees * 3.1415926535f / 180.0f;
        break;
    case rtvdb::camera_projection::orthographic:
        *out_param1 = camera.orthographic_height;
        *out_param0 = *out_param1 * aspect;
        break;
    case rtvdb::camera_projection::perspective:
    default: {
        const float fov_radians = camera.vertical_fov_degrees * 3.1415926535f / 180.0f;
        *out_param0 = std::tan(fov_radians * 0.5f);
        *out_param1 = 0.0f;
        break;
    }
    }
}

bool rt_scene_has_renderable_primitives(const rt_scene_build &build) {
    return !build.triangle_chunks.empty() || build.point_count != 0 || build.line_count != 0;
}

} // namespace rtvdb::viewer_backend
