#include "viewer_backend/rt_backend_common.h"

#include "viewer_diagnostics/output.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
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

void append_rt_diagnostics_log_line(std::string_view filename, std::string_view text) {
    if (!viewer_diagnostics::output_enabled() || text.empty()) {
        return;
    }

    try {
        const std::filesystem::path directory = viewer_diagnostics::output_directory();
        std::filesystem::create_directories(directory);
        const std::filesystem::path path =
            directory / (filename.empty() ? std::string_view{"rt.log"} : filename);
        std::ofstream file(path, std::ios::binary | std::ios::app);
        if (!file) {
            return;
        }

        const auto now = std::chrono::system_clock::now();
        const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        std::tm local_time{};
#if defined(_WIN32)
        localtime_s(&local_time, &now_time);
#else
        localtime_r(&now_time, &local_time);
#endif
        const auto milliseconds =
            std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
        file << '[' << std::put_time(&local_time, "%Y-%m-%d %H:%M:%S")
             << '.' << std::setw(3) << std::setfill('0') << milliseconds << "] "
             << text << '\n';
    } catch (...) {
    }
}

void append_rt_startup_log(std::string_view text) {
    append_rt_diagnostics_log_line("startup.log", text);
}

rt_pipeline_desc viewer_rt_pipeline_desc::pipeline() const {
    return {
        bindings.data(),
        bindings.size(),
        shaders.data(),
        shaders.size(),
        groups.data(),
        groups.size(),
        1,
        sizeof(float) * 8,
        sizeof(float) * 2};
}

rt_shader_table_desc viewer_rt_pipeline_desc::shader_table(rt_pipeline_handle pipeline_handle) const {
    return {
        pipeline_handle,
        {ray_generation_groups.data(), ray_generation_groups.size()},
        {miss_groups.data(), miss_groups.size()},
        {hit_groups.data(), hit_groups.size()},
        {}};
}

viewer_rt_pipeline_desc make_viewer_rt_pipeline_desc(
    const rt_shader_module_handle* modules,
    std::size_t module_count)
{
    viewer_rt_pipeline_desc result{};
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
    if (modules != nullptr && (module_count == 1 || module_count == result.shaders.size())) {
        const auto module = [modules, module_count](std::size_t index) {
            return modules[module_count == 1 ? 0 : index];
        };
        result.shaders = {{
            {module(0), rt_shader_stage::ray_generation, "RayGen"},
            {module(1), rt_shader_stage::ray_generation, "PickRayGen"},
            {module(2), rt_shader_stage::miss, "Miss"},
            {module(3), rt_shader_stage::closest_hit, "ClosestHitTriangle"},
            {module(4), rt_shader_stage::closest_hit, "ClosestHitPoint"},
            {module(5), rt_shader_stage::intersection, "IntersectionPoint"},
            {module(6), rt_shader_stage::closest_hit, "ClosestHitLine"},
            {module(7), rt_shader_stage::intersection, "IntersectionLine"},
        }};
    }
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
    result.ray_generation_groups = {{0, 1}};
    result.miss_groups = {{2}};
    result.hit_groups = {{3, 4, 5}};
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
    bool is_pick_pass,
    int pick_pixel_x,
    int pick_pixel_y)
{
    rt_viewer_constants constants{};
    constants.width = static_cast<std::uint32_t>((std::max)(width, 0));
    constants.height = static_cast<std::uint32_t>((std::max)(height, 0));
    constants.aspect = height == 0 ? 1.0f : static_cast<float>(width) / static_cast<float>(height);
    constants.display_mode = display_mode;
    constants.accumulation_sample_index = accumulation_sample_index;
    constants.hover_highlight_kind = hover_highlight_kind;
    constants.hover_primitive_index = hover_primitive_index;
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
    constants.pick_and_flags[3] = source.has_frame;
    constants.pick_params[0] = source.pick_pixel_x;
    constants.pick_params[1] = source.pick_pixel_y;
    constants.pick_params[2] = source.is_pick_pass;
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
    }
    if (continuous_render && state->sample_count >= kRtMaxAccumulationSamples) {
        state->sample_count = 0;
    }
    state->active = false;
    return changed;
}

void complete_rt_accumulation(rt_accumulation_state* state, bool continuous_render) {
    if (state == nullptr) {
        return;
    }
    if (state->sample_count < kRtMaxAccumulationSamples) {
        ++state->sample_count;
    }
    state->active = continuous_render || state->sample_count < kRtMaxAccumulationSamples;
}

rt_accumulation_key make_rt_accumulation_key(
    const frame_scene &scene,
    bool has_frame,
    const rt_scene_build &build,
    int width,
    int height,
    std::uint32_t display_mode)
{
    rt_accumulation_key key{};
    hover_highlight highlight{};
    get_hover_highlight(&highlight);
    key.build_revision = build.revision;
    key.width = static_cast<std::uint32_t>(width);
    key.height = static_cast<std::uint32_t>(height);
    key.display_mode = display_mode;
    key.has_frame = has_frame ? 1u : 0u;
    key.hover_highlight_kind = static_cast<std::uint32_t>(highlight.kind);
    key.hover_primitive_index = highlight.primitive_index;
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
    data.positions.resize(build.vertex_count);
    data.indices = build.indices;
    data.triangle_colors.resize(build.triangle_count);
    data.points.resize(build.point_count);
    data.lines.resize(build.line_count);
    data.point_aabbs.resize(build.point_count);
    data.line_aabbs.resize(build.line_count);
    data.instance_geometry.resize(plan.items.size() * kRtBlasChunkSetChunkCount);

    const auto encode_srgb_channel = [](float value) {
        const float x = (std::clamp)(value, 0.0f, 1.0f);
        if (x <= 0.0031308f) {
            return x * 12.92f;
        }
        return 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
    };

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
                encode_srgb_channel(color.r),
                encode_srgb_channel(color.g),
                encode_srgb_channel(color.b),
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

void copy_rt_diagnostics(scene_build_info* out_info, const scene_build_info &diagnostics) {
    if (out_info == nullptr) return;
    out_info->blas_reused_count = diagnostics.blas_reused_count;
    out_info->blas_rebuilt_count = diagnostics.blas_rebuilt_count;
    out_info->blas_reused_triangle_chunk_count = diagnostics.blas_reused_triangle_chunk_count;
    out_info->blas_rebuilt_triangle_chunk_count = diagnostics.blas_rebuilt_triangle_chunk_count;
    out_info->tlas_rebuild_count = diagnostics.tlas_rebuild_count;
    out_info->accel_build_ms = diagnostics.accel_build_ms;
    out_info->accel_host_prep_ms = diagnostics.accel_host_prep_ms;
    out_info->accel_instance_build_ms = diagnostics.accel_instance_build_ms;
    out_info->accel_procedural_aabb_ms = diagnostics.accel_procedural_aabb_ms;
    out_info->accel_command_record_ms = diagnostics.accel_command_record_ms;
    out_info->accel_resource_alloc_ms = diagnostics.accel_resource_alloc_ms;
    out_info->accel_build_call_record_ms = diagnostics.accel_build_call_record_ms;
    out_info->accel_prebuild_info_ms = diagnostics.accel_prebuild_info_ms;
    out_info->accel_chunk_blas_prebuild_info_ms = diagnostics.accel_chunk_blas_prebuild_info_ms;
    out_info->accel_chunk_blas_prebuild_info_count = diagnostics.accel_chunk_blas_prebuild_info_count;
    out_info->accel_group_blas_prebuild_info_ms = diagnostics.accel_group_blas_prebuild_info_ms;
    out_info->accel_group_blas_prebuild_info_count = diagnostics.accel_group_blas_prebuild_info_count;
    out_info->accel_point_blas_prebuild_info_ms = diagnostics.accel_point_blas_prebuild_info_ms;
    out_info->accel_point_blas_prebuild_info_count = diagnostics.accel_point_blas_prebuild_info_count;
    out_info->accel_line_blas_prebuild_info_ms = diagnostics.accel_line_blas_prebuild_info_ms;
    out_info->accel_line_blas_prebuild_info_count = diagnostics.accel_line_blas_prebuild_info_count;
    out_info->accel_tlas_prebuild_info_ms = diagnostics.accel_tlas_prebuild_info_ms;
    out_info->accel_tlas_prebuild_info_count = diagnostics.accel_tlas_prebuild_info_count;
    out_info->accel_startup_prebuild_warmup_ms = diagnostics.accel_startup_prebuild_warmup_ms;
    out_info->accel_tlas_instance_upload_ms = diagnostics.accel_tlas_instance_upload_ms;
    out_info->accel_submit_cpu_ms = diagnostics.accel_submit_cpu_ms;
    out_info->accel_gpu_wait_ms = diagnostics.accel_gpu_wait_ms;
    out_info->accel_gpu_ms = diagnostics.accel_gpu_ms;
    out_info->dispatch_ms = diagnostics.dispatch_ms;
    out_info->dispatch_submit_cpu_ms = diagnostics.dispatch_submit_cpu_ms;
    out_info->dispatch_gpu_wait_ms = diagnostics.dispatch_gpu_wait_ms;
    out_info->dispatch_gpu_ms = diagnostics.dispatch_gpu_ms;
    out_info->readback_ms = diagnostics.readback_ms;
    out_info->accumulation_sample_count = diagnostics.accumulation_sample_count;
    out_info->accumulation_target_sample_count = diagnostics.accumulation_target_sample_count;
    out_info->accumulation_in_progress = diagnostics.accumulation_in_progress;
}

} // namespace rtvdb::viewer_backend
