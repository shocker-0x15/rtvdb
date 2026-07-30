#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;
#include "../shaders/rt_logic_shared_core.h"
using namespace rtvdb;

struct camera_gpu {
    float4 origin;
    float4 forward;
    float4 right;
    float4 up;
    float4 scene_bounds_min;
    float4 scene_bounds_max;
    uint width;
    uint height;
    uint projection;
    float aspect;
    float projection_param_from0;
    float projection_param_from1;
    float projection_param_to0;
    float projection_param_to1;
    uint projection_blend_from;
    uint projection_blend_to;
    float projection_blend_t;
    uint hover_highlight_kind;
    uint hover_primitive_index;
    float hover_highlight_mix;
    uint display_mode;
    uint accumulation_sample_index;
    float2 accumulation_jitter;
    uint scene_bounds_valid;
};

struct point_gpu {
    packed_float3 position;
    float radius;
    float4 color;
};

struct line_gpu {
    packed_float3 a;
    float radius;
    packed_float3 b;
    float pad;
    float4 color;
    uint flags;
    packed_float3 pad_flags;
};

struct pick_request_gpu {
    uint pixel_x;
    uint pixel_y;
};

struct pick_result_gpu {
    uint primitive_kind;
    uint primitive_index;
    float distance;
    uint pad;
};

struct triangle_instance_metadata_gpu {
    uint first_triangle;
    uint index_offset;
};

struct procedural_chunk_metadata_gpu {
    uint first_primitive;
    uint primitive_count;
    uint visible;
    uint pad;
};

struct procedural_chunk_count_gpu {
    uint point_chunk_count;
    uint line_chunk_count;
};

struct hit_info {
    bool hit;
    float distance;
    float3 normal;
    float4 color;
    uint primitive_id;
    uint kind;
    uint geometry_index;
    uint instance_index;
    uint flags;
};

constant float kAlphaOpaqueThreshold = 0.999f;
constant float kRayMinDistanceFallback = 1.0e-6f;
constant float kRayHitAdvanceBiasFallback = 1.0e-6f;
constant uint kMaxTransparencyLayers = 16u;
constant uint kLineFlagFixedColor = 1u << 0;
constant uint kLineFlagNonPickable = 1u << 1;

float scene_scale(constant camera_gpu &camera) {
    return core::scene_scale(
        camera.scene_bounds_valid,
        camera.scene_bounds_min.xyz,
        camera.scene_bounds_max.xyz);
}

float scene_intersection_t_min(constant camera_gpu &camera) {
    return max(core::scene_intersection_t_min(scene_scale(camera)), kRayMinDistanceFallback);
}

float scene_hit_advance_bias(constant camera_gpu &camera) {
    return max(core::scene_hit_advance_bias(scene_scale(camera)), kRayHitAdvanceBiasFallback);
}

float scene_length_sq_epsilon(constant camera_gpu &camera) {
    return core::scene_length_sq_epsilon(scene_scale(camera));
}

hit_info trace_points(
    ray view_ray,
    constant camera_gpu &camera,
    primitive_acceleration_structure point_scene,
    constant point_gpu* points,
    constant procedural_chunk_metadata_gpu* point_chunks,
    uint point_chunk_count);

hit_info trace_lines(
    ray view_ray,
    constant camera_gpu &camera,
    primitive_acceleration_structure line_scene,
    constant line_gpu* lines,
    constant procedural_chunk_metadata_gpu* line_chunks,
    uint line_chunk_count,
    bool is_pick_pass);

bool procedural_primitive_visible(
    uint primitive_index,
    constant procedural_chunk_metadata_gpu* groups,
    uint group_count)
{
    if (groups == nullptr) {
        return true;
    }
    for (uint i = 0u; i < group_count; ++i) {
        const procedural_chunk_metadata_gpu group = groups[i];
        if (primitive_index >= group.first_primitive &&
            primitive_index < group.first_primitive + group.primitive_count) {
            return group.visible != 0u;
        }
    }
    return true;
}

bool intersect_sphere(
    ray view_ray,
    float3 center,
    float radius,
    float min_hit_distance,
    thread float &out_distance,
    thread float3 &out_normal) {
    core::ray core_ray;
    core_ray.origin = view_ray.origin;
    core_ray.direction = view_ray.direction;
    core_ray.min_distance = min_hit_distance;
    core_ray.max_distance = view_ray.max_distance;
    const core::intersection hit = core::intersect_sphere(
        core_ray,
        center,
        radius,
        min_hit_distance);
    if (!hit.hit) {
        return false;
    }
    out_distance = hit.distance;
    out_normal = hit.normal;
    return true;
}

bool intersect_capsule(
    ray view_ray,
    line_gpu primitive,
    float min_hit_distance,
    float length_sq_epsilon,
    thread float &out_distance,
    thread float3 &out_normal) {
    core::ray core_ray;
    core_ray.origin = view_ray.origin;
    core_ray.direction = view_ray.direction;
    core_ray.min_distance = min_hit_distance;
    core_ray.max_distance = view_ray.max_distance;
    const core::intersection hit = core::intersect_capsule(
        core_ray,
        float3(primitive.a),
        float3(primitive.b),
        primitive.radius,
        min_hit_distance,
        length_sq_epsilon);
    if (!hit.hit) {
        return false;
    }
    out_distance = hit.distance;
    out_normal = hit.normal;
    return true;
}

float4 shade_hit(const thread hit_info &hit, constant camera_gpu &camera) {
    float4 color = hit.color;
    if ((hit.flags & kLineFlagFixedColor) == 0u) {
        const uint primitive_seed = hit.kind == 0u
            ? hit.primitive_id
            : hit.kind == 1u
                ? 1000000u + hit.primitive_id
                : 2000000u + hit.primitive_id;
        color = core::apply_display_mode(
            hit.color,
            hit.normal,
            primitive_seed,
            hit.geometry_index,
            hit.instance_index,
            camera.display_mode);
    }

    if ((hit.flags & kLineFlagFixedColor) == 0u) {
        color = core::apply_hover_highlight(
            color,
            hit.kind + 1u,
            hit.primitive_id,
            camera.hover_highlight_kind,
            camera.hover_primitive_index,
            camera.hover_highlight_mix);
    }
    return color;
}

hit_info trace_triangles(
    ray view_ray,
    instance_acceleration_structure scene,
    constant packed_float3* positions,
    constant uint* indices,
    constant float4* triangle_colors,
    constant uint* triangle_geometry_indices,
    constant uint* triangle_instance_indices,
    constant triangle_instance_metadata_gpu* triangle_instance_metadata)
{
    hit_info result{};
    if (is_null_acceleration_structure(scene) ||
        positions == nullptr ||
        indices == nullptr ||
        triangle_colors == nullptr) {
        return result;
    }

    intersector<triangle_data, instancing> intersector;
    intersector.set_triangle_cull_mode(triangle_cull_mode::none);
    const auto triangle_hit = intersector.intersect(view_ray, scene);
    if (triangle_hit.type == intersection_type::none) {
        return result;
    }

    const uint instance_index = triangle_hit.instance_id;
    const uint geometry_index = triangle_hit.geometry_id;
    const triangle_instance_metadata_gpu metadata =
        triangle_instance_metadata[instance_index * 4u + geometry_index];
    const uint primitive_index = metadata.first_triangle + triangle_hit.primitive_id;
    const uint index_base = metadata.index_offset + triangle_hit.primitive_id * 3u;
    const uint ia = indices[index_base + 0];
    const uint ib = indices[index_base + 1];
    const uint ic = indices[index_base + 2];
    const float3 a = float3(positions[ia]);
    const float3 b = float3(positions[ib]);
    const float3 c = float3(positions[ic]);
    result.hit = true;
    result.distance = triangle_hit.distance;
    result.normal = core::triangle_normal(a, b, c);
    result.color = triangle_colors[primitive_index];
    result.primitive_id = primitive_index;
    result.kind = 0u;
    result.geometry_index = triangle_geometry_indices != nullptr ? triangle_geometry_indices[primitive_index] : 0u;
    result.instance_index = triangle_instance_indices != nullptr ? triangle_instance_indices[primitive_index] : 0u;
    result.flags = 0u;
    return result;
}

hit_info trace_nearest_hit(
    ray view_ray,
    constant camera_gpu &camera,
    instance_acceleration_structure scene,
    primitive_acceleration_structure point_scene,
    primitive_acceleration_structure line_scene,
    constant packed_float3* positions,
    constant uint* indices,
    constant float4* triangle_colors,
    constant point_gpu* points,
    constant line_gpu* lines,
    constant uint* triangle_geometry_indices,
    constant uint* triangle_instance_indices,
    constant triangle_instance_metadata_gpu* triangle_instance_metadata,
    constant procedural_chunk_metadata_gpu* point_chunks,
    uint point_chunk_count,
    constant procedural_chunk_metadata_gpu* line_chunks,
    uint line_chunk_count,
    bool is_pick_pass)
{
    hit_info best_hit = trace_triangles(
        view_ray,
        scene,
        positions,
        indices,
        triangle_colors,
        triangle_geometry_indices,
        triangle_instance_indices,
        triangle_instance_metadata);

    const hit_info point_hit = trace_points(view_ray, camera, point_scene, points, point_chunks, point_chunk_count);
    if (point_hit.hit && (!best_hit.hit || point_hit.distance < best_hit.distance)) {
        best_hit = point_hit;
    }

    const hit_info line_hit = trace_lines(
        view_ray,
        camera,
        line_scene,
        lines,
        line_chunks,
        line_chunk_count,
        is_pick_pass);
    if (line_hit.hit && (!best_hit.hit || line_hit.distance < best_hit.distance)) {
        best_hit = line_hit;
    }
    return best_hit;
}

hit_info trace_points(
    ray view_ray,
    constant camera_gpu &camera,
    primitive_acceleration_structure point_scene,
    constant point_gpu* points,
    constant procedural_chunk_metadata_gpu* point_chunks,
    uint point_chunk_count)
{
    hit_info result{};
    if (is_null_acceleration_structure(point_scene) || points == nullptr) {
        return result;
    }

    const float min_hit_distance = scene_intersection_t_min(camera);
    intersection_query<> query(view_ray, point_scene);
    while (query.next()) {
        const uint primitive_index = query.get_candidate_primitive_id();
        if (!procedural_primitive_visible(primitive_index, point_chunks, point_chunk_count)) {
            continue;
        }
        const point_gpu primitive = points[primitive_index];
        if (core::point_contains(
                camera.origin.xyz,
                float3(primitive.position),
                primitive.radius,
                scene_length_sq_epsilon(camera))) {
            continue;
        }
        float distance = 0.0f;
        float3 normal = float3(0.0f);
        if (intersect_sphere(view_ray, float3(primitive.position), primitive.radius, min_hit_distance, distance, normal)) {
            query.commit_bounding_box_intersection(distance);
        }
    }

    if (query.get_committed_intersection_type() == intersection_type::none) {
        return result;
    }

    const uint primitive_index = query.get_committed_primitive_id();
    if (!procedural_primitive_visible(primitive_index, point_chunks, point_chunk_count)) {
        return result;
    }
    const point_gpu primitive = points[primitive_index];
    if (core::point_contains(
            camera.origin.xyz,
            float3(primitive.position),
            primitive.radius,
            scene_length_sq_epsilon(camera))) {
        return result;
    }
    float distance = 0.0f;
    float3 normal = float3(0.0f);
    if (!intersect_sphere(view_ray, float3(primitive.position), primitive.radius, min_hit_distance, distance, normal)) {
        return result;
    }

    result.hit = true;
    result.distance = distance;
    result.normal = normal;
    result.color = primitive.color;
    result.primitive_id = primitive_index;
    result.kind = 1u;
    result.geometry_index = 0u;
    result.instance_index = 0u;
    result.flags = 0u;
    return result;
}

hit_info trace_lines(
    ray view_ray,
    constant camera_gpu &camera,
    primitive_acceleration_structure line_scene,
    constant line_gpu* lines,
    constant procedural_chunk_metadata_gpu* line_chunks,
    uint line_chunk_count,
    bool is_pick_pass)
{
    hit_info result{};
    if (is_null_acceleration_structure(line_scene) || lines == nullptr) {
        return result;
    }

    const float min_hit_distance = scene_intersection_t_min(camera);
    const float length_sq_epsilon = scene_length_sq_epsilon(camera);
    intersection_query<> query(view_ray, line_scene);
    while (query.next()) {
        const uint primitive_index = query.get_candidate_primitive_id();
        if (!procedural_primitive_visible(primitive_index, line_chunks, line_chunk_count)) {
            continue;
        }
        const line_gpu primitive = lines[primitive_index];
        if (is_pick_pass && (primitive.flags & kLineFlagNonPickable) != 0u) {
            continue;
        }
        if (core::capsule_contains(
                camera.origin.xyz,
                float3(primitive.a),
                float3(primitive.b),
                primitive.radius,
                length_sq_epsilon,
                length_sq_epsilon)) {
            continue;
        }
        float distance = 0.0f;
        float3 normal = float3(0.0f);
        if (intersect_capsule(view_ray, primitive, min_hit_distance, length_sq_epsilon, distance, normal)) {
            query.commit_bounding_box_intersection(distance);
        }
    }

    if (query.get_committed_intersection_type() == intersection_type::none) {
        return result;
    }

    const uint primitive_index = query.get_committed_primitive_id();
    if (!procedural_primitive_visible(primitive_index, line_chunks, line_chunk_count)) {
        return result;
    }
    const line_gpu primitive = lines[primitive_index];
    if (is_pick_pass && (primitive.flags & kLineFlagNonPickable) != 0u) {
        return result;
    }
    if (core::capsule_contains(
            camera.origin.xyz,
            float3(primitive.a),
            float3(primitive.b),
            primitive.radius,
            length_sq_epsilon,
            length_sq_epsilon)) {
        return result;
    }
    float distance = 0.0f;
    float3 normal = float3(0.0f);
    if (!intersect_capsule(view_ray, primitive, min_hit_distance, length_sq_epsilon, distance, normal)) {
        return result;
    }

    result.hit = true;
    result.distance = distance;
    result.normal = normal;
    result.color = primitive.color;
    result.primitive_id = primitive_index;
    result.kind = 2u;
    result.geometry_index = 0u;
    result.instance_index = 0u;
    result.flags = primitive.flags;
    return result;
}

uchar4 to_bgra8(float3 color) {
    const float3 clamped = clamp(color, 0.0f, 1.0f);
    return uchar4(
        uchar(clamped.z * 255.0f),
        uchar(clamped.y * 255.0f),
        uchar(clamped.x * 255.0f),
        uchar(255.0f));
}

float4 to_rgb_linear(float3 color) {
    const float3 clamped = clamp(color, 0.0f, 1.0f);
    return float4(clamped, 1.0f);
}

void build_projection_ray(
    constant camera_gpu &camera,
    uint projection,
    float2 ndc,
    float projection_param0,
    float projection_param1,
    thread float3 &ray_origin,
    thread float3 &direction)
{
    const core::projection_ray result = core::build_projection_ray(
        projection,
        ndc,
        projection_param0,
        projection_param1,
        camera.origin.xyz,
        camera.forward.xyz,
        camera.right.xyz,
        camera.up.xyz,
        camera.aspect,
        camera.scene_bounds_valid,
        camera.scene_bounds_min.xyz,
        camera.scene_bounds_max.xyz);
    ray_origin = result.origin;
    direction = result.direction;
}

kernel void rtvdb_trace_kernel(
    device uchar4* output_pixels [[buffer(0)]],
    constant camera_gpu &camera [[buffer(1)]],
    constant packed_float3* positions [[buffer(2)]],
    constant uint* indices [[buffer(3)]],
    constant float4* triangle_colors [[buffer(4)]],
    instance_acceleration_structure scene [[buffer(5)]],
    primitive_acceleration_structure point_scene [[buffer(6)]],
    primitive_acceleration_structure line_scene [[buffer(7)]],
    constant point_gpu* points [[buffer(8)]],
    constant line_gpu* lines [[buffer(9)]],
    constant uint* triangle_geometry_indices [[buffer(10)]],
    constant uint* triangle_instance_indices [[buffer(11)]],
    device float4* accumulation_pixels [[buffer(13)]],
    constant triangle_instance_metadata_gpu* triangle_instance_metadata [[buffer(14)]],
    constant procedural_chunk_metadata_gpu* point_chunks [[buffer(15)]],
    constant procedural_chunk_metadata_gpu* line_chunks [[buffer(16)]],
    constant procedural_chunk_count_gpu &chunk_counts [[buffer(17)]],
    uint2 tid [[thread_position_in_grid]])
{
    if (tid.x >= camera.width || tid.y >= camera.height) {
        return;
    }

    const uint pixel_index = tid.y * camera.width + tid.x;
    const float2 uv = (float2(tid) + 0.5f + camera.accumulation_jitter) / float2(camera.width, camera.height);
    const float2 ndc = float2(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f);
    float3 ray_origin = camera.origin.xyz;
    float3 direction = camera.forward.xyz;
    build_projection_ray(
        camera,
        camera.projection_blend_from,
        ndc,
        camera.projection_param_from0,
        camera.projection_param_from1,
        ray_origin,
        direction);
    const float blend_t = clamp(camera.projection_blend_t, 0.0f, 1.0f);
    if (camera.projection_blend_from != camera.projection_blend_to && blend_t < 1.0f) {
        float3 ray_origin_b;
        float3 direction_b;
        build_projection_ray(
            camera,
            camera.projection_blend_to,
            ndc,
            camera.projection_param_to0,
            camera.projection_param_to1,
            ray_origin_b,
            direction_b);
        ray_origin = mix(ray_origin, ray_origin_b, blend_t);
        direction = normalize(mix(direction, direction_b, blend_t));
    }

    float3 accum = float3(0.0f);
    float3 throughput = float3(1.0f);

    for (uint layer = 0u; layer < kMaxTransparencyLayers; ++layer) {
        ray view_ray(ray_origin, direction, scene_intersection_t_min(camera), INFINITY);
        const hit_info best_hit = trace_nearest_hit(
            view_ray,
            camera,
            scene,
            point_scene,
            line_scene,
            positions,
            indices,
            triangle_colors,
            points,
            lines,
            triangle_geometry_indices,
            triangle_instance_indices,
            triangle_instance_metadata,
            point_chunks,
            chunk_counts.point_chunk_count,
            line_chunks,
            chunk_counts.line_chunk_count,
            false);
        if (!best_hit.hit) {
            break;
        }

        const float4 shaded = shade_hit(best_hit, camera);
        const float alpha = clamp(shaded.a, 0.0f, 1.0f);
        if (alpha >= kAlphaOpaqueThreshold) {
            accum += throughput * shaded.rgb;
            break;
        }

        accum += throughput * shaded.rgb * alpha;
        throughput *= 1.0f - alpha;
        ray_origin += direction * (best_hit.distance + scene_hit_advance_bias(camera));
        if (max(throughput.r, max(throughput.g, throughput.b)) <= 0.001f) {
            break;
        }
    }

    const float4 sample_rgb_linear = to_rgb_linear(accum);
    if (accumulation_pixels != nullptr) {
        if (camera.accumulation_sample_index == 0u) {
            accumulation_pixels[pixel_index] = sample_rgb_linear;
        } else {
            accumulation_pixels[pixel_index] += sample_rgb_linear;
        }
        const float inv_count = 1.0f / float(camera.accumulation_sample_index + 1u);
        output_pixels[pixel_index] = to_bgra8(accumulation_pixels[pixel_index].xyz * inv_count);
        return;
    }

    output_pixels[pixel_index] = to_bgra8(accum);
}

kernel void rtvdb_pick_kernel(
    device pick_result_gpu* pick_result [[buffer(0)]],
    constant camera_gpu &camera [[buffer(1)]],
    constant packed_float3* positions [[buffer(2)]],
    constant uint* indices [[buffer(3)]],
    constant float4* triangle_colors [[buffer(4)]],
    instance_acceleration_structure scene [[buffer(5)]],
    primitive_acceleration_structure point_scene [[buffer(6)]],
    primitive_acceleration_structure line_scene [[buffer(7)]],
    constant point_gpu* points [[buffer(8)]],
    constant line_gpu* lines [[buffer(9)]],
    constant uint* triangle_geometry_indices [[buffer(10)]],
    constant uint* triangle_instance_indices [[buffer(11)]],
    constant pick_request_gpu &pick_request [[buffer(12)]],
    constant triangle_instance_metadata_gpu* triangle_instance_metadata [[buffer(14)]],
    constant procedural_chunk_metadata_gpu* point_chunks [[buffer(15)]],
    constant procedural_chunk_metadata_gpu* line_chunks [[buffer(16)]],
    constant procedural_chunk_count_gpu &chunk_counts [[buffer(17)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0u || pick_result == nullptr) {
        return;
    }

    pick_result[0].primitive_kind = 0u;
    pick_result[0].primitive_index = 0u;
    pick_result[0].distance = 0.0f;
    pick_result[0].pad = 0u;

    if (camera.width == 0u || camera.height == 0u ||
        pick_request.pixel_x >= camera.width || pick_request.pixel_y >= camera.height) {
        return;
    }

    const float2 uv = (float2(float(pick_request.pixel_x), float(pick_request.pixel_y)) + 0.5f) /
        float2(camera.width, camera.height);
    const float2 ndc = float2(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f);
    float3 ray_origin = camera.origin.xyz;
    float3 direction = camera.forward.xyz;
    build_projection_ray(
        camera,
        camera.projection_blend_from,
        ndc,
        camera.projection_param_from0,
        camera.projection_param_from1,
        ray_origin,
        direction);

    const float blend_t = clamp(camera.projection_blend_t, 0.0f, 1.0f);
    if (camera.projection_blend_from != camera.projection_blend_to && blend_t < 1.0f) {
        float3 ray_origin_b;
        float3 direction_b;
        build_projection_ray(
            camera,
            camera.projection_blend_to,
            ndc,
            camera.projection_param_to0,
            camera.projection_param_to1,
            ray_origin_b,
            direction_b);
        ray_origin = mix(ray_origin, ray_origin_b, blend_t);
        direction = normalize(mix(direction, direction_b, blend_t));
    }

    ray view_ray(ray_origin, direction, scene_intersection_t_min(camera), INFINITY);
    const hit_info best_hit = trace_nearest_hit(
        view_ray,
        camera,
        scene,
        point_scene,
        line_scene,
        positions,
        indices,
        triangle_colors,
        points,
        lines,
        triangle_geometry_indices,
        triangle_instance_indices,
        triangle_instance_metadata,
        point_chunks,
        chunk_counts.point_chunk_count,
        line_chunks,
        chunk_counts.line_chunk_count,
        true);
    if (!best_hit.hit) {
        return;
    }

    pick_result[0].primitive_kind = best_hit.kind + 1u;
    pick_result[0].primitive_index = best_hit.primitive_id;
    pick_result[0].distance = best_hit.distance;
}
