#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;

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

constant uint kProjectionFisheye = 1;
constant uint kProjectionOrthographic = 2;

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

struct procedural_group_metadata_gpu {
    uint first_primitive;
    uint primitive_count;
    uint visible;
    uint pad;
};

struct procedural_group_count_gpu {
    uint point_group_count;
    uint line_group_count;
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
    if (camera.scene_bounds_valid != 0u) {
        return max(length(camera.scene_bounds_max.xyz - camera.scene_bounds_min.xyz), 1.0e-3f);
    }
    return 1.0f;
}

float scene_intersection_t_min(constant camera_gpu &camera) {
    return max(scene_scale(camera) * 1.0e-5f, kRayMinDistanceFallback);
}

float scene_hit_advance_bias(constant camera_gpu &camera) {
    return max(scene_scale(camera) * 1.0e-5f, kRayHitAdvanceBiasFallback);
}

float scene_length_sq_epsilon(constant camera_gpu &camera) {
    const float scale = scene_scale(camera);
    return max(scale * scale * 1.0e-12f, 1.0e-12f);
}

hit_info trace_points(
    ray view_ray,
    constant camera_gpu &camera,
    primitive_acceleration_structure point_scene,
    constant point_gpu* points,
    constant procedural_group_metadata_gpu* point_groups,
    uint point_group_count);

hit_info trace_lines(
    ray view_ray,
    constant camera_gpu &camera,
    primitive_acceleration_structure line_scene,
    constant line_gpu* lines,
    constant procedural_group_metadata_gpu* line_groups,
    uint line_group_count,
    bool is_pick_pass);

bool procedural_primitive_visible(
    uint primitive_index,
    constant procedural_group_metadata_gpu* groups,
    uint group_count)
{
    if (groups == nullptr) {
        return true;
    }
    for (uint i = 0u; i < group_count; ++i) {
        const procedural_group_metadata_gpu group = groups[i];
        if (primitive_index >= group.first_primitive &&
            primitive_index < group.first_primitive + group.primitive_count) {
            return group.visible != 0u;
        }
    }
    return true;
}

float3 hash_color(uint seed) {
    const float x = fmod(float(seed) * 0.1031f, 1.0f);
    const float y = fmod(float(seed) * 0.11369f, 1.0f);
    const float z = fmod(float(seed) * 0.13787f, 1.0f);
    const float3 h0 = float3(x, y, z);
    const float d = dot(h0, float3(h0.y + 19.19f, h0.z + 19.19f, h0.x + 19.19f));
    const float3 h1 = float3(fmod(h0.x + d, 1.0f), fmod(h0.y + d, 1.0f), fmod(h0.z + d, 1.0f));
    return float3(
        fmod((h1.x + h1.y) * h1.z, 1.0f),
        fmod((h1.x + h1.z) * h1.y, 1.0f),
        fmod((h1.y + h1.z) * h1.x, 1.0f));
}

float approximate_ray_shift(ray view_ray, float3 target, float min_t, float max_t) {
    const float direction_len_sq = dot(view_ray.direction, view_ray.direction);
    if (direction_len_sq <= 1.0e-20f || max_t < min_t) {
        return 0.0f;
    }

    const float projected_t = dot(target - view_ray.origin, view_ray.direction) / direction_len_sq;
    return clamp(projected_t, min_t, max_t);
}

bool intersect_sphere(
    ray view_ray,
    float3 center,
    float radius,
    float min_hit_distance,
    thread float &out_distance,
    thread float3 &out_normal) {
    const float a = dot(view_ray.direction, view_ray.direction);
    if (a <= 1.0e-20f) {
        return false;
    }
    const float shift_t = approximate_ray_shift(view_ray, center, min_hit_distance, view_ray.max_distance);
    const float3 shifted_origin = view_ray.origin + view_ray.direction * shift_t;
    const float3 oc = shifted_origin - center;
    const float b = dot(oc, view_ray.direction);
    const float c = dot(oc, oc) - radius * radius;
    const float discriminant = b * b - a * c;
    if (discriminant < 0.0f) {
        return false;
    }
    const float sqrt_disc = sqrt(discriminant);
    float t = (-b - sqrt_disc) / a + shift_t;
    if (t <= min_hit_distance) {
        t = (-b + sqrt_disc) / a + shift_t;
    }
    if (t <= min_hit_distance) {
        return false;
    }
    const float3 position = view_ray.origin + view_ray.direction * t;
    out_distance = t;
    out_normal = normalize(position - center);
    return true;
}

bool intersect_capsule(
    ray view_ray,
    line_gpu primitive,
    float min_hit_distance,
    float length_sq_epsilon,
    thread float &out_distance,
    thread float3 &out_normal) {
    const float3 a = float3(primitive.a);
    const float3 b = float3(primitive.b);
    const float3 ba = b - a;
    const float baba = dot(ba, ba);
    const float rdrd = dot(view_ray.direction, view_ray.direction);
    if (rdrd <= 1.0e-20f) {
        return false;
    }
    const float3 shift_target = baba <= length_sq_epsilon ? a : mix(a, b, 0.5f);
    const float shift_t = approximate_ray_shift(view_ray, shift_target, min_hit_distance, view_ray.max_distance);
    const float3 shifted_origin = view_ray.origin + view_ray.direction * shift_t;

    const float bard = dot(ba, view_ray.direction);
    const float3 shifted_oa = shifted_origin - a;
    const float baoa = dot(ba, shifted_oa);
    const float rdoa = dot(view_ray.direction, shifted_oa);
    const float oaoa = dot(shifted_oa, shifted_oa);
    const float radius_sq = primitive.radius * primitive.radius;
    float best_distance = INFINITY;
    float3 best_normal = float3(0.0f, 1.0f, 0.0f);

    if (baba <= length_sq_epsilon) {
        float sphere_distance = 0.0f;
        float3 sphere_normal = float3(0.0f);
        if (!intersect_sphere(view_ray, a, primitive.radius, min_hit_distance, sphere_distance, sphere_normal)) {
            return false;
        }
        out_distance = sphere_distance;
        out_normal = sphere_normal;
        return true;
    }

    {
        const float qa = baba * rdrd - bard * bard;
        const float qb = baba * rdoa - baoa * bard;
        const float qc = baba * oaoa - baoa * baoa - radius_sq * baba;
        const float h = qb * qb - qa * qc;
        if (h >= 0.0f && fabs(qa) > length_sq_epsilon) {
            const float s = sqrt(h);
            const float local_t0 = (-qb - s) / qa;
            const float t0 = local_t0 + shift_t;
            const float y0 = baoa + local_t0 * bard;
            if (t0 >= min_hit_distance && t0 < best_distance && y0 >= 0.0f && y0 <= baba) {
                best_distance = t0;
            }

            const float local_t1 = (-qb + s) / qa;
            const float t1 = local_t1 + shift_t;
            const float y1 = baoa + local_t1 * bard;
            if (t1 >= min_hit_distance && t1 < best_distance && y1 >= 0.0f && y1 <= baba) {
                best_distance = t1;
            }
        }
    }

    float sphere_distance = 0.0f;
    float3 sphere_normal = float3(0.0f);
    if (intersect_sphere(view_ray, a, primitive.radius, min_hit_distance, sphere_distance, sphere_normal) &&
        sphere_distance < best_distance) {
        best_distance = sphere_distance;
        best_normal = sphere_normal;
    }
    if (intersect_sphere(view_ray, b, primitive.radius, min_hit_distance, sphere_distance, sphere_normal) &&
        sphere_distance < best_distance) {
        best_distance = sphere_distance;
        best_normal = sphere_normal;
    }

    if (!isfinite(best_distance)) {
        return false;
    }
    const float3 position = view_ray.origin + view_ray.direction * best_distance;
    const float u = clamp(dot(position - a, ba) / baba, 0.0f, 1.0f);
    const float3 axis_point = a + ba * u;
    best_normal = normalize(position - axis_point);
    out_distance = best_distance;
    out_normal = best_normal;
    return true;
}

float4 shade_hit(const thread hit_info &hit, constant camera_gpu &camera) {
    float4 color = hit.color;
    if ((hit.flags & kLineFlagFixedColor) == 0u) {
        switch (camera.display_mode) {
        case 0u:
            color = float4(hit.normal * 0.5f + 0.5f, 1.0f);
            break;
        case 1u:
            color = hit.color;
            break;
        case 2u: {
            const float3 light_direction = normalize(float3(0.35f, 0.70f, 0.62f));
            const float diffuse = max(dot(hit.normal, light_direction), 0.0f);
            const float shaded = 0.18f + (1.0f - 0.18f) * diffuse;
            color = float4(float3(shaded), 1.0f);
            break;
        }
        case 3u:
            color = float4(hash_color(hit.primitive_id + hit.kind * 4099u + 1u), 1.0f);
            break;
        case 4u:
            color = float4(hash_color(hit.geometry_index + 1u), 1.0f);
            break;
        case 5u:
            color = float4(hash_color(hit.instance_index + 1u), 1.0f);
            break;
        default:
            color = hit.color;
            break;
        }
    }

    if ((hit.flags & kLineFlagFixedColor) == 0u &&
        camera.hover_highlight_kind != 0u &&
        camera.hover_highlight_kind == hit.kind + 1u &&
        camera.hover_primitive_index == hit.primitive_id) {
        const float luminance = dot(color.rgb, float3(0.299f, 0.587f, 0.114f));
        const float3 complement = 1.0f - color.rgb;
        const float3 bias = luminance >= 0.45f
            ? float3(0.15f, 0.15f, 0.15f)
            : float3(0.35f, 0.35f, 0.35f);
        const float3 target = clamp(complement + bias, 0.0f, 1.0f);
        color.rgb = mix(color.rgb, target, clamp(camera.hover_highlight_mix, 0.0f, 1.0f));
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
    const triangle_instance_metadata_gpu metadata = triangle_instance_metadata[instance_index];
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
    result.normal = normalize(cross(b - a, c - a));
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
    constant procedural_group_metadata_gpu* point_groups,
    uint point_group_count,
    constant procedural_group_metadata_gpu* line_groups,
    uint line_group_count,
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

    const hit_info point_hit = trace_points(view_ray, camera, point_scene, points, point_groups, point_group_count);
    if (point_hit.hit && (!best_hit.hit || point_hit.distance < best_hit.distance)) {
        best_hit = point_hit;
    }

    const hit_info line_hit = trace_lines(
        view_ray,
        camera,
        line_scene,
        lines,
        line_groups,
        line_group_count,
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
    constant procedural_group_metadata_gpu* point_groups,
    uint point_group_count)
{
    hit_info result{};
    if (is_null_acceleration_structure(point_scene) || points == nullptr) {
        return result;
    }

    const float min_hit_distance = scene_intersection_t_min(camera);
    intersection_query<> query(view_ray, point_scene);
    while (query.next()) {
        const uint primitive_index = query.get_candidate_primitive_id();
        if (!procedural_primitive_visible(primitive_index, point_groups, point_group_count)) {
            continue;
        }
        const point_gpu primitive = points[primitive_index];
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
    if (!procedural_primitive_visible(primitive_index, point_groups, point_group_count)) {
        return result;
    }
    const point_gpu primitive = points[primitive_index];
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
    constant procedural_group_metadata_gpu* line_groups,
    uint line_group_count,
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
        if (!procedural_primitive_visible(primitive_index, line_groups, line_group_count)) {
            continue;
        }
        const line_gpu primitive = lines[primitive_index];
        if (is_pick_pass && (primitive.flags & kLineFlagNonPickable) != 0u) {
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
    if (!procedural_primitive_visible(primitive_index, line_groups, line_group_count)) {
        return result;
    }
    const line_gpu primitive = lines[primitive_index];
    if (is_pick_pass && (primitive.flags & kLineFlagNonPickable) != 0u) {
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

float minimum_forward_projection(float3 bounds_min, float3 bounds_max, float3 forward) {
    float min_projection = dot(bounds_min, forward);
    for (uint corner_index = 1u; corner_index < 8u; ++corner_index) {
        const float3 corner(
            (corner_index & 1u) != 0u ? bounds_max.x : bounds_min.x,
            (corner_index & 2u) != 0u ? bounds_max.y : bounds_min.y,
            (corner_index & 4u) != 0u ? bounds_max.z : bounds_min.z);
        min_projection = min(min_projection, dot(corner, forward));
    }
    return min_projection;
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
    ray_origin = camera.origin.xyz;
    direction = camera.forward.xyz;
    if (projection == kProjectionOrthographic) {
        const float ortho_width = max(projection_param0, 0.001f);
        const float ortho_height = max(projection_param1, 0.001f);
        ray_origin +=
            camera.right.xyz * (ndc.x * ortho_width * 0.5f) +
            camera.up.xyz * (ndc.y * ortho_height * 0.5f);
        if (camera.scene_bounds_valid != 0u) {
            const float min_forward = minimum_forward_projection(
                camera.scene_bounds_min.xyz,
                camera.scene_bounds_max.xyz,
                camera.forward.xyz);
            const float scene_depth = max(length(camera.scene_bounds_max.xyz - camera.scene_bounds_min.xyz), 0.001f);
            const float forward_margin = max(scene_depth * 0.001f, 0.001f);
            const float origin_forward = dot(ray_origin, camera.forward.xyz);
            const float max_origin_forward = min_forward - forward_margin;
            if (origin_forward > max_origin_forward) {
                ray_origin -= camera.forward.xyz * (origin_forward - max_origin_forward);
            }
        }
        return;
    }
    if (projection == kProjectionFisheye) {
        const float yaw = ndc.x * projection_param0 * 0.5f;
        const float pitch = ndc.y * projection_param1 * 0.5f;
        direction = normalize(
            camera.forward.xyz * (cos(yaw) * cos(pitch)) +
            camera.right.xyz * sin(yaw) +
            camera.up.xyz * sin(pitch));
        return;
    }

    const float tan_half_fov = projection_param0;
    direction = normalize(
        camera.forward.xyz +
        camera.right.xyz * (ndc.x * camera.aspect * tan_half_fov) +
        camera.up.xyz * (ndc.y * tan_half_fov));
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
    constant procedural_group_metadata_gpu* point_groups [[buffer(15)]],
    constant procedural_group_metadata_gpu* line_groups [[buffer(16)]],
    constant procedural_group_count_gpu &group_counts [[buffer(17)]],
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
            point_groups,
            group_counts.point_group_count,
            line_groups,
            group_counts.line_group_count,
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
        throughput *= mix(float3(1.0f), shaded.rgb, alpha);
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
    constant procedural_group_metadata_gpu* point_groups [[buffer(15)]],
    constant procedural_group_metadata_gpu* line_groups [[buffer(16)]],
    constant procedural_group_count_gpu &group_counts [[buffer(17)]],
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
        point_groups,
        group_counts.point_group_count,
        line_groups,
        group_counts.line_group_count,
        true);
    if (!best_hit.hit) {
        return;
    }

    pick_result[0].primitive_kind = best_hit.kind + 1u;
    pick_result[0].primitive_index = best_hit.primitive_id;
    pick_result[0].distance = best_hit.distance;
}
