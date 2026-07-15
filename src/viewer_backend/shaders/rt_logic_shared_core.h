struct rtvdb_core_ray
{
    float3 origin;
    float3 direction;
    float min_distance;
    float max_distance;
};

struct rtvdb_core_projection_ray
{
    float3 origin;
    float3 direction;
};

struct rtvdb_core_intersection
{
    bool hit;
    float distance;
    float3 normal;
};

float rtvdb_core_minimum_forward_projection(float3 bounds_min, float3 bounds_max, float3 forward)
{
    float min_projection = dot(bounds_min, forward);
    for (uint corner_index = 1u; corner_index < 8u; ++corner_index)
    {
        const float3 corner = float3(
            (corner_index & 1u) != 0u ? bounds_max.x : bounds_min.x,
            (corner_index & 2u) != 0u ? bounds_max.y : bounds_min.y,
            (corner_index & 4u) != 0u ? bounds_max.z : bounds_min.z);
        min_projection = min(min_projection, dot(corner, forward));
    }
    return min_projection;
}

float rtvdb_core_scene_scale(uint bounds_valid, float3 bounds_min, float3 bounds_max)
{
    if (bounds_valid != 0u)
    {
        return max(length(bounds_max - bounds_min), 1.0e-3);
    }
    return 1.0;
}

float rtvdb_core_scene_intersection_t_min(float scale)
{
    return max(scale * 1.0e-5, 1.0e-6);
}

float rtvdb_core_scene_hit_advance_bias(float scale)
{
    return max(scale * 1.0e-5, 1.0e-6);
}

float rtvdb_core_scene_length_sq_epsilon(float scale)
{
    return max(scale * scale * 1.0e-12, 1.0e-12);
}

float rtvdb_core_approximate_ray_shift(
    float3 ray_origin,
    float3 ray_direction,
    float3 target,
    float min_t,
    float max_t)
{
    const float direction_len_sq = dot(ray_direction, ray_direction);
    if (direction_len_sq <= 1.0e-20 || max_t < min_t)
    {
        return 0.0;
    }

    const float projected_t = dot(target - ray_origin, ray_direction) / direction_len_sq;
    return clamp(projected_t, min_t, max_t);
}

float rtvdb_core_encode_srgb_channel(float value)
{
    const float x = clamp(value, 0.0, 1.0);
    if (x <= 0.0031308)
    {
        return x * 12.92;
    }
    return 1.055 * pow(x, 1.0 / 2.4) - 0.055;
}

float3 rtvdb_core_hash_color(uint seed)
{
    float3 h = float3(float(seed), float(seed), float(seed)) * float3(0.1031, 0.11369, 0.13787);
    h -= floor(h);
    h += dot(h, h.yzx + 19.19);
    h -= floor(h);
    float3 result = float3(
        (h.x + h.y) * h.z,
        (h.x + h.z) * h.y,
        (h.y + h.z) * h.x);
    result -= floor(result);
    return result;
}

float3 rtvdb_core_safe_normalize(float3 value, float3 fallback)
{
    const float length_sq = dot(value, value);
    if (length_sq <= 1.0e-12)
    {
        return fallback;
    }
    return value * rsqrt(length_sq);
}

float3 rtvdb_core_triangle_normal(float3 a, float3 b, float3 c)
{
    return rtvdb_core_safe_normalize(cross(b - a, c - a), float3(0.0, 1.0, 0.0));
}

float3 rtvdb_core_point_normal(float3 center, float3 hit_position)
{
    return rtvdb_core_safe_normalize(hit_position - center, float3(0.0, 1.0, 0.0));
}

float3 rtvdb_core_line_normal(float3 a, float3 b, float3 hit_position)
{
    const float3 ab = b - a;
    const float ab_len_sq = dot(ab, ab);
    if (ab_len_sq <= 1.0e-12)
    {
        return rtvdb_core_point_normal(a, hit_position);
    }

    const float u = clamp(dot(hit_position - a, ab) / ab_len_sq, 0.0, 1.0);
    const float3 closest = a + (b - a) * u;
    return rtvdb_core_safe_normalize(hit_position - closest, float3(0.0, 1.0, 0.0));
}

bool rtvdb_core_point_contains(
    float3 position,
    float3 center,
    float radius,
    float distance_sq_epsilon)
{
    const float radius_sq = radius * radius;
    if (radius_sq <= distance_sq_epsilon)
    {
        return false;
    }
    return dot(position - center, position - center) < radius_sq;
}

bool rtvdb_core_capsule_contains(
    float3 position,
    float3 a,
    float3 b,
    float radius,
    float length_sq_epsilon,
    float distance_sq_epsilon)
{
    const float radius_sq = radius * radius;
    if (radius_sq <= distance_sq_epsilon)
    {
        return false;
    }

    const float3 ab = b - a;
    const float ab_len_sq = dot(ab, ab);
    const float u = ab_len_sq <= length_sq_epsilon
        ? 0.0
        : clamp(dot(position - a, ab) / ab_len_sq, 0.0, 1.0);
    const float3 closest = a + ab * u;
    return dot(position - closest, position - closest) < radius_sq;
}

float4 rtvdb_core_apply_display_mode(
    float4 client_color,
    float3 normal,
    uint primitive_seed,
    uint geometry_index,
    uint instance_index,
    uint display_mode)
{
    const float alpha = clamp(client_color.a, 0.0, 1.0);
    if (display_mode == 1u)
    {
        return float4(client_color.rgb, alpha);
    }
    if (display_mode == 2u)
    {
        const float3 light_dir = normalize(float3(0.35, 0.70, 0.62));
        const float diffuse = max(0.0, dot(normal, light_dir));
        const float intensity = 0.18 + (1.0 - 0.18) * diffuse;
        const float shaded = rtvdb_core_encode_srgb_channel(intensity);
        return float4(shaded, shaded, shaded, 1.0);
    }
    if (display_mode == 3u)
    {
        return float4(rtvdb_core_hash_color(primitive_seed + 1u), 1.0);
    }
    if (display_mode == 4u)
    {
        return float4(rtvdb_core_hash_color(geometry_index + 1u), 1.0);
    }
    if (display_mode == 5u)
    {
        return float4(rtvdb_core_hash_color(instance_index + 1u), 1.0);
    }
    return float4(normal * 0.5 + 0.5, 1.0);
}

float4 rtvdb_core_apply_hover_highlight(
    float4 color,
    uint primitive_kind,
    uint primitive_index,
    uint hover_kind,
    uint hover_primitive_index,
    float hover_mix)
{
    if (hover_kind == 0u ||
        primitive_kind != hover_kind ||
        primitive_index != hover_primitive_index)
    {
        return color;
    }

    const float luminance = dot(color.rgb, float3(0.299, 0.587, 0.114));
    const float3 complement = 1.0 - color.rgb;
    const float3 bias = luminance >= 0.45
        ? float3(0.15, 0.15, 0.15)
        : float3(0.35, 0.35, 0.35);
    const float3 target = clamp(complement + bias, 0.0, 1.0);
    color.rgb = color.rgb + (target - color.rgb) * clamp(hover_mix, 0.0, 1.0);
    return color;
}

rtvdb_core_projection_ray rtvdb_core_build_projection_ray(
    uint projection,
    float2 uv,
    float projection_param0,
    float projection_param1,
    float3 origin,
    float3 forward,
    float3 right,
    float3 up,
    float aspect,
    uint scene_bounds_valid,
    float3 scene_bounds_min,
    float3 scene_bounds_max)
{
    rtvdb_core_projection_ray result;
    result.origin = origin;
    result.direction = forward;
    if (projection == 2u)
    {
        const float ortho_width = max(projection_param0, 0.001);
        const float ortho_height = max(projection_param1, 0.001);
        result.origin += right * (uv.x * ortho_width * 0.5) + up * (uv.y * ortho_height * 0.5);
        if (scene_bounds_valid != 0u)
        {
            const float min_forward = rtvdb_core_minimum_forward_projection(
                scene_bounds_min,
                scene_bounds_max,
                forward);
            const float scene_depth = max(length(scene_bounds_max - scene_bounds_min), 0.001);
            const float forward_margin = max(scene_depth * 0.001, 0.001);
            const float origin_forward = dot(result.origin, forward);
            const float max_origin_forward = min_forward - forward_margin;
            if (origin_forward > max_origin_forward)
            {
                result.origin -= forward * (origin_forward - max_origin_forward);
            }
        }
        return result;
    }
    if (projection == 1u)
    {
        const float yaw = uv.x * projection_param0 * 0.5;
        const float pitch = uv.y * projection_param1 * 0.5;
        result.direction = normalize(
            forward * (cos(yaw) * cos(pitch)) +
            right * sin(yaw) +
            up * sin(pitch));
        return result;
    }

    result.direction = normalize(
        forward + uv.x * aspect * projection_param0 * right + uv.y * projection_param0 * up);
    return result;
}

rtvdb_core_intersection rtvdb_core_intersect_sphere(
    rtvdb_core_ray view_ray,
    float3 center,
    float radius,
    float min_hit_distance)
{
    rtvdb_core_intersection result;
    result.hit = false;
    result.distance = 0.0;
    result.normal = float3(0.0, 1.0, 0.0);

    const float a = dot(view_ray.direction, view_ray.direction);
    if (a <= 1.0e-20)
    {
        return result;
    }
    const float shift_t = rtvdb_core_approximate_ray_shift(
        view_ray.origin,
        view_ray.direction,
        center,
        min_hit_distance,
        view_ray.max_distance);
    const float3 shifted_origin = view_ray.origin + view_ray.direction * shift_t;
    const float3 oc = shifted_origin - center;
    const float b = dot(oc, view_ray.direction);
    const float c = dot(oc, oc) - radius * radius;
    const float discriminant = b * b - a * c;
    if (discriminant < 0.0)
    {
        return result;
    }

    const float sqrt_discriminant = sqrt(discriminant);
    float distance = (-b - sqrt_discriminant) / a + shift_t;
    if (distance <= min_hit_distance)
    {
        distance = (-b + sqrt_discriminant) / a + shift_t;
    }
    if (distance <= min_hit_distance || distance > view_ray.max_distance)
    {
        return result;
    }

    const float3 position = view_ray.origin + view_ray.direction * distance;
    result.hit = true;
    result.distance = distance;
    result.normal = rtvdb_core_safe_normalize(position - center, float3(0.0, 1.0, 0.0));
    return result;
}

rtvdb_core_intersection rtvdb_core_intersect_capsule(
    rtvdb_core_ray view_ray,
    float3 a,
    float3 b,
    float radius,
    float min_hit_distance,
    float length_sq_epsilon)
{
    rtvdb_core_intersection result;
    result.hit = false;
    result.distance = 0.0;
    result.normal = float3(0.0, 1.0, 0.0);

    const float3 ba = b - a;
    const float baba = dot(ba, ba);
    const float rdrd = dot(view_ray.direction, view_ray.direction);
    if (rdrd <= 1.0e-20)
    {
        return result;
    }

    const float3 shift_target = baba <= length_sq_epsilon ? a : a + ba * 0.5;
    const float shift_t = rtvdb_core_approximate_ray_shift(
        view_ray.origin,
        view_ray.direction,
        shift_target,
        min_hit_distance,
        view_ray.max_distance);
    const float3 shifted_origin = view_ray.origin + view_ray.direction * shift_t;
    const float bard = dot(ba, view_ray.direction);
    const float3 shifted_oa = shifted_origin - a;
    const float baoa = dot(ba, shifted_oa);
    const float rdoa = dot(view_ray.direction, shifted_oa);
    const float oaoa = dot(shifted_oa, shifted_oa);
    const float radius_sq = radius * radius;
    float best_distance = view_ray.max_distance;

    if (baba <= length_sq_epsilon)
    {
        return rtvdb_core_intersect_sphere(view_ray, a, radius, min_hit_distance);
    }

    {
        const float qa = baba * rdrd - bard * bard;
        const float qb = baba * rdoa - baoa * bard;
        const float qc = baba * oaoa - baoa * baoa - radius_sq * baba;
        const float h = qb * qb - qa * qc;
        if (h >= 0.0 && abs(qa) > length_sq_epsilon)
        {
            const float s = sqrt(h);
            const float local_t0 = (-qb - s) / qa;
            const float t0 = local_t0 + shift_t;
            const float y0 = baoa + local_t0 * bard;
            if (t0 >= min_hit_distance && t0 <= best_distance && y0 >= 0.0 && y0 <= baba)
            {
                best_distance = t0;
            }

            const float local_t1 = (-qb + s) / qa;
            const float t1 = local_t1 + shift_t;
            const float y1 = baoa + local_t1 * bard;
            if (t1 >= min_hit_distance && t1 <= best_distance && y1 >= 0.0 && y1 <= baba)
            {
                best_distance = t1;
            }
        }
    }

    const rtvdb_core_intersection hit_a = rtvdb_core_intersect_sphere(
        view_ray,
        a,
        radius,
        min_hit_distance);
    if (hit_a.hit && hit_a.distance < best_distance)
    {
        best_distance = hit_a.distance;
    }
    const rtvdb_core_intersection hit_b = rtvdb_core_intersect_sphere(
        view_ray,
        b,
        radius,
        min_hit_distance);
    if (hit_b.hit && hit_b.distance < best_distance)
    {
        best_distance = hit_b.distance;
    }

    if (best_distance >= view_ray.max_distance)
    {
        return result;
    }
    const float3 position = view_ray.origin + view_ray.direction * best_distance;
    const float u = clamp(dot(position - a, ba) / baba, 0.0, 1.0);
    const float3 axis_point = a + ba * u;
    result.hit = true;
    result.distance = best_distance;
    result.normal = rtvdb_core_safe_normalize(position - axis_point, float3(0.0, 1.0, 0.0));
    return result;
}
