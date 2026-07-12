float minimum_forward_projection(float3 bounds_min, float3 bounds_max, float3 forward)
{
    float min_projection = dot(bounds_min, forward);
    LOOP_UNROLL
    for (uint32_t corner_index = 1; corner_index < 8; ++corner_index)
    {
        const float3 corner = float3(
            (corner_index & 1u) != 0u ? bounds_max.x : bounds_min.x,
            (corner_index & 2u) != 0u ? bounds_max.y : bounds_min.y,
            (corner_index & 4u) != 0u ? bounds_max.z : bounds_min.z);
        min_projection = min(min_projection, dot(corner, forward));
    }
    return min_projection;
}

float scene_scale()
{
    if (shared_scene_bounds_valid() != 0u)
    {
        return max(length(shared_scene_bounds_max() - shared_scene_bounds_min()), 1.0e-3);
    }
    return 1.0;
}

float scene_intersection_t_min()
{
    return max(scene_scale() * 1.0e-5, kRayTMinFallback);
}

float scene_hit_advance_bias()
{
    return max(scene_scale() * 1.0e-5, kHitAdvanceBiasFallback);
}

float scene_length_sq_epsilon()
{
    const float scale = scene_scale();
    return max(scale * scale * 1.0e-12, 1.0e-12);
}

float approximate_ray_shift(
  float3 ray_origin, float3 ray_direction, float3 target, float min_t, float max_t)
{
    const float direction_len_sq = dot(ray_direction, ray_direction);
    if (direction_len_sq <= 1.0e-20 || max_t < min_t)
    {
        return 0.0;
    }

    const float projected_t = dot(target - ray_origin, ray_direction) / direction_len_sq;
    return clamp(projected_t, min_t, max_t);
}

float encode_srgb_channel(float value)
{
    float x = saturate(value);
    if (x <= 0.0031308)
    {
        return x * 12.92;
    }
    return 1.055 * pow(x, 1.0 / 2.4) - 0.055;
}

float3 hash_color(uint32_t seed)
{
    float3 h = frac(float3(seed * 0.1031, seed * 0.11369, seed * 0.13787));
    h += dot(h, h.yzx + 19.19);
    return frac(float3((h.x + h.y) * h.z, (h.x + h.z) * h.y, (h.y + h.z) * h.x));
}

float3 safe_normalize(float3 v, float3 fallback)
{
    const float len_sq = dot(v, v);
    if (len_sq <= 1.0e-12)
    {
        return fallback;
    }
    return v * rsqrt(len_sq);
}

float3 triangle_normal(uint32_t index_offset)
{
    const uint32_t ia = shared_scene_index(index_offset + 0);
    const uint32_t ib = shared_scene_index(index_offset + 1);
    const uint32_t ic = shared_scene_index(index_offset + 2);
    const float3 a = shared_scene_position(ia);
    const float3 b = shared_scene_position(ib);
    const float3 c = shared_scene_position(ic);
    return safe_normalize(cross(b - a, c - a), float3(0.0, 1.0, 0.0));
}

float3 point_normal(const SharedPointPrimitive point_primitive, float3 hit_position)
{
    return safe_normalize(hit_position - point_primitive.position, float3(0.0, 1.0, 0.0));
}

float3 line_normal(const SharedLinePrimitive line_primitive, float3 hit_position)
{
    const float3 ab = line_primitive.b - line_primitive.a;
    const float ab_len_sq = dot(ab, ab);
    if (ab_len_sq <= 1.0e-12)
    {
        return safe_normalize(hit_position - line_primitive.a, float3(0.0, 1.0, 0.0));
    }

    const float u = saturate(dot(hit_position - line_primitive.a, ab) / ab_len_sq);
    const float3 closest = lerp(line_primitive.a, line_primitive.b, u);
    return safe_normalize(hit_position - closest, float3(0.0, 1.0, 0.0));
}

float4 apply_display_mode(
  float4 client_color, float3 n, uint32_t primitive_seed,
  uint32_t geometry_index, uint32_t instance_index)
{
    const float alpha = saturate(client_color.a);
    if (shared_display_mode() == 1u)
    {
        return float4(client_color.rgb, alpha);
    }
    if (shared_display_mode() == 2u)
    {
        const float3 light_dir = normalize(float3(0.35, 0.70, 0.62));
        const float diffuse = max(0.0, dot(n, light_dir));
        const float intensity = 0.18 + (1.0 - 0.18) * diffuse;
        const float shaded = encode_srgb_channel(intensity);
        return float4(shaded, shaded, shaded, 1.0);
    }
    if (shared_display_mode() == 3u)
    {
        return float4(hash_color(primitive_seed + 1), 1.0);
    }
    if (shared_display_mode() == 4u)
    {
        return float4(hash_color(geometry_index + 1), 1.0);
    }
    if (shared_display_mode() == 5u)
    {
        return float4(hash_color(instance_index + 1), 1.0);
    }
    return float4(n * 0.5 + 0.5, 1.0);
}

float4 apply_hover_highlight(float4 color, uint32_t primitive_kind, uint32_t primitive_index)
{
    if (shared_hover_highlight_kind() == 0u ||
        primitive_kind != shared_hover_highlight_kind() ||
        primitive_index != shared_hover_primitive_index())
    {
        return color;
    }

    const float luminance = dot(color.rgb, float3(0.299, 0.587, 0.114));
    const float3 complement = 1.0 - color.rgb;
    const float3 bias = luminance >= 0.45
        ? float3(0.15, 0.15, 0.15)
        : float3(0.35, 0.35, 0.35);
    const float3 target = saturate(complement + bias);
    color.rgb = lerp(color.rgb, target, saturate(shared_hover_highlight_mix()));
    return color;
}

float4 triangle_surface_rgba(
  uint32_t triangle_index, uint32_t index_offset,
  uint32_t geometry_index, uint32_t instance_index)
{
    return apply_hover_highlight(
        apply_display_mode(
            shared_triangle_color(triangle_index),
            triangle_normal(index_offset),
            triangle_index,
            geometry_index,
            instance_index),
        1u,
        triangle_index);
}

float4 point_surface_rgba(uint32_t point_index, float3 hit_position, uint32_t instance_index)
{
    const SharedPointPrimitive point_primitive = shared_point_primitive(point_index);
    return apply_hover_highlight(
        apply_display_mode(
            point_primitive.color,
            point_normal(point_primitive, hit_position),
            kPointPrimitiveSeedBase + point_index,
            0u,
            instance_index),
        2u,
        point_index);
}

float4 line_surface_rgba(uint32_t line_index, float3 hit_position, uint32_t instance_index)
{
    const SharedLinePrimitive line_primitive = shared_line_primitive(line_index);
    if ((line_primitive.flags & kLineFlagFixedColor) != 0u)
    {
        return line_primitive.color;
    }
    return apply_hover_highlight(
        apply_display_mode(
            line_primitive.color,
            line_normal(line_primitive, hit_position),
            kLinePrimitiveSeedBase + line_index,
            0u,
            instance_index),
        3u,
        line_index);
}

void build_projection_ray(
    uint32_t projection,
    float2 uv,
    float projection_param0,
    float projection_param1,
    ARG_OUT(float3, ray_origin),
    ARG_OUT(float3, ray_direction))
{
    ray_origin = shared_origin();
    ray_direction = shared_forward();
    if (projection == kProjectionOrthographic)
    {
        const float ortho_width = max(projection_param0, 0.001);
        const float ortho_height = max(projection_param1, 0.001);
        ray_origin +=
            shared_right() * (uv.x * ortho_width * 0.5) +
            shared_up() * (uv.y * ortho_height * 0.5);
        if (shared_scene_bounds_valid() != 0u)
        {
            const float min_forward = minimum_forward_projection(
                shared_scene_bounds_min(),
                shared_scene_bounds_max(),
                shared_forward());
            const float scene_depth = max(length(shared_scene_bounds_max() - shared_scene_bounds_min()), 0.001);
            const float forward_margin = max(scene_depth * 0.001, 0.001);
            const float origin_forward = dot(ray_origin, shared_forward());
            const float max_origin_forward = min_forward - forward_margin;
            if (origin_forward > max_origin_forward)
            {
                ray_origin -= shared_forward() * (origin_forward - max_origin_forward);
            }
        }
        return;
    }
    if (projection == kProjectionFisheye)
    {
        const float yaw = uv.x * projection_param0 * 0.5;
        const float pitch = uv.y * projection_param1 * 0.5;
        ray_direction = normalize(
            shared_forward() * (cos(yaw) * cos(pitch)) +
            shared_right() * sin(yaw) +
            shared_up() * sin(pitch));
        return;
    }

    const float tan_half_fov = projection_param0;
    ray_direction = normalize(
        shared_forward() + uv.x * shared_aspect() * tan_half_fov * shared_right() +
        uv.y * tan_half_fov * shared_up());
}

void shared_raygen()
{
    const uint2 pixel = DispatchRaysIndex().xy;
    const uint32_t sample_index = shared_accumulation_sample_index();
    const float2 jitter = shared_accumulation_jitter();
    float2 uv = ((float2(pixel) + 0.5 + jitter) / float2(shared_width(), shared_height())) * 2.0 - 1.0;
    uv.y = -uv.y;
    float3 accum = float3(0.0, 0.0, 0.0);
    float3 throughput = float3(1.0, 1.0, 1.0);
    float3 ray_origin = shared_origin();
    float3 direction = shared_forward();
    build_projection_ray(
        shared_projection_blend_from(),
        uv,
        shared_projection_param_from0(),
        shared_projection_param_from1(),
        ray_origin,
        direction);
    const float blend_t = saturate(shared_projection_blend_t());
    if (shared_projection_blend_from() != shared_projection_blend_to() && blend_t < 1.0)
    {
        float3 ray_origin_b;
        float3 direction_b;
        build_projection_ray(
            shared_projection_blend_to(),
            uv,
            shared_projection_param_to0(),
            shared_projection_param_to1(),
            ray_origin_b,
            direction_b);
        ray_origin = lerp(ray_origin, ray_origin_b, blend_t);
        direction = normalize(lerp(direction, direction_b, blend_t));
    }

    LOOP_LOOP
    for (uint32_t layer = 0; layer < kMaxTransparencyLayers; ++layer)
    {
        RayDesc ray;
        ray.Origin = ray_origin;
        ray.Direction = direction;
        ray.TMin = scene_intersection_t_min();
        ray.TMax = kRayTMax;

        Payload payload;
        payload.color = float4(0.0, 0.0, 0.0, 0.0);
        payload.hit_t = 0.0;
        payload.hit = 0u;
        payload.primitive_kind = 0u;
        payload.primitive_index = 0u;
        TraceRay(g_scene, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);

        if (payload.hit == 0u)
        {
            accum += throughput * payload.color.rgb;
            break;
        }

        const float alpha = saturate(payload.color.a);
        if (alpha >= kAlphaOpaqueThreshold)
        {
            accum += throughput * payload.color.rgb;
            break;
        }

        accum += throughput * payload.color.rgb * alpha;
        throughput *= lerp(float3(1.0, 1.0, 1.0), payload.color.rgb, alpha);
        ray_origin += direction * (payload.hit_t + scene_hit_advance_bias());

        if (max(throughput.r, max(throughput.g, throughput.b)) <= 0.001)
        {
            break;
        }
    }

    const float4 previous = sample_index == 0 ? float4(0.0, 0.0, 0.0, 0.0) : g_accum[pixel];
    const float sample_count = float(sample_index + 1);
    const float3 averaged = (previous.rgb * float(sample_index) + accum) / sample_count;
    const float4 output = float4(averaged, 1.0);
    g_accum[pixel] = output;
    g_output[pixel] = output;
}

void shared_pick_raygen()
{
    const float2 uv =
        ((float2(shared_pick_pixel_x(), shared_pick_pixel_y()) + 0.5) / float2(shared_width(), shared_height())) *
            2.0 -
        1.0;
    float2 remapped_uv = uv;
    remapped_uv.y = -remapped_uv.y;

    float3 ray_origin = shared_origin();
    float3 direction = shared_forward();
    build_projection_ray(
        shared_projection_blend_from(),
        remapped_uv,
        shared_projection_param_from0(),
        shared_projection_param_from1(),
        ray_origin,
        direction);
    const float blend_t = saturate(shared_projection_blend_t());
    if (shared_projection_blend_from() != shared_projection_blend_to() && blend_t < 1.0)
    {
        float3 ray_origin_b;
        float3 direction_b;
        build_projection_ray(
            shared_projection_blend_to(),
            remapped_uv,
            shared_projection_param_to0(),
            shared_projection_param_to1(),
            ray_origin_b,
            direction_b);
        ray_origin = lerp(ray_origin, ray_origin_b, blend_t);
        direction = normalize(lerp(direction, direction_b, blend_t));
    }

    RayDesc ray;
    ray.Origin = ray_origin;
    ray.Direction = direction;
    ray.TMin = scene_intersection_t_min();
    ray.TMax = kRayTMax;

    Payload payload;
    payload.color = float4(0.0, 0.0, 0.0, 0.0);
    payload.hit_t = 0.0;
    payload.hit = 0u;
    payload.primitive_kind = 0u;
    payload.primitive_index = 0u;
    TraceRay(g_scene, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);

    PickResult result;
    result.primitive_kind = payload.primitive_kind;
    result.primitive_index = payload.primitive_index;
    result.distance = payload.hit_t;
    result.hit = payload.hit;
    g_pick_output[0] = result;
}

void shared_miss(ARG_INOUT(Payload, payload))
{
    payload.color = float4(0.0, 0.0, 0.0, 1.0);
    payload.hit_t = 0.0;
    payload.hit = 0u;
    payload.primitive_kind = 0u;
    payload.primitive_index = 0u;
}

void shared_closest_hit_triangle(
    ARG_INOUT(Payload, payload), uint32_t geometry_index, uint32_t instance_index)
{
    const uint32_t metadata_index = instance_index * kMaxInstanceGeometryCount + geometry_index;
    const GeometryMetadata metadata = shared_instance_metadata(metadata_index);
    uint32_t local_primitive_index = PrimitiveIndex();
    if (local_primitive_index >= metadata.primitive_count)
    {
        if (local_primitive_index < metadata.primitive_offset ||
            local_primitive_index >= metadata.primitive_offset + metadata.primitive_count)
        {
            payload.color = float4(1.0, 0.0, 1.0, 1.0);
            payload.hit_t = RayTCurrent();
            payload.hit = 1u;
            payload.primitive_kind = 1u;
            payload.primitive_index = 0u;
            return;
        }
        local_primitive_index -= metadata.primitive_offset;
    }

    const uint32_t triangle_index = metadata.first_triangle + local_primitive_index;
    const uint32_t index_offset = metadata.index_offset + local_primitive_index * 3;
    payload.color = triangle_surface_rgba(triangle_index, index_offset, geometry_index, instance_index);
    payload.hit_t = RayTCurrent();
    payload.hit = 1u;
    payload.primitive_kind = 1u;
    payload.primitive_index = triangle_index;
}

void shared_intersection_point()
{
    const uint32_t point_index = shared_procedural_primitive_offset(InstanceID()) + PrimitiveIndex();
    const SharedPointPrimitive point_primitive = shared_point_primitive(point_index);
    const float3 ray_origin = ObjectRayOrigin();
    const float3 direction = ObjectRayDirection();
    const float min_t = max(RayTMin(), scene_intersection_t_min());
    const float shift_t = approximate_ray_shift(
        ray_origin,
        direction,
        point_primitive.position,
        min_t,
        RayTCurrent());
    const float3 shifted_origin = ray_origin + direction * shift_t;
    const float3 oc = shifted_origin - point_primitive.position;
    const float a = dot(direction, direction);
    const float b = dot(oc, direction);
    const float c = dot(oc, oc) - point_primitive.radius * point_primitive.radius;
    const float h = b * b - a * c;
    if (h < 0.0)
    {
        return;
    }

    const float sqrt_h = sqrt(h);
    const float inv_a = 1.0 / a;
    const float t0 = (-b - sqrt_h) * inv_a + shift_t;
    const float t1 = (-b + sqrt_h) * inv_a + shift_t;

    ProceduralAttributes attrs;
    attrs.uv = float2(0.0, 0.0);
    if (t0 >= min_t && t0 <= RayTCurrent())
    {
        ReportHit(t0, 0, attrs);
        return;
    }
    if (t1 >= min_t && t1 <= RayTCurrent())
    {
        ReportHit(t1, 0, attrs);
    }
}

void shared_closest_hit_point(ARG_INOUT(Payload, payload), uint32_t instance_index)
{
    const uint32_t point_index = shared_procedural_primitive_offset(instance_index) + PrimitiveIndex();
    const float3 hit_position = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    payload.color = point_surface_rgba(point_index, hit_position, instance_index);
    payload.hit_t = RayTCurrent();
    payload.hit = 1u;
    payload.primitive_kind = 2u;
    payload.primitive_index = point_index;
}

void shared_intersection_line()
{
    const uint32_t line_index = shared_procedural_primitive_offset(InstanceID()) + PrimitiveIndex();
    const SharedLinePrimitive line_primitive = shared_line_primitive(line_index);
    if (shared_is_pick_pass() != 0u && (line_primitive.flags & kLineFlagNonPickable) != 0u)
    {
        return;
    }
    const float3 pa = line_primitive.a;
    const float3 pb = line_primitive.b;
    const float radius = line_primitive.radius;
    const float radius_sq = radius * radius;
    const float3 ray_origin = ObjectRayOrigin();
    const float3 rd = ObjectRayDirection();
    const float3 ba = pb - pa;
    const float baba = dot(ba, ba);
    const float rdrd = dot(rd, rd);
    const float min_t = max(RayTMin(), scene_intersection_t_min());
    const float length_sq_epsilon = scene_length_sq_epsilon();
    if (rdrd <= 1.0e-20)
    {
        return;
    }

    const float3 shift_target = baba <= length_sq_epsilon ? pa : lerp(pa, pb, 0.5);
    const float shift_t = approximate_ray_shift(ray_origin, rd, shift_target, min_t, RayTCurrent());
    const float3 ro = ray_origin + rd * shift_t;
    const float3 oa = ro - pa;

    const float bard = dot(ba, rd);
    const float baoa = dot(ba, oa);
    const float rdoa = dot(rd, oa);
    const float oaoa = dot(oa, oa);

    ProceduralAttributes attrs;
    attrs.uv = float2(0.0, 0.0);
    float best_t = RayTCurrent();

    if (baba <= length_sq_epsilon)
    {
        const float b = dot(oa, rd);
        const float c = oaoa - radius_sq;
        const float h = b * b - rdrd * c;
        if (h < 0.0)
        {
            return;
        }

        const float s = sqrt(h);
        const float t0 = (-b - s) / rdrd + shift_t;
        const float t1 = (-b + s) / rdrd + shift_t;
        if (t0 >= min_t && t0 <= best_t)
        {
            best_t = t0;
        }
        if (t1 >= min_t && t1 <= best_t)
        {
            best_t = t1;
        }
        if (best_t <= RayTCurrent())
        {
            ReportHit(best_t, 0, attrs);
        }
        return;
    }

    {
        const float a = baba * rdrd - bard * bard;
        const float b = baba * rdoa - baoa * bard;
        const float c = baba * oaoa - baoa * baoa - radius_sq * baba;
        const float h = b * b - a * c;
        if (h >= 0.0 && abs(a) > length_sq_epsilon)
        {
            const float s = sqrt(h);

            const float local_t0 = (-b - s) / a;
            const float t0 = local_t0 + shift_t;
            const float y0 = baoa + local_t0 * bard;
            if (t0 >= min_t && t0 <= best_t && y0 >= 0.0 && y0 <= baba)
            {
                best_t = t0;
            }

            const float local_t1 = (-b + s) / a;
            const float t1 = local_t1 + shift_t;
            const float y1 = baoa + local_t1 * bard;
            if (t1 >= min_t && t1 <= best_t && y1 >= 0.0 && y1 <= baba)
            {
                best_t = t1;
            }
        }
    }

    {
        const float3 oc = ro - pa;
        const float b = dot(rd, oc);
        const float c = dot(oc, oc) - radius_sq;
        const float h = b * b - rdrd * c;
        if (h >= 0.0)
        {
            const float s = sqrt(h);
            const float t0 = (-b - s) / rdrd + shift_t;
            const float t1 = (-b + s) / rdrd + shift_t;
            if (t0 >= min_t && t0 <= best_t)
            {
                best_t = t0;
            }
            if (t1 >= min_t && t1 <= best_t)
            {
                best_t = t1;
            }
        }
    }

    {
        const float3 oc = ro - pb;
        const float b = dot(rd, oc);
        const float c = dot(oc, oc) - radius_sq;
        const float h = b * b - rdrd * c;
        if (h >= 0.0)
        {
            const float s = sqrt(h);
            const float t0 = (-b - s) / rdrd + shift_t;
            const float t1 = (-b + s) / rdrd + shift_t;
            if (t0 >= min_t && t0 <= best_t)
            {
                best_t = t0;
            }
            if (t1 >= min_t && t1 <= best_t)
            {
                best_t = t1;
            }
        }
    }

    if (best_t < RayTCurrent())
    {
        ReportHit(best_t, 0, attrs);
    }
}

void shared_closest_hit_line(ARG_INOUT(Payload, payload), uint32_t instance_index)
{
    const uint32_t line_index = shared_procedural_primitive_offset(instance_index) + PrimitiveIndex();
    const float3 hit_position = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    payload.color = line_surface_rgba(line_index, hit_position, instance_index);
    payload.hit_t = RayTCurrent();
    payload.hit = 1u;
    payload.primitive_kind = 3u;
    payload.primitive_index = line_index;
}
