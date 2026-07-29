#include "rt_logic_shared_core.h"

using namespace rtvdb;

float scene_scale()
{
    return core::scene_scale(
        shared_scene_bounds_valid(),
        shared_scene_bounds_min(),
        shared_scene_bounds_max());
}

float scene_intersection_t_min()
{
    return max(core::scene_intersection_t_min(scene_scale()), kRayTMinFallback);
}

float scene_hit_advance_bias()
{
    return max(core::scene_hit_advance_bias(scene_scale()), kHitAdvanceBiasFallback);
}

float scene_length_sq_epsilon()
{
    return core::scene_length_sq_epsilon(scene_scale());
}

float3 triangle_normal(uint32_t index_offset)
{
    const uint32_t ia = shared_scene_index(index_offset + 0);
    const uint32_t ib = shared_scene_index(index_offset + 1);
    const uint32_t ic = shared_scene_index(index_offset + 2);
    const float3 a = shared_scene_position(ia);
    const float3 b = shared_scene_position(ib);
    const float3 c = shared_scene_position(ic);
    return core::triangle_normal(a, b, c);
}

float3 point_normal(const SharedPointPrimitive point_primitive, float3 hit_position)
{
    return core::point_normal(point_primitive.position, hit_position);
}

float3 line_normal(const SharedLinePrimitive line_primitive, float3 hit_position)
{
    return core::line_normal(line_primitive.a, line_primitive.b, hit_position);
}

float4 apply_display_mode(
  float4 client_color, float3 n, uint32_t primitive_seed,
  uint32_t geometry_index, uint32_t instance_index)
{
    return core::apply_display_mode(
        client_color,
        n,
        primitive_seed,
        geometry_index,
        instance_index,
        shared_display_mode());
}

float4 apply_hover_highlight(float4 color, uint32_t primitive_kind, uint32_t primitive_index)
{
    return core::apply_hover_highlight(
        color,
        primitive_kind,
        primitive_index,
        shared_hover_highlight_kind(),
        shared_hover_primitive_index(),
        shared_hover_highlight_mix());
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
    const core::projection_ray result = core::build_projection_ray(
        projection,
        uv,
        projection_param0,
        projection_param1,
        shared_origin(),
        shared_forward(),
        shared_right(),
        shared_up(),
        shared_aspect(),
        shared_scene_bounds_valid(),
        shared_scene_bounds_min(),
        shared_scene_bounds_max());
    ray_origin = result.origin;
    ray_direction = result.direction;
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
        throughput *= 1.0 - alpha;
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

    const uint32_t triangle_index = metadata.primitive_base + local_primitive_index;
    const uint32_t index_offset = metadata.index_offset + local_primitive_index * 3;
    payload.color = triangle_surface_rgba(triangle_index, index_offset, geometry_index, instance_index);
    payload.hit_t = RayTCurrent();
    payload.hit = 1u;
    payload.primitive_kind = 1u;
    payload.primitive_index = triangle_index;
}

void shared_intersection_point()
{
    const uint32_t point_index = shared_procedural_primitive_offset(InstanceID(), GeometryIndex()) + PrimitiveIndex();
    const SharedPointPrimitive point_primitive = shared_point_primitive(point_index);
    if (core::point_contains(
            shared_origin(),
            point_primitive.position,
            point_primitive.radius,
            scene_length_sq_epsilon()))
    {
        return;
    }
    const float3 ray_origin = ObjectRayOrigin();
    const float3 direction = ObjectRayDirection();
    const float min_t = max(RayTMin(), scene_intersection_t_min());
    core::ray ray;
    ray.origin = ray_origin;
    ray.direction = direction;
    ray.min_distance = min_t;
    ray.max_distance = RayTCurrent();
    const core::intersection hit = core::intersect_sphere(
        ray,
        point_primitive.position,
        point_primitive.radius,
        min_t);

    ProceduralAttributes attrs;
    attrs.uv = float2(0.0, 0.0);
    if (hit.hit)
    {
        ReportHit(hit.distance, 0, attrs);
    }
}

void shared_closest_hit_point(ARG_INOUT(Payload, payload), uint32_t instance_index, uint32_t geometry_index)
{
    const uint32_t point_index = shared_procedural_primitive_offset(instance_index, geometry_index) + PrimitiveIndex();
    const float3 hit_position = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    payload.color = point_surface_rgba(point_index, hit_position, instance_index);
    payload.hit_t = RayTCurrent();
    payload.hit = 1u;
    payload.primitive_kind = 2u;
    payload.primitive_index = point_index;
}

void shared_intersection_line()
{
    const uint32_t line_index = shared_procedural_primitive_offset(InstanceID(), GeometryIndex()) + PrimitiveIndex();
    const SharedLinePrimitive line_primitive = shared_line_primitive(line_index);
    if (shared_is_pick_pass() != 0u && (line_primitive.flags & kLineFlagNonPickable) != 0u)
    {
        return;
    }
    if (core::capsule_contains(
            shared_origin(),
            line_primitive.a,
            line_primitive.b,
            line_primitive.radius,
            scene_length_sq_epsilon(),
            scene_length_sq_epsilon()))
    {
        return;
    }
    const float3 ray_origin = ObjectRayOrigin();
    const float3 direction = ObjectRayDirection();
    const float min_t = max(RayTMin(), scene_intersection_t_min());
    core::ray ray;
    ray.origin = ray_origin;
    ray.direction = direction;
    ray.min_distance = min_t;
    ray.max_distance = RayTCurrent();
    const core::intersection hit = core::intersect_capsule(
        ray,
        line_primitive.a,
        line_primitive.b,
        line_primitive.radius,
        min_t,
        scene_length_sq_epsilon());

    ProceduralAttributes attrs;
    attrs.uv = float2(0.0, 0.0);
    if (hit.hit)
    {
        ReportHit(hit.distance, 0, attrs);
    }
}

void shared_closest_hit_line(ARG_INOUT(Payload, payload), uint32_t instance_index, uint32_t geometry_index)
{
    const uint32_t line_index = shared_procedural_primitive_offset(instance_index, geometry_index) + PrimitiveIndex();
    const float3 hit_position = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    payload.color = line_surface_rgba(line_index, hit_position, instance_index);
    payload.hit_t = RayTCurrent();
    payload.hit = 1u;
    payload.primitive_kind = 3u;
    payload.primitive_index = line_index;
}
