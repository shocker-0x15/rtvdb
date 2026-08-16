#include <metal_stdlib>
#include <metal_raytracing>

using namespace metal;
using namespace metal::raytracing;

#include "../shaders/rt_logic_shared_core.h"

using namespace rtvdb;

struct geometry_metadata_gpu {
    uint primitive_base;
    uint index_offset;
    uint primitive_offset;
    uint primitive_count;
};

struct viewer_constants_gpu {
    float4 origin;
    float4 forward;
    float4 right;
    float4 up;
    float4 scene_bounds_min;
    float4 scene_bounds_max;
    uint4 size_and_mode;
    float4 projection_from;
    float4 projection_to;
    uint4 projection_modes;
    float4 blend_and_jitter;
    uint4 pick_and_flags;
    uint4 pick_params;
};

struct point_gpu {
    float4 position_radius;
    float4 color;
};

struct line_gpu {
    float4 a_radius;
    float4 b_pad;
    float4 color;
    uint flags;
    uint pad0;
    uint pad1;
    uint pad2;
};

struct pick_result_gpu {
    uint primitive_kind;
    uint primitive_index;
    float distance;
    uint hit;
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

static_assert(sizeof(geometry_metadata_gpu) == 16);
static_assert(sizeof(viewer_constants_gpu) == 208);
static_assert(sizeof(point_gpu) == 32);
static_assert(sizeof(line_gpu) == 64);
static_assert(sizeof(pick_result_gpu) == 16);

constant float kAlphaOpaqueThreshold = 0.999f;
constant float kRayMinDistanceFallback = 1.0e-6f;
constant float kRayHitAdvanceBiasFallback = 1.0e-6f;
constant float kRayMaxDistance = 10000.0f;
constant uint kMaxTransparencyLayers = 16u;
constant uint kMaxInstanceGeometryCount = 4u;
constant uint kInstanceKindTriangle = 0u;
constant uint kInstanceKindPoint = 1u;
constant uint kInstanceKindLine = 2u;
constant uint kLineFlagFixedColor = 1u << 0;
constant uint kLineFlagNonPickable = 1u << 1;

uint view_width(constant viewer_constants_gpu &view) {
    return view.size_and_mode.x;
}

uint view_height(constant viewer_constants_gpu &view) {
    return view.size_and_mode.y;
}

uint view_display_mode(constant viewer_constants_gpu &view) {
    return view.size_and_mode.z;
}

uint view_accumulation_sample_index(constant viewer_constants_gpu &view) {
    return view.size_and_mode.w;
}

uint view_scene_bounds_valid(constant viewer_constants_gpu &view) {
    return view.scene_bounds_max.w != 0.0f ? 1u : 0u;
}

float scene_scale(constant viewer_constants_gpu &view) {
    return core::scene_scale(
        view_scene_bounds_valid(view),
        view.scene_bounds_min.xyz,
        view.scene_bounds_max.xyz);
}

float scene_intersection_t_min(constant viewer_constants_gpu &view) {
    return max(core::scene_intersection_t_min(scene_scale(view)), kRayMinDistanceFallback);
}

float scene_hit_advance_bias(constant viewer_constants_gpu &view) {
    return max(core::scene_hit_advance_bias(scene_scale(view)), kRayHitAdvanceBiasFallback);
}

float scene_length_sq_epsilon(constant viewer_constants_gpu &view) {
    return core::scene_length_sq_epsilon(scene_scale(view));
}

float4 apply_highlights(
    float4 color,
    uint primitive_kind,
    uint primitive_index,
    constant viewer_constants_gpu &view,
    bool allow_hover)
{
    if (view.pick_and_flags.w != 0u &&
        primitive_kind == view.pick_and_flags.w &&
        primitive_index == view.pick_params.w)
    {
        return core::apply_selection_highlight(
            color,
            primitive_kind,
            primitive_index,
            view.pick_and_flags.w,
            view.pick_params.w);
    }
    if (!allow_hover)
    {
        return color;
    }
    return core::apply_hover_highlight(
        color,
        primitive_kind,
        primitive_index,
        view.projection_modes.z,
        view.projection_modes.w,
        view.blend_and_jitter.w);
}

bool resolve_local_primitive(
    geometry_metadata_gpu metadata,
    uint native_primitive_index,
    thread uint &out_local_primitive)
{
    if (metadata.primitive_count == 0u) {
        return false;
    }
    uint local_primitive = native_primitive_index;
    if (local_primitive >= metadata.primitive_count) {
        if (local_primitive < metadata.primitive_offset ||
            local_primitive - metadata.primitive_offset >= metadata.primitive_count) {
            return false;
        }
        local_primitive -= metadata.primitive_offset;
    }
    out_local_primitive = local_primitive;
    return true;
}

bool intersect_sphere(
    ray view_ray,
    float3 center,
    float radius,
    float min_hit_distance,
    thread float &out_distance,
    thread float3 &out_normal)
{
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
    thread float3 &out_normal)
{
    core::ray core_ray;
    core_ray.origin = view_ray.origin;
    core_ray.direction = view_ray.direction;
    core_ray.min_distance = min_hit_distance;
    core_ray.max_distance = view_ray.max_distance;
    const core::intersection hit = core::intersect_capsule(
        core_ray,
        primitive.a_radius.xyz,
        primitive.b_pad.xyz,
        primitive.a_radius.w,
        min_hit_distance,
        length_sq_epsilon);
    if (!hit.hit) {
        return false;
    }
    out_distance = hit.distance;
    out_normal = hit.normal;
    return true;
}

ray transform_ray_to_object_space(ray world_ray, float4x3 world_to_object) {
    return ray(
        world_to_object * float4(world_ray.origin, 1.0f),
        world_to_object * float4(world_ray.direction, 0.0f),
        world_ray.min_distance,
        world_ray.max_distance);
}

float3 transform_normal_to_world_space(float3 object_normal, float4x3 world_to_object) {
    const float3x3 normal_transform = transpose(float3x3(
        world_to_object[0],
        world_to_object[1],
        world_to_object[2]));
    return normalize(normal_transform * object_normal);
}

float4 shade_hit(const thread hit_info &hit, constant viewer_constants_gpu &view) {
    float4 color = hit.color;
    const bool allow_hover = (hit.flags & kLineFlagFixedColor) == 0u;
    if (allow_hover) {
        const uint primitive_seed = hit.kind == kInstanceKindTriangle
            ? hit.primitive_id
            : hit.kind == kInstanceKindPoint
                ? 1000000u + hit.primitive_id
                : 2000000u + hit.primitive_id;
        color = core::apply_display_mode(
            hit.color,
            hit.normal,
            primitive_seed,
            hit.geometry_index,
            hit.instance_index,
            view_display_mode(view));
    }
    return apply_highlights(color, hit.kind + 1u, hit.primitive_id, view, allow_hover);
}

hit_info trace_nearest_hit(
    ray view_ray,
    constant viewer_constants_gpu &view,
    instance_acceleration_structure scene,
    constant float4* triangle_colors,
    constant geometry_metadata_gpu* instance_metadata,
    constant float4* positions,
    constant uint* indices,
    constant point_gpu* points,
    constant line_gpu* lines,
    constant uint* instance_kinds,
    bool is_pick_pass)
{
    hit_info result{};
    if (is_null_acceleration_structure(scene) || instance_metadata == nullptr ||
        instance_kinds == nullptr) {
        return result;
    }

    const float min_hit_distance = scene_intersection_t_min(view);
    const float length_sq_epsilon = scene_length_sq_epsilon(view);
    intersection_query<instancing, triangle_data> query(view_ray, scene);
    while (query.next()) {
        const intersection_type candidate_type = query.get_candidate_intersection_type();
        if (candidate_type == intersection_type::triangle) {
            query.commit_triangle_intersection();
            continue;
        }
        if (candidate_type != intersection_type::bounding_box) {
            continue;
        }

        const uint system_instance_id = query.get_candidate_instance_id();
        const uint user_instance_id = query.get_candidate_user_instance_id();
        const uint geometry_index = query.get_candidate_geometry_id();
        const uint native_primitive_index = query.get_candidate_primitive_id();
        const uint instance_kind = instance_kinds[system_instance_id];
        const geometry_metadata_gpu metadata =
            instance_metadata[user_instance_id * kMaxInstanceGeometryCount + geometry_index];
        uint local_primitive = 0u;
        if (!resolve_local_primitive(metadata, native_primitive_index, local_primitive)) {
            continue;
        }
        const uint primitive_index = metadata.primitive_base + local_primitive;
        const ray object_ray = transform_ray_to_object_space(
            view_ray,
            query.get_candidate_world_to_object_transform());
        float distance = 0.0f;
        float3 normal = float3(0.0f);
        if (instance_kind == kInstanceKindPoint && points != nullptr) {
            const point_gpu primitive = points[primitive_index];
            if (core::point_contains(
                    object_ray.origin,
                    primitive.position_radius.xyz,
                    primitive.position_radius.w,
                    length_sq_epsilon)) {
                continue;
            }
            if (intersect_sphere(
                    object_ray,
                    primitive.position_radius.xyz,
                    primitive.position_radius.w,
                    min_hit_distance,
                    distance,
                    normal)) {
                query.commit_bounding_box_intersection(distance);
            }
        } else if (instance_kind == kInstanceKindLine && lines != nullptr) {
            const line_gpu primitive = lines[primitive_index];
            if ((is_pick_pass && (primitive.flags & kLineFlagNonPickable) != 0u) ||
                core::capsule_contains(
                    object_ray.origin,
                    primitive.a_radius.xyz,
                    primitive.b_pad.xyz,
                    primitive.a_radius.w,
                    length_sq_epsilon,
                    length_sq_epsilon)) {
                continue;
            }
            if (intersect_capsule(
                    object_ray,
                    primitive,
                    min_hit_distance,
                    length_sq_epsilon,
                    distance,
                    normal)) {
                query.commit_bounding_box_intersection(distance);
            }
        }
    }

    const intersection_type committed_type = query.get_committed_intersection_type();
    if (committed_type == intersection_type::none) {
        return result;
    }
    const uint system_instance_id = query.get_committed_instance_id();
    const uint user_instance_id = query.get_committed_user_instance_id();
    const uint geometry_index = query.get_committed_geometry_id();
    const uint native_primitive_index = query.get_committed_primitive_id();
    const uint instance_kind = instance_kinds[system_instance_id];
    const geometry_metadata_gpu metadata =
        instance_metadata[user_instance_id * kMaxInstanceGeometryCount + geometry_index];
    uint local_primitive = 0u;
    if (!resolve_local_primitive(metadata, native_primitive_index, local_primitive)) {
        return result;
    }

    const uint primitive_index = metadata.primitive_base + local_primitive;
    result.hit = true;
    result.distance = query.get_committed_distance();
    result.primitive_id = primitive_index;
    result.kind = instance_kind;
    result.geometry_index = instance_kind == kInstanceKindTriangle ? geometry_index : 0u;
    result.instance_index = user_instance_id;
    result.flags = 0u;
    const float4x3 world_to_object = query.get_committed_world_to_object_transform();
    if (committed_type == intersection_type::triangle &&
        instance_kind == kInstanceKindTriangle && positions != nullptr &&
        indices != nullptr && triangle_colors != nullptr) {
        const uint index_offset = metadata.index_offset + local_primitive * 3u;
        const uint ia = indices[index_offset + 0u];
        const uint ib = indices[index_offset + 1u];
        const uint ic = indices[index_offset + 2u];
        result.normal = transform_normal_to_world_space(
            core::triangle_normal(
                positions[ia].xyz,
                positions[ib].xyz,
                positions[ic].xyz),
            world_to_object);
        result.color = triangle_colors[primitive_index];
        return result;
    }
    if (committed_type != intersection_type::bounding_box) {
        return hit_info{};
    }

    float distance = 0.0f;
    float3 normal = float3(0.0f);
    const ray object_ray = transform_ray_to_object_space(view_ray, world_to_object);
    if (instance_kind == kInstanceKindPoint && points != nullptr) {
        const point_gpu primitive = points[primitive_index];
        if (!intersect_sphere(
                object_ray,
                primitive.position_radius.xyz,
                primitive.position_radius.w,
                min_hit_distance,
                distance,
                normal)) {
            return hit_info{};
        }
        result.distance = distance;
        result.normal = transform_normal_to_world_space(normal, world_to_object);
        result.color = primitive.color;
        return result;
    }
    if (instance_kind == kInstanceKindLine && lines != nullptr) {
        const line_gpu primitive = lines[primitive_index];
        if ((is_pick_pass && (primitive.flags & kLineFlagNonPickable) != 0u) ||
            !intersect_capsule(
                object_ray,
                primitive,
                min_hit_distance,
                length_sq_epsilon,
                distance,
                normal)) {
            return hit_info{};
        }
        result.distance = distance;
        result.normal = transform_normal_to_world_space(normal, world_to_object);
        result.color = primitive.color;
        result.flags = primitive.flags;
        return result;
    }
    return hit_info{};
}

void build_projection_ray(
    constant viewer_constants_gpu &view,
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
        view.origin.xyz,
        view.forward.xyz,
        view.right.xyz,
        view.up.xyz,
        view.projection_from.w,
        view_scene_bounds_valid(view),
        view.scene_bounds_min.xyz,
        view.scene_bounds_max.xyz);
    ray_origin = result.origin;
    direction = result.direction;
}

void make_view_ray(
    constant viewer_constants_gpu &view,
    float2 pixel,
    float2 jitter,
    thread float3 &ray_origin,
    thread float3 &direction)
{
    const float2 uv = (pixel + 0.5f + jitter) / float2(view_width(view), view_height(view));
    const float2 ndc = float2(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f);
    ray_origin = view.origin.xyz;
    direction = view.forward.xyz;
    build_projection_ray(
        view,
        view.projection_modes.x,
        ndc,
        view.projection_from.x,
        view.projection_from.y,
        ray_origin,
        direction);
    const float blend_t = clamp(view.blend_and_jitter.x, 0.0f, 1.0f);
    if (view.projection_modes.x != view.projection_modes.y && blend_t < 1.0f) {
        float3 ray_origin_b;
        float3 direction_b;
        build_projection_ray(
            view,
            view.projection_modes.y,
            ndc,
            view.projection_to.x,
            view.projection_to.y,
            ray_origin_b,
            direction_b);
        ray_origin = mix(ray_origin, ray_origin_b, blend_t);
        direction = normalize(mix(direction, direction_b, blend_t));
    }
}

kernel void rtvdb_trace_kernel(
    instance_acceleration_structure scene [[buffer(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant float4* triangle_colors [[buffer(2)]],
    constant geometry_metadata_gpu* instance_metadata [[buffer(3)]],
    constant float4* positions [[buffer(4)]],
    constant uint* indices [[buffer(5)]],
    constant point_gpu* points [[buffer(6)]],
    constant line_gpu* lines [[buffer(7)]],
    constant viewer_constants_gpu &view [[buffer(8)]],
    texture2d<half, access::read_write> accumulation [[texture(10)]],
    constant uint* instance_kinds [[buffer(11)]],
    uint2 tid [[thread_position_in_grid]])
{
    if (tid.x >= view_width(view) || tid.y >= view_height(view)) {
        return;
    }

    float3 ray_origin;
    float3 direction;
    make_view_ray(view, float2(tid), view.blend_and_jitter.yz, ray_origin, direction);
    float3 accumulated_color = float3(0.0f);
    float accumulated_alpha = 0.0f;
    float throughput = 1.0f;
    for (uint layer = 0u; layer < kMaxTransparencyLayers; ++layer) {
        ray view_ray(
            ray_origin,
            direction,
            scene_intersection_t_min(view),
            kRayMaxDistance);
        const hit_info hit = trace_nearest_hit(
            view_ray,
            view,
            scene,
            triangle_colors,
            instance_metadata,
            positions,
            indices,
            points,
            lines,
            instance_kinds,
            false);
        if (!hit.hit) {
            break;
        }

        const float4 shaded = shade_hit(hit, view);
        const float alpha = clamp(shaded.a, 0.0f, 1.0f);
        if (alpha >= kAlphaOpaqueThreshold) {
            accumulated_color += throughput * shaded.rgb;
            accumulated_alpha += throughput;
            break;
        }
        accumulated_color += throughput * shaded.rgb * alpha;
        accumulated_alpha += throughput * alpha;
        throughput *= 1.0f - alpha;
        ray_origin += direction * (hit.distance + scene_hit_advance_bias(view));
        if (throughput <= 0.001f) {
            break;
        }
    }

    const uint sample_index = view_accumulation_sample_index(view);
    const float4 previous = sample_index == 0u
        ? float4(0.0f)
        : float4(accumulation.read(tid));
    const float sample_count = float(sample_index + 1u);
    const float4 averaged_premultiplied =
        (previous * float(sample_index) + float4(accumulated_color, accumulated_alpha)) / sample_count;
    const float output_alpha = clamp(averaged_premultiplied.w, 0.0f, 1.0f);
    const float3 output_rgb = output_alpha > 1.0e-6f
        ? averaged_premultiplied.xyz / output_alpha
        : float3(0.0f);
    accumulation.write(half4(averaged_premultiplied), tid);
    output.write(float4(output_rgb, output_alpha), tid);
}

kernel void rtvdb_pick_kernel(
    instance_acceleration_structure scene [[buffer(0)]],
    constant float4* triangle_colors [[buffer(2)]],
    constant geometry_metadata_gpu* instance_metadata [[buffer(3)]],
    constant float4* positions [[buffer(4)]],
    constant uint* indices [[buffer(5)]],
    constant point_gpu* points [[buffer(6)]],
    constant line_gpu* lines [[buffer(7)]],
    constant viewer_constants_gpu &view [[buffer(8)]],
    device pick_result_gpu* pick_result [[buffer(9)]],
    constant uint* instance_kinds [[buffer(11)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0u || pick_result == nullptr) {
        return;
    }

    pick_result[0] = {};
    const uint pixel_x = view.pick_params.x;
    const uint pixel_y = view.pick_params.y;
    if (view_width(view) == 0u || view_height(view) == 0u ||
        pixel_x >= view_width(view) || pixel_y >= view_height(view)) {
        return;
    }

    float3 ray_origin;
    float3 direction;
    make_view_ray(view, float2(pixel_x, pixel_y), float2(0.0f), ray_origin, direction);
    ray view_ray(
        ray_origin,
        direction,
        scene_intersection_t_min(view),
        kRayMaxDistance);
    const hit_info hit = trace_nearest_hit(
        view_ray,
        view,
        scene,
        triangle_colors,
        instance_metadata,
        positions,
        indices,
        points,
        lines,
        instance_kinds,
        true);
    if (!hit.hit) {
        return;
    }
    pick_result[0].primitive_kind = hit.kind + 1u;
    pick_result[0].primitive_index = hit.primitive_id;
    pick_result[0].distance = hit.distance;
    pick_result[0].hit = 1u;
}
