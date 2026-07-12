#include "../shaders/rt_logic_shared_types.h"

[[vk::binding(0, 0)]]
RaytracingAccelerationStructure g_scene REGISTER(t0);

[[vk::binding(1, 0)]]
[[vk::image_format("rgba8")]]
RWTexture2D<float4> g_output REGISTER(u0);

[[vk::binding(10, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> g_accum REGISTER(u1);

[[vk::binding(9, 0)]]
RWStructuredBuffer<PickResult> g_pick_output REGISTER(u2);

[[vk::binding(2, 0)]]
StructuredBuffer<float4> g_triangle_color REGISTER(t1);

[[vk::binding(3, 0)]]
StructuredBuffer<GeometryMetadata> g_instance_metadata REGISTER(t2);

[[vk::binding(4, 0)]]
StructuredBuffer<float4> g_scene_positions REGISTER(t3);

[[vk::binding(5, 0)]]
StructuredBuffer<uint32_t> g_scene_indices REGISTER(t4);

struct PointPrimitiveVk
{
    float4 position_radius;
    float4 color;
};

struct LinePrimitiveVk
{
    float4 a_radius;
    float4 b_pad;
    float4 color;
    uint32_t flags;
    float pad0;
    float pad1;
    float pad2;
};

[[vk::binding(6, 0)]]
StructuredBuffer<PointPrimitiveVk> g_points REGISTER(t5);

[[vk::binding(7, 0)]]
StructuredBuffer<LinePrimitiveVk> g_lines REGISTER(t6);

struct ViewerConstantsVk
{
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

[[vk::binding(8, 0)]]
ConstantBuffer<ViewerConstantsVk> g_view REGISTER(b0);

float4 shared_triangle_color(uint32_t triangle_index) { return g_triangle_color[triangle_index]; }
GeometryMetadata shared_instance_metadata(uint32_t metadata_index) { return g_instance_metadata[metadata_index]; }
uint32_t shared_procedural_primitive_offset(uint32_t instance_index) { return instance_index; }
float3 shared_scene_position(uint32_t vertex_index) { return g_scene_positions[vertex_index].xyz; }
uint32_t shared_scene_index(uint32_t index_index) { return g_scene_indices[index_index]; }
SharedPointPrimitive shared_point_primitive(uint32_t point_index)
{
    PointPrimitiveVk point_value = g_points[point_index];
    SharedPointPrimitive result;
    result.position = point_value.position_radius.xyz;
    result.radius = point_value.position_radius.w;
    result.color = point_value.color;
    return result;
}
SharedLinePrimitive shared_line_primitive(uint32_t line_index)
{
    LinePrimitiveVk line_value = g_lines[line_index];
    SharedLinePrimitive result;
    result.a = line_value.a_radius.xyz;
    result.radius = line_value.a_radius.w;
    result.b = line_value.b_pad.xyz;
    result.color = line_value.color;
    result.flags = line_value.flags;
    return result;
}

float3 shared_origin() { return g_view.origin.xyz; }
float3 shared_forward() { return g_view.forward.xyz; }
float3 shared_right() { return g_view.right.xyz; }
float3 shared_up() { return g_view.up.xyz; }
float3 shared_scene_bounds_min() { return g_view.scene_bounds_min.xyz; }
uint32_t shared_scene_bounds_valid() { return asuint(g_view.scene_bounds_max.w); }
float3 shared_scene_bounds_max() { return g_view.scene_bounds_max.xyz; }
uint32_t shared_width() { return g_view.size_and_mode.x; }
uint32_t shared_height() { return g_view.size_and_mode.y; }
uint32_t shared_projection() { return g_view.projection_modes.x; }
float shared_aspect() { return g_view.projection_from.w; }
float shared_projection_param_from0() { return g_view.projection_from.x; }
float shared_projection_param_from1() { return g_view.projection_from.y; }
float shared_projection_param_to0() { return g_view.projection_to.x; }
float shared_projection_param_to1() { return g_view.projection_to.y; }
uint32_t shared_projection_blend_from() { return g_view.projection_modes.x; }
uint32_t shared_projection_blend_to() { return g_view.projection_modes.y; }
float shared_projection_blend_t() { return g_view.blend_and_jitter.x; }
uint32_t shared_pick_pixel_x() { return g_view.pick_params.x; }
uint32_t shared_pick_pixel_y() { return g_view.pick_params.y; }
uint32_t shared_display_mode() { return g_view.size_and_mode.z; }
uint32_t shared_accumulation_sample_index() { return g_view.size_and_mode.w; }
float2 shared_accumulation_jitter() { return g_view.blend_and_jitter.yz; }
uint32_t shared_hover_highlight_kind() { return g_view.projection_modes.z; }
uint32_t shared_hover_primitive_index() { return g_view.projection_modes.w; }
float shared_hover_highlight_mix() { return g_view.blend_and_jitter.w; }
uint32_t shared_is_pick_pass() { return g_view.pick_params.z; }

#include "../shaders/rt_logic_shared_impl.h"

#if defined(RTVDB_VULKAN_COMPILE_RAYGEN)
[shader("raygeneration")]
void RayGen()
{
    shared_raygen();
}
#endif

#if defined(RTVDB_VULKAN_COMPILE_PICK_RAYGEN)
[shader("raygeneration")]
void PickRayGen()
{
    shared_pick_raygen();
}
#endif

#if defined(RTVDB_VULKAN_COMPILE_MISS)
[shader("miss")]
void Miss(ARG_INOUT(Payload, payload))
{
    shared_miss(payload);
}
#endif

#if defined(RTVDB_VULKAN_COMPILE_TRIANGLE_CLOSEST_HIT)
[shader("closesthit")]
void ClosestHitTriangle(ARG_INOUT(Payload, payload), ARG_IN(BuiltInTriangleIntersectionAttributes, attr))
{
    shared_closest_hit_triangle(payload, GeometryIndex(), InstanceID());
}
#endif

#if defined(RTVDB_VULKAN_COMPILE_POINT_CLOSEST_HIT)
[shader("closesthit")]
void ClosestHitPoint(ARG_INOUT(Payload, payload), ARG_IN(ProceduralAttributes, attr))
{
    shared_closest_hit_point(payload, InstanceID());
}
#endif

#if defined(RTVDB_VULKAN_COMPILE_POINT_INTERSECTION)
[shader("intersection")]
void IntersectionPoint()
{
    shared_intersection_point();
}
#endif

#if defined(RTVDB_VULKAN_COMPILE_LINE_CLOSEST_HIT)
[shader("closesthit")]
void ClosestHitLine(ARG_INOUT(Payload, payload), ARG_IN(ProceduralAttributes, attr))
{
    shared_closest_hit_line(payload, InstanceID());
}
#endif

#if defined(RTVDB_VULKAN_COMPILE_LINE_INTERSECTION)
[shader("intersection")]
void IntersectionLine()
{
    shared_intersection_line();
}
#endif
