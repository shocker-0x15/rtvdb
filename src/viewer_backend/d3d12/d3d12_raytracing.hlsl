#include "../shaders/rt_logic_shared_types.h"

RaytracingAccelerationStructure g_scene REGISTER(t0);
RWTexture2D<float4> g_output REGISTER(u0);
RWTexture2D<float4> g_accum REGISTER(u1);
RWStructuredBuffer<PickResult> g_pick_output REGISTER(u2);
StructuredBuffer<float4> g_triangle_color REGISTER(t1);
StructuredBuffer<GeometryMetadata> g_instance_metadata REGISTER(t2);
StructuredBuffer<float3> g_scene_positions REGISTER(t3);
StructuredBuffer<uint32_t> g_scene_indices REGISTER(t4);

struct PointPrimitive
{
    float3 position;
    float radius;
    float4 color;
};

struct LinePrimitive
{
    float3 a;
    float radius;
    float3 b;
    float pad;
    float4 color;
    uint32_t flags;
    float3 pad2;
};

StructuredBuffer<PointPrimitive> g_points REGISTER(t5);
StructuredBuffer<LinePrimitive> g_lines REGISTER(t6);

struct ViewerConstants
{
    float3 origin;
    float pad0;
    float3 forward;
    float pad1;
    float3 right;
    float pad2;
    float3 up;
    float pad3;
    float3 scene_bounds_min;
    uint32_t scene_bounds_valid;
    float3 scene_bounds_max;
    float pad4;
    uint32_t width;
    uint32_t height;
    uint32_t projection;
    float aspect;
    float projection_param_from0;
    float projection_param_from1;
    float projection_param_to0;
    float projection_param_to1;
    uint32_t projection_blend_from;
    uint32_t projection_blend_to;
    float projection_blend_t;
    uint32_t pick_pixel_x;
    uint32_t pick_pixel_y;
    uint32_t display_mode;
    uint32_t accumulation_sample_index;
    uint32_t pad5;
    float2 accumulation_jitter;
    uint32_t hover_highlight_kind;
    uint32_t hover_primitive_index;
    float hover_highlight_mix;
    uint32_t is_pick_pass;
    float2 pad6;
};

ConstantBuffer<ViewerConstants> g_view REGISTER(b0);

float4 shared_triangle_color(uint32_t triangle_index) { return g_triangle_color[triangle_index]; }
GeometryMetadata shared_instance_metadata(uint32_t metadata_index) { return g_instance_metadata[metadata_index]; }
uint32_t shared_procedural_primitive_offset(uint32_t instance_index)
{
    return shared_instance_metadata(instance_index).primitive_offset;
}
float3 shared_scene_position(uint32_t vertex_index) { return g_scene_positions[vertex_index]; }
uint32_t shared_scene_index(uint32_t index_index) { return g_scene_indices[index_index]; }
SharedPointPrimitive shared_point_primitive(uint32_t point_index)
{
    PointPrimitive point_value = g_points[point_index];
    SharedPointPrimitive result;
    result.position = point_value.position;
    result.radius = point_value.radius;
    result.color = point_value.color;
    return result;
}
SharedLinePrimitive shared_line_primitive(uint32_t line_index)
{
    LinePrimitive line_value = g_lines[line_index];
    SharedLinePrimitive result;
    result.a = line_value.a;
    result.radius = line_value.radius;
    result.b = line_value.b;
    result.color = line_value.color;
    result.flags = line_value.flags;
    return result;
}

float3 shared_origin() { return g_view.origin; }
float3 shared_forward() { return g_view.forward; }
float3 shared_right() { return g_view.right; }
float3 shared_up() { return g_view.up; }
float3 shared_scene_bounds_min() { return g_view.scene_bounds_min; }
uint32_t shared_scene_bounds_valid() { return g_view.scene_bounds_valid; }
float3 shared_scene_bounds_max() { return g_view.scene_bounds_max; }
uint32_t shared_width() { return g_view.width; }
uint32_t shared_height() { return g_view.height; }
uint32_t shared_projection() { return g_view.projection; }
float shared_aspect() { return g_view.aspect; }
float shared_projection_param_from0() { return g_view.projection_param_from0; }
float shared_projection_param_from1() { return g_view.projection_param_from1; }
float shared_projection_param_to0() { return g_view.projection_param_to0; }
float shared_projection_param_to1() { return g_view.projection_param_to1; }
uint32_t shared_projection_blend_from() { return g_view.projection_blend_from; }
uint32_t shared_projection_blend_to() { return g_view.projection_blend_to; }
float shared_projection_blend_t() { return g_view.projection_blend_t; }
uint32_t shared_pick_pixel_x() { return g_view.pick_pixel_x; }
uint32_t shared_pick_pixel_y() { return g_view.pick_pixel_y; }
uint32_t shared_display_mode() { return g_view.display_mode; }
uint32_t shared_accumulation_sample_index() { return g_view.accumulation_sample_index; }
float2 shared_accumulation_jitter() { return g_view.accumulation_jitter; }
uint32_t shared_hover_highlight_kind() { return g_view.hover_highlight_kind; }
uint32_t shared_hover_primitive_index() { return g_view.hover_primitive_index; }
float shared_hover_highlight_mix() { return g_view.hover_highlight_mix; }
uint32_t shared_is_pick_pass() { return g_view.is_pick_pass; }

#include "../shaders/rt_logic_shared_impl.h"

[shader("raygeneration")]
void RayGen()
{
    shared_raygen();
}

[shader("raygeneration")]
void PickRayGen()
{
    shared_pick_raygen();
}

[shader("miss")]
void Miss(ARG_INOUT(Payload, payload))
{
    shared_miss(payload);
}

[shader("closesthit")]
void ClosestHitTriangle(ARG_INOUT(Payload, payload), ARG_IN(BuiltInTriangleIntersectionAttributes, attr))
{
    shared_closest_hit_triangle(payload, GeometryIndex(), InstanceID());
}

[shader("intersection")]
void IntersectionPoint()
{
    shared_intersection_point();
}

[shader("closesthit")]
void ClosestHitPoint(ARG_INOUT(Payload, payload), ARG_IN(ProceduralAttributes, attr))
{
    shared_closest_hit_point(payload, InstanceID());
}

[shader("intersection")]
void IntersectionLine()
{
    shared_intersection_line();
}

[shader("closesthit")]
void ClosestHitLine(ARG_INOUT(Payload, payload), ARG_IN(ProceduralAttributes, attr))
{
    shared_closest_hit_line(payload, InstanceID());
}
