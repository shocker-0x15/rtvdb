#include "../shaders/rt_logic_shared_types.h"

RaytracingAccelerationStructure g_scene REGISTER(t0);
RWTexture2D<float4> g_output REGISTER(u1);
RWTexture2D<float4> g_accum REGISTER(u10);
RWStructuredBuffer<PickResult> g_pick_output REGISTER(u9);
StructuredBuffer<float4> g_triangle_color REGISTER(t2);
StructuredBuffer<GeometryMetadata> g_instance_metadata REGISTER(t3);
StructuredBuffer<float4> g_scene_positions REGISTER(t4);
StructuredBuffer<uint32_t> g_scene_indices REGISTER(t5);

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

StructuredBuffer<PointPrimitive> g_points REGISTER(t6);
StructuredBuffer<LinePrimitive> g_lines REGISTER(t7);

ConstantBuffer<ViewerConstants> g_view REGISTER(b8);

float4 shared_triangle_color(uint32_t triangle_index) { return g_triangle_color[triangle_index]; }
GeometryMetadata shared_instance_metadata(uint32_t metadata_index) { return g_instance_metadata[metadata_index]; }
uint32_t shared_procedural_primitive_offset(uint32_t instance_index, uint32_t geometry_index)
{
    // The first metadata field is the source primitive base for every geometry kind.
    return shared_instance_metadata(instance_index * kMaxInstanceGeometryCount + geometry_index).primitive_base;
}
float3 shared_scene_position(uint32_t vertex_index) { return g_scene_positions[vertex_index].xyz; }
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
#include "../shaders/rt_shader_entrypoints.h"
