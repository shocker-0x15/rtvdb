#include "../shaders/rt_logic_shared_types.h"

RaytracingAccelerationStructure g_scene REGISTER(t0);
RWTexture2D<float4> g_output REGISTER(u1);
RWTexture2D<float4> g_accum REGISTER(u10);
RWStructuredBuffer<PickResult> g_pick_output REGISTER(u9);
StructuredBuffer<float4> g_triangle_color REGISTER(t2);
StructuredBuffer<GeometryMetadata> g_instance_metadata REGISTER(t3);
StructuredBuffer<float4> g_scene_positions REGISTER(t4);
StructuredBuffer<uint32_t> g_scene_indices REGISTER(t5);

StructuredBuffer<PointPrimitive> g_points REGISTER(t6);
StructuredBuffer<LinePrimitive> g_lines REGISTER(t7);

ConstantBuffer<ViewerConstants> g_view REGISTER(b8);

#include "../shaders/rt_logic_shared_impl.h"
#include "../shaders/rt_shader_entrypoints.h"
