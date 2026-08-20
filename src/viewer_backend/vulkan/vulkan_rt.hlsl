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

[[vk::binding(6, 0)]]
StructuredBuffer<PointPrimitive> g_points REGISTER(t5);

[[vk::binding(7, 0)]]
StructuredBuffer<LinePrimitive> g_lines REGISTER(t6);

[[vk::binding(8, 0)]]
ConstantBuffer<ViewerConstants> g_view REGISTER(b0);

#include "../shaders/rt_logic_shared_impl.h"
#define RTVDB_VULKAN_SHADER 1
#include "../shaders/rt_shader_entrypoints.h"
