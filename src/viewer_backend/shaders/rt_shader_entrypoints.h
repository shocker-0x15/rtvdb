// Resource declarations are API-specific. Every RT entry point body is shared.
// DXIL compiles the complete library; SPIR-V selects one exported stage per pass.

#if !defined(RTVDB_VULKAN_SHADER) || defined(RTVDB_VULKAN_COMPILE_RAYGEN)
[shader("raygeneration")]
void RayGen()
{
    shared_raygen();
}
#endif

#if !defined(RTVDB_VULKAN_SHADER) || defined(RTVDB_VULKAN_COMPILE_PICK_RAYGEN)
[shader("raygeneration")]
void PickRayGen()
{
    shared_pick_raygen();
}
#endif

#if !defined(RTVDB_VULKAN_SHADER) || defined(RTVDB_VULKAN_COMPILE_MISS)
[shader("miss")]
void Miss(ARG_INOUT(Payload, payload))
{
    shared_miss(payload);
}
#endif

#if !defined(RTVDB_VULKAN_SHADER) || defined(RTVDB_VULKAN_COMPILE_TRIANGLE_CLOSEST_HIT)
[shader("closesthit")]
void ClosestHitTriangle(ARG_INOUT(Payload, payload), ARG_IN(BuiltInTriangleIntersectionAttributes, attr))
{
    shared_closest_hit_triangle(payload, GeometryIndex(), InstanceID());
}
#endif

#if !defined(RTVDB_VULKAN_SHADER) || defined(RTVDB_VULKAN_COMPILE_POINT_CLOSEST_HIT)
[shader("closesthit")]
void ClosestHitPoint(ARG_INOUT(Payload, payload), ARG_IN(ProceduralAttributes, attr))
{
    shared_closest_hit_point(payload, InstanceID(), GeometryIndex());
}
#endif

#if !defined(RTVDB_VULKAN_SHADER) || defined(RTVDB_VULKAN_COMPILE_POINT_INTERSECTION)
[shader("intersection")]
void IntersectionPoint()
{
    shared_intersection_point();
}
#endif

#if !defined(RTVDB_VULKAN_SHADER) || defined(RTVDB_VULKAN_COMPILE_LINE_CLOSEST_HIT)
[shader("closesthit")]
void ClosestHitLine(ARG_INOUT(Payload, payload), ARG_IN(ProceduralAttributes, attr))
{
    shared_closest_hit_line(payload, InstanceID(), GeometryIndex());
}
#endif

#if !defined(RTVDB_VULKAN_SHADER) || defined(RTVDB_VULKAN_COMPILE_LINE_INTERSECTION)
[shader("intersection")]
void IntersectionLine()
{
    shared_intersection_line();
}
#endif
