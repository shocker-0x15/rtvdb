#include "viewer_backend/d3d12/d3d12_rt_shaders.h"

#include "viewer_backend/rt_render_plan.h"

#include "rtvdb_d3d12_raytracing_dxil.h"

#include <array>

namespace rtvdb::viewer_backend {
namespace {

constexpr std::array<rt_shader_module_desc, 1> kD3d12RtShaderModules{{
    {
        rt_shader_binary_format::dxil_library,
        kD3d12RaytracingDxil,
        kD3d12RaytracingDxilSize,
    },
}};
constexpr std::array<std::uint32_t, kViewerRtShaderEntryCount> kD3d12RtShaderEntryModules{};
constexpr std::array<rt_logical_shader_entry, kViewerRtShaderEntryCount> kD3d12RtLogicalEntries{{
    rt_logical_shader_entry::render,
    rt_logical_shader_entry::pick,
    rt_logical_shader_entry::miss,
    rt_logical_shader_entry::triangle_closest_hit,
    rt_logical_shader_entry::point_closest_hit,
    rt_logical_shader_entry::point_intersection,
    rt_logical_shader_entry::line_closest_hit,
    rt_logical_shader_entry::line_intersection,
}};

} // namespace

rt_shader_package_desc d3d12_rt_shader_package() {
    return {
        rt_pipeline_model::native_ray_tracing,
        kD3d12RtShaderModules.data(),
        kD3d12RtShaderModules.size(),
        kD3d12RtShaderEntryModules.data(),
        kD3d12RtLogicalEntries.data(),
        kD3d12RtShaderEntryModules.size(),
    };
}

} // namespace rtvdb::viewer_backend
