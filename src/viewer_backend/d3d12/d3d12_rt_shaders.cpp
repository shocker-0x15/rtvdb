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
} // namespace

rt_shader_package_desc d3d12_rt_shader_package() {
    return make_viewer_rt_shader_package(
        kD3d12RtShaderModules.data(),
        kD3d12RtShaderModules.size());
}

} // namespace rtvdb::viewer_backend
