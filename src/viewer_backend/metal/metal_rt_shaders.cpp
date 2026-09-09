#include "viewer_backend/metal/metal_rt_shaders.h"

#include "viewer_backend/rt_render_plan.h"

#include "rtvdb_metal_rt_metallib.h"

#include <array>

namespace rtvdb::viewer_backend {
namespace {

constexpr std::array<rt_shader_module_desc, 1> kMetalRtShaderModules{{
    {
        rt_shader_binary_format::metallib,
        kMetalRtMetallib,
        kMetalRtMetallibSize,
    },
}};
} // namespace

rt_shader_package_desc metal_rt_shader_package() {
    return make_viewer_rt_shader_package(
        kMetalRtShaderModules.data(),
        kMetalRtShaderModules.size());
}

} // namespace rtvdb::viewer_backend
