#include "viewer_backend/metal/metal_rt_shaders.h"

#include "rtvdb_metal_rt_metallib.h"

#include <array>

namespace rtvdb::viewer_backend {
namespace {

constexpr std::size_t kMetalRtShaderEntryCount = 2;
constexpr std::array<rt_shader_module_desc, 1> kMetalRtShaderModules{{
    {
        rt_shader_binary_format::metallib,
        kMetalRtMetallib,
        kMetalRtMetallibSize,
    },
}};
constexpr std::array<std::uint32_t, kMetalRtShaderEntryCount> kMetalRtShaderEntryModules{};
constexpr std::array<rt_logical_shader_entry, kMetalRtShaderEntryCount> kMetalRtLogicalEntries{{
    rt_logical_shader_entry::render,
    rt_logical_shader_entry::pick,
}};

} // namespace

rt_shader_package_desc metal_rt_shader_package() {
    return {
        rt_pipeline_model::compute_intersector,
        kMetalRtShaderModules.data(),
        kMetalRtShaderModules.size(),
        kMetalRtShaderEntryModules.data(),
        kMetalRtLogicalEntries.data(),
        kMetalRtShaderEntryModules.size(),
    };
}

} // namespace rtvdb::viewer_backend
