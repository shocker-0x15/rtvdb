#include "viewer_backend/vulkan/vulkan_rt_shaders.h"

#include "viewer_backend/rt_render_plan.h"

#include "rtvdb_vulkan_rt_line_rchit_spv.h"
#include "rtvdb_vulkan_rt_line_rint_spv.h"
#include "rtvdb_vulkan_rt_pick_rgen_spv.h"
#include "rtvdb_vulkan_rt_point_rchit_spv.h"
#include "rtvdb_vulkan_rt_point_rint_spv.h"
#include "rtvdb_vulkan_rt_rchit_spv.h"
#include "rtvdb_vulkan_rt_rgen_spv.h"
#include "rtvdb_vulkan_rt_rmiss_spv.h"

#include <array>

namespace rtvdb::viewer_backend {
namespace {

constexpr std::array<rt_shader_module_desc, kViewerRtShaderEntryCount> kVulkanRtShaderModules{{
    {rt_shader_binary_format::spirv, kVulkanRtRaygenSpirv, kVulkanRtRaygenSpirvSize},
    {rt_shader_binary_format::spirv, kVulkanRtPickRaygenSpirv, kVulkanRtPickRaygenSpirvSize},
    {rt_shader_binary_format::spirv, kVulkanRtMissSpirv, kVulkanRtMissSpirvSize},
    {rt_shader_binary_format::spirv, kVulkanRtClosestHitSpirv, kVulkanRtClosestHitSpirvSize},
    {rt_shader_binary_format::spirv, kVulkanRtPointClosestHitSpirv, kVulkanRtPointClosestHitSpirvSize},
    {rt_shader_binary_format::spirv, kVulkanRtPointIntersectionSpirv, kVulkanRtPointIntersectionSpirvSize},
    {rt_shader_binary_format::spirv, kVulkanRtLineClosestHitSpirv, kVulkanRtLineClosestHitSpirvSize},
    {rt_shader_binary_format::spirv, kVulkanRtLineIntersectionSpirv, kVulkanRtLineIntersectionSpirvSize},
}};
constexpr std::array<std::uint32_t, kViewerRtShaderEntryCount> kVulkanRtShaderEntryModules{{
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
}};
constexpr std::array<rt_logical_shader_entry, kViewerRtShaderEntryCount> kVulkanRtLogicalEntries{{
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

rt_shader_package_desc vulkan_rt_shader_package() {
    return {
        rt_pipeline_model::native_ray_tracing,
        kVulkanRtShaderModules.data(),
        kVulkanRtShaderModules.size(),
        kVulkanRtShaderEntryModules.data(),
        kVulkanRtLogicalEntries.data(),
        kVulkanRtShaderEntryModules.size(),
    };
}

} // namespace rtvdb::viewer_backend
