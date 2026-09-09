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
} // namespace

rt_shader_package_desc vulkan_rt_shader_package() {
    return make_viewer_rt_shader_package(
        kVulkanRtShaderModules.data(),
        kVulkanRtShaderModules.size());
}

} // namespace rtvdb::viewer_backend
