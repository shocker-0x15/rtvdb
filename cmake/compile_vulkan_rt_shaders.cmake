if(NOT DEFINED DXC_EXECUTABLE OR NOT DEFINED HLSL_INPUT OR NOT DEFINED EMBED_SCRIPT OR
   NOT DEFINED OUTPUT_DIRECTORY OR NOT DEFINED STAMP_OUTPUT)
    message(FATAL_ERROR "Vulkan shader compilation requires all output and tool paths.")
endif()

file(MAKE_DIRECTORY "${OUTPUT_DIRECTORY}")

function(run_dxc output entry_point define_name)
    execute_process(
        COMMAND "${DXC_EXECUTABLE}"
            -spirv
            -fspv-target-env=vulkan1.3
            -T lib_6_6
            -E "${entry_point}"
            -D "${define_name}=1"
            -HV 2021
            -Fo "${output}"
            "${HLSL_INPUT}"
        RESULT_VARIABLE _dxc_result
    )
    if(NOT _dxc_result EQUAL 0)
        message(FATAL_ERROR "Vulkan HLSL compilation for ${entry_point} failed with exit code ${_dxc_result}.")
    endif()
endfunction()

function(embed_spirv input output symbol_name size_name)
    execute_process(
        COMMAND "${CMAKE_COMMAND}"
            "-DINPUT_FILE=${input}"
            "-DOUTPUT_FILE=${output}"
            "-DSYMBOL_NAME=${symbol_name}"
            "-DSIZE_NAME=${size_name}"
            -P "${EMBED_SCRIPT}"
        RESULT_VARIABLE _embed_result
    )
    if(NOT _embed_result EQUAL 0)
        message(FATAL_ERROR "Embedding the Vulkan shader binary ${input} failed with exit code ${_embed_result}.")
    endif()
endfunction()

run_dxc("${OUTPUT_DIRECTORY}/vulkan_rt.rgen.spv" RayGen RTVDB_VULKAN_COMPILE_RAYGEN)
run_dxc("${OUTPUT_DIRECTORY}/vulkan_rt_pick.rgen.spv" PickRayGen RTVDB_VULKAN_COMPILE_PICK_RAYGEN)
run_dxc("${OUTPUT_DIRECTORY}/vulkan_rt.rmiss.spv" Miss RTVDB_VULKAN_COMPILE_MISS)
run_dxc("${OUTPUT_DIRECTORY}/vulkan_rt.rchit.spv" ClosestHitTriangle RTVDB_VULKAN_COMPILE_TRIANGLE_CLOSEST_HIT)
run_dxc("${OUTPUT_DIRECTORY}/vulkan_rt_point.rchit.spv" ClosestHitPoint RTVDB_VULKAN_COMPILE_POINT_CLOSEST_HIT)
run_dxc("${OUTPUT_DIRECTORY}/vulkan_rt_point.rint.spv" IntersectionPoint RTVDB_VULKAN_COMPILE_POINT_INTERSECTION)
run_dxc("${OUTPUT_DIRECTORY}/vulkan_rt_line.rchit.spv" ClosestHitLine RTVDB_VULKAN_COMPILE_LINE_CLOSEST_HIT)
run_dxc("${OUTPUT_DIRECTORY}/vulkan_rt_line.rint.spv" IntersectionLine RTVDB_VULKAN_COMPILE_LINE_INTERSECTION)

embed_spirv(
    "${OUTPUT_DIRECTORY}/vulkan_rt.rgen.spv"
    "${OUTPUT_DIRECTORY}/rtvdb_vulkan_rt_rgen_spv.h"
    kVulkanRtRaygenSpirv
    kVulkanRtRaygenSpirvSize)
embed_spirv(
    "${OUTPUT_DIRECTORY}/vulkan_rt_pick.rgen.spv"
    "${OUTPUT_DIRECTORY}/rtvdb_vulkan_rt_pick_rgen_spv.h"
    kVulkanRtPickRaygenSpirv
    kVulkanRtPickRaygenSpirvSize)
embed_spirv(
    "${OUTPUT_DIRECTORY}/vulkan_rt.rmiss.spv"
    "${OUTPUT_DIRECTORY}/rtvdb_vulkan_rt_rmiss_spv.h"
    kVulkanRtMissSpirv
    kVulkanRtMissSpirvSize)
embed_spirv(
    "${OUTPUT_DIRECTORY}/vulkan_rt.rchit.spv"
    "${OUTPUT_DIRECTORY}/rtvdb_vulkan_rt_rchit_spv.h"
    kVulkanRtClosestHitSpirv
    kVulkanRtClosestHitSpirvSize)
embed_spirv(
    "${OUTPUT_DIRECTORY}/vulkan_rt_point.rchit.spv"
    "${OUTPUT_DIRECTORY}/rtvdb_vulkan_rt_point_rchit_spv.h"
    kVulkanRtPointClosestHitSpirv
    kVulkanRtPointClosestHitSpirvSize)
embed_spirv(
    "${OUTPUT_DIRECTORY}/vulkan_rt_point.rint.spv"
    "${OUTPUT_DIRECTORY}/rtvdb_vulkan_rt_point_rint_spv.h"
    kVulkanRtPointIntersectionSpirv
    kVulkanRtPointIntersectionSpirvSize)
embed_spirv(
    "${OUTPUT_DIRECTORY}/vulkan_rt_line.rchit.spv"
    "${OUTPUT_DIRECTORY}/rtvdb_vulkan_rt_line_rchit_spv.h"
    kVulkanRtLineClosestHitSpirv
    kVulkanRtLineClosestHitSpirvSize)
embed_spirv(
    "${OUTPUT_DIRECTORY}/vulkan_rt_line.rint.spv"
    "${OUTPUT_DIRECTORY}/rtvdb_vulkan_rt_line_rint_spv.h"
    kVulkanRtLineIntersectionSpirv
    kVulkanRtLineIntersectionSpirvSize)

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E touch "${STAMP_OUTPUT}"
    RESULT_VARIABLE _stamp_result
)
if(NOT _stamp_result EQUAL 0)
    message(FATAL_ERROR "Updating the Vulkan shader stamp failed with exit code ${_stamp_result}.")
endif()
