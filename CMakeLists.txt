cmake_minimum_required(VERSION 3.24)

project(rtvdb VERSION 0.1.0 LANGUAGES CXX)

if(APPLE)
    enable_language(OBJCXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(WIN32)
    option(RTVDB_ENABLE_D3D12_DXR "Build the viewer against the D3D12 DXR backend" ON)
    option(RTVDB_ENABLE_VULKAN_RT "Build the viewer against the Vulkan RT backend" OFF)
elseif(UNIX AND NOT APPLE)
    option(RTVDB_ENABLE_VULKAN_RT "Build the viewer against the Vulkan RT backend" ON)
else()
    option(RTVDB_ENABLE_D3D12_DXR "Build the viewer against the D3D12 DXR backend" OFF)
    option(RTVDB_ENABLE_VULKAN_RT "Build the viewer against the Vulkan RT backend" OFF)
endif()

if((WIN32 AND RTVDB_ENABLE_D3D12_DXR) OR RTVDB_ENABLE_VULKAN_RT)
    set(_rtvdb_dxc_search_paths)
    if(WIN32 AND DEFINED ENV{WindowsSdkVerBinPath})
        list(APPEND _rtvdb_dxc_search_paths "$ENV{WindowsSdkVerBinPath}x64")
    endif()
    if(WIN32 AND DEFINED ENV{WindowsSdkDir} AND CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION)
        list(APPEND _rtvdb_dxc_search_paths
            "$ENV{WindowsSdkDir}bin/${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}/x64")
    endif()

    find_program(RTVDB_DXC_EXECUTABLE NAMES dxc HINTS ${_rtvdb_dxc_search_paths})
    if(NOT RTVDB_DXC_EXECUTABLE)
        message(FATAL_ERROR "Shader builds require dxc. Install DirectX Shader Compiler or a compatible Vulkan/Windows SDK that provides dxc.")
    endif()
endif()

set(RTVDB_SHADER_SHARED_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/viewer_backend/shaders/rt_logic_shared_types.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/viewer_backend/shaders/rt_logic_shared_impl.h
)

if(WIN32 AND RTVDB_ENABLE_D3D12_DXR)
    set(RTVDB_D3D12_RAYTRACING_HLSL ${CMAKE_CURRENT_SOURCE_DIR}/src/viewer_backend/d3d12/d3d12_raytracing.hlsl)
    set(RTVDB_D3D12_RAYTRACING_DXIL ${CMAKE_CURRENT_BINARY_DIR}/generated/d3d12_raytracing.dxil)
    set(RTVDB_D3D12_RAYTRACING_HEADER ${CMAKE_CURRENT_BINARY_DIR}/generated/rtvdb_d3d12_raytracing_dxil.h)

    add_custom_command(
        OUTPUT ${RTVDB_D3D12_RAYTRACING_DXIL} ${RTVDB_D3D12_RAYTRACING_HEADER}
        MAIN_DEPENDENCY ${RTVDB_D3D12_RAYTRACING_HLSL}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/generated
        COMMAND ${RTVDB_DXC_EXECUTABLE}
            -T lib_6_6
            -HV 2021
            -Zi
            -Qembed_debug
            -Fo ${RTVDB_D3D12_RAYTRACING_DXIL}
            ${RTVDB_D3D12_RAYTRACING_HLSL}
        COMMAND ${CMAKE_COMMAND}
            -DINPUT_FILE=${RTVDB_D3D12_RAYTRACING_DXIL}
            -DOUTPUT_FILE=${RTVDB_D3D12_RAYTRACING_HEADER}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary_as_header.cmake
        DEPENDS
            ${RTVDB_D3D12_RAYTRACING_HLSL}
            ${RTVDB_SHADER_SHARED_HEADERS}
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary_as_header.cmake
        VERBATIM
    )

    source_group("Shader Files" FILES ${RTVDB_D3D12_RAYTRACING_HLSL} ${RTVDB_SHADER_SHARED_HEADERS})
    source_group("CMake Files" FILES ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary_as_header.cmake)
endif()

if(RTVDB_ENABLE_VULKAN_RT)
    find_package(Vulkan REQUIRED)
    set(_rtvdb_spirv_dxc_search_paths)
    if(DEFINED ENV{VULKAN_SDK})
        list(APPEND _rtvdb_spirv_dxc_search_paths "$ENV{VULKAN_SDK}/Bin")
    endif()
    find_program(RTVDB_SPIRV_DXC_EXECUTABLE NAMES dxc HINTS ${_rtvdb_spirv_dxc_search_paths})
    if(NOT RTVDB_SPIRV_DXC_EXECUTABLE)
        set(RTVDB_SPIRV_DXC_EXECUTABLE ${RTVDB_DXC_EXECUTABLE})
    endif()
    if(NOT RTVDB_SPIRV_DXC_EXECUTABLE)
        message(FATAL_ERROR "RTVDB_ENABLE_VULKAN_RT=ON requires a SPIR-V-capable dxc.exe. Install the Vulkan SDK or DirectXShaderCompiler with SPIR-V CodeGen enabled.")
    endif()

    set(RTVDB_VULKAN_RT_HLSL ${CMAKE_CURRENT_SOURCE_DIR}/src/viewer_backend/vulkan/vulkan_rt.hlsl)
    set(RTVDB_VULKAN_RT_RGEN_SPV ${CMAKE_CURRENT_BINARY_DIR}/generated/vulkan_rt.rgen.spv)
    set(RTVDB_VULKAN_RT_PICK_RGEN_SPV ${CMAKE_CURRENT_BINARY_DIR}/generated/vulkan_rt_pick.rgen.spv)
    set(RTVDB_VULKAN_RT_RMISS_SPV ${CMAKE_CURRENT_BINARY_DIR}/generated/vulkan_rt.rmiss.spv)
    set(RTVDB_VULKAN_RT_RCHIT_SPV ${CMAKE_CURRENT_BINARY_DIR}/generated/vulkan_rt.rchit.spv)
    set(RTVDB_VULKAN_RT_POINT_RCHIT_SPV ${CMAKE_CURRENT_BINARY_DIR}/generated/vulkan_rt_point.rchit.spv)
    set(RTVDB_VULKAN_RT_POINT_RINT_SPV ${CMAKE_CURRENT_BINARY_DIR}/generated/vulkan_rt_point.rint.spv)
    set(RTVDB_VULKAN_RT_LINE_RCHIT_SPV ${CMAKE_CURRENT_BINARY_DIR}/generated/vulkan_rt_line.rchit.spv)
    set(RTVDB_VULKAN_RT_LINE_RINT_SPV ${CMAKE_CURRENT_BINARY_DIR}/generated/vulkan_rt_line.rint.spv)
    set(RTVDB_VULKAN_RT_RGEN_HEADER ${CMAKE_CURRENT_BINARY_DIR}/generated/rtvdb_vulkan_rt_rgen_spv.h)
    set(RTVDB_VULKAN_RT_PICK_RGEN_HEADER ${CMAKE_CURRENT_BINARY_DIR}/generated/rtvdb_vulkan_rt_pick_rgen_spv.h)
    set(RTVDB_VULKAN_RT_RMISS_HEADER ${CMAKE_CURRENT_BINARY_DIR}/generated/rtvdb_vulkan_rt_rmiss_spv.h)
    set(RTVDB_VULKAN_RT_RCHIT_HEADER ${CMAKE_CURRENT_BINARY_DIR}/generated/rtvdb_vulkan_rt_rchit_spv.h)
    set(RTVDB_VULKAN_RT_POINT_RCHIT_HEADER ${CMAKE_CURRENT_BINARY_DIR}/generated/rtvdb_vulkan_rt_point_rchit_spv.h)
    set(RTVDB_VULKAN_RT_POINT_RINT_HEADER ${CMAKE_CURRENT_BINARY_DIR}/generated/rtvdb_vulkan_rt_point_rint_spv.h)
    set(RTVDB_VULKAN_RT_LINE_RCHIT_HEADER ${CMAKE_CURRENT_BINARY_DIR}/generated/rtvdb_vulkan_rt_line_rchit_spv.h)
    set(RTVDB_VULKAN_RT_LINE_RINT_HEADER ${CMAKE_CURRENT_BINARY_DIR}/generated/rtvdb_vulkan_rt_line_rint_spv.h)

    add_custom_command(
        OUTPUT
            ${RTVDB_VULKAN_RT_RGEN_SPV}
            ${RTVDB_VULKAN_RT_PICK_RGEN_SPV}
            ${RTVDB_VULKAN_RT_RMISS_SPV}
            ${RTVDB_VULKAN_RT_RCHIT_SPV}
            ${RTVDB_VULKAN_RT_POINT_RCHIT_SPV}
            ${RTVDB_VULKAN_RT_POINT_RINT_SPV}
            ${RTVDB_VULKAN_RT_LINE_RCHIT_SPV}
            ${RTVDB_VULKAN_RT_LINE_RINT_SPV}
            ${RTVDB_VULKAN_RT_RGEN_HEADER}
            ${RTVDB_VULKAN_RT_PICK_RGEN_HEADER}
            ${RTVDB_VULKAN_RT_RMISS_HEADER}
            ${RTVDB_VULKAN_RT_RCHIT_HEADER}
            ${RTVDB_VULKAN_RT_POINT_RCHIT_HEADER}
            ${RTVDB_VULKAN_RT_POINT_RINT_HEADER}
            ${RTVDB_VULKAN_RT_LINE_RCHIT_HEADER}
            ${RTVDB_VULKAN_RT_LINE_RINT_HEADER}
        MAIN_DEPENDENCY ${RTVDB_VULKAN_RT_HLSL}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/generated
        COMMAND ${RTVDB_SPIRV_DXC_EXECUTABLE}
            -spirv
            -fspv-target-env=vulkan1.3
            -T lib_6_6
            -E RayGen
            -D RTVDB_VULKAN_COMPILE_RAYGEN=1
            -HV 2021
            -Fo ${RTVDB_VULKAN_RT_RGEN_SPV}
            ${RTVDB_VULKAN_RT_HLSL}
        COMMAND ${RTVDB_SPIRV_DXC_EXECUTABLE}
            -spirv
            -fspv-target-env=vulkan1.3
            -T lib_6_6
            -E PickRayGen
            -D RTVDB_VULKAN_COMPILE_PICK_RAYGEN=1
            -HV 2021
            -Fo ${RTVDB_VULKAN_RT_PICK_RGEN_SPV}
            ${RTVDB_VULKAN_RT_HLSL}
        COMMAND ${RTVDB_SPIRV_DXC_EXECUTABLE}
            -spirv
            -fspv-target-env=vulkan1.3
            -T lib_6_6
            -E Miss
            -D RTVDB_VULKAN_COMPILE_MISS=1
            -HV 2021
            -Fo ${RTVDB_VULKAN_RT_RMISS_SPV}
            ${RTVDB_VULKAN_RT_HLSL}
        COMMAND ${RTVDB_SPIRV_DXC_EXECUTABLE}
            -spirv
            -fspv-target-env=vulkan1.3
            -T lib_6_6
            -E ClosestHitTriangle
            -D RTVDB_VULKAN_COMPILE_TRIANGLE_CLOSEST_HIT=1
            -HV 2021
            -Fo ${RTVDB_VULKAN_RT_RCHIT_SPV}
            ${RTVDB_VULKAN_RT_HLSL}
        COMMAND ${RTVDB_SPIRV_DXC_EXECUTABLE}
            -spirv
            -fspv-target-env=vulkan1.3
            -T lib_6_6
            -E ClosestHitPoint
            -D RTVDB_VULKAN_COMPILE_POINT_CLOSEST_HIT=1
            -HV 2021
            -Fo ${RTVDB_VULKAN_RT_POINT_RCHIT_SPV}
            ${RTVDB_VULKAN_RT_HLSL}
        COMMAND ${RTVDB_SPIRV_DXC_EXECUTABLE}
            -spirv
            -fspv-target-env=vulkan1.3
            -T lib_6_6
            -E IntersectionPoint
            -D RTVDB_VULKAN_COMPILE_POINT_INTERSECTION=1
            -HV 2021
            -Fo ${RTVDB_VULKAN_RT_POINT_RINT_SPV}
            ${RTVDB_VULKAN_RT_HLSL}
        COMMAND ${RTVDB_SPIRV_DXC_EXECUTABLE}
            -spirv
            -fspv-target-env=vulkan1.3
            -T lib_6_6
            -E ClosestHitLine
            -D RTVDB_VULKAN_COMPILE_LINE_CLOSEST_HIT=1
            -HV 2021
            -Fo ${RTVDB_VULKAN_RT_LINE_RCHIT_SPV}
            ${RTVDB_VULKAN_RT_HLSL}
        COMMAND ${RTVDB_SPIRV_DXC_EXECUTABLE}
            -spirv
            -fspv-target-env=vulkan1.3
            -T lib_6_6
            -E IntersectionLine
            -D RTVDB_VULKAN_COMPILE_LINE_INTERSECTION=1
            -HV 2021
            -Fo ${RTVDB_VULKAN_RT_LINE_RINT_SPV}
            ${RTVDB_VULKAN_RT_HLSL}
        COMMAND ${CMAKE_COMMAND}
            -DINPUT_FILE=${RTVDB_VULKAN_RT_RGEN_SPV}
            -DOUTPUT_FILE=${RTVDB_VULKAN_RT_RGEN_HEADER}
            -DSYMBOL_NAME=kVulkanRtRaygenSpirv
            -DSIZE_NAME=kVulkanRtRaygenSpirvSize
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary_as_header.cmake
        COMMAND ${CMAKE_COMMAND}
            -DINPUT_FILE=${RTVDB_VULKAN_RT_PICK_RGEN_SPV}
            -DOUTPUT_FILE=${RTVDB_VULKAN_RT_PICK_RGEN_HEADER}
            -DSYMBOL_NAME=kVulkanRtPickRaygenSpirv
            -DSIZE_NAME=kVulkanRtPickRaygenSpirvSize
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary_as_header.cmake
        COMMAND ${CMAKE_COMMAND}
            -DINPUT_FILE=${RTVDB_VULKAN_RT_RMISS_SPV}
            -DOUTPUT_FILE=${RTVDB_VULKAN_RT_RMISS_HEADER}
            -DSYMBOL_NAME=kVulkanRtMissSpirv
            -DSIZE_NAME=kVulkanRtMissSpirvSize
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary_as_header.cmake
        COMMAND ${CMAKE_COMMAND}
            -DINPUT_FILE=${RTVDB_VULKAN_RT_RCHIT_SPV}
            -DOUTPUT_FILE=${RTVDB_VULKAN_RT_RCHIT_HEADER}
            -DSYMBOL_NAME=kVulkanRtClosestHitSpirv
            -DSIZE_NAME=kVulkanRtClosestHitSpirvSize
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary_as_header.cmake
        COMMAND ${CMAKE_COMMAND}
            -DINPUT_FILE=${RTVDB_VULKAN_RT_POINT_RCHIT_SPV}
            -DOUTPUT_FILE=${RTVDB_VULKAN_RT_POINT_RCHIT_HEADER}
            -DSYMBOL_NAME=kVulkanRtPointClosestHitSpirv
            -DSIZE_NAME=kVulkanRtPointClosestHitSpirvSize
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary_as_header.cmake
        COMMAND ${CMAKE_COMMAND}
            -DINPUT_FILE=${RTVDB_VULKAN_RT_POINT_RINT_SPV}
            -DOUTPUT_FILE=${RTVDB_VULKAN_RT_POINT_RINT_HEADER}
            -DSYMBOL_NAME=kVulkanRtPointIntersectionSpirv
            -DSIZE_NAME=kVulkanRtPointIntersectionSpirvSize
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary_as_header.cmake
        COMMAND ${CMAKE_COMMAND}
            -DINPUT_FILE=${RTVDB_VULKAN_RT_LINE_RCHIT_SPV}
            -DOUTPUT_FILE=${RTVDB_VULKAN_RT_LINE_RCHIT_HEADER}
            -DSYMBOL_NAME=kVulkanRtLineClosestHitSpirv
            -DSIZE_NAME=kVulkanRtLineClosestHitSpirvSize
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary_as_header.cmake
        COMMAND ${CMAKE_COMMAND}
            -DINPUT_FILE=${RTVDB_VULKAN_RT_LINE_RINT_SPV}
            -DOUTPUT_FILE=${RTVDB_VULKAN_RT_LINE_RINT_HEADER}
            -DSYMBOL_NAME=kVulkanRtLineIntersectionSpirv
            -DSIZE_NAME=kVulkanRtLineIntersectionSpirvSize
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary_as_header.cmake
        DEPENDS
            ${RTVDB_VULKAN_RT_HLSL}
            ${RTVDB_SHADER_SHARED_HEADERS}
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary_as_header.cmake
        VERBATIM
    )

    set(RTVDB_VULKAN_RT_GENERATED_HEADERS
        ${RTVDB_VULKAN_RT_RGEN_HEADER}
        ${RTVDB_VULKAN_RT_PICK_RGEN_HEADER}
        ${RTVDB_VULKAN_RT_RMISS_HEADER}
        ${RTVDB_VULKAN_RT_RCHIT_HEADER}
        ${RTVDB_VULKAN_RT_POINT_RCHIT_HEADER}
        ${RTVDB_VULKAN_RT_POINT_RINT_HEADER}
        ${RTVDB_VULKAN_RT_LINE_RCHIT_HEADER}
        ${RTVDB_VULKAN_RT_LINE_RINT_HEADER}
    )
    source_group("Shader Files" FILES ${RTVDB_VULKAN_RT_HLSL} ${RTVDB_SHADER_SHARED_HEADERS})
    source_group("CMake Files" FILES ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary_as_header.cmake)
endif()

if(APPLE)
    set(RTVDB_METAL_RT_SHADER_MSL ${CMAKE_CURRENT_SOURCE_DIR}/src/viewer_backend/metal/metal_rt.metal)
    find_program(RTVDB_METAL_EXECUTABLE NAMES metal)
    find_program(RTVDB_METALLIB_EXECUTABLE NAMES metallib)
    if(NOT RTVDB_METAL_EXECUTABLE OR NOT RTVDB_METALLIB_EXECUTABLE)
        find_program(_rtvdb_xcrun_executable NAMES xcrun)
        if(_rtvdb_xcrun_executable)
            if(NOT RTVDB_METAL_EXECUTABLE)
                execute_process(
                    COMMAND "${_rtvdb_xcrun_executable}" --find metal
                    RESULT_VARIABLE _rtvdb_metal_result
                    OUTPUT_VARIABLE _rtvdb_metal_path
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET
                )
                if(_rtvdb_metal_result EQUAL 0 AND _rtvdb_metal_path)
                    set(RTVDB_METAL_EXECUTABLE "${_rtvdb_metal_path}")
                endif()
            endif()
            if(NOT RTVDB_METALLIB_EXECUTABLE)
                execute_process(
                    COMMAND "${_rtvdb_xcrun_executable}" --find metallib
                    RESULT_VARIABLE _rtvdb_metallib_result
                    OUTPUT_VARIABLE _rtvdb_metallib_path
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET
                )
                if(_rtvdb_metallib_result EQUAL 0 AND _rtvdb_metallib_path)
                    set(RTVDB_METALLIB_EXECUTABLE "${_rtvdb_metallib_path}")
                endif()
            endif()
        endif()
    endif()
    if(NOT RTVDB_METAL_EXECUTABLE OR NOT RTVDB_METALLIB_EXECUTABLE)
        message(FATAL_ERROR "Apple Metal backend requires metal and metallib. Put them on PATH, set DEVELOPER_DIR for xcrun, or pass RTVDB_METAL_EXECUTABLE and RTVDB_METALLIB_EXECUTABLE.")
    endif()

    set(RTVDB_METAL_RT_SHADER_AIR ${CMAKE_CURRENT_BINARY_DIR}/generated/metal_rt.air)
    set(RTVDB_METAL_RT_SHADER_METALLIB ${CMAKE_CURRENT_BINARY_DIR}/generated/metal_rt.metallib)
    set(RTVDB_METAL_RT_SHADER_METALLIB_HEADER ${CMAKE_CURRENT_BINARY_DIR}/generated/rtvdb_metal_rt_metallib.h)
    set(RTVDB_METAL_RT_SHADER_MODULE_CACHE ${CMAKE_CURRENT_BINARY_DIR}/generated/metal-module-cache)
    add_custom_command(
        OUTPUT ${RTVDB_METAL_RT_SHADER_AIR} ${RTVDB_METAL_RT_SHADER_METALLIB} ${RTVDB_METAL_RT_SHADER_METALLIB_HEADER}
        MAIN_DEPENDENCY ${RTVDB_METAL_RT_SHADER_MSL}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/generated
        COMMAND ${CMAKE_COMMAND} -E make_directory ${RTVDB_METAL_RT_SHADER_MODULE_CACHE}
        COMMAND ${RTVDB_METAL_EXECUTABLE}
            -fmodules-cache-path=${RTVDB_METAL_RT_SHADER_MODULE_CACHE}
            -c ${RTVDB_METAL_RT_SHADER_MSL}
            -o ${RTVDB_METAL_RT_SHADER_AIR}
        COMMAND ${RTVDB_METALLIB_EXECUTABLE} ${RTVDB_METAL_RT_SHADER_AIR} -o ${RTVDB_METAL_RT_SHADER_METALLIB}
        COMMAND ${CMAKE_COMMAND}
            -DINPUT_FILE=${RTVDB_METAL_RT_SHADER_METALLIB}
            -DOUTPUT_FILE=${RTVDB_METAL_RT_SHADER_METALLIB_HEADER}
            -DSYMBOL_NAME=kMetalRtMetallib
            -DSIZE_NAME=kMetalRtMetallibSize
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary_as_header.cmake
        DEPENDS
            ${RTVDB_METAL_RT_SHADER_MSL}
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary_as_header.cmake
        VERBATIM
    )
    set(RTVDB_METAL_RT_GENERATED_HEADERS ${RTVDB_METAL_RT_SHADER_METALLIB_HEADER})

    source_group("Shader Files" FILES ${RTVDB_METAL_RT_SHADER_MSL})
    source_group("CMake Files" FILES ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary_as_header.cmake)
endif()

set(SDL_SHARED OFF CACHE BOOL "Build SDL shared library" FORCE)
set(SDL_STATIC ON CACHE BOOL "Build SDL static library" FORCE)
set(SDL_TEST_LIBRARY OFF CACHE BOOL "Build SDL test library" FORCE)
add_subdirectory(third_party/SDL EXCLUDE_FROM_ALL)
if(WIN32 AND NOT RTVDB_ENABLE_D3D12_DXR AND NOT RTVDB_ENABLE_VULKAN_RT)
    message(FATAL_ERROR "rtvdb viewer now requires at least one Windows RT backend.")
endif()
add_library(rtvdb_header INTERFACE)
target_include_directories(rtvdb_header INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(RTVDB_FPNG_SOURCES
    third_party/fpng/src/fpng.cpp
    src/viewer_capture/png.cpp
)

set(RTVDB_IMGUI_SOURCES
    third_party/imgui/imgui.cpp
    third_party/imgui/imgui_draw.cpp
    third_party/imgui/imgui_tables.cpp
    third_party/imgui/imgui_widgets.cpp
    third_party/imgui/backends/imgui_impl_sdl3.cpp
    third_party/imgui/backends/imgui_impl_sdlrenderer3.cpp
)

set(RTVDB_VIEWER_BACKEND_SOURCES
    src/viewer_backend/backend.cpp
    src/viewer_backend/helper_overlay.cpp
    src/viewer_backend/rt_scene_builder.cpp
)
if(WIN32 AND RTVDB_ENABLE_D3D12_DXR)
    list(APPEND RTVDB_VIEWER_BACKEND_SOURCES
        src/viewer_backend/d3d12/d3d12_dxr.cpp
    )
endif()
if(RTVDB_ENABLE_VULKAN_RT)
    list(APPEND RTVDB_VIEWER_BACKEND_SOURCES
        src/viewer_backend/vulkan/vulkan_rt.cpp
    )
elseif(APPLE)
    list(APPEND RTVDB_VIEWER_BACKEND_SOURCES
        src/viewer_backend/metal/metal_rt.mm
    )
endif()

add_library(rtvdb_viewer_backend ${RTVDB_VIEWER_BACKEND_SOURCES})
target_include_directories(rtvdb_viewer_backend PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/fpng/src
)
if(WIN32 AND RTVDB_ENABLE_D3D12_DXR)
    target_sources(
        rtvdb_viewer_backend PRIVATE
            ${RTVDB_D3D12_RAYTRACING_HLSL}
            ${RTVDB_SHADER_SHARED_HEADERS})
endif()
if(RTVDB_ENABLE_VULKAN_RT)
    target_sources(
        rtvdb_viewer_backend PRIVATE
            ${RTVDB_VULKAN_RT_HLSL}
            ${RTVDB_SHADER_SHARED_HEADERS})
endif()
if(WIN32 AND RTVDB_ENABLE_D3D12_DXR)
    target_sources(
        rtvdb_viewer_backend PRIVATE
            ${RTVDB_D3D12_RAYTRACING_HEADER})
    target_include_directories(rtvdb_viewer_backend PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
endif()
if(APPLE)
    target_sources(rtvdb_viewer_backend PRIVATE ${RTVDB_METAL_RT_GENERATED_HEADERS})
    target_include_directories(rtvdb_viewer_backend PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
endif()
if(RTVDB_ENABLE_VULKAN_RT)
    target_sources(rtvdb_viewer_backend PRIVATE ${RTVDB_VULKAN_RT_GENERATED_HEADERS})
    target_include_directories(rtvdb_viewer_backend PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
endif()
if(WIN32 AND RTVDB_ENABLE_D3D12_DXR)
    target_compile_definitions(rtvdb_viewer_backend PUBLIC RTVDB_ENABLE_D3D12_DXR=1)
endif()
if(RTVDB_ENABLE_VULKAN_RT)
    target_compile_definitions(rtvdb_viewer_backend PUBLIC RTVDB_ENABLE_VULKAN_RT=1)
endif()
if(WIN32)
    target_link_libraries(rtvdb_viewer_backend PUBLIC dxgi)
    if(RTVDB_ENABLE_D3D12_DXR)
        target_link_libraries(rtvdb_viewer_backend PUBLIC d3d12)
    endif()
    if(RTVDB_ENABLE_VULKAN_RT)
        target_link_libraries(rtvdb_viewer_backend PUBLIC Vulkan::Vulkan)
    endif()
elseif(APPLE)
    target_link_libraries(rtvdb_viewer_backend PUBLIC "-framework Metal" "-framework QuartzCore" "-framework Foundation" "-framework AppKit" "-framework CoreVideo")
elseif(RTVDB_ENABLE_VULKAN_RT)
    target_link_libraries(rtvdb_viewer_backend PUBLIC Vulkan::Vulkan)
endif()

set(RTVDB_VIEWER_CAPTURE_SOURCES
    src/viewer_capture/capture.cpp
)

set(RTVDB_VIEWER_MAIN_GROUP_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/viewer/main.cpp
)
set(RTVDB_VIEWER_SOURCE_GROUP_FILES
    ${RTVDB_VIEWER_BACKEND_SOURCES}
    ${RTVDB_VIEWER_CAPTURE_SOURCES}
    src/viewer_capture/png.cpp
    src/viewer_session/session.cpp
    src/viewer_shell/shell_info.cpp
    src/viewer_shell/sdl3_shell.cpp
    src/viewer_shell/png_save_dialog.cpp
)
list(TRANSFORM RTVDB_VIEWER_SOURCE_GROUP_FILES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
set(RTVDB_IMGUI_GROUP_FILES ${RTVDB_IMGUI_SOURCES})
list(TRANSFORM RTVDB_IMGUI_GROUP_FILES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
set(RTVDB_FPNG_GROUP_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/fpng/src/fpng.cpp
)

if(APPLE)
    set_source_files_properties(
        third_party/fpng/src/fpng.cpp
        PROPERTIES
            COMPILE_OPTIONS "-Wno-tautological-constant-out-of-range-compare"
    )
endif()

source_group("Viewer" FILES ${RTVDB_VIEWER_MAIN_GROUP_FILES})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/src PREFIX "Source Files" FILES ${RTVDB_VIEWER_SOURCE_GROUP_FILES})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui PREFIX "External Libraries\\ImGui" FILES ${RTVDB_IMGUI_GROUP_FILES})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/fpng PREFIX "External Libraries\\fpng" FILES ${RTVDB_FPNG_GROUP_FILES})
if(WIN32 AND RTVDB_ENABLE_D3D12_DXR)
    source_group("Generated Files" FILES ${RTVDB_D3D12_RAYTRACING_HEADER})
elseif(RTVDB_ENABLE_VULKAN_RT)
    source_group("Generated Files" FILES ${RTVDB_VULKAN_RT_GENERATED_HEADERS})
elseif(APPLE)
    source_group("Generated Files" FILES ${RTVDB_METAL_RT_GENERATED_HEADERS})
endif()

add_library(rtvdb_viewer_session
    src/viewer_session/session.cpp
)
target_include_directories(rtvdb_viewer_session PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(rtvdb_viewer_session PUBLIC rtvdb_viewer_backend)
if(WIN32)
    target_link_libraries(rtvdb_viewer_session PUBLIC ws2_32)
endif()

add_library(rtvdb_viewer_shell
    src/viewer_shell/shell_info.cpp
    src/viewer_shell/sdl3_shell.cpp
    src/viewer_shell/png_save_dialog.cpp
)
if(APPLE)
    target_sources(rtvdb_viewer_shell PRIVATE src/viewer_shell/png_save_dialog_mac.mm)
endif()
target_include_directories(rtvdb_viewer_shell PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(rtvdb_viewer_shell PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/fpng/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui/backends
)
target_link_libraries(rtvdb_viewer_shell PUBLIC SDL3::SDL3)
if(RTVDB_ENABLE_VULKAN_RT)
    target_compile_definitions(rtvdb_viewer_shell PUBLIC RTVDB_ENABLE_VULKAN_RT=1)
    target_link_libraries(rtvdb_viewer_shell PUBLIC Vulkan::Vulkan)
endif()
if(WIN32)
    target_link_libraries(rtvdb_viewer_shell PUBLIC user32 gdi32)
elseif(APPLE)
    target_link_libraries(rtvdb_viewer_shell PUBLIC "-framework CoreVideo" "-framework AppKit" "-framework UniformTypeIdentifiers")
endif()
target_link_libraries(rtvdb_viewer_backend PRIVATE rtvdb_viewer_shell)

if(WIN32)
    add_executable(rtvdb_viewer WIN32
        viewer/main.cpp
    )
elseif(APPLE)
    add_executable(rtvdb_viewer MACOSX_BUNDLE
        viewer/main.cpp
    )
    set_target_properties(rtvdb_viewer PROPERTIES
        MACOSX_BUNDLE_GUI_IDENTIFIER "com.rtvdb.viewer"
        MACOSX_BUNDLE_BUNDLE_NAME "rtvdb Viewer"
        MACOSX_BUNDLE_BUNDLE_VERSION "${PROJECT_VERSION}"
        MACOSX_BUNDLE_SHORT_VERSION_STRING "${PROJECT_VERSION}"
        MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/cmake/rtvdb_viewer_Info.plist.in"
    )
else()
    add_executable(rtvdb_viewer
        viewer/main.cpp
    )
endif()
target_sources(rtvdb_viewer PRIVATE ${RTVDB_VIEWER_CAPTURE_SOURCES})
target_sources(rtvdb_viewer PRIVATE ${RTVDB_FPNG_SOURCES})
target_sources(rtvdb_viewer PRIVATE ${RTVDB_IMGUI_SOURCES})
target_include_directories(rtvdb_viewer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/fpng/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui/backends
)
target_link_libraries(rtvdb_viewer PRIVATE rtvdb_header rtvdb_viewer_session rtvdb_viewer_shell)
if(WIN32)
    target_link_libraries(rtvdb_viewer PRIVATE ws2_32 user32 gdi32 dbghelp)
endif()
if(WIN32 AND RTVDB_ENABLE_D3D12_DXR)
    add_custom_command(TARGET rtvdb_viewer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E rm -f $<TARGET_FILE_DIR:rtvdb_viewer>/dxcompiler.dll
    )
endif()

add_executable(rtvdb_example_client
    examples/basic_client.cpp
)
target_link_libraries(rtvdb_example_client PRIVATE rtvdb_header)
if(WIN32)
    target_link_libraries(rtvdb_example_client PRIVATE ws2_32)
endif()

add_executable(rtvdb_obj_stream_client
    examples/obj_stream_client.cpp
)
target_link_libraries(rtvdb_obj_stream_client PRIVATE rtvdb_header)
if(WIN32)
    target_link_libraries(rtvdb_obj_stream_client PRIVATE ws2_32)
endif()

add_executable(rtvdb_bt2020_volume_client
    examples/bt2020_volume_client.cpp
)
target_link_libraries(rtvdb_bt2020_volume_client PRIVATE rtvdb_header)
if(WIN32)
    target_link_libraries(rtvdb_bt2020_volume_client PRIVATE ws2_32)
endif()

install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/rtvdb/rtvdb.h
    ${CMAKE_CURRENT_SOURCE_DIR}/README.md
    ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE
    DESTINATION .
)

install(TARGETS rtvdb_viewer
    BUNDLE DESTINATION .
    RUNTIME DESTINATION .
)

install(TARGETS
    rtvdb_example_client
    rtvdb_bt2020_volume_client
    rtvdb_obj_stream_client
    RUNTIME DESTINATION examples
)

set(CPACK_GENERATOR ZIP)
set(CPACK_PACKAGE_NAME "rtvdb")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "rtvdb viewer and example clients")
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
if(WIN32)
    set(CPACK_PACKAGE_FILE_NAME "rtvdb-win")
elseif(APPLE)
    set(CPACK_PACKAGE_FILE_NAME "rtvdb-mac")
else()
    set(CPACK_PACKAGE_FILE_NAME "rtvdb-${CMAKE_SYSTEM_NAME}")
endif()
include(CPack)
