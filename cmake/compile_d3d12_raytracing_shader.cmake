if(NOT DEFINED DXC_EXECUTABLE OR NOT DEFINED HLSL_INPUT OR NOT DEFINED DXIL_OUTPUT OR
   NOT DEFINED EMBEDDED_HEADER_OUTPUT OR NOT DEFINED EMBED_SCRIPT OR NOT DEFINED STAMP_OUTPUT)
    message(FATAL_ERROR "D3D12 shader compilation requires all output and tool paths.")
endif()

get_filename_component(_d3d12_output_directory "${DXIL_OUTPUT}" DIRECTORY)
file(MAKE_DIRECTORY "${_d3d12_output_directory}")

execute_process(
    COMMAND "${DXC_EXECUTABLE}"
        -T lib_6_6
        -HV 2021
        -Zi
        -Qembed_debug
        -Fo "${DXIL_OUTPUT}"
        "${HLSL_INPUT}"
    RESULT_VARIABLE _d3d12_dxc_result
)
if(NOT _d3d12_dxc_result EQUAL 0)
    message(FATAL_ERROR "D3D12 HLSL compilation failed with exit code ${_d3d12_dxc_result}.")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}"
        "-DINPUT_FILE=${DXIL_OUTPUT}"
        "-DOUTPUT_FILE=${EMBEDDED_HEADER_OUTPUT}"
        -P "${EMBED_SCRIPT}"
    RESULT_VARIABLE _d3d12_embed_result
)
if(NOT _d3d12_embed_result EQUAL 0)
    message(FATAL_ERROR "Embedding the D3D12 shader binary failed with exit code ${_d3d12_embed_result}.")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E touch "${STAMP_OUTPUT}"
    RESULT_VARIABLE _d3d12_stamp_result
)
if(NOT _d3d12_stamp_result EQUAL 0)
    message(FATAL_ERROR "Updating the D3D12 shader stamp failed with exit code ${_d3d12_stamp_result}.")
endif()
