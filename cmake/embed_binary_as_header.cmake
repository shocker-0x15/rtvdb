if(NOT DEFINED INPUT_FILE OR NOT DEFINED OUTPUT_FILE)
    message(FATAL_ERROR "INPUT_FILE and OUTPUT_FILE are required.")
endif()

if(NOT DEFINED SYMBOL_NAME)
    set(SYMBOL_NAME kD3d12RaytracingDxil)
endif()

if(NOT DEFINED SIZE_NAME)
    set(SIZE_NAME ${SYMBOL_NAME}Size)
endif()

file(READ "${INPUT_FILE}" _rtvdb_hex HEX)
string(LENGTH "${_rtvdb_hex}" _rtvdb_hex_length)
math(EXPR _rtvdb_last_index "${_rtvdb_hex_length} - 2")

set(_rtvdb_lines "")
foreach(_rtvdb_index RANGE 0 ${_rtvdb_last_index} 2)
    string(SUBSTRING "${_rtvdb_hex}" ${_rtvdb_index} 2 _rtvdb_byte)
    string(APPEND _rtvdb_lines "0x${_rtvdb_byte},")
    math(EXPR _rtvdb_col "${_rtvdb_index} / 2 % 12")
    if(_rtvdb_col EQUAL 11)
        string(APPEND _rtvdb_lines "\n")
    else()
        string(APPEND _rtvdb_lines " ")
    endif()
endforeach()

file(WRITE "${OUTPUT_FILE}" "#pragma once\n\n#include <cstddef>\n#include <cstdint>\n\nnamespace rtvdb::viewer_backend {\n\ninline constexpr std::uint8_t ${SYMBOL_NAME}[] = {\n${_rtvdb_lines}\n};\ninline constexpr std::size_t ${SIZE_NAME} = sizeof(${SYMBOL_NAME});\n\n} // namespace rtvdb::viewer_backend\n")
