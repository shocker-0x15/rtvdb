cmake_minimum_required(VERSION 3.20)

get_filename_component(RTVDB_ROOT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(RTVDB_SDL_DIRECTORY "${RTVDB_ROOT_DIRECTORY}/third_party/SDL")
set(RTVDB_SDL_PATCH_DIRECTORY "${RTVDB_ROOT_DIRECTORY}/third_party/patches/SDL")

find_program(RTVDB_GIT_EXECUTABLE NAMES git REQUIRED)
file(GLOB RTVDB_SDL_PATCHES "${RTVDB_SDL_PATCH_DIRECTORY}/*.patch")
list(SORT RTVDB_SDL_PATCHES)

foreach(RTVDB_SDL_PATCH IN LISTS RTVDB_SDL_PATCHES)
    execute_process(
        COMMAND ${RTVDB_GIT_EXECUTABLE} -C ${RTVDB_SDL_DIRECTORY}
            apply --check --reverse ${RTVDB_SDL_PATCH}
        RESULT_VARIABLE RTVDB_SDL_PATCH_ALREADY_APPLIED
        OUTPUT_QUIET
        ERROR_QUIET)
    if(RTVDB_SDL_PATCH_ALREADY_APPLIED EQUAL 0)
        message(STATUS "SDL patch already applied: ${RTVDB_SDL_PATCH}")
        continue()
    endif()

    execute_process(
        COMMAND ${RTVDB_GIT_EXECUTABLE} -C ${RTVDB_SDL_DIRECTORY}
            apply --3way ${RTVDB_SDL_PATCH}
        RESULT_VARIABLE RTVDB_SDL_PATCH_RESULT
        OUTPUT_VARIABLE RTVDB_SDL_PATCH_OUTPUT
        ERROR_VARIABLE RTVDB_SDL_PATCH_ERROR)
    if(NOT RTVDB_SDL_PATCH_RESULT EQUAL 0)
        message(FATAL_ERROR
            "Failed to apply SDL patch: ${RTVDB_SDL_PATCH}\n"
            "${RTVDB_SDL_PATCH_OUTPUT}${RTVDB_SDL_PATCH_ERROR}")
    endif()
    message(STATUS "Applied SDL patch: ${RTVDB_SDL_PATCH}")
endforeach()
