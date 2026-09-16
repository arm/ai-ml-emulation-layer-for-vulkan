# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

###############################################################################
# Generate shaders
###############################################################################

# Get a list of all shaders under graph_op/, excluding the shared common header.
file(GLOB_RECURSE SOURCES_GRAPH_OP CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/graph_op/*.comp")
list(FILTER SOURCES_GRAPH_OP EXCLUDE REGEX "/common\\.comp$")
list(SORT SOURCES_GRAPH_OP)

set(GLSL_SOURCES "")
add_custom_target(mlel_generate_shader_comp)
macro(mlel_add_glsl_source SOURCE USE_COMMON)
    if("${USE_COMMON}")
        get_filename_component(NAME "${SOURCE}" NAME)
        set(OUT_FILE "${CMAKE_CURRENT_BINARY_DIR}/${NAME}")

        add_custom_command(
            OUTPUT "${OUT_FILE}"
            COMMAND ${CMAKE_COMMAND} -E cat "${CMAKE_CURRENT_SOURCE_DIR}/graph_op/common.comp" "${SOURCE}" > "${OUT_FILE}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/graph_op/common.comp" "${SOURCE}")
        add_custom_target("mlel_generate_shader_${NAME}" DEPENDS "${OUT_FILE}")
        add_dependencies(mlel_generate_shader_comp "mlel_generate_shader_${NAME}")
        list(APPEND GLSL_SOURCES "${OUT_FILE}")
    else()
        list(APPEND GLSL_SOURCES "${SOURCE}")
    endif()
endmacro()

foreach(SOURCE ${SOURCES_GRAPH_OP})
    mlel_add_glsl_source("${SOURCE}" TRUE)
endforeach()

file(GLOB SOURCES_OPTICAL_FLOW CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/optical_flow/*.comp")
list(SORT SOURCES_OPTICAL_FLOW)
foreach(SOURCE ${SOURCES_OPTICAL_FLOW})
    mlel_add_glsl_source("${SOURCE}" FALSE)
endforeach()

# Combine all outputs into a single target
add_custom_target(mlel_generate_all_shaders ALL DEPENDS ${GLSL_SOURCES} mlel_generate_shader_comp)

###############################################################################
# Generate pre-compiled shaders
###############################################################################

# Each module is compiled independently so edits to the pipeline cache do not
# recompile the embedded binaries, and changing one module rebuilds only its data.
macro(mlel_generate_spv_library)
    set(SPV_SOURCES "")
    set(SHORT_SPV_FILES "")
    foreach(SPV_FILE IN LISTS SPV_FILES)
        set(SOURCE "${SPV_FILE}.cpp")
        add_custom_command(
            OUTPUT "${SOURCE}"
            COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/generate_spv_source.cmake
                OUTPUT_FILE "${SOURCE}"
                INPUT_FILE "${SPV_FILE}"
            DEPENDS
                ${CMAKE_CURRENT_SOURCE_DIR}/generate_spv_source.cmake
                "${SPV_FILE}")
        list(APPEND SPV_SOURCES "${SOURCE}")
        get_filename_component(NAME "${SPV_FILE}" NAME)
        list(APPEND SHORT_SPV_FILES "${NAME}")
    endforeach()

    set(REGISTRY "${CMAKE_CURRENT_BINARY_DIR}/shaders_spv.cpp")
    add_custom_command(
        OUTPUT "${REGISTRY}"
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/generate_spv_registry.cmake
            OUTPUT_FILE "${REGISTRY}"
            INPUT_FILES ${SHORT_SPV_FILES}
        DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/generate_spv_registry.cmake
            ${SPV_FILES}
            mlel_generate_all_shaders
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    add_library(MLELPrecompiledShaders STATIC ${SPV_SOURCES} "${REGISTRY}")
    target_compile_options(MLELPrecompiledShaders PRIVATE ${VMEL_COMPILE_OPTIONS})
    target_include_directories(MLELPrecompiledShaders PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
endmacro()

# Cross-compilation requires a compiler that runs on the build host.
if((CMAKE_CROSSCOMPILING OR NOT TARGET glslang-standalone) AND NOT GLSLANG_EXECUTABLE)
    find_program(GLSLANG_EXECUTABLE NAMES glslang glslangValidator NO_CMAKE_FIND_ROOT_PATH)
    if(NOT GLSLANG_EXECUTABLE)
        message(FATAL_ERROR "Precompiled shaders require a host glslang compiler. Set GLSLANG_EXECUTABLE to its path.")
    endif()
endif()

if(GLSLANG_EXECUTABLE)
    execute_process(COMMAND "${GLSLANG_EXECUTABLE}" --version
        RESULT_VARIABLE GLSLANG_RESULT OUTPUT_QUIET ERROR_QUIET)
    if(NOT GLSLANG_RESULT STREQUAL "0")
        message(FATAL_ERROR "Cannot run host shader compiler: ${GLSLANG_EXECUTABLE}")
    endif()
endif()

if(GLSLANG_EXECUTABLE)
    set(GLSLANG_CMD "${GLSLANG_EXECUTABLE}")
    set(GLSLANG_DEP "")
else()
    set(GLSLANG_CMD "$<TARGET_FILE:glslang-standalone>")
    set(GLSLANG_DEP glslang-standalone)
endif()

set(WARP1D 64)
set(WARPX 8)
set(WARPY 8)
set(WARPZ 1)

function(mlel_generate_glsl)
    cmake_parse_arguments(ARGS "" "INPUT;OUTPUT" "REPLACE" ${ARGN})
    # Aliases (e.g. widened i4/i48 storage) may share a module, but different
    # shader substitutions must never silently share an output/key.
    string(SHA256 OUTPUT_ID "${ARGS_OUTPUT}")
    set(SIGNATURE "${ARGS_INPUT};${ARGS_REPLACE}")
    get_property(PREVIOUS DIRECTORY PROPERTY "MLEL_SPV_${OUTPUT_ID}")
    if(PREVIOUS)
        if(NOT PREVIOUS STREQUAL SIGNATURE)
            message(FATAL_ERROR "Conflicting shader variants for ${ARGS_OUTPUT}")
        endif()
        return()
    endif()
    set_property(DIRECTORY PROPERTY "MLEL_SPV_${OUTPUT_ID}" "${SIGNATURE}")
    # Generate SPV
    add_custom_command(
        OUTPUT ${ARGS_OUTPUT}
        COMMAND ${CMAKE_COMMAND}
            -DUSE_FLOAT_AS_DOUBLE=${VMEL_USE_FLOAT_AS_DOUBLE}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/generate_spv.cmake
            GLSLANG "${GLSLANG_CMD}"
            OUTPUT_FILE "${ARGS_OUTPUT}"
            INPUT_FILE "${ARGS_INPUT}"
            REPLACE ${ARGS_REPLACE}
        DEPENDS
            ${GLSLANG_DEP}
            ${CMAKE_CURRENT_SOURCE_DIR}/generate_spv.cmake
            ${ARGS_INPUT})
endfunction()

function(mlel_spv_recurse)
    cmake_parse_arguments(ARGS "" "INPUT;OUTPUT;OUTPUT_VARIABLE" "REPLACE;TYPES" ${ARGV})

    list(LENGTH ARGS_TYPES LENGTH)

    # Generate shader if there are no more types to handle
    if(LENGTH EQUAL 0)
        mlel_generate_glsl(
            INPUT ${ARGS_INPUT}
            OUTPUT ${ARGS_OUTPUT}.spv
            REPLACE ${ARGS_REPLACE}
            REPLACE "warpX=${WARP1D}")

        list(APPEND ${ARGS_OUTPUT_VARIABLE} ${ARGS_OUTPUT}.spv)
        set(${ARGS_OUTPUT_VARIABLE} "${${ARGS_OUTPUT_VARIABLE}}" PARENT_SCOPE)

        return()
    endif()

    # Pop first index from TYPES list
    list(GET ARGS_TYPES 0 TYPES)
    list(POP_FRONT ARGS_TYPES ARGS_TYPES)

    # Convert TYPES into a list
    # First index is the key for the string replacement
    string(REPLACE " " ";" TYPES ${TYPES})
    list(GET TYPES 0 KEY)
    list(POP_FRONT TYPES TYPES)

    foreach(TYPE ${TYPES})
        mlel_spv_recurse(
            INPUT ${ARGS_INPUT}
            OUTPUT ${ARGS_OUTPUT}_${TYPE}
            OUTPUT_VARIABLE ${ARGS_OUTPUT_VARIABLE}
            REPLACE ${ARGS_REPLACE}
            REPLACE "${KEY}=${TYPE}"
            TYPES ${ARGS_TYPES})
    endforeach()

    # Propagate output variable to parent scope
    set(${ARGS_OUTPUT_VARIABLE} "${${ARGS_OUTPUT_VARIABLE}}" PARENT_SCOPE)
endfunction()

macro(mlel_spv SHADER)
    mlel_spv_recurse(
        INPUT ${CMAKE_CURRENT_BINARY_DIR}/${SHADER}.comp
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${SHADER}
        OUTPUT_VARIABLE SPV_FILES
        TYPES ${ARGN})
endmacro()

# Convolutions
macro(mlel_spv_convolution OPERATION IN_T WEIGHT_T OUT_T ACC_T)
    set(INPUT "${CMAKE_CURRENT_BINARY_DIR}/${OPERATION}.comp")
    set(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${OPERATION}_${IN_T}_${WEIGHT_T}_${OUT_T}_${ACC_T}.spv")

    set(WARP ${WARP1D})
    if(${OPERATION} STREQUAL "conv2d")
        set(WARP ${WARPX})
    endif()

    mlel_generate_glsl(
        INPUT ${INPUT}
        OUTPUT ${OUTPUT}
        REPLACE "warpX=${WARP}"
        REPLACE "warpY=${WARPY}"
        REPLACE "warpZ=${WARPZ}"
        REPLACE "in_t=${IN_T}"
        REPLACE "out_t=${OUT_T}"
        REPLACE "weight_t=${WEIGHT_T}"
        REPLACE "acc_t=${ACC_T}")

    list(APPEND SPV_FILES ${OUTPUT})
endmacro()

# Elementwise
macro(mlel_spv_elementwise PIPELINE NAME OPERATION)
    foreach(IN_OUT_T ${ARGN})
        set(INPUT "${CMAKE_CURRENT_BINARY_DIR}/${PIPELINE}.comp")

        set(IN_T ${IN_OUT_T})
        set(OUT_T ${IN_OUT_T})

        if(${NAME} STREQUAL "equal" OR ${NAME} STREQUAL "greater" OR ${NAME} STREQUAL "greater_equal")
            set(OUT_T "bool")
        endif()

        string(TOUPPER "${NAME}" OP_NAME)
        set(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${PIPELINE}_${OP_NAME}_${IN_T}")
        if("${PIPELINE}" STREQUAL "elementwise_binary")
            string(APPEND OUTPUT "_${OUT_T}")
        endif()
        string(APPEND OUTPUT ".spv")

        mlel_generate_glsl(
            INPUT ${INPUT}
            OUTPUT ${OUTPUT}
            REPLACE "warpX=${WARP1D}"
            REPLACE "in_t=${IN_T}"
            REPLACE "out_t=${OUT_T}"
            REPLACE "in_out_t=${IN_OUT_T}"
            REPLACE "operation=\"${OPERATION}\"")

        list(APPEND SPV_FILES ${OUTPUT})
    endforeach()
endmacro()

# Reduce
macro(mlel_spv_reduce NAME INIT OPERATION TYPES)
    foreach(TYPE ${TYPES})
        set(INPUT "${CMAKE_CURRENT_BINARY_DIR}/reduce.comp")
        string(TOUPPER "reduce_${NAME}" OP_NAME)
        set(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/reduce_${OP_NAME}_${TYPE}.spv")

        mlel_generate_glsl(
            INPUT ${INPUT}
            OUTPUT ${OUTPUT}
            REPLACE "warpX=${WARP1D}"
            REPLACE "init=\"${INIT}\""
            REPLACE "in_out_t=${TYPE}"
            REPLACE "operation=\"${OPERATION}\"")

        list(APPEND SPV_FILES ${OUTPUT})
    endforeach()
endmacro()

if(WIN32 AND (CMAKE_GENERATOR MATCHES "Visual Studio"))
    set(ESCAPE_CHAR "%")
else()
    set(ESCAPE_CHAR "")
endif()

include(tosa_shader_variants.cmake)

# Block Match
macro(mlel_spv_block_match OPERATION SEARCH_TYPE COST_TYPE)
    set(INPUT "${CMAKE_CURRENT_BINARY_DIR}/${OPERATION}.comp")
    set(NAME "${OPERATION}_${SEARCH_TYPE}")
    if("${SEARCH_TYPE}" EQUAL 2)
        string(APPEND NAME "_${COST_TYPE}")
    endif()
    set(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${NAME}.spv")

    mlel_generate_glsl(
        INPUT ${INPUT}
        OUTPUT ${OUTPUT}
        REPLACE "warpX=8"
        REPLACE "warpY=8"
        REPLACE "search_type=${SEARCH_TYPE}"
        REPLACE "cost_t=${COST_TYPE}")

    list(APPEND SPV_FILES ${OUTPUT})
endmacro()

mlel_spv_block_match(block_match "0" uint16_t)
mlel_spv_block_match(block_match "1" uint16_t)
# RAW_SAD permits both 8-bit and 16-bit cost tensors.
mlel_spv_block_match(block_match "2" uint8_t)
mlel_spv_block_match(block_match "2" uint16_t)

# Optical Flow
macro(mlel_spv_optical_flow)
    foreach(SOURCE ${SOURCES_OPTICAL_FLOW})
        get_filename_component(NAME "${SOURCE}" NAME_WE)
        set(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${NAME}.spv")

        mlel_generate_glsl(
            INPUT "${SOURCE}"
            OUTPUT ${OUTPUT})

        list(APPEND SPV_FILES ${OUTPUT})
    endforeach()
endmacro()

mlel_spv_optical_flow()

# Generate independently compiled SPIR-V modules and their lookup registry
mlel_generate_spv_library()
