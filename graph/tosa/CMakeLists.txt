# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

###############################################################################
# Generate shaders
###############################################################################

# Get a list of all shader, excluding the common header
file(GLOB SOURCES "*.comp")
list(FILTER SOURCES EXCLUDE REGEX common.comp)

# Merge common header with GLSL sources
foreach(SOURCE ${SOURCES})
    get_filename_component(NAME ${SOURCE} NAME)

    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${NAME}"
        COMMAND ${CMAKE_COMMAND} -E cat "${CMAKE_CURRENT_SOURCE_DIR}/common.comp" "${SOURCE}" > "${CMAKE_CURRENT_BINARY_DIR}/${NAME}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/common.comp" "${SOURCE}")

    list(APPEND GLSL_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/${NAME}")
endforeach()

# Generate C++ header file
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shaders.hpp.inc
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/generate_glsl_header.cmake
        OUTPUT_FILE "${CMAKE_CURRENT_BINARY_DIR}/shaders.hpp.inc"
        SRCS ${GLSL_SOURCES}
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/generate_glsl_header.cmake
        ${GLSL_SOURCES})

add_custom_target(mlel_generate_glsl_header DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/shaders.hpp.inc)

###############################################################################
# Generate pre-compiled shaders
###############################################################################

# Macro for generating target
macro(mlel_generate_spv_inc)
    # Strip the path from the SPV files
    set(SHORT_SPV_FILES "")
    foreach(f ${SPV_FILES})
        get_filename_component(fname "${f}" NAME)
        list(APPEND SHORT_SPV_FILES "${fname}")
    endforeach()
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shaders_spv.hpp.inc
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/generate_spv_header.cmake
            OUTPUT_FILE "${CMAKE_CURRENT_BINARY_DIR}/shaders_spv.hpp.inc"
            INPUT_FILES ${SHORT_SPV_FILES}
        DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/generate_spv_header.cmake
            ${SHORT_SPV_FILES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    add_custom_target(mlel_generate_spv_header DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/shaders_spv.hpp.inc)
endmacro()

# Verify that required dependencies exist
if(NOT TARGET glslang-standalone OR VMEL_DISABLE_PRECOMPILE_SHADERS)
    # Generate include file with empty shader map
    mlel_generate_spv_inc()
    return()
endif()

set(WARP1D 64)
set(WARPX 8)
set(WARPY 8)
set(WARPZ 1)

function(mlel_generate_glsl)
    cmake_parse_arguments(ARGS "" "INPUT;OUTPUT" "REPLACE" ${ARGN})

    # Generate SPV
    add_custom_command(
        OUTPUT ${ARGS_OUTPUT}
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/generate_spv.cmake
            GLSLANG "$<TARGET_FILE:glslang-standalone>"
            OUTPUT_FILE "${ARGS_OUTPUT}"
            INPUT_FILE "${ARGS_INPUT}"
            REPLACE ${ARGS_REPLACE}
        DEPENDS
            glslang-standalone
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

mlel_spv(argmax "in_t int8_t int16_t float16_t float")
mlel_spv(avgpool2d "in_out_t int8_t int16_t float16_t float" "acc_t int float") # TODO acc_t float16_t
mlel_spv(cast "in_t float16_t float" "out_t int8_t int16_t int float16_t float")
mlel_spv(cast "in_t int8_t int16_t int" "out_t bool int8_t int16_t int float16_t float")
mlel_spv(cast "in_t bool" "out_t int8_t int16_t int")
mlel_spv(clamp "in_out_t int8_t int16_t float16_t float")
mlel_spv(concat "in_out_t bool int8_t int16_t int float16_t float")
mlel_spv(fft2d "in_out_t float")
mlel_spv(gather "in_out_t int8_t int16_t int float16_t float" "index_t int")
mlel_spv(matmul "in_t int8_t" "out_t int")
mlel_spv(matmul "in_t int16_t" "out_t int64_t")
mlel_spv(matmul "in_t float16_t" "out_t float16_t float")
mlel_spv(matmul "in_t float" "out_t float")
mlel_spv(maxpool2d "in_out_t int8_t int16_t float16_t float")
mlel_spv(mul "in_t int8_t int16_t int" "out_t int")
mlel_spv(mul "in_t float16_t" "out_t float16_t")
mlel_spv(mul "in_t float" "out_t float")
mlel_spv(negate "in_out_t int8_t int16_t int" "acc_t int")
mlel_spv(negate "in_out_t float16_t" "acc_t float16_t")
mlel_spv(negate "in_out_t float" "acc_t float")
mlel_spv(pad "in_out_t bool int8_t int16_t int float16_t float")
mlel_spv(rescale "in_t int8_t int16_t int int64_t" "out_t int8_t int16_t int" "mul_t int16_t int")
mlel_spv(reshape "in_out_t bool int8_t int16_t int float16_t float")
mlel_spv(resize "in_t int8_t" "out_t int8_t int")
mlel_spv(resize "in_t int16_t" "out_t int16_t int64_t")
mlel_spv(resize "in_t float16_t" "out_t float16_t")
mlel_spv(resize "in_t float" "out_t float")
mlel_spv(reverse "in_out_t bool int8_t int16_t int float16_t float")
mlel_spv(rfft2d "in_out_t float")
mlel_spv(scatter "in_out_t int8_t int16_t int float16_t float" "index_t int")
mlel_spv(select "in_out_t bool int8_t int16_t int float16_t float")
mlel_spv(slice "in_out_t bool int8_t int16_t int float16_t float")
mlel_spv(table "in_t int8_t" "out_t int8_t")
mlel_spv(table "in_t int16_t" "out_t int")
mlel_spv(tile "in_out_t bool int8_t int16_t int float16_t float")
mlel_spv(transpose "in_out_t bool int8_t int16_t int float16_t float")

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

foreach(OPERATION conv2d conv3d depthwise_conv2d transpose_conv2d)
    mlel_spv_convolution(${OPERATION} float16_t float16_t float16_t float16_t)
    mlel_spv_convolution(${OPERATION} float16_t float16_t float16_t float)
    mlel_spv_convolution(${OPERATION} float float float float)
    mlel_spv_convolution(${OPERATION} int8_t int8_t int int)
    mlel_spv_convolution(${OPERATION} int16_t int8_t int64_t int64_t)
endforeach()

# Elementwise
macro(mlel_spv_elementwise PIPELINE NAME OPERATION)
    foreach(IN_OUT_T ${ARGN})
        set(INPUT "${CMAKE_CURRENT_BINARY_DIR}/${PIPELINE}.comp")
        set(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${PIPELINE}_${NAME}_${IN_OUT_T}.spv")

        set(IN_T ${IN_OUT_T})
        set(OUT_T ${IN_OUT_T})

        if(${NAME} STREQUAL "equal" OR ${NAME} STREQUAL "greater" OR ${NAME} STREQUAL "greater_equal")
            set(OUT_T "bool")
        endif()

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

# Elementwise binary
mlel_spv_elementwise(elementwise_binary add "value1 + value2" int float16_t float)
mlel_spv_elementwise(elementwise_binary bitwise_and "value1 & value2" int8_t int16_t int)
mlel_spv_elementwise(elementwise_binary bitwise_or "value1 | value2" int8_t int16_t int)
mlel_spv_elementwise(elementwise_binary bitwise_xor "value1 ^ value2" int8_t int16_t int)
mlel_spv_elementwise(elementwise_binary intdiv "value1 / value2" int)
mlel_spv_elementwise(elementwise_binary logical_and "value1 && value2" bool)
mlel_spv_elementwise(elementwise_binary logical_left_shift "uint(value1) << uint(value2)" int8_t int16_t int)
mlel_spv_elementwise(elementwise_binary logical_right_shift "zeroExtend(value1) >> uint(value2)" int8_t int16_t int)
mlel_spv_elementwise(elementwise_binary logical_or "value1 || value2" bool)
mlel_spv_elementwise(elementwise_binary logical_xor "value1 ^^ value2" bool)
mlel_spv_elementwise(elementwise_binary maximum "applyMax(value1, value2, pushConstants.nanMode)" int float16_t float)
mlel_spv_elementwise(elementwise_binary minimum "applyMin(value1, value2, pushConstants.nanMode)" int float16_t float)
mlel_spv_elementwise(elementwise_binary pow "power(value1, value2)" float16_t float)
mlel_spv_elementwise(elementwise_binary sub "value1 - value2" int float16_t float)

# Elementwise unary
mlel_spv_elementwise(elementwise_unary abs "abs(value1)" int float16_t float)
mlel_spv_elementwise(elementwise_unary bitwise_not "~value1" int8_t int16_t int)
mlel_spv_elementwise(elementwise_unary ceil "ceil(value1)" float16_t float)
mlel_spv_elementwise(elementwise_unary clz "clz(value1)" int)
mlel_spv_elementwise(elementwise_unary cos "cos(value1)" float16_t float)
mlel_spv_elementwise(elementwise_unary exp "exp(value1)" float16_t float)
mlel_spv_elementwise(elementwise_unary floor "floor(value1)" float16_t float)
mlel_spv_elementwise(elementwise_unary log "log(value1)" float16_t float)
mlel_spv_elementwise(elementwise_unary logical_not "!value1" bool)
mlel_spv_elementwise(elementwise_unary reciprocal "1.0 / value1" float16_t float)
mlel_spv_elementwise(elementwise_unary rsqrt "inversesqrt(value1)" float16_t float)
mlel_spv_elementwise(elementwise_unary sin "sin(value1)" float16_t float)

# Comparison
mlel_spv_elementwise(elementwise_binary equal "value1 == value2" int float16_t float)
mlel_spv_elementwise(elementwise_binary greater "value1 > value2" int float16_t float)
mlel_spv_elementwise(elementwise_binary greater_equal "value1 >= value2" int float16_t float)

# Activation functions
mlel_spv_elementwise(elementwise_unary erf "erf(value1)" float16_t float)
mlel_spv_elementwise(elementwise_unary sigmoid "1.0 / (1.0 + exp(-value1))" float16_t float)
mlel_spv_elementwise(elementwise_unary tanh "tanh_clamped(value1)" float16_t float)

# Reduce
macro(mlel_spv_reduce NAME INIT OPERATION TYPES)
    foreach(TYPE ${TYPES})
        set(INPUT "${CMAKE_CURRENT_BINARY_DIR}/reduce.comp")
        set(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/reduce_${NAME}_${TYPE}.spv")

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

mlel_spv_reduce(all "true" "result && value" "bool")
mlel_spv_reduce(any "false" "result || value" "bool")
mlel_spv_reduce(
    max
    "(pushConstants.nanMode == NAN_MODE_IGNORE) ? IN_OUT_T(NAN) : IN_OUT_T(${ESCAPE_CHAR}%in_out_t_lowest${ESCAPE_CHAR}%)"
    "max(result, value)"
    "int8_t;int16_t;int;float16_t;float")
mlel_spv_reduce(
    min
    "(pushConstants.nanMode == NAN_MODE_IGNORE) ? IN_OUT_T(NAN) : IN_OUT_T(${ESCAPE_CHAR}%in_out_t_max${ESCAPE_CHAR}%)"
    "min(result, value)"
    "int8_t;int16_t;int;float16_t;float")
mlel_spv_reduce(product "1" "result * value" "float16_t;float")
mlel_spv_reduce(sum "0" "result + value" "int;float16_t;float")

# Generate header file with SPIR-V modules
mlel_generate_spv_inc()
