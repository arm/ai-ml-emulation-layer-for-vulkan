# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

macro(mlel_limits TYPE OUTPUT)
    if(${TYPE} MATCHES "bool")
        set(${OUTPUT}_LOWEST "0")
        set(${OUTPUT}_MAX "1")
        set(${OUTPUT}_TYPE "0x6231")
    elseif(${TYPE} MATCHES "int8_t")
        set(${OUTPUT}_LOWEST "-128")
        set(${OUTPUT}_MAX "127")
        set(${OUTPUT}_TYPE "0x6931")
    elseif(${TYPE} MATCHES "int16_t")
        set(${OUTPUT}_LOWEST "-32768")
        set(${OUTPUT}_MAX "32767")
        set(${OUTPUT}_TYPE "0x6932")
    elseif(${TYPE} MATCHES "int")
        set(${OUTPUT}_LOWEST "-2147483648")
        set(${OUTPUT}_MAX "2147483647")
        set(${OUTPUT}_TYPE "0x6934")
    elseif(${TYPE} MATCHES "float16_t")
        set(${OUTPUT}_LOWEST "-65504")
        set(${OUTPUT}_MAX "65504")
        set(${OUTPUT}_TYPE "0x6632")
    elseif(${TYPE} MATCHES "float")
        set(${OUTPUT}_LOWEST "-3.402823466e+38")
        set(${OUTPUT}_MAX "3.402823466e+38")
        set(${OUTPUT}_TYPE "0x6634")
    elseif(${TYPE} MATCHES "double")
        set(${OUTPUT}_LOWEST "-1.7976931348623158e+308")
        set(${OUTPUT}_MAX "1.7976931348623158e+308")
        set(${OUTPUT}_TYPE "0x6638")
    endif()
endmacro()

# Compile list of arguments, removing the first three
math(EXPR COUNT "${CMAKE_ARGC} - 1")
foreach(i RANGE 3 ${COUNT})
    list(APPEND ARGV "${CMAKE_ARGV${i}}")
endforeach()

# Parse command line arguments
cmake_parse_arguments(ARGS "" "INPUT_FILE;OUTPUT_FILE;GLSLANG" "REPLACE" ${ARGV})

# Read source file into memory
file(READ ${ARGS_INPUT_FILE} GLSL)

foreach(R ${ARGS_REPLACE})
    string(REGEX MATCH "^([^=]+)=(.*)$" R ${R})
    set(KEY ${CMAKE_MATCH_1})
    set(VAL ${CMAKE_MATCH_2})
    string(REPLACE "\\ " " " VAL ${VAL})

    mlel_limits(${VAL} ${KEY})

    string(REPLACE "%${KEY}%" "${VAL}" GLSL "${GLSL}")
    string(REPLACE "%${KEY}_lowest%" "${${KEY}_LOWEST}" GLSL "${GLSL}")
    string(REPLACE "%${KEY}_max%" "${${KEY}_MAX}" GLSL "${GLSL}")
    string(REPLACE "%${KEY}_type%" "${${KEY}_TYPE}" GLSL "${GLSL}")
endforeach()

file(WRITE ${ARGS_OUTPUT_FILE}.comp "${GLSL}")

set(GLSLANG_FLAGS -V --target-env vulkan1.3)

execute_process(
    COMMAND ${ARGS_GLSLANG} ${GLSLANG_FLAGS} -o ${ARGS_OUTPUT_FILE} ${ARGS_OUTPUT_FILE}.comp
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
