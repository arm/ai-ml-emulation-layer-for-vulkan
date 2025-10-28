# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

# Compile list of arguments, removing the first three
math(EXPR COUNT "${CMAKE_ARGC} - 1")
foreach(i RANGE 3 ${COUNT})
    list(APPEND ARGV "${CMAKE_ARGV${i}}")
endforeach()

# Parse command line arguments
cmake_parse_arguments(ARGS "" "OUTPUT_FILE" "SRCS" ${ARGV})

# Write header file header
file(WRITE "${ARGS_OUTPUT_FILE}" "#pragma once\n#include <map>\n#include <string_view>\nnamespace {\n")

set(SHADER_MAP "const std::map<std::string_view, std::string_view> glslMap {\n")

# Generate a string view for each shader
foreach(SHADER_FILE ${ARGS_SRCS})
    get_filename_component(SHADER_NAME ${SHADER_FILE} NAME_WE)

    # Read shader file
    file(READ "${SHADER_FILE}" SHADER_GLSL)

    # Get string length
    string(LENGTH "${SHADER_GLSL}" LENGTH)

    # Handle empty files
    if(LENGTH EQUAL 0)
        message(FATAL_ERROR "Shader file is empty: ${SHADER_FILE}")
    endif()

    # Debug output for troubleshooting
    message(STATUS "Processing ${SHADER_FILE}: length=${LENGTH}")

    # Append shader variable declaration
    file(APPEND "${ARGS_OUTPUT_FILE}" "constexpr std::string_view ${SHADER_NAME}GLSL =\n")

    # Split into chunks to avoid compiler string length limits
    set(OFFSET 0)
    set(CHUNK_SIZE 16000)

    while(OFFSET LESS LENGTH)
        string(SUBSTRING "${SHADER_GLSL}" ${OFFSET} ${CHUNK_SIZE} CHUNK)

        # Output chunk
        file(APPEND "${ARGS_OUTPUT_FILE}" "R\"(${CHUNK})\"\n")

        # Increment offset
        math(EXPR OFFSET "${OFFSET} + ${CHUNK_SIZE}")
    endwhile()

    file(APPEND "${ARGS_OUTPUT_FILE}" ";\n")

    string(APPEND SHADER_MAP "    { \"${SHADER_NAME}\", ${SHADER_NAME}GLSL },\n")
endforeach()

string(APPEND SHADER_MAP "};\n")
file(APPEND "${ARGS_OUTPUT_FILE}" "${SHADER_MAP}")

# Write header file footer
file(APPEND "${ARGS_OUTPUT_FILE}" "\n}\n")
