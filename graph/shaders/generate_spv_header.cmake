# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

# Compile list of arguments, removing the first three
math(EXPR COUNT "${CMAKE_ARGC} - 1")
foreach(i RANGE 3 ${COUNT})
    list(APPEND ARGV "${CMAKE_ARGV${i}}")
endforeach()

# Parse command line arguments
cmake_parse_arguments(ARGS "" "OUTPUT_FILE" "INPUT_FILES" ${ARGV})

# Write header file header
file(WRITE "${ARGS_OUTPUT_FILE}" "#pragma once
#include <map>
#include <string>
#include <tuple>

namespace {
")

# Generate a C array for each SPIR-V module
foreach(INPUT_FILE ${ARGS_INPUT_FILES})
    get_filename_component(NAME ${INPUT_FILE} NAME_WE)

    # Read shader file
    file(READ "${INPUT_FILE}" INPUT HEX)
    string(LENGTH "${INPUT}" LENGTH)
    math(EXPR LENGTH "${LENGTH} - 1")
    set(HEX "")

    foreach(i RANGE 0 ${LENGTH} 8)
        string(SUBSTRING "${INPUT}" ${i} 8 VALUE)
        string(SUBSTRING "${VALUE}" 0 2 BYTE0)
        string(SUBSTRING "${VALUE}" 2 2 BYTE1)
        string(SUBSTRING "${VALUE}" 4 2 BYTE2)
        string(SUBSTRING "${VALUE}" 6 2 BYTE3)

        string(APPEND HEX "0x${BYTE3}${BYTE2}${BYTE1}${BYTE0},\n")
    endforeach()

    file(APPEND "${ARGS_OUTPUT_FILE}" "static constexpr uint32_t ${NAME}[] = {\n${HEX}};\n")
endforeach()

# Generate lookup table
file(APPEND "${ARGS_OUTPUT_FILE}" "const std::map<std::string, std::tuple<const uint32_t * const, const std::size_t>> precompiledSpirvModules = {\n")

foreach(INPUT_FILE ${ARGS_INPUT_FILES})
    get_filename_component(NAME ${INPUT_FILE} NAME_WE)

    file(APPEND "${ARGS_OUTPUT_FILE}" "{ \"${NAME}\", { ${NAME}, sizeof(${NAME}) / sizeof(${NAME}[0])} }, \n")
endforeach()

# Write header file footer
file(APPEND "${ARGS_OUTPUT_FILE}" "}; // precompiledSpirvModules[]
} // namespace
")
