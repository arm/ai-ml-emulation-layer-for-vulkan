# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

# Compile list of arguments, removing the first three
math(EXPR COUNT "${CMAKE_ARGC} - 1")
foreach(i RANGE 3 ${COUNT})
    list(APPEND ARGV "${CMAKE_ARGV${i}}")
endforeach()

# Parse command line arguments
cmake_parse_arguments(ARGS "" "OUTPUT_FILE;INPUT_FILE" "" ${ARGV})

get_filename_component(NAME "${ARGS_INPUT_FILE}" NAME_WE)
file(READ "${ARGS_INPUT_FILE}" INPUT HEX)
string(LENGTH "${INPUT}" LENGTH)
math(EXPR REMAINDER "${LENGTH} % 8")
if(LENGTH EQUAL 0 OR NOT REMAINDER EQUAL 0)
    message(FATAL_ERROR "SPIR-V must contain whole 32-bit words: ${ARGS_INPUT_FILE}")
endif()

# Convert little-endian SPIR-V bytes into host-independent uint32_t literals.
string(REGEX REPLACE "(..)(..)(..)(..)" "0x\\4\\3\\2\\1,\n" HEX "${INPUT}")
file(CONFIGURE OUTPUT "${ARGS_OUTPUT_FILE}" CONTENT "#include <cstdint>

namespace mlsdk::el::compute::precompiled {
extern const uint32_t ${NAME}[] = {
${HEX}};
} // namespace mlsdk::el::compute::precompiled
" @ONLY)
