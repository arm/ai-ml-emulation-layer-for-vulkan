# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

# Compile list of arguments, removing the first three
math(EXPR COUNT "${CMAKE_ARGC} - 1")
foreach(i RANGE 3 ${COUNT})
    list(APPEND ARGV "${CMAKE_ARGV${i}}")
endforeach()

# Parse command line arguments
cmake_parse_arguments(ARGS "" "OUTPUT_FILE" "INPUT_FILES" ${ARGV})

set(DECLARATIONS "")
set(ENTRIES "")
foreach(INPUT_FILE IN LISTS ARGS_INPUT_FILES)
    get_filename_component(NAME "${INPUT_FILE}" NAME_WE)
    file(SIZE "${INPUT_FILE}" SIZE)
    math(EXPR WORD_COUNT "${SIZE} / 4")
    string(APPEND DECLARATIONS "extern const uint32_t ${NAME}[];\n")
    string(APPEND ENTRIES "    {\"${NAME}\", {precompiled::${NAME}, ${WORD_COUNT}}},\n")
endforeach()

# Preserve the timestamp when only shader contents, rather than names or sizes, change.
file(CONFIGURE OUTPUT "${ARGS_OUTPUT_FILE}" CONTENT "#include \"precompiled_shaders.hpp\"

namespace mlsdk::el::compute {
namespace precompiled {
${DECLARATIONS}} // namespace precompiled

const std::map<std::string_view, std::tuple<const uint32_t *const, const std::size_t>> precompiledSpirvModules = {
${ENTRIES}};
} // namespace mlsdk::el::compute
" @ONLY)
