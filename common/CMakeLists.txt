# SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

###############################################################################
# Common
###############################################################################

add_library(VkLayer_Common STATIC)

target_compile_options(VkLayer_Common PUBLIC ${VMEL_COMPILE_OPTIONS})

if(VMEL_USE_FLOAT_AS_DOUBLE)
    target_compile_definitions(VkLayer_Common PUBLIC USE_FLOAT_AS_DOUBLE)
    target_compile_definitions(VkLayer_Common PUBLIC real_t=float)
else()
    target_compile_definitions(VkLayer_Common PUBLIC real_t=double)
endif()

target_include_directories(VkLayer_Common PUBLIC
    include)

target_include_directories(VkLayer_Common SYSTEM PUBLIC
    ${SPIRV_TOOLS_PATH}
    ${spirv-tools_BINARY_DIR})

target_sources(VkLayer_Common PRIVATE
    log.cpp
    utils.cpp)

target_link_libraries(VkLayer_Common PUBLIC
    Vulkan::Headers
    SPIRV-Tools
    SPIRV-Tools-opt
    glslang::glslang
    glslang::glslang-default-resource-limits
    glslang::SPIRV)

set(COMMON_VERSION_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/version.hpp")
mlsdk_generate_version_header(
    TARGET VkLayer_Common
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/version.hpp.in"
    DESTINATION "${COMMON_VERSION_HEADER}"
    DEPENDENCIES
        ${VMEL_VERSION_DEPS}
)
