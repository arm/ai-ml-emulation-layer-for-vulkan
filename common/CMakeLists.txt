# SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

###############################################################################
# Logger
###############################################################################

add_library(VkLayer_Log INTERFACE)

target_sources(VkLayer_Log INTERFACE
    log.cpp)

target_include_directories(VkLayer_Log INTERFACE
    include)

###############################################################################
# Common
###############################################################################

add_library(VkLayer_Common INTERFACE)

target_compile_options(VkLayer_Common INTERFACE ${VMEL_COMPILE_OPTIONS})

if(VMEL_USE_FLOAT_AS_DOUBLE)
    target_compile_definitions(VkLayer_Common INTERFACE USE_FLOAT_AS_DOUBLE)
    target_compile_definitions(VkLayer_Common INTERFACE real_t=float)
else()
    target_compile_definitions(VkLayer_Common INTERFACE real_t=double)
endif()

target_include_directories(VkLayer_Common INTERFACE
    include)

target_include_directories(VkLayer_Common SYSTEM INTERFACE
    ${SPIRV_TOOLS_PATH}
    ${spirv-tools_BINARY_DIR})

target_sources(VkLayer_Common INTERFACE
    utils.cpp)

target_link_libraries(VkLayer_Common INTERFACE
    VkLayer_Log
    Vulkan::Headers
    SPIRV-Tools
    SPIRV-Tools-opt
    glslang::glslang
    glslang::glslang-default-resource-limits
    glslang::SPIRV)
