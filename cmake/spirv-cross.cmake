#
# SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

set(SPIRV_CROSS_PATH "SPIRV_CROSS-NOTFOUND" CACHE PATH "Path to SPIRV-Cross")

if(EXISTS ${SPIRV_CROSS_PATH}/CMakeLists.txt)
    if(NOT TARGET spirv-cross-glsl)
        option(SPIRV_CROSS_ENABLE_TESTS "" OFF)
        add_subdirectory(${SPIRV_CROSS_PATH} spirv-cross SYSTEM EXCLUDE_FROM_ALL)
    endif()
else()
    find_package(spirv_cross_core REQUIRED CONFIG)
    find_package(spirv_cross_glsl REQUIRED CONFIG)
endif()
