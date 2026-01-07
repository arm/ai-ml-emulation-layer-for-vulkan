# SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.25)

project(MLEmulationLayer
    DESCRIPTION "ML Emulation Layer for Vulkan"
    LANGUAGES C CXX)

# Select C/C++ version
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 99)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

if(DEFINED ANDROID_ABI)
    message(WARNING "Building for Android is currently an experimental feature.")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    add_compile_definitions(EXPERIMENTAL_MOLTEN_VK_SUPPORT)
endif()

include(CTest)
configure_file(CTestCustom.cmake CTestCustom.cmake)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

###############################################################################
# Options for ML Emulation Layer for Vulkan
###############################################################################
if(NOT (CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR) AND ML_SDK_GENERATE_CPACK)
    set(ML_SDK_EL_PACKAGE_NAME "ai_ml_sdk_for_vulkan")
    set(ML_SDK_EL_NAMESPACE "ai_ml_sdk_for_vulkan")
else()
    set(ML_SDK_EL_PACKAGE_NAME ${PROJECT_NAME})
    set(ML_SDK_EL_NAMESPACE ${PROJECT_NAME})
    set(CPACK_PACKAGE_NAME "ai_ml_emulation_layer_for_vulkan")
endif()

option(VMEL_TESTS_ENABLE "Enable tests" OFF)
option(VMEL_DISABLE_PRECOMPILE_SHADERS "Disable precompilation of SPIR-V shaders" OFF)
option(VMEL_USE_FLOAT_AS_DOUBLE "Use float as double precision type" OFF)
option(VMEL_BUILD_DOCS "Build documentation" OFF)

###############################################################################
# Vulkan and SPIR-V dependencies
###############################################################################

include(vulkan-headers)
include(spirv-headers)
include(spirv-cross)
include(spirv-tools)
include(glslang)

if(NOT EXISTS ${SPIRV_TOOLS_PATH}/CMakeLists.txt)
    message(FATAL_ERROR "ML Emulation Layer for Vulkan only supports building SPIR-V Tools from source")
endif()

include(setLayerEntrypoints)

###############################################################################
# Precision configuration
###############################################################################

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin" OR DEFINED ANDROID_ABI)
    set(VMEL_USE_FLOAT_AS_DOUBLE ON CACHE BOOL "Use float as double precision type" FORCE)
endif()

###############################################################################
# ML Emulation Layer for Vulkan
###############################################################################

add_subdirectory(common)
add_subdirectory(graph)
add_subdirectory(tensor)
add_subdirectory(utilities)

###############################################################################
# Tests
###############################################################################

if(VMEL_TESTS_ENABLE)
    include(gtest)

    add_subdirectory(tests)
endif()

###############################################################################
# Packaging
###############################################################################
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    include(cmake/package.cmake)
    mlsdk_package(PACKAGE_NAME ${ML_SDK_EL_PACKAGE_NAME}
        NAMESPACE ${ML_SDK_EL_NAMESPACE}
        LICENSES "${CMAKE_CURRENT_LIST_DIR}/LICENSES/Apache-2.0.txt")
endif()

###############################################################################
# Documentation
###############################################################################
if(ML_SDK_BUILD_DOCS OR VMEL_BUILD_DOCS)
    include(docs/docs.cmake)
endif()

unset(VMEL_BUILD_DOCS CACHE)
unset(ANDROID_ABI CACHE)
