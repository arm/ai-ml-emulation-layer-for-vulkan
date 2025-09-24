# SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

if(DEFINED ANDROID_ABI)
    message(WARNING "Building for Android is currently an experimental feature.")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    add_compile_definitions(MOLTEN_VK_SUPPORT)
endif()

include(CTest)
configure_file(CTestCustom.cmake CTestCustom.cmake)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

###############################################################################
# Options for ML Emulation Layer for Vulkan
###############################################################################
if(ML_SDK_GENERATE_CPACK)
    set(ML_SDK_EL_PACKAGE_NAME "ml-sdk")
    set(ML_SDK_EL_NAMESPACE "ml-sdk")
else()
    set(ML_SDK_EL_PACKAGE_NAME ${PROJECT_NAME})
    set(ML_SDK_EL_NAMESPACE ${PROJECT_NAME})
    set(CPACK_PACKAGE_NAME "ml-emulation-layer")
endif()

option(VMEL_TESTS_ENABLE "Enable tests" ON)
option(VMEL_ENABLE_LINT "Enable linting using cppcheck" OFF)
option(VMEL_DISABLE_PRECOMPILE_SHADERS "Disable precompilation of SPIR-V shaders" OFF)
option(VMEL_BUILD_DOCS "Build documentation" OFF)

###############################################################################
# Vulkan and SPIR-V dependencies
###############################################################################

include(gtest)
include(vulkan-headers)
include(spirv-headers)
include(spirv-cross)
include(spirv-tools)
include(glslang)

if(NOT EXISTS ${SPIRV_TOOLS_PATH}/CMakeLists.txt)
    message(FATAL_ERROR "ML Emulation Layer for Vulkan only supports building SPIR-V Tools from source")
endif()

# Enable Cppcheck
if(VMEL_ENABLE_LINT)
    message(STATUS "Cppcheck is enabled.")
    find_program(CPPCHECK_EXECUTABLE NAMES cppcheck)
    include(ProcessorCount)
    ProcessorCount(NCORES)
    set(CMAKE_CXX_CPPCHECK_OPTIONS
        -j${NCORES}
        --std=c++17
        --force
        --enable=performance,portability,information # reenable style
        --quiet
        --inline-suppr
        --error-exitcode=1
        --suppress=unreadVariable
        --suppress=unmatchedSuppression
        --suppress=noValidConfiguration
        --suppress=*:${CMAKE_BINARY_DIR}*
        )

        if(EXISTS ${VULKAN_HEADERS_PATH})
            list(APPEND CMAKE_CXX_CPPCHECK_OPTIONS --suppress=*:*${VULKAN_HEADERS_PATH}*)
        endif()

        if(EXISTS ${SPIRV_CROSS_PATH})
            list(APPEND CMAKE_CXX_CPPCHECK_OPTIONS --suppress=*:*${SPIRV_CROSS_PATH}*)
        endif()

        if(EXISTS ${SPIRV_TOOLS_PATH})
            list(APPEND CMAKE_CXX_CPPCHECK_OPTIONS --suppress=*:*${SPIRV_TOOLS_PATH}*)
        endif()

        if(EXISTS ${SPIRV_HEADERS_PATH})
            list(APPEND CMAKE_CXX_CPPCHECK_OPTIONS --suppress=*:*${SPIRV_HEADERS_PATH}*)
        endif()

        if(EXISTS ${GLSLANG_PATH})
            list(APPEND CMAKE_CXX_CPPCHECK_OPTIONS --suppress=*:*${GLSLANG_PATH}*)
        endif()

    set(CMAKE_CXX_CPPCHECK "${CPPCHECK_EXECUTABLE}" "${CMAKE_CXX_CPPCHECK_OPTIONS}")
endif()
unset(VMEL_ENABLE_LINT CACHE)

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

if(VMEL_TESTS_ENABLE AND TARGET GTest::gtest_main)
    add_subdirectory(tests)
endif()

###############################################################################
# Packaging
###############################################################################

include(package)
if(NOT ML_SDK_GENERATE_CPACK)
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
