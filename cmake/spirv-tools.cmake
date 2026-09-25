#
# SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

include(version)

set(SPIRV_TOOLS_PATH "SPIRV_TOOLS-NOTFOUND" CACHE PATH "Path to SPIR-V Tools")
set(SPIRV-Tools_VERSION "unknown")

if(EXISTS ${SPIRV_TOOLS_PATH}/CMakeLists.txt)
    if(NOT TARGET SPIRV-Tools)
        option(SPIRV_SKIP_TESTS "" ON)
        option(SPIRV_WERROR "" OFF)

        if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            add_subdirectory(${SPIRV_TOOLS_PATH} spirv-tools EXCLUDE_FROM_ALL)
        else()
            add_subdirectory(${SPIRV_TOOLS_PATH} spirv-tools SYSTEM EXCLUDE_FROM_ALL)
        endif()
    endif()

    # AppleClang searches /usr/local/include before command-line system include
    # paths. Use normal include ordering for the source checkout while retaining
    # system-header diagnostics for SPIR-V Tools headers.
    if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        foreach(SPIRV_TOOLS_INTERNAL_TARGET IN ITEMS
                SPIRV-Tools SPIRV-Tools-static SPIRV-Tools-shared SPIRV-Tools-opt)
            if(TARGET "${SPIRV_TOOLS_INTERNAL_TARGET}")
                get_target_property(SPIRV_TOOLS_ALIASED_TARGET
                    "${SPIRV_TOOLS_INTERNAL_TARGET}" ALIASED_TARGET)
                if(NOT SPIRV_TOOLS_ALIASED_TARGET)
                    target_compile_options("${SPIRV_TOOLS_INTERNAL_TARGET}" INTERFACE
                        "$<$<COMPILE_LANGUAGE:CXX>:--system-header-prefix=spirv-tools/>")
                endif()
            endif()
        endforeach()
    endif()

    mlsdk_get_git_revision(${SPIRV_TOOLS_PATH} SPIRV-Tools_VERSION)
else()
    find_package(SPIRV-Tools REQUIRED CONFIG)
    set(SPIRV-Tools_VERSION "unknown")
endif()
