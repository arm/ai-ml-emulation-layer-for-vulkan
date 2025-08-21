#
# SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

set(CMAKE_CROSSCOMPILING OFF)

find_program(CLANG_PATH clang)
if(NOT CLANG_PATH)
    message(FATAL_ERROR "clang not found")
endif()

find_program(CLANGXX_PATH clang++)
if(NOT CLANGXX_PATH)
    message(FATAL_ERROR "clang++ not found")
endif()

# Set the compilers
set(CMAKE_C_COMPILER "${CLANG_PATH}" CACHE FILEPATH "C compiler")
set(CMAKE_CXX_COMPILER "${CLANGXX_PATH}" CACHE FILEPATH "C++ compiler")

set(CMAKE_CXX_FLAGS_INIT "-stdlib=libc++")

# Use lld if available
if(UNIX AND NOT APPLE)
    find_program(LLD lld)
    if(LLD)
        message(STATUS "Using lld linker with Clang")
        set(CMAKE_EXE_LINKER_FLAGS_INIT "-fuse-ld=lld")
    else()
        message(WARNING "lld not found, using system default linker")
    endif()
endif()

include(${CMAKE_CURRENT_LIST_DIR}/gnu_compiler_options.cmake)
