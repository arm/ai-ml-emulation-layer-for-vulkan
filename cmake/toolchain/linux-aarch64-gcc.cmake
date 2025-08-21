#
# SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_CROSSCOMPILING TRUE)

find_program(AARCH64_GCC aarch64-linux-gnu-gcc)
if(NOT AARCH64_GCC)
    message(FATAL_ERROR "aarch64-linux-gnu-gcc not found — make sure the cross C compiler is installed")
endif()

find_program(AARCH64_GPP aarch64-linux-gnu-g++)
if(NOT AARCH64_GPP)
    message(FATAL_ERROR "aarch64-linux-gnu-g++ not found — make sure the cross C++ compiler is installed")
endif()

find_program(AARCH64_LD aarch64-linux-gnu-ld)
if(NOT AARCH64_LD)
    message(FATAL_ERROR "aarch64-linux-gnu-ld not found — make sure the cross linker is installed")
endif()

set(CMAKE_C_COMPILER "${AARCH64_GCC}" CACHE FILEPATH "C compiler")
set(CMAKE_CXX_COMPILER "${AARCH64_GPP}" CACHE FILEPATH "C++ compiler")
set(CMAKE_LINKER "${AARCH64_LD}" CACHE FILEPATH "Linker")

set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)   # Programs are found on the host
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)    # Libraries only in the target sysroot
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)    # Includes only in the target sysroot
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)    # Package files (e.g., pkg-config) in target sysroot


include(${CMAKE_CURRENT_LIST_DIR}/gnu_compiler_options.cmake)
