#
# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

# Set warnings as errors and enable all warnings
set(VMEL_COMPILE_OPTIONS -Werror -Wall -Wextra -Wsign-conversion -Wpedantic -fno-strict-aliasing) # FIXME: readd -Wconversion

if(NOT "${VMEL_GCC_SANITIZERS}" STREQUAL "")
    message(STATUS "GCC Sanitizer enabled: ${VMEL_GCC_SANITIZERS}")
    add_compile_options(-fsanitize=${VMEL_GCC_SANITIZERS})
    add_compile_options(-fno-sanitize=alignment -fno-sanitize-recover=all)
    add_link_options(-fsanitize=${VMEL_GCC_SANITIZERS})
endif()
