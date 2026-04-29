#
# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

# Use /Z7-style embedded debug info for MSVC debug-symbol configurations.
# This avoids compiler PDB contention when parallel launchers such as sccache
# are used during Windows builds.
set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT
    "$<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:Embedded>")

# Set warnings as errors and enable all warnings
# Disable warning C4324 for padding
# Disable warning C4996 for potentially unsafe functions, for example 'getenv'
set(VMEL_COMPILE_OPTIONS /EHa /WX /W4 /wd4324 /wd4996)
