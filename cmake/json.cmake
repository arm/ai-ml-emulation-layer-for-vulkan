#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

include(version)

set(JSON_PATH "JSON-NOTFOUND" CACHE PATH "Path to JSON")
set(nlohmann_json_VERSION "unknown")

if(EXISTS ${JSON_PATH}/CMakeLists.txt)
    if(NOT TARGET nlohmann_json)
        add_subdirectory(${JSON_PATH} json SYSTEM EXCLUDE_FROM_ALL)
    endif()

    mlsdk_get_git_revision(${JSON_PATH} nlohmann_json_VERSION)
else()
    find_package(nlohmann_json REQUIRED CONFIG)
endif()
