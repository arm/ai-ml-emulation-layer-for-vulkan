#
# SPDX-FileCopyrightText: Copyright 2024,2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

find_package(Git)

function(mlsdk_get_git_revision SRCDIR RETURN_GIT_REVISION)
    set(${RETURN_GIT_REVISION} "unknown" PARENT_SCOPE)

    if(NOT Git_FOUND)
        message(WARNING "Git not found")
        return()
    endif()

    if(NOT IS_DIRECTORY ${SRCDIR})
        message(WARNING "Unable to get git revision, ${SRCDIR} is not a directory")
        return()
    endif()

    # Refresh cached stat metadata before describe so clean checkouts do not
    # get a false -dirty suffix. Real local changes still report as dirty.
    execute_process(
        COMMAND ${GIT_EXECUTABLE} update-index -q --refresh
        WORKING_DIRECTORY ${SRCDIR})

    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --dirty --always --tag --broken --long
        WORKING_DIRECTORY ${SRCDIR}
        RESULT_VARIABLE GIT_RETURN_CODE
        OUTPUT_VARIABLE GIT_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(GIT_RETURN_CODE)
        message(WARNING "Git command returns error for ${SRCDIR}")
        return()
    endif()

    if(NOT ${GIT_OUTPUT} MATCHES "^[-A-Za-z0-9/._]+$")
        message(FATAL_ERROR "Invalid revision ${GIT_OUTPUT} for ${SRCDIR}")
        return()
    endif()

    set(${RETURN_GIT_REVISION} ${GIT_OUTPUT} PARENT_SCOPE)
endfunction()

function(mlsdk_generate_version_header)
    cmake_parse_arguments(ARGS "" "TARGET;SOURCE;DESTINATION" "DEPENDENCIES" ${ARGN})

    if(NOT EXISTS "${ARGS_SOURCE}")
        message(FATAL_ERROR "Provided source file ${ARGS_SOURCE} does not exist")
    endif()

    set(GIT_REVISION "unknown")
    mlsdk_get_git_revision("${CMAKE_CURRENT_SOURCE_DIR}" GIT_REVISION)

    set(DEP_GIT_REVISIONS "")
    foreach(dep ${ARGS_DEPENDENCIES})
        if(NOT dep MATCHES "^[-A-Za-z0-9_]+$")
            message(FATAL_ERROR "Invalid dependency name: ${dep}")
        endif()

        set(depVersionVar "${dep}_VERSION")
        set(depVersion "unknown")

        if(DEFINED ${depVersionVar})
            set(depVersion "${${depVersionVar}}")
        else()
            message(WARNING "Unable to get version for ${dep}: ${depVersionVar} is not set")
        endif()

        if(NOT depVersion MATCHES "^[-A-Za-z0-9/._]+$")
            message(FATAL_ERROR "Invalid version ${depVersion} for dependency ${dep}")
        endif()

        list(APPEND DEP_GIT_REVISIONS "\"${dep}=${depVersion}\"")
    endforeach()

    string(JOIN ",\n    " DEP_GIT_REVISIONS ${DEP_GIT_REVISIONS})

    if(DEFINED ML_SDK_PACKAGE_VERSION AND NOT "${ML_SDK_PACKAGE_VERSION}" STREQUAL "")
        set(RUNTIME_VERSION "${ML_SDK_PACKAGE_VERSION}")
    else()
        set(RUNTIME_VERSION "unknown")
    endif()

    configure_file("${ARGS_SOURCE}" "${ARGS_DESTINATION}")
    get_filename_component(GEN_DIR_PATH "${ARGS_DESTINATION}" DIRECTORY)

    target_include_directories(${ARGS_TARGET} PUBLIC "${GEN_DIR_PATH}")
endfunction()
