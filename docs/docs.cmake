#
# SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

# Both the native project and docs/CMakeLists.txt use this implementation.
get_filename_component(MLEL_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(MLEL_DOCS_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(MLEL_DOCS_DIR "${MLEL_DOCS_BINARY_DIR}/docs")
list(APPEND CMAKE_MODULE_PATH "${MLEL_REPO_ROOT}/cmake")

if(MLEL_DOCS_STANDALONE)
    find_package(Doxygen 1.9.1 REQUIRED)
    find_package(Sphinx REQUIRED)
else()
    find_package(Doxygen)
    find_package(Sphinx)
    if(NOT DOXYGEN_FOUND OR NOT SPHINX_FOUND)
        message(STATUS "Doxygen or Sphinx is not available. Cannot generate documentation.")
        return()
    endif()
    if(CMAKE_CROSSCOMPILING)
        message(WARNING "Cannot build the documentation when cross-compiling. Skipping.")
        return()
    endif()
endif()

set(SPHINX_SRC_DIR_IN "${MLEL_REPO_ROOT}/docs/source")
set(SPHINX_SRC_DIR "${MLEL_DOCS_DIR}/in")
set(SPHINX_GEN_DIR "${MLEL_DOCS_DIR}/generated")
set(DOC_STAGED_FILES)

foreach(SRC_IN README.md CONTRIBUTING.md SECURITY.md LICENSES/Apache-2.0.txt)
    set(DOC_SOURCE_FILE "${SPHINX_GEN_DIR}/${SRC_IN}")
    configure_file("${MLEL_REPO_ROOT}/${SRC_IN}" "${DOC_SOURCE_FILE}" COPYONLY)
    list(APPEND DOC_STAGED_FILES "${DOC_SOURCE_FILE}")
endforeach()

file(GLOB_RECURSE DOC_SRC_FILES CONFIGURE_DEPENDS LIST_DIRECTORIES false
    RELATIVE "${SPHINX_SRC_DIR_IN}" "${SPHINX_SRC_DIR_IN}/*")
foreach(SRC_IN IN LISTS DOC_SRC_FILES)
    set(DOC_SOURCE_FILE "${SPHINX_SRC_DIR}/${SRC_IN}")
    configure_file("${SPHINX_SRC_DIR_IN}/${SRC_IN}" "${DOC_SOURCE_FILE}" COPYONLY)
    list(APPEND DOC_STAGED_FILES "${DOC_SOURCE_FILE}")
endforeach()

# Track the Doxyfile's top-level graph and tensor inputs (RECURSIVE=NO).
# Doxygen reads checked-in sources directly, keeping relative includes intact.
file(GLOB DOC_CODE_FILES CONFIGURE_DEPENDS LIST_DIRECTORIES false
    RELATIVE "${MLEL_REPO_ROOT}"
    "${MLEL_REPO_ROOT}/graph/*.cpp" "${MLEL_REPO_ROOT}/graph/*.hpp"
    "${MLEL_REPO_ROOT}/tensor/*.cpp" "${MLEL_REPO_ROOT}/tensor/*.hpp")
set(DOC_DOXYGEN_FILES)
foreach(SRC_IN IN LISTS DOC_CODE_FILES)
    list(APPEND DOC_DOXYGEN_FILES "${MLEL_REPO_ROOT}/${SRC_IN}")
endforeach()

# Doxygen preprocesses these includes from the top-level tensor sources.
foreach(SRC_IN tensor/shaders/tensor.glsl tensor/shaders/copy.comp)
    list(APPEND DOC_DOXYGEN_FILES "${MLEL_REPO_ROOT}/${SRC_IN}")
endforeach()

set(DOC_STAGED_MANIFEST "${MLEL_DOCS_DIR}/staged-files.txt")
if(EXISTS "${DOC_STAGED_MANIFEST}")
    file(STRINGS "${DOC_STAGED_MANIFEST}" DOC_PREVIOUS_FILES)
    foreach(DOC_PREVIOUS_FILE IN LISTS DOC_PREVIOUS_FILES)
        cmake_path(IS_PREFIX MLEL_DOCS_DIR "${DOC_PREVIOUS_FILE}" NORMALIZE DOC_IS_STAGED)
        if(DOC_IS_STAGED AND NOT DOC_PREVIOUS_FILE IN_LIST DOC_STAGED_FILES)
            file(REMOVE "${DOC_PREVIOUS_FILE}")
        endif()
    endforeach()
endif()
list(JOIN DOC_STAGED_FILES "\n" DOC_STAGED_CONTENT)
file(CONFIGURE OUTPUT "${DOC_STAGED_MANIFEST}" CONTENT "${DOC_STAGED_CONTENT}\n" @ONLY)
set(DOC_DOXYGEN_MANIFEST "${MLEL_DOCS_DIR}/doxygen-inputs.txt")
list(JOIN DOC_DOXYGEN_FILES "\n" DOC_DOXYGEN_CONTENT)
file(CONFIGURE OUTPUT "${DOC_DOXYGEN_MANIFEST}" CONTENT "${DOC_DOXYGEN_CONTENT}\n" @ONLY)

include("${MLEL_REPO_ROOT}/cmake/doxygen.cmake")
include("${MLEL_REPO_ROOT}/cmake/sphinx.cmake")

add_custom_target(mlel_doc ALL
    DEPENDS mlel_doxy_doc mlel_sphx_doc
    SOURCES "${SPHINX_SRC_DIR}/index.rst")
