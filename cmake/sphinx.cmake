#
# SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

set(SPHINX_BLD_DIR "${MLEL_DOCS_DIR}/out")
set(SPHINX_INDEX_HTML "${SPHINX_BLD_DIR}/index.html")
set(SPHINX_STAMP "${MLEL_DOCS_DIR}/sphinx.stamp")

add_custom_command(
    OUTPUT "${SPHINX_STAMP}" "${SPHINX_INDEX_HTML}"
    DEPENDS ${DOC_STAGED_FILES} "${DOC_STAGED_MANIFEST}"
    COMMAND "${CMAKE_COMMAND}" -E rm -rf "${SPHINX_BLD_DIR}"
    COMMAND "${SPHINX_EXECUTABLE}" -b html -W "${SPHINX_SRC_DIR}" "${SPHINX_BLD_DIR}"
    COMMAND "${CMAKE_COMMAND}" -E touch "${SPHINX_STAMP}"
    WORKING_DIRECTORY "${MLEL_DOCS_BINARY_DIR}"
    COMMENT "Generating API documentation with Sphinx"
    VERBATIM)

add_custom_target(mlel_sphx_doc
    DEPENDS "${SPHINX_STAMP}" "${SPHINX_INDEX_HTML}")
