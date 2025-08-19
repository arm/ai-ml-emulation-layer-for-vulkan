#
# SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
import os
import sys

sys.path.insert(0, os.path.abspath("."))

# Emulation Layer project config
EL_project = "ML Emulation Layer for Vulkan®"
copyright = "2023-2025, Arm Limited and/or its affiliates <open-source-office@arm.com>"
author = "Arm Limited"
git_repo_tool_url = "https://gerrit.googlesource.com/git-repo"

# Set home project name
project = EL_project

rst_epilog = """
.. |EL_project| replace:: %s
.. |git_repo_tool_url| replace:: %s
""" % (
    EL_project,
    git_repo_tool_url,
)

# Enabled extensions
extensions = [
    "breathe",
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "myst_parser",
    "sphinxcontrib.plantuml",
]

# Disable superfluous warnings
suppress_warnings = [
    "sphinx.ext.autosectionlabel.*",
    "myst.header",
]

# Breathe Configuration
breathe_projects = {"VMEL": "../generated/xml"}
breathe_default_project = "VMEL"
breathe_domain_by_extension = {"h": "c"}

# Enable RTD theme
html_theme = "sphinx_rtd_theme"

# Stand-alone builds need to include some base docs (Security and Contributor guide)
tags.add("WITH_BASE_MD")
