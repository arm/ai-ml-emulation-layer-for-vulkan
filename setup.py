#
# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
import os
import pathlib
import platform
import shutil
import sys

from setuptools import Distribution
from setuptools import setup
from setuptools.command.build import build as setuptools_build
from setuptools.command.build_py import build_py

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:
    from wheel.bdist_wheel import bdist_wheel


EMULATION_LAYER_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(EMULATION_LAYER_DIR))

SKIP_NATIVE_BUILD_ENV = "EMULATION_LAYER_SKIP_NATIVE_BUILD"


class Build(setuptools_build):
    def initialize_options(self):
        super().initialize_options()
        self.build_base = str(pathlib.Path("build") / "python")


class BuildPy(build_py):
    def run(self):
        super().run()

        manifest_dir = (
            pathlib.Path("bin")
            if platform.system() == "Windows"
            else pathlib.Path("share") / "vulkan" / "explicit_layer.d"
        )
        staged_manifest = (
            EMULATION_LAYER_DIR
            / "pip_package"
            / "emulation_layer"
            / "deploy"
            / manifest_dir
            / "VkLayer_Graph.json"
        )
        if os.environ.get(SKIP_NATIVE_BUILD_ENV) == "1" or staged_manifest.is_file():
            return

        dependency_dir = EMULATION_LAYER_DIR.parent.parent / "dependencies"
        if not dependency_dir.is_dir():
            raise RuntimeError(
                "The Emulation Layer native build requires an ML SDK checkout. "
                f"Missing: {dependency_dir}"
            )

        missing_tools = [tool for tool in ("cmake", "ninja") if not shutil.which(tool)]
        if missing_tools:
            raise RuntimeError(
                "The Emulation Layer native build requires: " + ", ".join(missing_tools)
            )

        from scripts.build import build as build_emulation_layer

        build_command = self.get_finalized_command("build")
        native_build_dir = pathlib.Path(build_command.build_temp) / "emulation_layer"
        native_install_dir = pathlib.Path(self.build_lib) / "emulation_layer" / "deploy"

        result = build_emulation_layer(
            [
                "--build-dir",
                str(native_build_dir),
                "--install",
                str(native_install_dir),
                "--package-version",
                self.distribution.get_version(),
                "--threads",
                os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", str(os.cpu_count() or 1)),
            ]
        )
        if result:
            raise RuntimeError(
                f"Emulation Layer native build failed with code {result}"
            )


class BDistWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        _, _, platform_tag = super().get_tag()
        return ("py3", "none", platform_tag)


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(
    distclass=BinaryDistribution,
    cmdclass={"build": Build, "build_py": BuildPy, "bdist_wheel": BDistWheel},
)
