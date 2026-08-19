#
# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
import os
import pathlib
import platform
import shutil
import sys

from setuptools import setup
from setuptools.command.build import build as setuptools_build
from setuptools.command.build_py import build_py

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:
    from wheel.bdist_wheel import bdist_wheel


EMULATION_LAYER_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(EMULATION_LAYER_DIR))

from scripts.build import build as build_emulation_layer  # noqa: E402

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

        build_command = self.get_finalized_command("build")
        native_build_dir = pathlib.Path(build_command.build_temp) / "emulation_layer"
        native_install_dir = pathlib.Path(self.build_lib) / "emulation_layer" / "deploy"

        result = build_emulation_layer(
            [
                "--build-dir",
                str(native_build_dir),
                "--install",
                str(native_install_dir),
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
        system = platform.system()
        machine = platform.machine()
        if system == "Windows":
            assert machine == "AMD64"
            platformName = "win_amd64"
        elif system == "Linux":
            if machine == "aarch64":
                platformName = "manylinux2014_aarch64"
            else:
                assert machine == "x86_64"
                platformName = "manylinux2014_x86_64"
        elif system == "Darwin":
            assert machine == "arm64"
            platformName = "macosx_11_0_arm64"
        return ("py3", "none", platformName)


setup(cmdclass={"build": Build, "build_py": BuildPy, "bdist_wheel": BDistWheel})
