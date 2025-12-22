#
# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
import platform

from setuptools import setup
from wheel.bdist_wheel import bdist_wheel


class BDistWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        system = platform.system()
        machine = platform.machine()
        if system == "Windows":
            assert machine == "x86_64"
            platformName = "win_amd64"
        elif system == "Linux":
            if machine == "aarch64":
                platformName = "manyLinux2014_aarch64"
            else:
                assert machine == "x86_64"
                platformName = "manyLinux2014_x86_64"
        elif system == "Darwin":
            assert machine == "arm64"
            platformName = "macosx_10_13_arm64"
        return ("py3", "none", platformName)


setup(cmdclass={"bdist_wheel": BDistWheel})
