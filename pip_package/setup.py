#
# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
from setuptools import setup
from wheel.bdist_wheel import bdist_wheel


class BDistWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        platformName = super().get_tag()[2]
        return ("py3", "none", platformName)


setup(cmdclass={"bdist_wheel": BDistWheel})
