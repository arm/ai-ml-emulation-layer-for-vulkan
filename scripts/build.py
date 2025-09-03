#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
import argparse
import pathlib
import platform
import subprocess
import sys

try:
    import argcomplete
except:
    argcomplete = None

EMULATION_LAYER_DIR = pathlib.Path(__file__).resolve().parent / ".."
DEPENDENCY_DIR = EMULATION_LAYER_DIR / ".." / ".." / "dependencies"
CMAKE_TOOLCHAIN_PATH = EMULATION_LAYER_DIR / "cmake" / "toolchain"


class Builder:
    def __init__(self, args):
        self.build_dir = args.build_dir
        self.prefix_path = args.prefix_path
        self.test_dir = pathlib.Path(self.build_dir) / "tests"
        self.threads = args.threads
        self.run_tests = args.test
        self.build_type = args.build_type
        self.run_linting = args.lint
        self.enable_gcc_sanitizers = args.enable_gcc_sanitizers
        self.install = args.install
        self.package = args.package
        self.package_type = args.package_type
        self.target_platform = args.target_platform
        self.disable_precompile_shaders = args.disable_precompile_shaders
        self.doc = args.doc

        # Dependencies
        self.vulkan_headers_path = args.vulkan_headers_path
        self.spirv_headers_path = args.spirv_headers_path
        self.spirv_tools_path = args.spirv_tools_path
        self.spirv_cross_path = args.spirv_cross_path
        self.glslang_path = args.glslang_path
        self.gtest_path = args.gtest_path

    def setup_platform_build(self, cmake_cmd):
        if self.target_platform == "host":
            system = platform.system()
            if system == "Linux":
                cmake_cmd.append(
                    f"-DCMAKE_TOOLCHAIN_FILE={CMAKE_TOOLCHAIN_PATH / 'gcc.cmake'}"
                )
                return True
            if system == "Windows":
                cmake_cmd.append(
                    f"-DCMAKE_TOOLCHAIN_FILE={CMAKE_TOOLCHAIN_PATH / 'windows-msvc.cmake'}"
                )
                cmake_cmd.append("-DMSVC=ON")
                return True
            print(f"Unsupported host platform {system}", file=sys.stderr)
            return False

        if self.target_platform == "aarch64":
            cmake_cmd.append(
                f"-DCMAKE_TOOLCHAIN_FILE={CMAKE_TOOLCHAIN_PATH / 'linux-aarch64-gcc.cmake'}"
            )
            return True

        print(
            f"ERROR: Incorrect target platform option: {self.target_platform}",
            file=sys.stderr,
        )
        return False

    def run(self):
        cmake_setup_cmd = [
            "cmake",
            "-S",
            str(EMULATION_LAYER_DIR),
            "-B",
            self.build_dir,
        ]
        if self.build_type:
            cmake_setup_cmd.append(f"-DCMAKE_BUILD_TYPE={self.build_type}")

        if self.prefix_path:
            cmake_setup_cmd.append(f"-DCMAKE_PREFIX_PATH={self.prefix_path}")

        # Dependency injection
        if self.vulkan_headers_path:
            cmake_setup_cmd.append(f"-DVULKAN_HEADERS_PATH={self.vulkan_headers_path}")
        if self.spirv_headers_path:
            cmake_setup_cmd.append(f"-DSPIRV_HEADERS_PATH={self.spirv_headers_path}")
        if self.spirv_tools_path:
            cmake_setup_cmd.append(f"-DSPIRV_TOOLS_PATH={self.spirv_tools_path}")
        if self.spirv_cross_path:
            cmake_setup_cmd.append(f"-DSPIRV_CROSS_PATH={self.spirv_cross_path}")
        if self.glslang_path:
            cmake_setup_cmd.append(f"-DGLSLANG_PATH={self.glslang_path}")
        if self.gtest_path:
            cmake_setup_cmd.append(f"-DGTEST_PATH={self.gtest_path}")

        # Extra options
        if self.run_linting:
            cmake_setup_cmd.append("-DVMEL_ENABLE_LINT=ON")
        if self.disable_precompile_shaders:
            cmake_setup_cmd.append("-DVMEL_DISABLE_PRECOMPILE_SHADERS=ON")
        if self.doc:
            cmake_setup_cmd.append("-DVMEL_BUILD_DOCS=ON")

        if self.enable_gcc_sanitizers:
            cmake_setup_cmd.append("-DVMEL_GCC_SANITIZERS=undefined,address")
        if not self.setup_platform_build(cmake_setup_cmd):
            return 1

        cmake_build_cmd = [
            "cmake",
            "--build",
            self.build_dir,
            "-j",
            str(self.threads),
        ]

        if self.build_type:
            cmake_build_cmd.extend(["--config", self.build_type])

        try:
            subprocess.run(cmake_setup_cmd, check=True)
            subprocess.run(cmake_build_cmd, check=True)

            if self.install:
                cmake_install_cmd = [
                    "cmake",
                    "--install",
                    self.build_dir,
                    "--prefix",
                    self.install,
                ]
                if self.build_type:
                    cmake_install_cmd += ["--config", self.build_type]
                subprocess.run(cmake_install_cmd, check=True)

            if self.run_tests:
                test_cmd = [
                    "ctest",
                    "--test-dir",
                    str(self.test_dir),
                    "--output-on-failure",
                ]

                subprocess.run(test_cmd, check=True)

            if self.package:
                package_type = self.package_type or "tgz"
                cpack_generator = package_type.upper()

                cmake_package_cmd = [
                    "cpack",
                    "--config",
                    f"{self.build_dir}/CPackConfig.cmake",
                    "-C",
                    self.build_type,
                    "-G",
                    cpack_generator,
                    "-B",
                    self.package,
                    "-D",
                    "CPACK_INCLUDE_TOPLEVEL_DIRECTORY=OFF",
                ]
                subprocess.run(cmake_package_cmd, check=True)

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"EmulationLayerBuilder failed with {e}", file=sys.stderr)
            return 1

        return 0


def parse_arguments():
    parser = argparse.ArgumentParser(description="Build ML SDK Emulation Layer")
    parser.add_argument(
        "--build-dir",
        help="Name of folder where to build the ML SDK Emulation Layer. Default: build",
        default=f"{EMULATION_LAYER_DIR / 'build'}",
    )
    parser.add_argument(
        "--threads",
        "-j",
        type=int,
        help="Number of threads to use for building. Default: %(default)s",
        default=16,
    )
    parser.add_argument(
        "--prefix-path",
        help="Path to prefix directory.",
    )
    parser.add_argument(
        "-l",
        "--lint",
        help="Run linter. Default: %(default)s",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-t",
        "--test",
        help="Run unit tests after build. Default: %(default)s",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--build-type",
        help="Type of build to perform. Default: %(default)s",
        default=None,
    )
    parser.add_argument(
        "--target-platform",
        help="Specify the target build platform",
        choices=["host", "aarch64"],
        default="host",
    )
    parser.add_argument(
        "--doc",
        help="Build documentation. Default: %(default)s",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--install",
        help="Install build artifacts into a provided location",
    )
    parser.add_argument(
        "--package",
        help="Create a package with build artifacts and store it in a provided location",
    )
    parser.add_argument(
        "--package-type",
        choices=["zip", "tgz"],
        help="Package type",
    )
    parser.add_argument(
        "--enable-gcc-sanitizers",
        help="Enable GCC sanitizers. Default: %(default)s",
        action="store_true",
        default=False,
    )

    # Extras
    parser.add_argument(
        "--disable-precompile-shaders",
        help="Disable precompilation of SPIR-V shaders",
        action="store_true",
        default=False,
    )

    # Dependencies
    parser.add_argument(
        "--vulkan-headers-path", default=f"{DEPENDENCY_DIR / 'Vulkan-Headers'}"
    )
    parser.add_argument(
        "--spirv-headers-path", default=f"{DEPENDENCY_DIR / 'SPIRV-Headers'}"
    )
    parser.add_argument(
        "--spirv-tools-path", default=f"{DEPENDENCY_DIR / 'SPIRV-Tools'}"
    )
    parser.add_argument(
        "--spirv-cross-path", default=f"{DEPENDENCY_DIR / 'SPIRV-Cross'}"
    )
    parser.add_argument("--glslang-path", default=f"{DEPENDENCY_DIR / 'glslang'}")
    parser.add_argument("--gtest-path", default=f"{DEPENDENCY_DIR / 'googletest'}")

    if argcomplete:
        argcomplete.autocomplete(parser)

    return parser.parse_args()


def main():
    builder = Builder(parse_arguments())
    sys.exit(builder.run())


if __name__ == "__main__":
    main()
