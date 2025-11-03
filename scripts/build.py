#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
import argparse
import os
import pathlib
import platform
import shutil
import subprocess
import sys
from datetime import datetime

try:
    import argcomplete
except:
    argcomplete = None

EMULATION_LAYER_DIR = pathlib.Path(__file__).parent / ".."
EMULATION_LAYER_DIR = EMULATION_LAYER_DIR.resolve()
DEPENDENCY_DIR = EMULATION_LAYER_DIR / ".." / ".." / "dependencies"
DEPENDENCY_DIR = DEPENDENCY_DIR.resolve()
CMAKE_TOOLCHAIN_PATH = EMULATION_LAYER_DIR / "cmake" / "toolchain"


class Builder:
    def __init__(self, args):
        self.build_dir = str(pathlib.Path(args.build_dir).resolve())
        self.prefix_path = args.prefix_path
        self.test_dir = pathlib.Path(self.build_dir) / "tests"
        self.threads = args.threads
        self.run_tests = args.test
        self.build_type = args.build_type
        self.lint = args.lint
        self.enable_sanitizers = args.enable_sanitizers
        self.install = args.install
        self.target_platform = args.target_platform
        self.cmake_toolchain_for_android = args.cmake_toolchain_for_android
        self.disable_precompile_shaders = args.disable_precompile_shaders
        self.doc = args.doc
        self.clang_tidy_fix = args.clang_tidy_fix

        self.package_dir = args.package_dir or self.build_dir
        self.package_version = args.package_version
        self.package_tgz = "tgz" in args.package_type
        self.package_zip = "zip" in args.package_type
        self.package_pip = "pip" in args.package_type
        self.package_source_tgz = "source-tgz" in args.package_type
        self.package_source_zip = "source-zip" in args.package_type

        # Dependencies
        self.vulkan_headers_path = args.vulkan_headers_path
        self.spirv_headers_path = args.spirv_headers_path
        self.spirv_tools_path = args.spirv_tools_path
        self.spirv_cross_path = args.spirv_cross_path
        self.glslang_path = args.glslang_path
        self.gtest_path = args.gtest_path

        if not self.install and self.package_pip:
            self.install = "pip_install"

    def setup_platform_build(self, cmake_cmd):
        system = platform.system()
        if self.target_platform == "host":
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
            # Support for Darwin is experimental
            if system == "Darwin":
                cmake_cmd.append(
                    f"-DCMAKE_TOOLCHAIN_FILE={CMAKE_TOOLCHAIN_PATH / 'clang.cmake'}"
                )
                return True
            print(f"Unsupported host platform {system}", file=sys.stderr)
            return False
        if self.target_platform == "linux-clang":
            if system != "Linux":
                print(
                    f"ERROR: target {self.target_platform} only supported on Linux. Host platform {system}",
                    file=sys.stderr,
                )
                return False
            cmake_cmd.append(
                f"-DCMAKE_TOOLCHAIN_FILE={CMAKE_TOOLCHAIN_PATH / 'clang.cmake'}"
            )
            return True

        if self.target_platform == "aarch64":
            cmake_cmd.append(
                f"-DCMAKE_TOOLCHAIN_FILE={CMAKE_TOOLCHAIN_PATH / 'linux-aarch64-gcc.cmake'}"
            )
            return True

        if self.target_platform == "android":
            print(
                "WARNING: Cross-compiling Emulation Layer for Android is currently an experimental feature."
            )
            if not self.cmake_toolchain_for_android:
                print(
                    "No toolchain path specified for Android cross-compilation",
                    file=sys.stderr,
                )
                return False

            cmake_cmd.append(
                f"-DCMAKE_TOOLCHAIN_FILE={self.cmake_toolchain_for_android}"
            )
            cmake_cmd.append("-DCMAKE_FIND_ROOT_PATH=/")
            cmake_cmd.append("-DANDROID_ABI=arm64-v8a")
            cmake_cmd.append("-DANDROID_PLATFORM=android-21")
            cmake_cmd.append("-DANDROID_ALLOW_UNDEFINED_SYMBOLS=ON")
            cmake_cmd.append("-DANDROID_PIE=ON")
            return True

        print(
            f"ERROR: Incorrect target platform option: {self.target_platform}",
            file=sys.stderr,
        )
        return False

    def generate_cmake_package(self, generator, sourcePackage=False):
        config_file = (
            "CPackSourceConfig.cmake" if sourcePackage else "CPackConfig.cmake"
        )

        cmake_package_cmd = [
            "cpack",
            "--config",
            f"{self.build_dir}/{config_file}",
            "-C",
            self.build_type,
            "-G",
            generator,
            "-B",
            self.package_dir,
            "-D",
            "CPACK_INCLUDE_TOPLEVEL_DIRECTORY=OFF",
        ]
        subprocess.run(cmake_package_cmd, check=True)

    def run(self):
        cmake_setup_cmd = [
            "cmake",
            "-S",
            str(EMULATION_LAYER_DIR),
            "-B",
            self.build_dir,
            f"-DCMAKE_BUILD_TYPE={self.build_type}",
        ]

        if self.prefix_path:
            cmake_setup_cmd.append(f"-DCMAKE_PREFIX_PATH={self.prefix_path}")

        # Dependency injection
        cmake_setup_cmd.append(f"-DVULKAN_HEADERS_PATH={self.vulkan_headers_path}")
        cmake_setup_cmd.append(f"-DSPIRV_HEADERS_PATH={self.spirv_headers_path}")
        cmake_setup_cmd.append(f"-DSPIRV_TOOLS_PATH={self.spirv_tools_path}")
        cmake_setup_cmd.append(f"-DSPIRV_CROSS_PATH={self.spirv_cross_path}")
        cmake_setup_cmd.append(f"-DGLSLANG_PATH={self.glslang_path}")
        cmake_setup_cmd.append(f"-DGTEST_PATH={self.gtest_path}")

        # Extra options
        if self.run_tests:
            cmake_setup_cmd.append("-DVMEL_TESTS_ENABLE=ON")
        if self.lint:
            cmake_setup_cmd.append("-DCMAKE_EXPORT_COMPILE_COMMANDS=ON")
        if self.disable_precompile_shaders:
            cmake_setup_cmd.append("-DVMEL_DISABLE_PRECOMPILE_SHADERS=ON")
        if self.doc:
            cmake_setup_cmd.append("-DVMEL_BUILD_DOCS=ON")

        if self.enable_sanitizers:
            if self.target_platform != "host":
                print(
                    f"ERROR: sanitizer not supported for target platform: {self.target_platform}"
                )
                return 1

            system = platform.system()
            if system == "Linux":
                gcc_sanitizer_flags = [
                    "-g",
                    "-fsanitize=address",
                    "-fno-sanitize=vptr",
                    "-fno-sanitize=alignment",
                    "-fno-sanitize-recover=all",
                ]
                cmake_setup_cmd.append(
                    f"-DCMAKE_CXX_FLAGS={' '.join(gcc_sanitizer_flags)}"
                )
                cmake_setup_cmd.append(
                    "-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=undefined,address"
                )
            elif system == "Windows":
                cmake_setup_cmd.append("-DCMAKE_CXX_FLAGS=/fsanitize=address /Zi /MDd")
                cmake_setup_cmd.append(
                    "-DCMAKE_EXE_LINKER_FLAGS=/INFERASANLIBS /DEBUG /INCREMENTAL:NO"
                )
            else:
                print(f"ERROR: sanitizer is not supported on system: {system}")

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

            if self.clang_tidy_fix and not self.lint:
                print(
                    "WARNING: --clang-tidy-fix requires --lint to be enabled, argument ignored."
                )

            if self.lint:
                source_dirs = [
                    f"{EMULATION_LAYER_DIR / 'common'}",
                    f"{EMULATION_LAYER_DIR / 'graph'}",
                    f"{EMULATION_LAYER_DIR / 'tensor'}",
                    f"{EMULATION_LAYER_DIR / 'tests'}",
                    f"{EMULATION_LAYER_DIR / 'utilities'}",
                ]

                lint_cmd = [
                    "cppcheck",
                    f"-j{str(self.threads)}",
                    "--std=c++17",
                    "--error-exitcode=1",
                    "--inline-suppr",
                    f"--cppcheck-build-dir={self.build_dir}/cppcheck",
                    "--enable=information,performance,portability,style",
                    "--suppress=noValidConfiguration",
                    "--suppress=unassignedVariable",
                    "--suppress=unmatchedSuppression",
                    "--suppress=useStlAlgorithm",
                    "--suppress=*:MachineIndependent*",
                    f"--suppress=*:{DEPENDENCY_DIR}*",
                ] + source_dirs

                subprocess.run(lint_cmd, check=True)

                clang_tidy_cmd = [
                    "run-clang-tidy",
                    f"-j{self.threads}",
                    f"-p{self.build_dir}",
                ] + source_dirs

                if self.clang_tidy_fix:
                    clang_tidy_cmd.append("-fix")

                subprocess.run(clang_tidy_cmd, check=True)

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

            if self.package_tgz:
                self.generate_cmake_package("TGZ")

            if self.package_zip:
                self.generate_cmake_package("ZIP")

            if self.package_source_tgz:
                self.generate_cmake_package("TGZ", True)

            if self.package_source_zip:
                self.generate_cmake_package("ZIP", True)

            if self.package_pip:
                os.makedirs("pip_package/emulation_layer/deploy/", exist_ok=True)
                shutil.copytree(
                    self.install,
                    "pip_package/emulation_layer/deploy/",
                    dirs_exist_ok=True,
                )

                os.environ[
                    "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_AI_ML_EMULATION_LAYER_FOR_VULKAN"
                ] = self.package_version
                result = subprocess.Popen(
                    [sys.executable, "-m", "build"],
                    env=os.environ,
                    cwd="pip_package",
                )
                result.communicate()
                if result.returncode != 0:
                    print("ERROR: Failed to generate pip package")
                    return 1

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
        default="Release",
    )
    parser.add_argument(
        "--target-platform",
        help="Specify the target build platform",
        choices=["host", "android", "aarch64", "linux-clang"],
        default="host",
    )
    parser.add_argument(
        "--cmake-toolchain-for-android",
        help="Path to the cmake compiler toolchain. Default: %(default)s",
        default="",
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
        "--package-dir",
        help="Specify location for packages to be created. Default path is the build directory",
        default="",
    )
    parser.add_argument(
        "--package-type",
        choices=["zip", "tgz", "pip", "source-zip", "source-tgz"],
        action="append",
        help="Create a package of a certain type",
        default=[],
    )
    parser.add_argument(
        "--package-version",
        help="Manually specify pip package version number",
        default=datetime.today().strftime("%m.%d"),
    )
    parser.add_argument(
        "--enable-sanitizers",
        help="Enable sanitizers. Default: %(default)s",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--clang-tidy-fix",
        help="Enable clang-tidy fix (requires --lint). Default: %(default)s",
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
