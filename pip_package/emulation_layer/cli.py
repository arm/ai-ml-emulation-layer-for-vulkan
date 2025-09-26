#
# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
import os
import sys


def get_linux_deploy_paths():
    base = os.path.join(os.path.dirname(__file__), "deploy")
    return {
        "LD_LIBRARY_PATH": os.path.join(base, "lib"),
        "VK_ADD_LAYER_PATH": os.path.join(base, "share", "vulkan", "explicit_layer.d"),
    }


def get_windows_deploy_paths():
    base = os.path.join(os.path.dirname(__file__), "deploy")
    return {
        "VK_LAYER_PATH": os.path.join(base, "bin"),
    }


def main():
    print(
        "To use the ML SDK for Vulkan Emulation Layer, set these environment variables:"
    )
    if sys.platform.startswith("linux"):
        paths = get_linux_deploy_paths()
        print(f"export LD_LIBRARY_PATH={paths['LD_LIBRARY_PATH']}:$LD_LIBRARY_PATH")
        print(f"export VK_ADD_LAYER_PATH={paths['VK_ADD_LAYER_PATH']}")
        print(
            "export VK_INSTANCE_LAYERS=VK_LAYER_ML_Graph_Emulation:VK_LAYER_ML_Tensor_Emulation"
        )
    elif sys.platform.startswith("win"):
        paths = get_windows_deploy_paths()
        print(f"$env:VK_LAYER_PATH={paths['VK_LAYER_PATH']}")
        print(
            '$env:VK_INSTANCE_LAYERS="VK_LAYER_ML_Graph_Emulation:VK_LAYER_ML_Tensor_Emulation"'
        )
    else:
        print("ERROR: Unsupported platform")
