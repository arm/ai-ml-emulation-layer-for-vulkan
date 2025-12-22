#
# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
import os
import sys


def get_deploy_paths():
    base = os.path.join(os.path.dirname(__file__), "deploy")
    return {
        "DYLD_LIBRARY_PATH": os.path.join(base, "lib"),  # Darwin
        "LD_LIBRARY_PATH": os.path.join(base, "lib"),  # Linux
        "VK_LAYER_PATH_windows": os.path.join(base, "bin"),  # Windows
        "VK_LAYER_PATH": os.path.join(
            base, "share", "vulkan", "explicit_layer.d"
        ),  # Linux, Darwin
    }


def main():
    print(
        "To use the ML SDK for Vulkan Emulation Layer, set these environment variables:"
    )
    paths = get_deploy_paths()
    if sys.platform.startswith("linux"):
        print(f"export LD_LIBRARY_PATH={paths['LD_LIBRARY_PATH']}:$LD_LIBRARY_PATH")
        print(f"export VK_LAYER_PATH={paths['VK_ADD_LAYER_PATH']}:$VK_LAYER_PATH")
        print(
            "export VK_INSTANCE_LAYERS=VK_LAYER_ML_Graph_Emulation:VK_LAYER_ML_Tensor_Emulation"
        )
    elif sys.platform.startswith("win"):
        print(f"$env:VK_LAYER_PATH={paths['VK_LAYER_PATH_windows']};$VK_LAYER_PATH")
        print(
            '$env:VK_INSTANCE_LAYERS="VK_LAYER_ML_Graph_Emulation;VK_LAYER_ML_Tensor_Emulation"'
        )
    elif sys.platform.startswith("darwin"):
        print(
            f"export DYLD_LIBRARY_PATH={paths['DYLD_LIBRARY_PATH']}:$DYLD_LIBRARY_PATH"
        )
        print(f"$env:VK_LAYER_PATH={paths['VK_LAYER_PATH']}:$VK_LAYER_PATH")
        print(
            '$env:VK_INSTANCE_LAYERS="VK_LAYER_ML_Graph_Emulation:VK_LAYER_ML_Tensor_Emulation"'
        )
    else:
        print("ERROR: Unsupported platform")
