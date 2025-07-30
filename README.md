# ML Emulation Layer for Vulkan®

Arm® has approached the Khronos® group with a set of Machine Learning
extensions for the Vulkan® and SPIR-V™ APIs. On devices where these extensions
have not been implemented by the Vulkan® Installable Device Drivers (ICD), the
ML Emulation Layer is required.

The ML Emulation Layer for Vulkan® provides an implementation of the ML APIs
enabling ML workloads to be executed on any Vulkan® Compute capable device. The
Emulation Layer is split into separate graph, VK_ARM_data_graph, and tensor,
VK_ARM_tensors, layers that are inserted by the Vulkan® Loader.

### Building the Emulation Layer from source

The build system must have:

- CMake 3.25 or later.
- C/C++ 17 compiler: GCC, or optionally Clang on Linux and MSVC on Windows®.

The following dependencies are also needed:

- [glslang](https://github.com/KhronosGroup/glslang).
- [SPIRV-Headers](https://github.com/KhronosGroup/SPIRV-Headers).
- [SPIRV-Tools](https://github.com/KhronosGroup/SPIRV-Tools).
- [SPIRV-Cross](https://github.com/KhronosGroup/SPIRV-Cross).
- [Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers).
- [GoogleTest](https://github.com/google/googletest). Optional, for testing.

For the preferred dependency versions see the manifest file.

### Building for Linux

Building the code requires CMake. Also required are Vulkan® and SPIR-V™
dependencies including the ML extensions for Vulkan®.

To build the Emulation Layer, run:

```shell
$ cmake -B build                                               \
   -DGLSLANG_PATH=${REPO}/dependencies/glslang                 \
   -DSPIRV_CROSS_PATH=${REPO}/dependencies/SPIRV-Cross         \
   -DSPIRV_HEADERS_PATH=${REPO}/dependencies/SPIRV-Headers     \
   -DSPIRV_TOOLS_PATH=${REPO}/dependencies/SPIRV-Tools         \
   -DVULKAN_HEADERS_PATH=${REPO}/dependencies/Vulkan-Headers

$ cmake --build build
```

To build the documentation, add the `-DVMEL_BUILD_DOCS` build flag.

To install the Emulation Layer into a `deploy` folder, run:

```shell
cmake --install build --prefix deploy
```

### Usage on Linux

You can enable the graph and tensor layers using environment variables only,
without modifying the Vulkan® application. The following environment variables
are used:

- Use the `LD_LIBRARY_PATH` environment variable to point at the `VkLayer_Graph`
  and `VkLayer_Tensor` libraries.
- Use the `VK_ADD_LAYER_PATH` environment variable to point at the
  `VkLayer_Graph.json` and `VkLayer_Tensor.json` manifest file.
- You must enable the graph layer before the tensor layer. To do this, use the
  `VK_INSTANCE_LAYERS` environment variable.

If you have installed the Emulation Layer into a deploy folder, use the
following environment variables to enable the layers:

```shell
export LD_LIBRARY_PATH=$PWD/deploy/lib:$LD_LIBRARY_PATH
export VK_ADD_LAYER_PATH=$PWD/deploy/share/vulkan/explicit_layer.d
export VK_INSTANCE_LAYERS=VK_LAYER_ML_Graph_Emulation:VK_LAYER_ML_Tensor_Emulation
```

You can also enable logging using environment variables. Logging must be set
before the application is started. Logging severity can be one of `error`,
`warning`, `info`, or `debug`. Logging severity is set independently for the
graph and tensor layer using the following commands:

```shell
export VMEL_GRAPH_SEVERITY=debug
export VMEL_TENSOR_SEVERITY=info
```

Common severity for both layers can be set using the following command:

```shell
export VMEL_COMMON_SEVERITY=debug
```

### Building for Windows®

To build on Windows®, the machine must have a Vulkan® ICD and Vulkan® Loader
installed. Vulkan® API 1.3 must also be supported.

The reference environment that the Emulation Layer has been built and tested
with is:

- OS: Win10
- Build Tool: Visual Studio 17 2022
- Compiler: MSVC 19.37.32825.0
- CMake: 3.27.7
- Terminal: PowerShell

To build the Emulation Layer, run:

```powershell
$env:REPO="path\to\repo"
cmake -B build                                                 `
   -DGLSLANG_PATH="$env:REPO\dependencies\glslang"             `
   -DSPIRV_CROSS_PATH="$env:REPO\dependencies\SPIRV-Cross"     `
   -DSPIRV_HEADERS_PATH="$env:REPO\dependencies\SPIRV-Headers" `
   -DSPIRV_TOOLS_PATH="$env:REPO\dependencies\SPIRV-Tools"     `
   -DVULKAN_HEADERS_PATH="$env:REPO\dependencies\Vulkan-Headers"
cmake --build build --config Release
```

To install the Emulation Layer into a `deploy` folder, run:

```powershell
cmake --install build --prefix deploy
```

### Usage on Windows®

You can enable the graph and tensor layers using environment variables only,
without modifying the Vulkan® application. The following environment variables
are used:

- Use the `VK_ADD_LAYER_PATH` environment variable to point at the
  `VkLayer_Graph.json` and `VkLayer_Tensor.json` manifest files.
- You must enable the graph layer before the tensor layer. To do this, use the
  `VK_INSTANCE_LAYERS` environment variable.

If you have installed the Emulation Layer into a deploy folder, use the
following environment variables to enable the layers:

```powershell
$env:VK_LAYER_PATH="$PWD\deploy\bin"
$env:VK_INSTANCE_LAYERS="VK_LAYER_ML_Graph_Emulation;VK_LAYER_ML_Tensor_Emulation"
```

Alternatively, you can use the Windows® registry keys to load the manifest
files. This can be done using the Windows® GUI. Or, if you have installed the
emulation layer into a deploy folder, you set the path to the manifest files
using:

```powershell
reg add HKEY_LOCAL_MACHINE\SOFTWARE\Khronos\Vulkan\ExplicitLayers /v `
{ABSOLUTE_PATH}\deploy\bin /t REG_DWORD /d 0 /f

$env:VK_INSTANCE_LAYERS="VK_LAYER_ML_Graph_Emulation;VK_LAYER_ML_Tensor_Emulation"
```

```{note}
If running a Windows® terminal with elevated permissions, `VK_ADD_LAYER_PATH` is ignored
for security reasons. However, if `VK_ADD_LAYER_PATH` is set and not ignored, then Vulkan
skips searching the registry keys for manifest files.
```

You can also enable logging using environment variables. Logging must be set
before the application is started. Logging severity can be one of `error`,
`warning`, `info`, or `debug`. Logging severity is set independently for the
graph and tensor layer using the following commands:

```powershell
$env:VMEL_GRAPH_SEVERITY="debug"
$env:VMEL_TENSOR_SEVERITY="info"
```

### Building for Android (Experimental)

The Android NDK toolset is required to build the Emulation layer for an Android
device. The Android device must have Vulkan® API 1.3 support.

To build the Emulation Layer, run:

```shell
$ cmake -B build
   -DCMAKE_TOOLCHAIN_FILE=${NDK}/build/cmake/android.toolchain.cmake \
   -DANDROID_ABI=arm64-v8a                                           \
   -DGLSLANG_PATH=${REPO}/dependencies/glslang                       \
   -DSPIRV_CROSS_PATH=${REPO}/dependencies/SPIRV-Cross               \
   -DSPIRV_HEADERS_PATH=${REPO}/dependencies/SPIRV-Headers           \
   -DSPIRV_TOOLS_PATH=${REPO}/dependencies/SPIRV-Tools               \
   -DVULKAN_HEADERS_PATH=${REPO}/dependencies/Vulkan-Headers

$ cmake --build build
```

### Usage on Android

You can pack the graph and tensor layer libraries into the Application Package
Kit (APK) or push to the `/data/local/debug/vulkan` directory for Android to
discover the Emulation Layer. Applications can enable the layers during Vulkan
instance creation or you can enable the layers without modifying the application
by using following commands:

```shell
$ adb shell settings put global enable_gpu_debug_layers 1
$ adb shell settings put global gpu_debug_app ${package_name}
$ adb shell settings put global gpu_debug_layers \
    VK_LAYER_ML_Graph_Emulation:VK_LAYER_ML_Tensor_Emulation
```

### Vulkan® Layer Documentation

For more information about using layers, see the
[Vulkan® Layer Documentation](https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderLayerInterface.md).

### Troubleshooting

#### All zero output from AMD GPUs on Linux

Some workloads may cause silent GPU crashes due to timeout errors. You can check
for related kernel messages with the following command:

```shell
dmesg | grep -i amdgpu
```

To change the timeout, follow these steps (applies if your system uses GRUB as
the bootloader):

1. Edit the GRUB configuration file:

```shell
sudo nano /etc/default/grub
```

2. Add or modify the `GRUB_CMDLINE_LINUX` line to include a longer timeout value
   in milliseconds:

```
GRUB_CMDLINE_LINUX="quiet splash amdgpu.lockup_timeout=20000"
```

3. Update the GRUB configuration:

```shell
sudo update-grub
```

4. Reboot the system:

```shell
sudo reboot
```

## Known Limitations

- Resources created with `VK_IMAGE_TILING_OPTIMAL` and
  `VK_TENSOR_TILING_OPTIMAL_ARM` flags cannot be used with memory aliasing.
- Data graph pipeline creation without a shader module is not supported.
- The following `VK_ARM_data_graph` functions are not implemented:

  - `vkGetDataGraphPipelineAvailablePropertiesARM`
  - `vkGetDataGraphPipelinePropertiesARM`
  - `vkGetPhysicalDeviceQueueFamilyDataGraphPropertiesARM`
  - `vkGetPhysicalDeviceQueueFamilyDataGraphProcessingEnginePropertiesARM`

## License

The ML Emulation Layer for Vulkan® is provided under an Apache-2.0 license.
Please see [Apache-2.0.txt](LICENSES/Apache-2.0.txt) for more information.

## Contributions

Please see [Contributing](CONTRIBUTING.md).

## Security

Please see [Security](SECURITY.md).

## Trademark notice

Arm® is a registered trademarks of Arm Limited (or its subsidiaries) in the US
and/or elsewhere.

Khronos®, Vulkan® and SPIR-V™ are registered trademarks of the
[Khronos® Group](https://www.khronos.org/legal/trademarks).
