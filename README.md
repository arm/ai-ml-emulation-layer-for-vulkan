# ML Emulation Layer for Vulkan®

Arm® has approached the Khronos® group with a set of Machine Learning
extensions for the Vulkan® and SPIR-V™ APIs. On devices where these extensions
have not been implemented by the Vulkan® Installable Device Drivers (ICD), the
ML Emulation Layer is required.

The ML Emulation Layer for Vulkan® provides an implementation of the ML APIs
enabling ML workloads to be executed on any Vulkan® Compute capable device. The
ML Emulation Layer for Vulkan® is split into separate graph and tensor layers
that are inserted by the Vulkan® Loader.

The graph layer exposes:

- `VK_ARM_data_graph`
- `VK_ARM_data_graph_instruction_set_tosa`
- `VK_ARM_data_graph_optical_flow`

The tensor layer exposes:

- `VK_ARM_tensors`

The corresponding SPIR-V™ extensions and extended instruction sets currently
used by the ML Emulation Layer for Vulkan® are:

- `SPV_ARM_graph`
- `SPV_ARM_tensors`
- `TOSA.001000.1`
- `Arm.MotionEngine.100`

## Cloning the repository

To clone the ML Emulation Layer for Vulkan® as a stand-alone repository,
you can use regular git clone commands. However, for better management of
dependencies and to ensure everything is placed in the appropriate directories,
we recommend using the `git-repo` tool to clone the repository as part of the ML
SDK for Vulkan® suite. [Repo tool](https://gerrit.googlesource.com/git-repo).

For a minimal build and to initialize only the ML Emulation Layer for
Vulkan® and its dependencies, run:

```bash
repo init -u https://github.com/arm/ai-ml-sdk-manifest -g emulation-layer
```

Alternatively, to initialize the repo structure for the entire ML SDK for
Vulkan®, including the ML Emulation Layer for Vulkan®, run:

```bash
repo init -u https://github.com/arm/ai-ml-sdk-manifest -g all
```

After the repo is initialized, you can fetch the contents with:

```bash
repo sync --no-clone-bundle
```

### Cloning on Windows®

To ensure nested submodules do not exceed the maximum long path length, you must
enable long paths on Windows®, and you must clone close to the root directory
or use a symlink. Make sure to use Git for Windows.

Using **PowerShell**:

```powershell
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1
git config --global core.longpaths true
git --version # Ensure you are using Git for Windows, for example 2.50.1.windows.1
git clone <git-repo-tool-url>
python <path-to-git-repo>\git-repo\repo init -u <manifest-url> -g all
    python <path-to-git-repo>\git-repo\repo sync --no-clone-bundle
```

Using **Git Bash**:

```powershell
cmd.exe "/c reg.exe add \"HKLM\System\CurrentControlSet\Control\FileSystem"" /v LongPathsEnabled /t REG_DWORD /d 1 /f"
git config --global core.longpaths true
git --version # Ensure you are using the Git for Windows, for example 2.50.1.windows.1
git clone <git-repo-tool-url>
python <path-to-git-repo>/git-repo/repo init -u <manifest-url> -g all
python <path-to-git-repo>/git-repo/repo sync --no-clone-bundle
```

After the sync command completes successfully, you can find the ML SDK Emulation
Layer for Vulkan® in `<repo_root>/sw/emulation-layer/`. You can also find all
the dependencies required by the ML Emulation Layer for Vulkan® in
`<repo_root>/dependencies/`.

## Building the ML Emulation Layer for Vulkan® from source

The build system must have:

- CMake 3.25 or later.
- C/C++ 17 compiler: GCC, or optionally Clang on Linux and MSVC on Windows®.
- Ninja 1.10 or later.
- Vulkan® SDK 1.4.328.1 or later.

The following dependencies are also needed:

- [glslang](https://github.com/KhronosGroup/glslang).
- [SPIRV-Headers](https://github.com/KhronosGroup/SPIRV-Headers).
- [SPIRV-Tools](https://github.com/KhronosGroup/SPIRV-Tools).
- [SPIRV-Cross](https://github.com/KhronosGroup/SPIRV-Cross).
- [Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers).
- [GoogleTest](https://github.com/google/googletest). Optional, for testing.

For the preferred dependency versions see the manifest file.

<a name="building-with-the-script"></a>

## Building with the script

To make the build configuration options easily discoverable, we provide a python
build script. When you run the script from a git-repo manifest checkout, the
script uses default paths and does not require any additional arguments. If you
do not use the script, you must specify paths to the dependencies.

To build on the current platform, for example on Linux or Windows®, run the
following command:

```bash
python3 $SDK_PATH/sw/emulation-layer/scripts/build.py -j $(nproc)
```

To cross compile for AArch64 architecture, add the following option:

```bash
python3 $SDK_PATH/sw/emulation-layer/scripts/build.py -j $(nproc) --target-platform aarch64
```

To enable and run tests, use the `--test` option. To lint the tests, use the
`--lint` option. To build the documentation, use the `--doc` option. To build
the documentation, you must have `sphinx` and `doxygen` installed on your
machine.

You can install the build artifacts for this project into a specified location.
To install the build artifacts, pass the `--install` option with the required
path.

To create an archive with the build artifacts option, you must add the
`--package` option. The archive is stored in the provided location.

For more command line options, see the help output:

```bash
python3 $SDK_PATH/sw/emulation-layer/scripts/build.py --help
```

## Usage

The ML Emulation Layer for Vulkan® is loaded as two explicit Vulkan® layers.
The platform-specific sections below show the exact commands for each
operating system, but the same setup sequence applies on all platforms:

1. Make the Vulkan® loader discover the layer manifest files.
2. Make the platform dynamic loader discover the graph and tensor layer
   libraries.
3. Enable the graph layer before the tensor layer, either during Vulkan®
   instance creation or with `VK_INSTANCE_LAYERS`.
4. Configure any optional logging or profiling environment variables before
   starting the application.

The layer names are:

- `VK_LAYER_ML_Graph_Emulation`
- `VK_LAYER_ML_Tensor_Emulation`

The manifest files are:

- `VkLayer_Graph.json`
- `VkLayer_Tensor.json`

For more information about using explicit Vulkan® layers, see the
[Vulkan® Layer Documentation](https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderLayerInterface.md).

### Logging

You can enable logging using environment variables. Logging must be set before
the application is started. Logging severity can be one of `error`, `warning`,
`info`, or `debug`. Logging severity is set independently for the graph and
tensor layer.

Using **shell**:

```shell
export VMEL_GRAPH_SEVERITY=debug
export VMEL_TENSOR_SEVERITY=info
```

Using **PowerShell**:

```powershell
$env:VMEL_GRAPH_SEVERITY="debug"
$env:VMEL_TENSOR_SEVERITY="info"
```

Common severity for both layers can be set using the following variable:

```shell
export VMEL_COMMON_SEVERITY=debug
```

```powershell
$env:VMEL_COMMON_SEVERITY="debug"
```

### Graph Profiling

You can enable per-pipeline graph profiling with Vulkan® timestamp queries
using environment variables before starting the application. Profiling covers
TOSA graph operators, MotionEngine graph operators, and optical-flow compute
pipelines. Profiling is disabled by default. When enabled, graph command-buffer
submits remain asynchronous. Timestamp results are collected when the application
waits on fences, waits for a queue or device to become idle, or when the
profiling property is queried. Profiling results are saved only as a queryable
data graph pipeline property.

Using **shell**:

```shell
export VMEL_GRAPH_PROFILING=1
```

Using **PowerShell**:

```powershell
$env:VMEL_GRAPH_PROFILING="1"
```

The profiling property returns JSON with a `samples` array containing one entry
per profiled internal compute dispatch, including `pipeline_kind`,
`operator_name`, raw cycle counts, and `time_ms`, plus a `by_operator` summary
with total, average, minimum, and maximum time per profiled pipeline.

## Usage on Linux

You can enable the graph and tensor layers using environment variables only,
without modifying the Vulkan® application. The following environment variables
are used:

- Use the `LD_LIBRARY_PATH` environment variable to point at the `VkLayer_Graph`
  and `VkLayer_Tensor` libraries.
- Use the `VK_ADD_LAYER_PATH` environment variable to point at the
  `VkLayer_Graph.json` and `VkLayer_Tensor.json` manifest file.
  - If your loader ignores `VK_ADD_LAYER_PATH` (older SDKs before 1.4.328.1), use `VK_LAYER_PATH`.
- You must enable the graph layer before the tensor layer. To do this, use the
  `VK_INSTANCE_LAYERS` environment variable.

If you have installed the ML Emulation Layer for Vulkan® into a deploy folder, use the
following environment variables to enable the layers:

```shell
export LD_LIBRARY_PATH=$PWD/deploy/lib:$LD_LIBRARY_PATH
export VK_ADD_LAYER_PATH=$PWD/deploy/share/vulkan/explicit_layer.d
export VK_INSTANCE_LAYERS=VK_LAYER_ML_Graph_Emulation:VK_LAYER_ML_Tensor_Emulation
```

## Usage on Windows®

You can enable the graph and tensor layers using environment variables only,
without modifying the Vulkan® application. The following environment variables
are used:

- Use the `VK_ADD_LAYER_PATH` environment variable to point at the
  `VkLayer_Graph.json` and `VkLayer_Tensor.json` manifest files.
- You must enable the graph layer before the tensor layer. To do this, use the
  `VK_INSTANCE_LAYERS` environment variable.

If you have installed the ML Emulation Layer for Vulkan® into a deploy folder, use the
following environment variables to enable the layers:

```powershell
$env:VK_LAYER_PATH="$PWD\deploy\bin"
$env:VK_INSTANCE_LAYERS="VK_LAYER_ML_Graph_Emulation;VK_LAYER_ML_Tensor_Emulation"
```

Alternatively, you can use the Windows® registry keys to load the manifest
files. This can be done using the Windows® GUI. Or, if you have installed the
ML Emulation Layer for Vulkan® into a deploy folder, you set the path to the manifest files
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

## Building for Android™ (Experimental)

The Android™ NDK toolset is required to build the ML Emulation Layer for Vulkan® for an Android™
device. The Android™ device must have Vulkan® API 1.3 support.

To build the ML Emulation Layer for Vulkan®, run:

```shell
cmake -B build
   -DCMAKE_TOOLCHAIN_FILE=${NDK}/build/cmake/android.toolchain.cmake \
   -DANDROID_ABI=arm64-v8a                                           \
   -DGLSLANG_PATH=${REPO}/dependencies/glslang                       \
   -DSPIRV_CROSS_PATH=${REPO}/dependencies/SPIRV-Cross               \
   -DSPIRV_HEADERS_PATH=${REPO}/dependencies/SPIRV-Headers           \
   -DSPIRV_TOOLS_PATH=${REPO}/dependencies/SPIRV-Tools               \
   -DVULKAN_HEADERS_PATH=${REPO}/dependencies/Vulkan-Headers

cmake --build build
```

## Usage on Android™ (Experimental)

You can pack the graph and tensor layer libraries into the Application Package
Kit (APK) or push to the `/data/local/debug/vulkan` directory for Android™ to
discover the ML Emulation Layer for Vulkan®. Applications can enable the layers during Vulkan
instance creation or you can enable the layers without modifying the application
by using following commands:

```shell
adb shell settings put global enable_gpu_debug_layers 1
adb shell settings put global gpu_debug_app $TARGET_APP_PKG
adb shell settings put global gpu_debug_layers \
    VK_LAYER_ML_Graph_Emulation:VK_LAYER_ML_Tensor_Emulation
```

### APK Packaging

If you want to package the ML Emulation Layer for Vulkan® as an Android™ APK,
set the following variables first:

```shell
export EMULATION_LAYER_ROOT=/path/to/emulation-layer
export NDK=/path/to/android-ndk
export ANDROID_HOME=/path/to/android-sdk
export TARGET_APP_PKG=com.example.targetapp
```

The Android™ packaging flow in `scripts/build.py` requires the Android™ NDK
toolchain for the native build and Gradle 8.4 or later with `ANDROID_HOME` set
for APK generation. The Android™ SDK installation pointed to by
`ANDROID_HOME` should include `build-tools;34.0.0` and
`platforms;android-34`, or other compatible versions. A typical APK packaging
command looks like:

```shell
python3 $EMULATION_LAYER_ROOT/scripts/build.py \
    --build-type Android \
    --target-platform android \
    --cmake-toolchain-for-android $NDK/build/cmake/android.toolchain.cmake \
    --install $EMULATION_LAYER_ROOT/apk_install \
    --package-type apk \
    -j $(nproc)
```

This produces an Android™ project in `apk_package/` and Gradle builds the debug
APK from there. The layer APK package name is currently
`com.arm.ai_ml_emulation_layer_for_vulkan`.

To enable the packaged layers for a target application, use Android™ GPU debug
layer settings:

```shell
adb shell settings put global enable_gpu_debug_layers 1
adb shell settings put global gpu_debug_app $TARGET_APP_PKG
adb shell settings put global gpu_debug_layers \
    VK_LAYER_ML_Graph_Emulation:VK_LAYER_ML_Tensor_Emulation
adb shell settings put global gpu_debug_layer_app \
    com.arm.ai_ml_emulation_layer_for_vulkan
```

If you only want to enable a single layer, the command will likely look like:

```shell
adb shell settings put global gpu_debug_layers VK_LAYER_KHRONOS_validation
```

The layer package must also be visible to the debug app. On Android™ 11 and
later, if the target app does not already query the layer package, add a
package visibility entry such as:

```xml
<queries>
    <package android:name="com.arm.ai_ml_emulation_layer_for_vulkan" />
</queries>
```

Refer to the Android™ validation layer guide for background on APK packaging,
debug layer settings, and package visibility:
[Use Vulkan validation layers on Android](https://developer.android.com/ndk/guides/graphics/validation-layer).

## Building for Darwin (Experimental)

MoltenVK is required to build and run the ML Emulation Layer for Vulkan® for a Darwin
device. The MoltenVK version must have Vulkan® API 1.3 support.
In this example we install into a deploy folder and build using the script.

To build the ML Emulation Layer for Vulkan®, run:

```shell
python3 ${REPO}/sw/emulation-layer/scripts/build.py --install $SDK_PATH/deploy
```

Install MoltenVK by following the documentation:
<https://vulkan.lunarg.com/doc/sdk/1.4.328.1/mac/getting_started.html>

## Usage on Darwin (Experimental)

We need to link to both the MoltenVK and the ML Emulation Layer for Vulkan® build output, when running on a Darwin device.
On Darwin, `LD_LIBRARY_PATH` is instead `DYLD_LIBRARY_PATH`.

```shell

export PATH=${MOLTEN_VK_PATH}/macOS/bin:${PATH}
export VK_ICD_FILENAMES=${MOLTEN_VK_PATH}/macOS/share/vulkan/icd.d/MoltenVK_icd.json
export DYLD_LIBRARY_PATH=${MOLTEN_VK_PATH}/macOS/lib:${SDK_PATH}/deploy/lib
export VK_LAYER_PATH=${REPO}/deploy/share/vulkan/explicit_layer.d
export VK_INSTANCE_LAYERS=VK_LAYER_ML_Graph_Emulation:VK_LAYER_ML_Tensor_Emulation
```

## Cross compilation for AArch64 on x86-64 (Experimental)

The shader pre-compilation step requires a glslang compiler. There are three
ways to accomplish this when cross-compiling:

1. Provide a custom glslang executable. You can direct CMake to a custom
   glslang executable file using the `GLSLANG_EXECUTABLE` option. First, build
   glslang inside its repo. When the repository is initialized using the repo
   manifest, the glslang source is checked out in
   `<repo_root>/dependencies/glslang/` For building glslang, see
   [Building (CMake)](https://github.com/KhronosGroup/glslang?tab=readme-ov-file#building-cmake).

2. Install glslang to the system. Under cross compilation, when no custom
   glslang executable is provided, it will be searched from the system using
   CMake's `find_package`. On Ubuntu, you can install it with
   `sudo apt install glslang-tools` or from the source code following the
   previously mentioned documentation. Note that we require version > 15.4.0,
   which may not yet be available in Ubuntu’s official package repositories.

3. Disable shader pre-compilation. This can be done by adding the
   flag `--disable-precompile-shaders` to the build script command. By doing so,
   the shaders will be compiled at runtime.

An example build flow using the option 1 would be:

First, build the glslang standalone under `<repo_root>/dependencies/glslang/`:

```shell
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DENABLE_GLSLANG_BINARIES=ON -DENABLE_OPT=OFF -DBUILD_SHARED_LIBS=OFF

cmake --build build --target glslang-standalone
```

After building, the binary will be at
`<repo_root>/dependencies/glslang/build/StandAlone/glslang`. Then run the
following under `<repo_root>/sw/emulation-layer/`:

```shell
cmake -B build \
   -DCMAKE_TOOLCHAIN_FILE=${REPO}/sw/emulation-layer/cmake/toolchain/linux-aarch64-gcc.cmake \
   -DGLSLANG_PATH=${REPO}/dependencies/glslang \
   -DSPIRV_CROSS_PATH=${REPO}/dependencies/SPIRV-Cross \
   -DSPIRV_HEADERS_PATH=${REPO}/dependencies/SPIRV-Headers \
   -DSPIRV_TOOLS_PATH=${REPO}/dependencies/SPIRV-Tools \
   -DVULKAN_HEADERS_PATH=${REPO}/dependencies/Vulkan-Headers \
   -DGLSLANG_EXECUTABLE=${REPO}/dependencies/glslang/build/StandAlone/glslang

cmake --build build
```

## Troubleshooting

### All zero output from AMD GPUs on Linux

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

   ```text
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

## PyPI

The ML Emulation Layer for Vulkan® is available on PyPI as the [ai-ml-emulation-layer-for-vulkan](https://pypi.org/project/ai-ml-emulation-layer-for-vulkan) package.

Install:

```bash
pip install ai-ml-emulation-layer-for-vulkan
```

## Known Limitations

- Resources created with `VK_IMAGE_TILING_OPTIMAL` and
  `VK_TENSOR_TILING_OPTIMAL_ARM` flags cannot be used with memory aliasing.
- Data graph pipeline creation without a shader module is not supported.
- Accuracy of MATMUL computation might be incorrect for mixed reduce float types
  such as fp8e5m2xfp8e4m3 and fp8e4m3xfp8e5m2.
- Usage of the 'shaderFloat64' feature requires support from the underlying ICD.
  This relates to high-precision types. Support of it can be checked with:

  ```bash
  vulkaninfo 2>&1 | grep -e 'shaderFloat64\|deviceName'
  ```

  If this feature is not available, add the `--use-float-as-double` flag to
  the build script command to use 32-bit `float` instead of `double`.
  This behavior is automatically enabled on Darwin and Android™.

MoltenVK currently does not have full Vulkan® coverage, some notable issues are:

- Several Vulkan® extensions are not available in MoltenVK, e.g. [custom border color](https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_custom_border_color.html).
- High-precision types in buffers/push constants is currently not supported, which forces lower precision to be used instead.
- Passing Shader Storage Buffer Objects, SSBOs, to functions is currently not supported in MoltenVK.

## License

The ML Emulation Layer for Vulkan® is distributed under the software licenses in
LICENSES directory.

## Trademark notice

Arm® is a registered trademarks of Arm Limited (or its subsidiaries) in the US
and/or elsewhere.

Khronos®, Vulkan® and SPIR-V™ are registered trademarks of the
[Khronos® Group](https://www.khronos.org/legal/trademarks).
