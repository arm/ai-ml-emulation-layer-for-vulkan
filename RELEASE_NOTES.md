# Emulation Layer — Release Notes

---

## Version 0.9.0 – *Datatype Support & API Coverage*

### New Datatypes

- Added BF16 and FP8 (`fp8e4m3`, `fp8e5m2`) support across SPIR-V™ passes and
  kernel execution paths.

### Tensor & Graph API coverage

- Added interception coverage for additional Vulkan® instance and physical-device
  query paths, including `vkGetPhysicalDeviceProperties2KHR`.
- Added support for `VK_EXT_tooling_info` and completed missing instance
  intercept-initialization paths.
- Updated graph/tensor behavior with higher-rank constant support and queue
  family property fixes.
- Added `VK_KHR_SYNCHRONIZATION_2` requirement in the graph layer.

### Build, Packaging & Developer Experience

- Added `use-float-as-double` build option for shader compatibility testing.
- Fixed Android™ shader source path handling
- Fixed Darwin usage documentation details.
- Fixed shader/runtime issues, including Reduce and Pad ops.
- Fixed `vkSetDebugUtilsObjectNameEXT` dispatch to use the device loader.

### Bug Fixes

- Fixed `vkCreateDevice` extension filtering.
- Fixed Pad/Reduce shader issues and tensor out-of-bounds reads.
- Fixed `vkSetDebugUtilsObjectNameEXT` to dispatch through the device loader.

## Version 0.8.0 – *API Coverage & Platform Expansion*

### New API

- Added Arm® Motion Engine Extended Instruction Set for SPIR-V™.

### Tensor & Graph API coverage

- Implemented `vkGetPhysicalDeviceExternalTensorPropertiesARM`, introduced new
  function skeletons, and exposed `vkEnumerateInstanceExtensionProperties` plus
  `vkGetPhysicalDeviceFeatures2KHR` so tooling can query the layers without
  falling back to core Vulkan® entry points.

### Build, Packaging & Developer Experience

- Modernized the pip package: switched to `pyproject.toml`, added the missing
  metadata, and fixed package naming/installation ordering issues that affected
  `--install`.
- Defaulted the build system to Ninja, refined the CMake packaging flow.
- Introduced `clang-tidy` configuration and streamlined cppcheck
  invocation/CLI integration (including build-script driven execution).

### Platform & Compliance

- Added Darwin targets for AArch64 to the pip packaging matrix.
- Refreshed SBOM data and adopted usage of `REUSE.toml`.

### Bug Fixes

- Fixed build-script packaging issues, including `setup.py` relative paths, bad
  package names, and SDK-root `--install` failures.
- Fixed `vkNegotiateLoaderLayerInterfaceVersion` handling in the Vulkan loader
  integration.
- Fixed `VkMemoryAllocateFlagsInfo` handling.

### Supported Platforms

The following platform combinations are supported:

- Linux - AArch64 and x86-64
- Windows® - x86-64
- Darwin - AArch64 via MoltenVK (experimental)
- Android™ - AArch64 (experimental)

---

## Version 0.7.0 – *Initial Public Release*

## Purpose

Provides a **layered Vulkan® implementation** of ML APIs for environments lacking
native ML extension support.

## Features

### Vulkan® Loader Integration

- **Dual-Layer Architecture**: Two separate layers - graph (`VK_ARM_data_graph`)
  and tensor (`VK_ARM_tensors`) — providing modular ML extensions for Vulkan® support. The
  graph layer depends on the tensor layer.
- **Vulkan® Loader Integration**: Seamlessly injected by the Vulkan® Loader
  without requiring application modifications

### ML Extensions

- **Universal Compatibility**: Enables ML workloads to run on any Vulkan®
  compute-capable device, regardless of native ML extensions for Vulkan® support
- **Fallback Implementation**: Acts as a bridge solution while native driver
  support for ML extensions matures across the ecosystem

### Shader compilation

- **SPIR-V™ Processing**: Advanced shader compilation and optimization through
  SPIRV-Tools, and SPIRV-Cross integration
- **Runtime Shader Compilation**: Supports both pre-compiled and runtime shader
  compilation modes for flexible deployment scenarios

## Platform Support

The following platform combinations are supported:

- Linux - X86-64
- Windows® - X86-64

---
