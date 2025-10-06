# Emulation Layer — Release Notes

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
