/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include "mlel/pipeline.hpp"
#include "mlel/tensor.hpp"
#include "mlel/utils.hpp"
#include "vulkan_test_utils.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <vulkan/vulkan_beta.h>

namespace mlsdk::el::tests {

using namespace utilities;

namespace {

using mlsdk::el::tests::createDevice;

vk::raii::Instance createInstance(vk::raii::Context &ctx, std::vector<const char *> enabledLayers = {},
                                  std::vector<const char *> enabledExtensions = {}) {
    const vk::ApplicationInfo applicationInfo{
        "ML Emulation Layer",            // application name
        VK_MAKE_API_VERSION(1, 3, 0, 0), // application version
        "ML Emulation Layer",            // engine name
        VK_MAKE_API_VERSION(1, 3, 0, 0), // engine version
        VK_API_VERSION_1_3,              // api version
    };
    vk::InstanceCreateFlags flags{};
    const auto extensionProperties = ctx.enumerateInstanceExtensionProperties();
    if (mlsdk::el::utils::hasExtension(extensionProperties, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        enabledExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        flags = vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
    }

    const vk::InstanceCreateInfo instanceCreateInfo{
        flags,                                           // flags
        &applicationInfo,                                // application info
        static_cast<uint32_t>(enabledLayers.size()),     // enabled layer count
        enabledLayers.data(),                            // enabled layers
        static_cast<uint32_t>(enabledExtensions.size()), // enabled extension count
        enabledExtensions.data(),                        // enabled extensions
    };

    return vk::raii::Instance{ctx, instanceCreateInfo};
}

bool hasExtensionProperties(const vk::raii::PhysicalDevice &physicalDevice,
                            const std::vector<const char *> &extensions) {
    const auto extensionProperties = physicalDevice.enumerateDeviceExtensionProperties();
    return std::all_of(extensions.begin(), extensions.end(), [&](const auto &extension) {
        return mlsdk::el::utils::hasExtension(extensionProperties, extension);
    });
}

std::array<const float, 16> queuePriorities = {1.0f};

std::vector<vk::DeviceQueueCreateInfo> getQueueCreateInfo(const vk::raii::PhysicalDevice &physicalDevice,
                                                          const vk::QueueFlags flags) {
    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfo;
    const auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

    for (uint32_t i = 0; i < queueFamilyProperties.size(); i++) {
        const auto &property = queueFamilyProperties[i];

        if (property.queueFlags & flags) {
            queueCreateInfo.push_back(vk::DeviceQueueCreateInfo{
                {},                     // flags
                i,                      // queue family index
                property.queueCount,    // queue count
                queuePriorities.data(), // queue priorities
            });
        }
    }

    return queueCreateInfo;
}

std::tuple<vk::raii::Device, vk::raii::PhysicalDevice>
createDevice(vk::raii::Instance &instance, std::vector<const char *> enabledLayers = {},
             const std::vector<const char *> &enabledExtensions = {}) {
    for (const auto &physicalDevice : vk::raii::PhysicalDevices{instance}) {
        // Verify that device supports compute queues
        const auto queueCreateInfo = getQueueCreateInfo(physicalDevice, vk::QueueFlagBits::eCompute);
        if (queueCreateInfo.empty()) {
            continue;
        }
        auto deviceExtensions = enabledExtensions;
        const auto extensionProperties = physicalDevice.enumerateDeviceExtensionProperties();
        if (mlsdk::el::utils::hasExtension(extensionProperties, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME)) {
            deviceExtensions.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
        }

        // Verify that device supports all enabled extensions
        if (!hasExtensionProperties(physicalDevice, deviceExtensions)) {
            continue;
        }

        const vk::DeviceCreateInfo deviceCreateInfo{
            {},                                             // flags
            static_cast<uint32_t>(queueCreateInfo.size()),  // queue create info count
            queueCreateInfo.data(),                         // queue create infos
            static_cast<uint32_t>(enabledLayers.size()),    // enabled layer count
            enabledLayers.data(),                           // enabled layers
            static_cast<uint32_t>(deviceExtensions.size()), // enabled extension count
            deviceExtensions.data(),                        // enabled extensions
        };

        return {vk::raii::Device{physicalDevice, deviceCreateInfo}, physicalDevice};
    };
    throw std::runtime_error("Could not create device");
}

vk::raii::TensorARM createTensor(vk::raii::Device &device, const std::vector<int64_t> &dimensions,
                                 const std::vector<int64_t> &strides) {
    vk::TensorDescriptionARM tensorDescription{
        vk::TensorTilingARM::eLinear, // tiling
        vk::Format::eR8Sint,          // format
        uint32_t(dimensions.size()),  // dimensions count
        dimensions.data(),            // dimensions
        nullptr,                      // strides
        {},                           // usage flags
    };

    if (!strides.empty()) {
        tensorDescription.setPStrides(strides.data());
    }

    const vk::TensorCreateInfoARM tensorCreateInfo{
        {},                          // flags
        &tensorDescription,          // tensor description
        vk::SharingMode::eExclusive, // sharing mode
        0,                           // queue family index count
        nullptr                      // queue family indices
    };

    return {device, tensorCreateInfo};
}

vk::MemoryRequirements2 getTensorMemoryRequirements(vk::raii::Device &device, const vk::raii::TensorARM &tensor) {
    const vk::TensorMemoryRequirementsInfoARM requirementsInfo{
        *tensor,
    };
    vk::MemoryRequirements2 memoryRequirements = device.getTensorMemoryRequirementsARM(requirementsInfo);

    return memoryRequirements;
}

vk::raii::DeviceMemory allocateTensorMemory(vk::raii::Device &device, vk::raii::PhysicalDevice &physicalDevice,
                                            const vk::raii::TensorARM &tensor) {
    const auto requirements = getTensorMemoryRequirements(device, tensor);
    const auto memoryProperties = physicalDevice.getMemoryProperties();

    uint32_t memoryTypeIndex = 0;
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
        if (!((1U << i) & requirements.memoryRequirements.memoryTypeBits)) {
            continue;
        }
        if (memoryProperties.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eHostCoherent) {
            memoryTypeIndex = i;
            break;
        }
    }
    // Allocate memory
    const vk::MemoryAllocateInfo allocateInfo{
        requirements.memoryRequirements.size, // size
        memoryTypeIndex,                      // memory type index
    };
    vk::raii::DeviceMemory deviceMemory{device, allocateInfo};

    return deviceMemory;
}

void bindTensor(vk::raii::Device &device, const vk::raii::TensorARM &tensor, const vk::raii::DeviceMemory &memory) {
    const vk::BindTensorMemoryInfoARM bindInfo{
        *tensor, // tensor
        *memory, // device memory
        {}       // memory offset
    };

    device.bindTensorMemoryARM(bindInfo);
}

vk::raii::TensorViewARM createTensorView(vk::raii::Device &device, const vk::raii::TensorARM &tensor,
                                         vk::Format format) {
    const vk::TensorViewCreateInfoARM tensorViewCreateInfo{
        {},      // flags
        *tensor, // tensor
        format,  // format
    };
    vk::raii::TensorViewARM tensorView{device, tensorViewCreateInfo};

    return tensorView;
}

TEST(MLEmulationLayerForVulkan, EnumerateLayers) { // cppcheck-suppress syntaxError
    vk::raii::Context ctx{};

    const auto layerProperties = ctx.enumerateInstanceLayerProperties();

    for (const auto &property : layerProperties) {
        std::cout << "name=" << property.layerName << ", description=" << property.description << std::endl;
    }
}

TEST(MLEmulationLayerForVulkan, EnumerateInstanceExtensions) {
    vk::raii::Context ctx{};

    const auto extensionProperties = ctx.enumerateInstanceExtensionProperties();

    for (const auto &property : extensionProperties) {
        std::cout << "name=" << property.extensionName << std::endl;
    }
}

TEST(MLEmulationLayerForVulkan, CreateInstance) {
    vk::raii::Context ctx{};
    [[maybe_unused]] auto instance = createInstance(ctx, {"VK_LAYER_ML_Tensor_Emulation"});
}

TEST(MLEmulationLayerForVulkan, EnumeratePhysicalDevices) {
    vk::raii::Context ctx{};
    auto instance = createInstance(ctx, {"VK_LAYER_ML_Tensor_Emulation"}, {});
    auto physicalDevices = vk::raii::PhysicalDevices{instance};

    std::cout << "Physical devices:" << std::endl;
    for (auto &physicalDevice : physicalDevices) {
        std::cout << "  Handle=" << (*physicalDevice) << std::endl;

        vk::PhysicalDeviceProperties physicalDeviceProperties = physicalDevice.getProperties();
        std::cout << "  Name=" << physicalDeviceProperties.deviceName << std::endl;
        std::cout << "  Type=" << vk::to_string(physicalDeviceProperties.deviceType) << std::endl;

        std::cout << "  Queue families:" << std::endl;
        auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
        for (auto &property : queueFamilyProperties) {
            std::cout << "    flags=" << vk::to_string(property.queueFlags) << ", count=" << property.queueCount
                      << std::endl;
        }

        std::cout << "  Extensions:" << std::endl;
        auto extensionProperties = physicalDevice.enumerateDeviceExtensionProperties();
        for (const auto &property : extensionProperties) {
            std::cout << "    name=" << property.extensionName << std::endl;
        }
    }
}

TEST(MLEmulationLayerForVulkan, CreateDevice) {
    vk::raii::Context ctx{};
    auto instance = createInstance(ctx, {"VK_LAYER_ML_Tensor_Emulation"});
    auto [device, physicalDevice] = createDevice(instance, {"VK_LAYER_ML_Tensor_Emulation"});
}

TEST(MLEmulationLayerForVulkan, CreateDeviceWithUnifiedFeatureStructs) {
    const std::vector<const char *> layers = {"VK_LAYER_ML_Graph_Emulation", "VK_LAYER_ML_Tensor_Emulation"};
    const std::vector<const char *> extensions = {
        VK_ARM_DATA_GRAPH_EXTENSION_NAME,
        VK_ARM_DATA_GRAPH_INSTRUCTION_SET_TOSA_EXTENSION_NAME,
        VK_ARM_DATA_GRAPH_OPTICAL_FLOW_EXTENSION_NAME,
        VK_ARM_TENSORS_EXTENSION_NAME,
    };

    auto context = std::make_shared<vk::raii::Context>();
    auto instance = std::make_shared<Instance>(context, layers);
    auto physicalDevice = std::make_shared<PhysicalDevice>(instance, extensions);

    vk::PhysicalDeviceVulkan13Features unified13{};
    vk::PhysicalDeviceVulkan12Features unified12{};
    unified12.shaderFloat16 = VK_TRUE;
    unified12.pNext = &unified13;
    vk::PhysicalDeviceVulkan11Features unified11{};
    unified11.pNext = &unified12;
    vk::PhysicalDeviceFeatures2 features2{};
    features2.pNext = &unified11;

    auto device = std::make_shared<Device>(physicalDevice, extensions, &features2);

    auto graph = makeMaxPoolProfilingGraph(device);
    submitGraphWithoutFence(device, graph, true);
}

TEST(MLEmulationLayerForVulkan, CreateDeviceWithIndividualPromotedFeatureStructs) {
    const std::vector<const char *> layers = {"VK_LAYER_ML_Graph_Emulation", "VK_LAYER_ML_Tensor_Emulation"};
    const std::vector<const char *> extensions = {
        VK_ARM_DATA_GRAPH_EXTENSION_NAME,
        VK_ARM_DATA_GRAPH_INSTRUCTION_SET_TOSA_EXTENSION_NAME,
        VK_ARM_DATA_GRAPH_OPTICAL_FLOW_EXTENSION_NAME,
        VK_ARM_TENSORS_EXTENSION_NAME,
    };

    auto context = std::make_shared<vk::raii::Context>();
    auto instance = std::make_shared<Instance>(context, layers);
    auto physicalDevice = std::make_shared<PhysicalDevice>(instance, extensions);

    vk::PhysicalDeviceShaderFloat16Int8Features individualFloat16Int8{};
    individualFloat16Int8.shaderFloat16 = VK_TRUE;
    vk::PhysicalDeviceFeatures2 features2{};
    features2.pNext = &individualFloat16Int8;

    auto device = std::make_shared<Device>(physicalDevice, extensions, &features2);

    auto graph = makeMaxPoolProfilingGraph(device);
    submitGraphWithoutFence(device, graph, true);
}

TEST(MLEmulationLayerForVulkan, ToolingInfo) {
    vk::raii::Context ctx{};
    auto instance = createInstance(ctx, {"VK_LAYER_ML_Graph_Emulation", "VK_LAYER_ML_Tensor_Emulation"});
    auto [device, physicalDevice] = createDevice(instance, {"VK_EXT_tooling_info"});

    auto tools = physicalDevice.getToolPropertiesEXT();

    bool graphLayerTool = false;
    bool tensorLayerTool = false;

    for (auto t : tools) {
        tensorLayerTool |= std::strcmp(t.name, "Tensor Layer") == 0;
        graphLayerTool |= std::strcmp(t.name, "Graph Layer") == 0;
    }

    ASSERT_TRUE(graphLayerTool && tensorLayerTool) << "Tooling Info feature failed!";
}

TEST(MLEmulationLayerForVulkan, CheckTensorFeature) {
    vk::raii::Context ctx{};
    auto instance = createInstance(ctx, {"VK_LAYER_ML_Tensor_Emulation"});
    auto [device, physicalDevice] = createDevice(instance, {"VK_LAYER_ML_Tensor_Emulation"});
    auto features = physicalDevice.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceTensorFeaturesARM>();
    const auto &tensorFeature = features.template get<vk::PhysicalDeviceTensorFeaturesARM>();
    ASSERT_TRUE(tensorFeature.shaderTensorAccess) << "shaderTensorAccess not supported!";
}

TEST(MLEmulationLayerForVulkan, CreateTensor) {
    vk::raii::Context ctx{};
    auto instance = createInstance(ctx, {"VK_LAYER_ML_Tensor_Emulation"});
    auto [device, physicalDevice] =
        createDevice(instance, {"VK_LAYER_ML_Tensor_Emulation"}, {VK_ARM_TENSORS_EXTENSION_NAME});

    vk::raii::TensorARM tensor = createTensor(device, {1, 32, 32, 3}, {});
    vk::raii::DeviceMemory memory = allocateTensorMemory(device, physicalDevice, tensor);
    bindTensor(device, tensor, memory);

    [[maybe_unused]] auto tensorView = createTensorView(device, tensor, vk::Format::eR8Sint);
}

TEST(MLEmulationLayerForVulkan, ApplicationFixedAddressAllocation) {
    auto device = createDevice();
    const auto &vkDevice = &(*device);
    const auto &vkPhysicalDevice = &(*device->getPhysicalDevice());

    const auto features =
        vkPhysicalDevice.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan12Features>();
    if (!features.template get<vk::PhysicalDeviceVulkan12Features>().bufferDeviceAddressCaptureReplay) {
        GTEST_SKIP() << "Device does not support BDA capture/replay";
    }

    const auto memSize = 0x4000;
    auto memAddr = uint64_t{};
    auto goodIndex = uint32_t{};

    // First allocate freely so we can get a valid GPU address and index
    auto flagsInfo = vk::MemoryAllocateFlagsInfo{vk::MemoryAllocateFlagBits::eDeviceAddress |
                                                 vk::MemoryAllocateFlagBits::eDeviceAddressCaptureReplay};
    auto addrInfo = vk::MemoryOpaqueCaptureAddressAllocateInfo{0x0, &flagsInfo};

    const auto memoryTypeIndices =
        device->getPhysicalDevice()->getMemoryTypeIndices(vk::MemoryPropertyFlagBits::eDeviceLocal, 0xffffffff);
    for (auto index : memoryTypeIndices) {
        try {
            const auto memoryAllocateInfo = vk::MemoryAllocateInfo{memSize, index, &addrInfo};
            const auto devMem = vk::raii::DeviceMemory{vkDevice, memoryAllocateInfo};

            memAddr = vkDevice.getMemoryOpaqueCaptureAddress(vk::DeviceMemoryOpaqueCaptureAddressInfo{devMem});
            goodIndex = index;
            break;
        } catch (const vk::OutOfDeviceMemoryError &) {
            // Ignore exception and try next memory index
        }
    }

    ASSERT_TRUE(memAddr != 0x0) << "Failed to allocate any memory or failed to acquire initial memory address";

    // Allocate at the fixed address
    {
        addrInfo.opaqueCaptureAddress = memAddr;
        const auto memoryAllocateInfo = vk::MemoryAllocateInfo{memSize, goodIndex, &addrInfo};
        const auto devMem = vk::raii::DeviceMemory{vkDevice, memoryAllocateInfo};

        const auto newMemAddr =
            vkDevice.getMemoryOpaqueCaptureAddress(vk::DeviceMemoryOpaqueCaptureAddressInfo{devMem});
        ASSERT_TRUE(newMemAddr == memAddr) << "Memory address does not match requested";
    }

    // Allocate again but with allocation pNext chain reversed
    {
        addrInfo = vk::MemoryOpaqueCaptureAddressAllocateInfo{memAddr};
        flagsInfo = vk::MemoryAllocateFlagsInfo{vk::MemoryAllocateFlagBits::eDeviceAddress |
                                                    vk::MemoryAllocateFlagBits::eDeviceAddressCaptureReplay,
                                                0x0, &addrInfo};
        const auto memoryAllocateInfo = vk::MemoryAllocateInfo{memSize, goodIndex, &flagsInfo};
        const auto devMem = vk::raii::DeviceMemory{vkDevice, memoryAllocateInfo};

        const auto newMemAddr =
            vkDevice.getMemoryOpaqueCaptureAddress(vk::DeviceMemoryOpaqueCaptureAddressInfo{devMem});
        ASSERT_TRUE(newMemAddr == memAddr) << "Memory address does not match requested";
    }
}

} // namespace
} // namespace mlsdk::el::tests
