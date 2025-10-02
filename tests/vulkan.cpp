/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include <gtest/gtest.h>

#include "mlel/device.hpp"
#include "mlel/exception.hpp"
#include "mlel/float.hpp"
#include "mlel/log.hpp"
#include "mlel/pipeline.hpp"
#include "mlel/tensor.hpp"
#include "mlel/utils.hpp"

#include <spirv-tools/libspirv.hpp>
#include <vulkan/vulkan.hpp>

#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <glslang/Public/ResourceLimits.h>
#include <iostream>
#include <iterator>
#include <numeric>
#include <tuple>
#include <vector>

using namespace mlsdk::el::utilities;

/*******************************************************************************
 * Helpers
 *******************************************************************************/

#define STR(str) #str
#define TOSTRING(str) STR(str)

namespace {

class MLEmulationLayerForVulkan : public ::testing::Test {
  public:
    ~MLEmulationLayerForVulkan() override {}

    // Override this to define how to tear down the environment.
    void TearDown() override {}
};

vk::raii::Instance createInstance(vk::raii::Context &ctx, std::vector<const char *> enabledLayers = {},
                                  std::vector<const char *> enabledExtensions = {}) {
    const vk::ApplicationInfo applicationInfo{
        "ML Emulation Layer",            // application name
        VK_MAKE_API_VERSION(1, 3, 0, 0), // application version
        "ML Emulation Layer",            // engine name
        VK_MAKE_API_VERSION(1, 3, 0, 0), // engine version
        VK_MAKE_API_VERSION(1, 3, 0, 0), // api version
    };
    vk::InstanceCreateFlags flags;
#ifdef MOLTEN_VK_SUPPORT
    enabledExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    flags = vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
#endif

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

    for (auto &extension : extensions) {
        auto it = std::find_if(extensionProperties.begin(), extensionProperties.end(), [&](const auto &property) {
            return std::strcmp(property.extensionName, extension) == 0;
        });
        if (it == extensionProperties.end()) {
            return false;
        }
    }

    return true;
}

std::array<const float, 16> queuePriorities = {1.0f};

std::vector<vk::DeviceQueueCreateInfo> getQueueCreateInfo(const vk::raii::PhysicalDevice &physicalDevice,
                                                          const vk::QueueFlags flags) {
    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfo;
    const auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

    for (uint32_t i = 0; i < queueFamilyProperties.size(); i++) {
        auto &property = queueFamilyProperties[i];

        if (property.queueFlags & flags)
            queueCreateInfo.push_back(vk::DeviceQueueCreateInfo{
                {},                     // flags
                i,                      // queue family index
                property.queueCount,    // queue count
                queuePriorities.data(), // queue priorities
            });
    }

    return queueCreateInfo;
}

std::tuple<vk::raii::Device, vk::raii::PhysicalDevice> createDevice(vk::raii::Instance &instance,
                                                                    std::vector<const char *> enabledLayers = {},
                                                                    std::vector<const char *> enabledExtensions = {}) {
    for (auto physicalDevice : vk::raii::PhysicalDevices{instance}) {
        // Verify that device supports compute queues
        const auto queueCreateInfo = getQueueCreateInfo(physicalDevice, vk::QueueFlagBits::eCompute);
        if (queueCreateInfo.size() == 0) {
            continue;
        }
#ifdef MOLTEN_VK_SUPPORT
        enabledExtensions.push_back("VK_KHR_portability_subset");
#endif
        // Verify that device supports all enabled extensions
        if (!hasExtensionProperties(physicalDevice, enabledExtensions)) {
            continue;
        }

        const vk::DeviceCreateInfo deviceCreateInfo{
            {},                                              // flags
            static_cast<uint32_t>(queueCreateInfo.size()),   // queue create info count
            queueCreateInfo.data(),                          // queue create infos
            static_cast<uint32_t>(enabledLayers.size()),     // enabled layer count
            enabledLayers.data(),                            // enabled layers
            static_cast<uint32_t>(enabledExtensions.size()), // enabled extension count
            enabledExtensions.data(),                        // enabled extensions
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

    if (strides.size() > 0)
        tensorDescription.setPStrides(strides.data());

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

std::string fileToString(const std::string &filename) {
    const std::filesystem::path path = std::filesystem::path(TOSTRING(SHADER_SOURCE_DIR)) / filename;
    std::ifstream ifs{path};
    VK_ASSERT_EQ(!ifs, false, std::string("Failed to open ") + filename);

    std::string str(std::istreambuf_iterator<char>{ifs}, {});
    return str;
}

void sprivMessageConsumer(spv_message_level_t level, const char *, const spv_position_t &position,
                          const char *message) {
    std::string levelstr;
    switch (level) {
    case SPV_MSG_FATAL:
        levelstr = "FATAL";
        break;
    case SPV_MSG_INTERNAL_ERROR:
        levelstr = "INTERNAL ERROR";
        break;
    case SPV_MSG_ERROR:
        levelstr = "ERROR";
        break;
    case SPV_MSG_WARNING:
        levelstr = "WARNING";
        break;
    case SPV_MSG_INFO:
        levelstr = "INFO";
        break;
    case SPV_MSG_DEBUG:
        levelstr = "DEBUG";
        break;
    }

    std::cout << levelstr << ": message=" << message << ", position=" << position.index << std::endl;
}

std::vector<uint32_t> assembleSpirv(const std::string &text) {
    spvtools::SpirvTools tools{SPV_ENV_UNIVERSAL_1_6};

    if (!tools.IsValid()) {
        VK_ASSERT(false, "Failed to instantiate SPIR-V tools");
        return {};
    }

    tools.SetMessageConsumer(sprivMessageConsumer);

    std::vector<uint32_t> spirvModule;

    if (!tools.Assemble(text, &spirvModule)) {
        VK_ASSERT(false, "Failed to assemble SPIR-V program");
        return {};
    }

    if (!tools.Validate(spirvModule)) {
        VK_ASSERT(false, "Failed to validate SPIR-V program");
        return {};
    }

    return spirvModule;
}

vk::raii::ShaderModule createShaderModule(const vk::raii::Device &device, const std::vector<uint32_t> &code) {
    const vk::ShaderModuleCreateInfo info{
        {},                                                    // flags
        static_cast<uint32_t>(code.size() * sizeof(uint32_t)), // code size
        code.data()                                            // code
    };

    return vk::raii::ShaderModule(device, info);
}

std::shared_ptr<Device> createDevice() {
    std::vector<const char *> layers = {"VK_LAYER_ML_Graph_Emulation", "VK_LAYER_ML_Tensor_Emulation"};
    std::vector<const char *> extensions = {
        VK_ARM_DATA_GRAPH_EXTENSION_NAME,
        VK_ARM_TENSORS_EXTENSION_NAME,
    };

    const auto envValidation = std::getenv("VMEL_VALIDATION");
    if (envValidation && std::string(envValidation) != "" && std::string(envValidation) != "0") {
        layers.emplace_back("VK_LAYER_KHRONOS_validation");
    }

    // Enable base features
    vk::PhysicalDeviceFeatures baseFeatures = {};
    baseFeatures.shaderInt64 = VK_TRUE;
    baseFeatures.shaderFloat64 = VK_TRUE;

    // Create the features2 wrapper
    vk::PhysicalDeviceFeatures2 features2 = {};
    features2.setFeatures(baseFeatures);

    return makeDevice(layers, extensions, &features2);
}

} // namespace

/*******************************************************************************
 * Test cases
 *******************************************************************************/
TEST_F(MLEmulationLayerForVulkan, EnumerateLayers) {
    vk::raii::Context ctx{};

    const auto layerProperties = ctx.enumerateInstanceLayerProperties();

    for (auto &property : layerProperties) {
        std::cout << "name=" << property.layerName << ", description=" << property.description << std::endl;
    }
}

TEST_F(MLEmulationLayerForVulkan, EnumerateInstanceExtensions) {
    vk::raii::Context ctx{};

    const auto extensionProperties = ctx.enumerateInstanceExtensionProperties();

    for (auto &property : extensionProperties) {
        std::cout << "name=" << property.extensionName << std::endl;
    }
}

TEST_F(MLEmulationLayerForVulkan, CreateInstance) {
    vk::raii::Context ctx{};
    [[maybe_unused]] auto instance = createInstance(ctx, {"VK_LAYER_ML_Tensor_Emulation"});
}

TEST_F(MLEmulationLayerForVulkan, EnumeratePhysicalDevices) {
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
        for (auto &property : extensionProperties) {
            std::cout << "    name=" << property.extensionName << std::endl;
        }
    }
}

TEST_F(MLEmulationLayerForVulkan, CreateDevice) {
    vk::raii::Context ctx{};
    auto instance = createInstance(ctx, {"VK_LAYER_ML_Tensor_Emulation"});
    auto [device, physicalDevice] = createDevice(instance, {"VK_LAYER_ML_Tensor_Emulation"});
}

TEST_F(MLEmulationLayerForVulkan, CheckTensorFeature) {
    vk::raii::Context ctx{};
    auto instance = createInstance(ctx, {"VK_LAYER_ML_Tensor_Emulation"});
    auto [device, physicalDevice] = createDevice(instance, {"VK_LAYER_ML_Tensor_Emulation"});
    auto features = physicalDevice.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceTensorFeaturesARM>();
    const auto &tensorFeature = features.template get<vk::PhysicalDeviceTensorFeaturesARM>();
    if (!tensorFeature.shaderTensorAccess) {
        throw std::runtime_error("shaderTensorAccess not supported!");
    }
}

TEST_F(MLEmulationLayerForVulkan, CreateTensor) {
    vk::raii::Context ctx{};
    auto instance = createInstance(ctx, {"VK_LAYER_ML_Tensor_Emulation"});
    auto [device, physicalDevice] =
        createDevice(instance, {"VK_LAYER_ML_Tensor_Emulation"}, {VK_ARM_TENSORS_EXTENSION_NAME});

    vk::raii::TensorARM tensor = createTensor(device, {1, 32, 32, 3}, {});
    vk::raii::DeviceMemory memory = allocateTensorMemory(device, physicalDevice, tensor);
    bindTensor(device, tensor, memory);

    [[maybe_unused]] auto tensorView = createTensorView(device, tensor, vk::Format::eR8Sint);
}

TEST_F(MLEmulationLayerForVulkan, MaxPool2D) {
    auto device = createDevice();

    auto inputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 16, 16, 3}});
    auto outputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 8, 8, 3}});
    const GraphPipeline::DescriptorMap descriptorMap = {
        {
            // set 0
            {
                0,                          // binding
                {inputTensor, inputTensor}, // tensor
            },
            {
                1,              // binding
                {outputTensor}, // tensor
            },
        },
    };
    const auto spirv = assembleSpirv(fileToString("maxpool.spvasm"));
    auto graphPipeline = std::make_shared<GraphPipeline>(device, descriptorMap, GraphConstants{}, spirv);

    for (size_t i = 0; i < inputTensor->size(); i += 3) {
        *(inputTensor->data() + i) = uint8_t(-128 + static_cast<int>(i / 3));
    }

    graphPipeline->dispatchSubmit();

    std::cout << "INPUT" << std::endl;
    inputTensor->print();

    std::cout << "OUTPUT" << std::endl;
    outputTensor->print();

    const uint8_t ref[] = {
        0x91, 0x00, 0x00, 0x93, 0x00, 0x00, 0x95, 0x00, 0x00, 0x97, 0x00, 0x00, 0x99, 0x00, 0x00, 0x9b, 0x00, 0x00,
        0x9d, 0x00, 0x00, 0x9f, 0x00, 0x00, 0xb1, 0x00, 0x00, 0xb3, 0x00, 0x00, 0xb5, 0x00, 0x00, 0xb7, 0x00, 0x00,
        0xb9, 0x00, 0x00, 0xbb, 0x00, 0x00, 0xbd, 0x00, 0x00, 0xbf, 0x00, 0x00, 0xd1, 0x00, 0x00, 0xd3, 0x00, 0x00,
        0xd5, 0x00, 0x00, 0xd7, 0x00, 0x00, 0xd9, 0x00, 0x00, 0xdb, 0x00, 0x00, 0xdd, 0x00, 0x00, 0xdf, 0x00, 0x00,
        0xf1, 0x00, 0x00, 0xf3, 0x00, 0x00, 0xf5, 0x00, 0x00, 0xf7, 0x00, 0x00, 0xf9, 0x00, 0x00, 0xfb, 0x00, 0x00,
        0xfd, 0x00, 0x00, 0xff, 0x00, 0x00, 0x11, 0x00, 0x00, 0x13, 0x00, 0x00, 0x15, 0x00, 0x00, 0x17, 0x00, 0x00,
        0x19, 0x00, 0x00, 0x1b, 0x00, 0x00, 0x1d, 0x00, 0x00, 0x1f, 0x00, 0x00, 0x31, 0x00, 0x00, 0x33, 0x00, 0x00,
        0x35, 0x00, 0x00, 0x37, 0x00, 0x00, 0x39, 0x00, 0x00, 0x3b, 0x00, 0x00, 0x3d, 0x00, 0x00, 0x3f, 0x00, 0x00,
        0x51, 0x00, 0x00, 0x53, 0x00, 0x00, 0x55, 0x00, 0x00, 0x57, 0x00, 0x00, 0x59, 0x00, 0x00, 0x5b, 0x00, 0x00,
        0x5d, 0x00, 0x00, 0x5f, 0x00, 0x00, 0x71, 0x00, 0x00, 0x73, 0x00, 0x00, 0x75, 0x00, 0x00, 0x77, 0x00, 0x00,
        0x79, 0x00, 0x00, 0x7b, 0x00, 0x00, 0x7d, 0x00, 0x00, 0x7f, 0x00, 0x00};

    if (!outputTensor->compare(reinterpret_cast<const int8_t *>(&ref[0]), sizeof(ref))) {
        throw std::runtime_error("Output mismatch");
    }
}

TEST_F(MLEmulationLayerForVulkan, TwoLayerMaxPool2D) {
    auto device = createDevice();

    auto inputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 16, 16, 3}});
    auto outputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 4, 4, 3}});
    const GraphPipeline::DescriptorMap descriptorMap = {
        {
            // set 0
            {
                0,             // binding
                {inputTensor}, // tensor
            },
            {
                1,              // binding
                {outputTensor}, // tensor
            },
        },
    };
    const auto spirv = assembleSpirv(fileToString("twolayer-maxpool.spvasm"));
    auto graphPipeline = std::make_shared<GraphPipeline>(device, descriptorMap, GraphConstants{}, spirv);

    for (size_t i = 0; i < inputTensor->size(); i += 3) {
        *(inputTensor->data() + i) = uint8_t(-128 + static_cast<int>(i / 3));
    }

    graphPipeline->dispatchSubmit();

    std::cout << "INPUT" << std::endl;
    inputTensor->print();

    std::cout << "OUTPUT" << std::endl;
    outputTensor->print();

    const uint8_t ref[] = {
        0xb3, 0x00, 0x00, 0xb7, 0x00, 0x00, 0xbb, 0x00, 0x00, 0xbf, 0x00, 0x00, 0xf3, 0x00, 0x00, 0xf7,
        0x00, 0x00, 0xfb, 0x00, 0x00, 0xff, 0x00, 0x00, 0x33, 0x00, 0x00, 0x37, 0x00, 0x00, 0x3b, 0x00,
        0x00, 0x3f, 0x00, 0x00, 0x73, 0x00, 0x00, 0x77, 0x00, 0x00, 0x7b, 0x00, 0x00, 0x7f, 0x00, 0x00,
    };

    if (!outputTensor->compare(reinterpret_cast<const int8_t *>(&ref[0]), sizeof(ref))) {
        throw std::runtime_error("Output mismatch");
    }
}

TEST_F(MLEmulationLayerForVulkan, SamePipelineLayout) {
    auto device = createDevice();

    auto inputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 16, 16, 3}});
    auto outputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 8, 8, 3}});
    const GraphPipeline::DescriptorMap descriptorMap = {
        {
            // set 0
            {
                0,                          // binding
                {inputTensor, inputTensor}, // tensor
            },
            {
                1,              // binding
                {outputTensor}, // tensor
            },
        },
    };
    const auto spirv = assembleSpirv(fileToString("maxpool.spvasm"));
    auto graphPipeline = std::make_shared<GraphPipeline>(device, descriptorMap, GraphConstants{}, spirv);

    auto outputTensor1 = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 8, 8, 3}});
    const GraphPipeline::DescriptorMap descriptorMap1 = {
        {
            // set 0
            {
                0,                          // binding
                {inputTensor, inputTensor}, // tensor
            },
            {
                1,               // binding
                {outputTensor1}, // tensor
            },
        },
    };

    auto graphPipeline1 = std::make_shared<GraphPipeline>(device, descriptorMap1, graphPipeline->getPipelineLayout(),
                                                          GraphConstants{}, spirv);

    for (size_t i = 0; i < inputTensor->size(); i += 3) {
        *(inputTensor->data() + i) = uint8_t(-128 + static_cast<int>(i / 3));
    }

    graphPipeline->dispatchSubmit();
    graphPipeline1->dispatchSubmit();

    std::cout << "INPUT" << std::endl;
    inputTensor->print();

    std::cout << "OUTPUT" << std::endl;
    outputTensor->print();

    std::cout << "OUTPUT1" << std::endl;
    outputTensor1->print();

    const uint8_t ref[] = {
        0x91, 0x00, 0x00, 0x93, 0x00, 0x00, 0x95, 0x00, 0x00, 0x97, 0x00, 0x00, 0x99, 0x00, 0x00, 0x9b, 0x00, 0x00,
        0x9d, 0x00, 0x00, 0x9f, 0x00, 0x00, 0xb1, 0x00, 0x00, 0xb3, 0x00, 0x00, 0xb5, 0x00, 0x00, 0xb7, 0x00, 0x00,
        0xb9, 0x00, 0x00, 0xbb, 0x00, 0x00, 0xbd, 0x00, 0x00, 0xbf, 0x00, 0x00, 0xd1, 0x00, 0x00, 0xd3, 0x00, 0x00,
        0xd5, 0x00, 0x00, 0xd7, 0x00, 0x00, 0xd9, 0x00, 0x00, 0xdb, 0x00, 0x00, 0xdd, 0x00, 0x00, 0xdf, 0x00, 0x00,
        0xf1, 0x00, 0x00, 0xf3, 0x00, 0x00, 0xf5, 0x00, 0x00, 0xf7, 0x00, 0x00, 0xf9, 0x00, 0x00, 0xfb, 0x00, 0x00,
        0xfd, 0x00, 0x00, 0xff, 0x00, 0x00, 0x11, 0x00, 0x00, 0x13, 0x00, 0x00, 0x15, 0x00, 0x00, 0x17, 0x00, 0x00,
        0x19, 0x00, 0x00, 0x1b, 0x00, 0x00, 0x1d, 0x00, 0x00, 0x1f, 0x00, 0x00, 0x31, 0x00, 0x00, 0x33, 0x00, 0x00,
        0x35, 0x00, 0x00, 0x37, 0x00, 0x00, 0x39, 0x00, 0x00, 0x3b, 0x00, 0x00, 0x3d, 0x00, 0x00, 0x3f, 0x00, 0x00,
        0x51, 0x00, 0x00, 0x53, 0x00, 0x00, 0x55, 0x00, 0x00, 0x57, 0x00, 0x00, 0x59, 0x00, 0x00, 0x5b, 0x00, 0x00,
        0x5d, 0x00, 0x00, 0x5f, 0x00, 0x00, 0x71, 0x00, 0x00, 0x73, 0x00, 0x00, 0x75, 0x00, 0x00, 0x77, 0x00, 0x00,
        0x79, 0x00, 0x00, 0x7b, 0x00, 0x00, 0x7d, 0x00, 0x00, 0x7f, 0x00, 0x00};

    if (!outputTensor->compare(reinterpret_cast<const int8_t *>(&ref[0]), sizeof(ref))) {
        throw std::runtime_error("Output mismatch");
    }

    if (!outputTensor1->compare(reinterpret_cast<const int8_t *>(&ref[0]), sizeof(ref))) {
        throw std::runtime_error("Output mismatch");
    }
}

TEST_F(MLEmulationLayerForVulkan, UpdateAfterDispatch) {
    auto device = createDevice();

    auto inputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 16, 16, 3}});
    auto outputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 8, 8, 3}});

    const GraphPipeline::DescriptorMap descriptorMap = {
        {
            // set 0
            {
                0,                          // binding
                {inputTensor, inputTensor}, // tensor
            },
            {
                1,              // binding
                {outputTensor}, // tensor
            },
        },
    };

    for (size_t i = 0; i < inputTensor->size(); i += 3) {
        *(inputTensor->data() + i) = uint8_t(-128 + static_cast<int>(i / 3));
    }

    const auto spirv = assembleSpirv(fileToString("maxpool.spvasm"));
    auto graphPipeline = std::make_shared<GraphPipeline>(device, descriptorMap, GraphConstants{}, spirv);

    graphPipeline->dispatchUpdateSubmit();

    std::cout << "INPUT" << std::endl;
    inputTensor->print();

    std::cout << "OUTPUT" << std::endl;
    outputTensor->print();

    const uint8_t ref[] = {
        0x91, 0x00, 0x00, 0x93, 0x00, 0x00, 0x95, 0x00, 0x00, 0x97, 0x00, 0x00, 0x99, 0x00, 0x00, 0x9b, 0x00, 0x00,
        0x9d, 0x00, 0x00, 0x9f, 0x00, 0x00, 0xb1, 0x00, 0x00, 0xb3, 0x00, 0x00, 0xb5, 0x00, 0x00, 0xb7, 0x00, 0x00,
        0xb9, 0x00, 0x00, 0xbb, 0x00, 0x00, 0xbd, 0x00, 0x00, 0xbf, 0x00, 0x00, 0xd1, 0x00, 0x00, 0xd3, 0x00, 0x00,
        0xd5, 0x00, 0x00, 0xd7, 0x00, 0x00, 0xd9, 0x00, 0x00, 0xdb, 0x00, 0x00, 0xdd, 0x00, 0x00, 0xdf, 0x00, 0x00,
        0xf1, 0x00, 0x00, 0xf3, 0x00, 0x00, 0xf5, 0x00, 0x00, 0xf7, 0x00, 0x00, 0xf9, 0x00, 0x00, 0xfb, 0x00, 0x00,
        0xfd, 0x00, 0x00, 0xff, 0x00, 0x00, 0x11, 0x00, 0x00, 0x13, 0x00, 0x00, 0x15, 0x00, 0x00, 0x17, 0x00, 0x00,
        0x19, 0x00, 0x00, 0x1b, 0x00, 0x00, 0x1d, 0x00, 0x00, 0x1f, 0x00, 0x00, 0x31, 0x00, 0x00, 0x33, 0x00, 0x00,
        0x35, 0x00, 0x00, 0x37, 0x00, 0x00, 0x39, 0x00, 0x00, 0x3b, 0x00, 0x00, 0x3d, 0x00, 0x00, 0x3f, 0x00, 0x00,
        0x51, 0x00, 0x00, 0x53, 0x00, 0x00, 0x55, 0x00, 0x00, 0x57, 0x00, 0x00, 0x59, 0x00, 0x00, 0x5b, 0x00, 0x00,
        0x5d, 0x00, 0x00, 0x5f, 0x00, 0x00, 0x71, 0x00, 0x00, 0x73, 0x00, 0x00, 0x75, 0x00, 0x00, 0x77, 0x00, 0x00,
        0x79, 0x00, 0x00, 0x7b, 0x00, 0x00, 0x7d, 0x00, 0x00, 0x7f, 0x00, 0x00};

    if (!outputTensor->compare(reinterpret_cast<const int8_t *>(&ref[0]), sizeof(ref))) {
        throw std::runtime_error("Output mismatch");
    }
}

TEST_F(MLEmulationLayerForVulkan, SequentialDispatch) {
    auto device = createDevice();

    auto inputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 16, 16, 3}});
    auto outputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 8, 8, 3}});
    auto outputTensor1 = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 8, 8, 3}});

    const GraphPipeline::DescriptorMap descriptorMap = {
        {
            // set 0
            {
                0,                          // binding
                {inputTensor, inputTensor}, // tensor
            },
            {
                1,              // binding
                {outputTensor}, // tensor
            },
        },
    };

    const GraphPipeline::DescriptorMap descriptorMap1 = {
        {
            // set 0
            {
                0,                          // binding
                {inputTensor, inputTensor}, // tensor
            },
            {
                1,               // binding
                {outputTensor1}, // tensor
            },
        },
    };

    for (size_t i = 0; i < inputTensor->size(); i += 3) {
        *(inputTensor->data() + i) = uint8_t(-128 + static_cast<int>(i / 3));
    }

    const auto spirv = assembleSpirv(fileToString("maxpool.spvasm"));
    auto graphPipeline = std::make_shared<GraphPipeline>(device, descriptorMap, GraphConstants{}, spirv);

    auto commandBuffer = graphPipeline->createCommandBuffer();
    auto [descriptorPool, descriptorSets] = graphPipeline->createDescriptorSets(descriptorMap);
    auto [descriptorPool1, descriptorSets1] = graphPipeline->createDescriptorSets(descriptorMap1);

    commandBuffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    graphPipeline->dispatch(commandBuffer, descriptorSets);
    graphPipeline->dispatch(commandBuffer, descriptorSets1);
    commandBuffer.end();

    graphPipeline->submitWork(commandBuffer);

    std::cout << "INPUT" << std::endl;
    inputTensor->print();

    std::cout << "OUTPUT" << std::endl;
    outputTensor->print();

    std::cout << "OUTPUT1" << std::endl;
    outputTensor1->print();

    const uint8_t ref[] = {
        0x91, 0x00, 0x00, 0x93, 0x00, 0x00, 0x95, 0x00, 0x00, 0x97, 0x00, 0x00, 0x99, 0x00, 0x00, 0x9b, 0x00, 0x00,
        0x9d, 0x00, 0x00, 0x9f, 0x00, 0x00, 0xb1, 0x00, 0x00, 0xb3, 0x00, 0x00, 0xb5, 0x00, 0x00, 0xb7, 0x00, 0x00,
        0xb9, 0x00, 0x00, 0xbb, 0x00, 0x00, 0xbd, 0x00, 0x00, 0xbf, 0x00, 0x00, 0xd1, 0x00, 0x00, 0xd3, 0x00, 0x00,
        0xd5, 0x00, 0x00, 0xd7, 0x00, 0x00, 0xd9, 0x00, 0x00, 0xdb, 0x00, 0x00, 0xdd, 0x00, 0x00, 0xdf, 0x00, 0x00,
        0xf1, 0x00, 0x00, 0xf3, 0x00, 0x00, 0xf5, 0x00, 0x00, 0xf7, 0x00, 0x00, 0xf9, 0x00, 0x00, 0xfb, 0x00, 0x00,
        0xfd, 0x00, 0x00, 0xff, 0x00, 0x00, 0x11, 0x00, 0x00, 0x13, 0x00, 0x00, 0x15, 0x00, 0x00, 0x17, 0x00, 0x00,
        0x19, 0x00, 0x00, 0x1b, 0x00, 0x00, 0x1d, 0x00, 0x00, 0x1f, 0x00, 0x00, 0x31, 0x00, 0x00, 0x33, 0x00, 0x00,
        0x35, 0x00, 0x00, 0x37, 0x00, 0x00, 0x39, 0x00, 0x00, 0x3b, 0x00, 0x00, 0x3d, 0x00, 0x00, 0x3f, 0x00, 0x00,
        0x51, 0x00, 0x00, 0x53, 0x00, 0x00, 0x55, 0x00, 0x00, 0x57, 0x00, 0x00, 0x59, 0x00, 0x00, 0x5b, 0x00, 0x00,
        0x5d, 0x00, 0x00, 0x5f, 0x00, 0x00, 0x71, 0x00, 0x00, 0x73, 0x00, 0x00, 0x75, 0x00, 0x00, 0x77, 0x00, 0x00,
        0x79, 0x00, 0x00, 0x7b, 0x00, 0x00, 0x7d, 0x00, 0x00, 0x7f, 0x00, 0x00};

    if (!outputTensor->compare(reinterpret_cast<const int8_t *>(&ref[0]), sizeof(ref))) {
        throw std::runtime_error("Output mismatch");
    }

    if (!outputTensor1->compare(reinterpret_cast<const int8_t *>(&ref[0]), sizeof(ref))) {
        throw std::runtime_error("Output mismatch");
    }
}

TEST_F(MLEmulationLayerForVulkan, Conv2D) {
    auto device = createDevice();

    auto inputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 8, 8, 3}});
    std::iota(inputTensor->data(), inputTensor->data() + inputTensor->size(), uint8_t{});

    auto weightTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{3, 2, 2, 3}});
    std::fill(weightTensor->data(), weightTensor->data() + weightTensor->size(), 1);

    auto biasTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR32Sint, std::vector<int64_t>{3}});
    for (size_t i = 0; i < biasTensor->size(); i++) {
        if ((i % 4) == 0) {
            *(biasTensor->data() + i) = uint8_t(i / 4);
        } else {
            *(biasTensor->data() + i) = 0;
        }
    }

    auto outputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR32Sint, std::vector<int64_t>{1, 4, 4, 3}});
    const GraphPipeline::DescriptorMap descriptorMap = {
        {
            // set 0
            {
                0,             // binding
                {inputTensor}, // tensor
            },
            {
                1,              // binding
                {weightTensor}, // tensor
            },
            {
                2,            // binding
                {biasTensor}, // tensor
            },
            {
                3,              // binding
                {outputTensor}, // tensor
            },
        },
    };

    GraphConstants graphConstants;

    const auto spirv = assembleSpirv(fileToString("conv2d.spvasm"));
    auto graphPipeline = std::make_shared<GraphPipeline>(device, descriptorMap, graphConstants, spirv);

    graphPipeline->dispatchSubmit();

    std::cout << "INPUT" << std::endl;
    inputTensor->print();

    std::cout << "OUTPUT" << std::endl;
    outputTensor->print();

    const uint32_t ref[1][4][4][3] = {
        // batch
        {
            // y=0
            {
                //      c=0         c=1         c=2        b  y  x  c
                {0x000000ae, 0x000000af, 0x000000b0}, // (0, 0, 0, c)
                {0x000000f6, 0x000000f7, 0x000000f8}, // (0, 0, 1, c)
                {0x0000013e, 0x0000013f, 0x00000140}, // (0, 0, 2, c)
                {0x00000186, 0x00000187, 0x00000188}, // (0, 0, 3, c)
            },
            // y=1
            {
                {0x000002ee, 0x000002ef, 0x000002f0}, // (0, 1, 0, c)
                {0x00000336, 0x00000337, 0x00000338}, // (0, 1, 1, c)
                {0x0000037e, 0x0000037f, 0x00000380}, // (0, 1, 2, c)
                {0x000003c6, 0x000003c7, 0x000003c8}, // (0, 1, 3, c)
            },
            // y=2
            {
                {0x0000052e, 0x0000052f, 0x00000530}, // (0, 2, 0, c)
                {0x00000176, 0x00000177, 0x00000178}, // (0, 2, 1, c)
                {0xffffffbe, 0xffffffbf, 0xffffffc0}, // (0, 2, 2, c)
                {0x00000006, 0x00000007, 0x00000008}, // (0, 2, 3, c)
            },
            // y=3
            {
                {0xfffffb6e, 0xfffffb6f, 0xfffffb70}, // (0, 3, 0, c)
                {0xfffffbb6, 0xfffffbb7, 0xfffffbb8}, // (0, 3, 1, c)
                {0xfffffbfe, 0xfffffbff, 0xfffffc00}, // (0, 3, 2, c)
                {0xfffffc46, 0xfffffc47, 0xfffffc48}, // (0, 3, 3, c)
            },
        },
    };

    if (!outputTensor->compare(reinterpret_cast<const int32_t *>(&ref[0][0][0][0]), sizeof(ref))) {
        throw std::runtime_error("Output mismatch");
    }
}

TEST_F(MLEmulationLayerForVulkan, Concat) {
    auto device = createDevice();

    auto inputTensor0 = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 2, 2, 2}});
    auto inputTensor1 = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 2, 2, 2}});
    auto outputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 4, 2, 2}});
    const GraphPipeline::DescriptorMap descriptorMap = {
        {
            // set 0
            {
                0,              // binding
                {inputTensor0}, // tensor
            },
            {
                1,              // binding
                {inputTensor1}, // tensor
            },
            {
                2,              // binding
                {outputTensor}, // tensor
            },
        },
    };

    std::iota(inputTensor0->data(), inputTensor0->data() + inputTensor0->size(), uint8_t{});

    std::iota(inputTensor1->data(), inputTensor1->data() + inputTensor1->size(), uint8_t{});

    const auto spirv = assembleSpirv(fileToString("concat.spvasm"));
    auto graphPipeline = std::make_shared<GraphPipeline>(device, descriptorMap, GraphConstants{}, spirv);

    graphPipeline->dispatchSubmit();

    std::cout << "INPUT" << std::endl;
    inputTensor0->print();

    std::cout << "INPUT" << std::endl;
    inputTensor1->print();

    std::cout << "OUTPUT" << std::endl;
    outputTensor->print();

    const uint8_t ref[1][4][2][2] = {
        // batch
        {
            // height
            {
                // width
                {0x00, 0x01}, // channel
                {0x02, 0x03}, // channel
            },
            {
                // width
                {0x04, 0x05}, // channel
                {0x06, 0x07}, // channel
            },
            // height
            {
                // width
                {0x00, 0x01}, // channel
                {0x02, 0x03}, // channel
            },
            {
                // width
                {0x04, 0x05}, // channel
                {0x06, 0x07}, // channel
            },
        },
    };

    if (!outputTensor->compare(&ref[0][0][0][0], sizeof(ref))) {
        throw std::runtime_error("Output mismatch");
    }
}

TEST_F(MLEmulationLayerForVulkan, GraphConstantARM) {
    auto device = createDevice();

    GraphConstants graphConstants;

    graphConstants.makeGraphPipelineConstantTensor(0, Shape{vk::Format::eR32Sint, {1}}, std::vector<int32_t>{2});
    graphConstants.makeGraphPipelineConstantTensor(1, Shape{vk::Format::eR8Sint, {1}}, std::vector<int8_t>{0});

    graphConstants.makeGraphPipelineConstantTensor(2, Shape{vk::Format::eR32Sint, {1}}, std::vector<int32_t>{3});
    graphConstants.makeGraphPipelineConstantTensor(3, Shape{vk::Format::eR8Sint, {1}}, std::vector<int8_t>{0});

    graphConstants.makeGraphPipelineConstantTensor(4, Shape{vk::Format::eR32Sint, {1}}, std::vector<int32_t>{4});
    graphConstants.makeGraphPipelineConstantTensor(5, Shape{vk::Format::eR8Sint, {1}}, std::vector<int8_t>{0});

    auto inputTensor0 = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{5}});
    auto inputTensor1 = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{5}});
    auto outputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{5}});
    const GraphPipeline::DescriptorMap descriptorMap = {
        {
            // set 0
            {
                0,              // binding
                {inputTensor0}, // tensor
            },
            {
                1,              // binding
                {inputTensor1}, // tensor
            },
            {
                2,              // binding
                {outputTensor}, // tensor
            },
        },
    };

    std::iota(inputTensor0->data(), inputTensor0->data() + inputTensor0->size(), uint8_t{});
    std::iota(inputTensor1->data(), inputTensor1->data() + inputTensor1->size(), uint8_t{});

    const auto spirv = assembleSpirv(fileToString("graph-constant-arm.spvasm"));
    auto graphPipeline = std::make_shared<GraphPipeline>(device, descriptorMap, graphConstants, spirv);

    graphPipeline->dispatchSubmit();

    std::cout << "INPUT" << std::endl;
    inputTensor0->print();

    std::cout << "INPUT" << std::endl;
    inputTensor1->print();

    std::cout << "OUTPUT" << std::endl;
    outputTensor->print();

    const uint8_t ref[5] = {0, 20, 40, 60, 80};

    if (!outputTensor->compare(&ref[0], sizeof(ref))) {
        throw std::runtime_error("Output mismatch");
    }
}

TEST_F(MLEmulationLayerForVulkan, Maximum) {
    auto device = createDevice();

    auto inputTensor0 = std::make_shared<Tensor>(device, Shape{vk::Format::eR32Sint, std::vector<int64_t>{1, 2, 2, 2}});
    auto inputTensor1 = std::make_shared<Tensor>(device, Shape{vk::Format::eR32Sint, std::vector<int64_t>{1, 2, 2, 2}});
    auto outputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR32Sint, std::vector<int64_t>{1, 2, 2, 2}});
    const GraphPipeline::DescriptorMap descriptorMap = {
        {
            // set 0
            {
                0,              // binding
                {inputTensor0}, // tensor
            },
            {
                1,              // binding
                {inputTensor1}, // tensor
            },
            {
                2,              // binding
                {outputTensor}, // tensor
            },
        },
    };

    for (size_t i = 0; i < (inputTensor0->size() / sizeof(int32_t)); i++) {
        *(reinterpret_cast<uint32_t *>(inputTensor0->data()) + i) = uint32_t(i);
    }

    for (size_t i = 0; i < (inputTensor1->size() / sizeof(int32_t)); i++) {
        *(reinterpret_cast<uint32_t *>(inputTensor1->data()) + i) = static_cast<uint32_t>(-16) + uint32_t(i * 4);
    }

    const auto spirv = assembleSpirv(fileToString("maximum.spvasm"));
    auto graphPipeline = std::make_shared<GraphPipeline>(device, descriptorMap, GraphConstants{}, spirv);

    graphPipeline->dispatchSubmit();

    std::cout << "INPUT" << std::endl;
    inputTensor0->print();

    std::cout << "INPUT" << std::endl;
    inputTensor1->print();

    std::cout << "OUTPUT" << std::endl;
    outputTensor->print();

    const int32_t ref[1][2][2][2] = {
        // batch
        {
            // height
            {
                // width
                {0x00, 0x01}, // channel
                {0x02, 0x03}, // channel
            },
            {
                // width
                {0x04, 0x05}, // channel
                {0x08, 0x0c}, // channel
            },
        },
    };

    if (!outputTensor->compare(&ref[0][0][0][0], sizeof(ref))) {
        throw std::runtime_error("Output mismatch");
    }
}

TEST_F(MLEmulationLayerForVulkan, Slice) {
    auto device = createDevice();

    auto inputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 8, 8, 3}});
    auto outputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 2, 2, 2}});
    const GraphPipeline::DescriptorMap descriptorMap = {
        {
            // set 0
            {
                0,             // binding
                {inputTensor}, // tensor
            },
            {
                1,              // binding
                {outputTensor}, // tensor
            },
        },
    };

    std::iota(inputTensor->data(), inputTensor->data() + inputTensor->size(), uint8_t{});

    const auto spirv = assembleSpirv(fileToString("slice.spvasm"));
    auto graphPipeline = std::make_shared<GraphPipeline>(device, descriptorMap, GraphConstants{}, spirv);

    graphPipeline->dispatchSubmit();

    const uint8_t ref[1][2][2][2] = {
        // batch
        {
            // height
            {
                // width
                {0x6d, 0x6e}, // channel
                {0x70, 0x71}, // channel
            },
            {
                // width
                {0x85, 0x86}, // channel
                {0x88, 0x89}, // channel
            },
        },
    };

    std::cout << "INPUT" << std::endl;
    inputTensor->print();

    std::cout << "OUTPUT" << std::endl;
    outputTensor->print();

    if (!outputTensor->compare(reinterpret_cast<const int8_t *>(&ref[0][0][0][0]), sizeof(ref))) {
        throw std::runtime_error("Output mismatch");
    }
}

TEST_F(MLEmulationLayerForVulkan, FFT2D) {
    auto device = createDevice();

    auto inputTensor0 =
        std::make_shared<Tensor>(device, Shape{vk::Format::eR32Sfloat, std::vector<int64_t>{1, 4, 4096}});
    auto inputTensor1 =
        std::make_shared<Tensor>(device, Shape{vk::Format::eR32Sfloat, std::vector<int64_t>{1, 4, 4096}});
    auto outputTensor0 =
        std::make_shared<Tensor>(device, Shape{vk::Format::eR32Sfloat, std::vector<int64_t>{1, 4, 4096}});
    auto outputTensor1 =
        std::make_shared<Tensor>(device, Shape{vk::Format::eR32Sfloat, std::vector<int64_t>{1, 4, 4096}});
    const GraphPipeline::DescriptorMap descriptorMap = {
        {
            // set 0
            {
                0,              // binding
                {inputTensor0}, // tensor
            },
            {
                1,              // binding
                {inputTensor1}, // tensor
            },
            {
                2,               // binding
                {outputTensor0}, // tensor
            },
            {
                3,               // binding
                {outputTensor1}, // tensor
            },
        },
    };
    const auto spirv = assembleSpirv(fileToString("fft2d.spvasm"));
    auto graphPipeline = std::make_shared<GraphPipeline>(device, descriptorMap, GraphConstants{}, spirv);

    graphPipeline->dispatchSubmit();

    std::cout << "INPUT 0" << std::endl;
    inputTensor0->print();

    std::cout << "INPUT 1" << std::endl;
    inputTensor1->print();

    std::cout << "OUTPUT 0" << std::endl;
    outputTensor0->print();

    std::cout << "OUTPUT 1" << std::endl;
    outputTensor1->print();
}

TEST_F(MLEmulationLayerForVulkan, CreateTensorComputeShader) {
    auto device = createDevice();

    const auto spirvModule = mlsdk::el::utils::glslToSpirv(fileToString("tensor_all_access.comp"));
    [[maybe_unused]] const auto shaderModule = createShaderModule((&(*device)), spirvModule);
}

TEST_F(MLEmulationLayerForVulkan, TensorArray) {
    auto device = createDevice();

    const auto spirvModule = mlsdk::el::utils::glslToSpirv(fileToString("tensor_array.comp"));
    [[maybe_unused]] const auto shaderModule = createShaderModule((&(*device)), spirvModule);
    std::vector<std::shared_ptr<Tensor>> inputTensors;
    std::vector<std::shared_ptr<Tensor>> outputTensors;
    for ([[maybe_unused]] auto _ : {1, 2, 3, 4}) {
        auto inputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{4}});
        inputTensors.push_back(std::move(inputTensor));
    }
    for ([[maybe_unused]] auto _ : {1, 2, 3, 4}) {
        auto outputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{4}});
        outputTensors.push_back(std::move(outputTensor));
    }

    const TensorComputePipeline::DescriptorMap descriptorMap = {
        {
            // set 0
            {
                0,                                                                    // binding
                {inputTensors[0], inputTensors[1], inputTensors[2], inputTensors[3]}, // tensor
            },
            {
                1,                                                                        // binding
                {outputTensors[0], outputTensors[1], outputTensors[2], outputTensors[3]}, // tensor
            },
        },
    };
    auto computePipeline = std::make_shared<TensorComputePipeline>(device, descriptorMap, spirvModule);

    uint8_t start = 17;
    for (auto &inputTensor : inputTensors) {
        std::iota(inputTensor->data(), inputTensor->data() + inputTensor->size(), start);
        start += uint8_t(inputTensor->size());
    }
    for (auto &outputTensor : outputTensors) {
        std::fill(outputTensor->data(), outputTensor->data() + outputTensor->size(), 0xFF);
    }

    computePipeline->dispatchSubmit(16, 1, 1);

    for (auto &inputTensor : inputTensors) {
        std::cout << "INPUT" << std::endl;
        inputTensor->print();
    }
    for (auto &outputTensor : outputTensors) {
        std::cout << "OUTPUT" << std::endl;
        outputTensor->print();
    }

    for (auto i : {0U, 1U, 2U, 3U}) {
        if (!outputTensors[i]->compare(inputTensors[i]->data(), inputTensors[i]->size())) {
            throw std::runtime_error("Output mismatch");
        }
    }
}

TEST_F(MLEmulationLayerForVulkan, CreateTensorComputePipeline) {
    auto device = createDevice();

    const auto spirv = mlsdk::el::utils::glslToSpirv(fileToString("tensor.comp"));
    auto inputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 2, 2, 2}});
    auto outputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 2, 2, 2}});
    const TensorComputePipeline::DescriptorMap descriptorMap = {
        {
            // set 0
            {
                0,             // binding
                {inputTensor}, // tensor
            },
            {
                1,              // binding
                {outputTensor}, // tensor
            },
        },
    };
    auto computePipeline = std::make_shared<TensorComputePipeline>(device, descriptorMap, spirv);

    std::iota(inputTensor->data(), inputTensor->data() + inputTensor->size(), uint8_t{});
    computePipeline->dispatchSubmit(16, 16, 3);

    std::cout << "INPUT" << std::endl;
    inputTensor->print();

    std::cout << "OUTPUT" << std::endl;
    outputTensor->print();

    const uint8_t ref[1][2][2][2] = {
        // batch
        {// height
         {
             // width
             {0x00, 0x01}, // channel
             {0x02, 0x03}, // channel
         },
         {
             // width
             {0x04, 0x05}, // channel
             {0x06, 0x07}, // channel
         }},
    };

    if (!outputTensor->compare(&ref[0][0][0][0], sizeof(ref))) {
        throw std::runtime_error("Output mismatch");
    }
}

TEST_F(MLEmulationLayerForVulkan, LoggerDefaultLogLevelHighSeverity) {
    using namespace mlsdk::el::log;
    Log testLog("VMEL_TEST_SEVERITY", "TestLog");

    testLog(Severity::Debug) << "Serverity is Debug(3)\n";
}

TEST_F(MLEmulationLayerForVulkan, LoggerDefaultLogLevelLowSeverity) {
    using namespace mlsdk::el::log;
    Log testLog("VMEL_TEST_SEVERITY", "TestLog");

    testLog(Severity::Error) << "Serverity is Error(0)\n";
}

TEST_F(MLEmulationLayerForVulkan, LoggerStdFunctions) {
    using namespace mlsdk::el::log;
    Log testLog("VMEL_TEST_SEVERITY", "TestLog");

    testLog(Severity::Error) << "Serverity is Error(0)" << std::endl;
    testLog(Severity::Info) << "Serverity is Info(2)" << std::endl;
    testLog(Severity::Error) << "Serverity is Error(0)" << std::endl;
}

TEST_F(MLEmulationLayerForVulkan, LoggerVectors) {
    using namespace mlsdk::el::log;
    Log testLog("VMEL_TEST_SEVERITY", "TestLog");
    const std::vector<std::string> strVector{"Hello", "World", "!"};
    const std::vector<int> intVector{1, 2, 3, 4, 5};

    testLog(Severity::Error) << strVector << "\n";
    testLog(Severity::Error) << intVector << "\n";
}

TEST_F(MLEmulationLayerForVulkan, LoggerLineNumbers) {
    using namespace mlsdk::el::log;
    Log testLog("VMEL_TEST_SEVERITY", "TestLog");
    const std::string str("Hello world\nThis is line 2\nFinal line");

    testLog(Severity::Error) << StringLineNumber(str) << std::endl;
}

TEST_F(MLEmulationLayerForVulkan, LoggerHexDump) {
    using namespace mlsdk::el::log;
    Log testLog("VMEL_TEST_SEVERITY", "TestLog");
    const uint8_t testchar[]{"Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor "
                             "incididunt ut labore et dolore magna "
                             "aliqua."};
    const auto *charPointer{testchar};

    testLog(Severity::Error) << HexDump(charPointer, sizeof(testchar));
}

TEST_F(MLEmulationLayerForVulkan, NOPOutputs) {
    auto device = createDevice();

    auto inputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 8, 8, 3}});
    auto outputTensor_0 =
        std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 2, 2, 2}});
    auto outputTensor_1 =
        std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 8, 8, 3}});
    auto outputTensor_2 =
        std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 8, 8, 3}});

    const GraphPipeline::DescriptorMap descriptorMap = {
        {
            // set 0
            {
                0,             // binding
                {inputTensor}, // tensor
            },
            {
                1,                // binding
                {outputTensor_0}, // tensor
            },
            {
                2,                // binding
                {outputTensor_1}, // tensor
            },
            {
                3,                // binding
                {outputTensor_2}, // tensor
            },
        },
    };

    std::vector<int8_t> constTensorData(192);
    std::iota(std::begin(constTensorData), std::end(constTensorData), 0);

    GraphConstants graphConstants;
    graphConstants.makeGraphPipelineConstantTensor(0, Shape{vk::Format::eR8Sint, {1, 8, 8, 3}}, constTensorData);

    std::iota(inputTensor->data(), inputTensor->data() + inputTensor->size(), uint8_t{});

    const auto spirv = assembleSpirv(fileToString("nop-outputs.spvasm"));
    auto graphPipeline = std::make_shared<GraphPipeline>(device, descriptorMap, graphConstants, spirv);

    graphPipeline->dispatchSubmit();

    const uint8_t ref[1][2][2][2] = {
        // batch
        {
            // height
            {
                // width
                {0x6d, 0x6e}, // channel
                {0x70, 0x71}, // channel
            },
            {
                // width
                {0x85, 0x86}, // channel
                {0x88, 0x89}, // channel
            },
        },
    };

    std::cout << "INPUT" << std::endl;
    inputTensor->print();

    std::cout << "OUTPUT_0 from Slice" << std::endl;
    outputTensor_0->print();

    std::cout << "OUTPUT_1 from Input" << std::endl;
    outputTensor_1->print();

    std::cout << "CONST TENSOR" << std::endl;
    graphConstants[0].print();

    std::cout << "OUTPUT_2 from Const Tensor" << std::endl;
    outputTensor_2->print();

    if (!outputTensor_0->compare(reinterpret_cast<const int8_t *>(&ref[0][0][0][0]), sizeof(ref))) {
        throw std::runtime_error("Output mismatch in OUTPUT_0");
    }

    if (!outputTensor_1->compare(reinterpret_cast<const int8_t *>(inputTensor->data()), inputTensor->size())) {
        throw std::runtime_error("Output mismatch in OUTPUT_1");
    }

    if (!outputTensor_2->compare(reinterpret_cast<const int8_t *>(graphConstants[0].data()),
                                 graphConstants[0].size())) {
        throw std::runtime_error("Output mismatch in OUTPUT_2");
    }
}

TEST_F(MLEmulationLayerForVulkan, CopyLargeNonPackedTensor) {
    auto device = createDevice();
    const std::vector<int64_t> dimensions{1, 1920, 1080, 3};
    const std::vector<int64_t> strides = {12441600, 6480, 6, 2};

    const std::vector<uint8_t> data;
    const auto useForCopy = true;
    auto inputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, dimensions}, data, useForCopy);
    auto outputTensor =
        std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, dimensions, strides}, data, useForCopy);
    for (size_t i = 0; i < inputTensor->size(); ++i) {
        *(inputTensor->data() + i) = -128 + static_cast<int>(i % 256);
    }
    // command pool
    const vk::CommandPoolCreateInfo commandPoolCreateInfo{
        {},                                                  // flags
        device->getPhysicalDevice()->getComputeFamilyIndex() // queue family index
    };
    auto commandPool = vk::raii::CommandPool(&(*device), commandPoolCreateInfo);

    // command buffer
    const vk::CommandBufferAllocateInfo commandBufferAllocInfo{
        *commandPool,                     // command pool
        vk::CommandBufferLevel::ePrimary, // command buffer level
        1                                 // command buffer count
    };
    vk::raii::CommandBuffers commandBuffers(&(*device), commandBufferAllocInfo);
    auto commandBuffer = std::move(commandBuffers.front());

    // command copy tensor
    const vk::CommandBufferBeginInfo commandBufferBeginInfo{
        vk::CommandBufferUsageFlagBits::eOneTimeSubmit, // flags
    };
    commandBuffer.begin(commandBufferBeginInfo);
    const vk::TensorCopyARM region{static_cast<uint32_t>(dimensions.size())};
    const vk::CopyTensorInfoARM copyInfo{
        &(*inputTensor),  // srcTensor
        &(*outputTensor), // dstTensor
        1,                // regionCount
        &region           // pRegions
    };
    commandBuffer.copyTensorARM(copyInfo);
    commandBuffer.end();

    // submit
    vk::raii::Queue queue(&(*device), device->getPhysicalDevice()->getComputeFamilyIndex(), 0);
    vk::raii::Fence fence(&(*device), vk::FenceCreateInfo());
    const vk::SubmitInfo submitInfo{
        0,                 // wait semaphore count
        nullptr,           // wait semaphore
        nullptr,           // pipeline stage flags
        1,                 // command buffer count
        &(*commandBuffer), // command buffers
        0,                 // signal semaphore count
    };
    auto begin = std::chrono::steady_clock::now();

    queue.submit({1, &submitInfo}, *fence);
    while (vk::Result::eTimeout == (&(*device)).waitForFences({*fence}, vk::True, uint64_t(-1)))
        ;

    auto end = std::chrono::steady_clock::now();

    std::cout << "Runtime " << std::dec << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << " ms" << std::endl;

    if (!outputTensor->stridedCompare<int8_t, int8_t>(*inputTensor)) {
        throw std::runtime_error("Output mismatch");
    }
}

template <typename T> void checkFloat(T v) {
    float8 f8{v};
    float16 f16{v};
    float32 f32{v};
    float64 f64{v};

    EXPECT_NEAR(double(f8), double(v), 0.5);
    ASSERT_EQ(double(f16), v);
    ASSERT_EQ(double(f32), v);
    ASSERT_EQ(double(f64), v);

    v += 1;
    f8 = v;
    f16 = v;
    f32 = v;
    f64 = v;

    EXPECT_NEAR(double(f8), double(v), 0.5);
    ASSERT_EQ(double(f16), v);
    ASSERT_EQ(double(f32), v);
    ASSERT_EQ(double(f64), v);
}

TEST_F(MLEmulationLayerForVulkan, Float) {
    ASSERT_EQ(sizeof(float8), 1);
    ASSERT_EQ(sizeof(float16), 2);
    ASSERT_EQ(sizeof(float32), 4);
    ASSERT_EQ(sizeof(float64), 8);

    checkFloat(float(-10.5));
    checkFloat(float(-0.5));
    checkFloat(float(0));
    checkFloat(float(0.5));
    checkFloat(float(10.0));

    checkFloat(int8_t(1));
    checkFloat(uint8_t(2));
    checkFloat(int16_t(3));
    checkFloat(uint16_t(4));
    checkFloat(int32_t(5));
    checkFloat(uint32_t(6));
    checkFloat(int64_t(7));
    checkFloat(uint64_t(8));

    checkFloat(float8(10.5));
    checkFloat(float16(11.5));
    checkFloat(float32(12.5));
    checkFloat(float64(13.5));

    auto f = float(float16(10.5));
    ASSERT_EQ(double(f), 10.5);

    float16 f16;

    f16 = f16 + 10;
    ASSERT_EQ(double(f16), 10);

    f16 += 10;
    ASSERT_EQ(double(f16), 20);

    f16 = f16 - 5;
    ASSERT_EQ(double(f16), 15);

    f16 -= 5;
    ASSERT_EQ(double(f16), 10);

    f16 = f16 * 2;
    ASSERT_EQ(double(f16), 20);

    f16 *= 2;
    ASSERT_EQ(double(f16), 40);

    f16 = f16 / 4;
    ASSERT_EQ(double(f16), 10);

    f16 /= 4;
    ASSERT_EQ(double(f16), 2.5);

    ASSERT_TRUE(f16 < 5);
    ASSERT_FALSE(5 < f16);

    ASSERT_TRUE(f16 <= 5);
    ASSERT_FALSE(5 <= f16);

    ASSERT_FALSE(f16 > 5);
    ASSERT_TRUE(5 > f16);

    ASSERT_FALSE(f16 >= 5);
    ASSERT_TRUE(5 >= f16);

    ASSERT_FALSE(f16 == 5);
    ASSERT_TRUE(f16 == 2.5);

    ASSERT_TRUE(f16 != 5);
    ASSERT_FALSE(f16 != 2.5);

    uint32_t overflow = 0U << 31 | 250 << 23 | 1 << 22;
    void *overflowPtr = &overflow;
    f16 = *reinterpret_cast<float *>(overflowPtr);
    ASSERT_FALSE(f16.isnan());
    ASSERT_TRUE(f16.isinf());
    ASSERT_FALSE(std::isnan(f16));
    ASSERT_TRUE(std::isinf(f16));
    ASSERT_FALSE(std::isnormal(float(f16)));

    uint32_t nan = 0xffffffff;
    void *nanPtr = &nan;
    f16 = *reinterpret_cast<float *>(nanPtr);
    ASSERT_TRUE(f16.isnan());
    ASSERT_FALSE(f16.isinf());
    ASSERT_TRUE(std::isnan(f16));
    ASSERT_FALSE(std::isinf(f16));
    ASSERT_FALSE(std::isnormal(float(f16)));

    uint32_t pinf = 0U << 31 | 0xffU << 23 | 0;
    void *pinfPtr = &pinf;
    f16 = *reinterpret_cast<float *>(pinfPtr);
    ASSERT_FALSE(f16.isnan());
    ASSERT_TRUE(f16.isinf());
    ASSERT_FALSE(std::isnan(f16));
    ASSERT_TRUE(std::isinf(f16));
    ASSERT_FALSE(std::isnormal(float(f16)));

    uint32_t ninf = 1U << 31 | 0xffU << 23 | 0;
    void *ninfPtr = &ninf;
    f16 = *reinterpret_cast<float *>(ninfPtr);
    ASSERT_FALSE(f16.isnan());
    ASSERT_TRUE(f16.isinf());
    ASSERT_FALSE(std::isnan(f16));
    ASSERT_TRUE(std::isinf(f16));
    ASSERT_FALSE(std::isnormal(float(f16)));
}

TEST_F(MLEmulationLayerForVulkan, MultiSessionsInOneCommandBuffer) {
    auto device = createDevice();

    auto inputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 16, 16, 3}});
    for (size_t i = 0; i < inputTensor->size(); i += 3) {
        *(inputTensor->data() + i) = uint8_t(-128 + static_cast<int>(i / 3));
    }

    std::vector<std::shared_ptr<Tensor>> outputTensors;
    std::vector<GraphPipeline::DescriptorMap> descriptorMaps;
    // First pipeline, 2 sessions. Second pipeline, 3 sessions.
    for ([[maybe_unused]] auto _ : {1, 2, 3, 4, 5}) {
        auto outputTensor =
            std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 4, 4, 3}});
        GraphPipeline::DescriptorMap descriptorMap = {
            {
                // set 0
                {
                    0,             // binding
                    {inputTensor}, // tensor
                },
                {
                    1,              // binding
                    {outputTensor}, // tensor
                },
            },
        };
        outputTensors.emplace_back(std::move(outputTensor));
        descriptorMaps.emplace_back(std::move(descriptorMap));
    }

    const auto spirv = assembleSpirv(fileToString("twolayer-maxpool.spvasm"));

    // Create pipeline, based on first descriptor map
    auto graphPipeline = std::make_shared<GraphPipeline>(device, descriptorMaps[0], GraphConstants{}, spirv);

    // Create pipeline from layout
    auto graphPipeline2 = std::make_shared<GraphPipeline>(device, descriptorMaps[0], graphPipeline->getPipelineLayout(),
                                                          GraphConstants{}, spirv);

    // Create command buffer
    auto commandBuffer = graphPipeline->createCommandBuffer();
    commandBuffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    std::vector<vk::raii::DescriptorPool> vkDescriptorPools;
    std::vector<vk::raii::DescriptorSets> vkDescriptorSets;

    // First pipeline
    auto pipeline = graphPipeline;
    for (auto i : {0U, 1U}) {
        auto [descriptorPool, descriptorSets] = pipeline->createDescriptorSets(descriptorMaps[i]);
        pipeline->dispatch(commandBuffer, descriptorSets);
        vkDescriptorPools.emplace_back(std::move(descriptorPool));
        vkDescriptorSets.emplace_back(std::move(descriptorSets));
    }

    // Second pipeline
    pipeline = graphPipeline2;
    for (auto i : {2U, 3U, 4U}) {
        auto [descriptorPool, descriptorSets] = pipeline->createDescriptorSets(descriptorMaps[i]);
        pipeline->dispatch(commandBuffer, descriptorSets);
        vkDescriptorPools.emplace_back(std::move(descriptorPool));
        vkDescriptorSets.emplace_back(std::move(descriptorSets));
    }

    commandBuffer.end();

    // graphPipeline->printGraphPipelineSessionMemory();
    // graphPipeline2->printGraphPipelineSessionMemory();

    graphPipeline->submitWork(commandBuffer);

    std::cout << "INPUT" << std::endl;
    inputTensor->print();

    std::cout << "OUTPUT" << std::endl;
    outputTensors[0]->print();

    const uint8_t ref[] = {
        0xb3, 0x00, 0x00, 0xb7, 0x00, 0x00, 0xbb, 0x00, 0x00, 0xbf, 0x00, 0x00, 0xf3, 0x00, 0x00, 0xf7,
        0x00, 0x00, 0xfb, 0x00, 0x00, 0xff, 0x00, 0x00, 0x33, 0x00, 0x00, 0x37, 0x00, 0x00, 0x3b, 0x00,
        0x00, 0x3f, 0x00, 0x00, 0x73, 0x00, 0x00, 0x77, 0x00, 0x00, 0x7b, 0x00, 0x00, 0x7f, 0x00, 0x00,
    };

    for (const auto &tensor : outputTensors) {
        if (!tensor->compare(reinterpret_cast<const int8_t *>(&ref[0]), sizeof(ref))) {
            throw std::runtime_error("Output mismatch");
        }
    }
}

TEST_F(MLEmulationLayerForVulkan, MultiSessionsOneAtTheTime) {
    auto device = createDevice();

    auto inputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 16, 16, 3}});
    for (size_t i = 0; i < inputTensor->size(); i += 3) {
        *(inputTensor->data() + i) = uint8_t(-128 + static_cast<int>(i / 3));
    }

    std::cout << "INPUT" << std::endl;
    inputTensor->print();

    const uint8_t ref[] = {
        0xb3, 0x00, 0x00, 0xb7, 0x00, 0x00, 0xbb, 0x00, 0x00, 0xbf, 0x00, 0x00, 0xf3, 0x00, 0x00, 0xf7,
        0x00, 0x00, 0xfb, 0x00, 0x00, 0xff, 0x00, 0x00, 0x33, 0x00, 0x00, 0x37, 0x00, 0x00, 0x3b, 0x00,
        0x00, 0x3f, 0x00, 0x00, 0x73, 0x00, 0x00, 0x77, 0x00, 0x00, 0x7b, 0x00, 0x00, 0x7f, 0x00, 0x00,
    };

    auto outputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 4, 4, 3}});
    const GraphPipeline::DescriptorMap descriptorMapRef = {
        {
            // set 0
            {
                0,             // binding
                {inputTensor}, // tensor
            },
            {
                1,              // binding
                {outputTensor}, // tensor
            },
        },
    };
    const std::vector<GraphPipeline::DescriptorMap> descriptorMaps(2, descriptorMapRef);

    const auto spirv = assembleSpirv(fileToString("twolayer-maxpool.spvasm"));

    // Create pipeline
    auto graphPipeline = std::make_shared<GraphPipeline>(device, descriptorMapRef, GraphConstants{}, spirv);

    // Create second pipeline, based on first layout
    auto graphPipeline2 = std::make_shared<GraphPipeline>(device, descriptorMapRef, graphPipeline->getPipelineLayout(),
                                                          GraphConstants{}, spirv);

    for (auto pipeline : {graphPipeline, graphPipeline2}) {
        for (const auto &descriptorMap : descriptorMaps) {
            // Clear output tensor and any stored sessions
            outputTensor->clear();
            pipeline->clearSessions();

            // create command buffer
            auto commandBuffer = pipeline->createCommandBuffer();

            // create descriptor set
            auto [descriptorPool, descriptorSets] = pipeline->createDescriptorSets(descriptorMap);

            // Dispatch command buffer
            commandBuffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
            pipeline->dispatch(commandBuffer, descriptorSets);
            commandBuffer.end();

            // pipeline->printGraphPipelineSessionMemory();
            pipeline->submitWork(commandBuffer);

            // std::cout << "OUTPUT" << std::endl;
            // outputTensor->print();

            if (!outputTensor->compare(reinterpret_cast<const int8_t *>(&ref[0]), sizeof(ref))) {
                throw std::runtime_error("Output mismatch");
            }

            // descriptor set and pool are destroyed
        }
    }
}

TEST_F(MLEmulationLayerForVulkan, PipelineCreationFeedback) {
    auto device = createDevice();

    auto inputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 16, 16, 3}});
    auto outputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 8, 8, 3}});
    const GraphPipeline::DescriptorMap descriptorMap = {
        {
            // set 0
            {
                0,                          // binding
                {inputTensor, inputTensor}, // tensor
            },
            {
                1,              // binding
                {outputTensor}, // tensor
            },
        },
    };
    const auto spirv = assembleSpirv(fileToString("maxpool.spvasm"));

    std::vector<vk::DataGraphPipelineResourceInfoARM> graphPipelineResourceInfos;
    uint32_t set = 0;

    for (auto &bindingMap : descriptorMap) {
        for (const auto &[binding, tensors] : bindingMap) {
            for (uint32_t i = 0; i < tensors.size(); i++) {
                const auto &tensor = tensors[i];

                graphPipelineResourceInfos.push_back(vk::DataGraphPipelineResourceInfoARM{
                    set,                            // descriptor set
                    binding,                        // binding
                    i,                              // array element
                    &tensor->getTensorDescription() // next
                });
            }
        }
        set++;
    }

    const vk::ShaderModuleCreateInfo info{
        {},                                                     // flags
        static_cast<uint32_t>(spirv.size() * sizeof(uint32_t)), // code size
        spirv.data()                                            // code
    };

    vk::raii::ShaderModule shaderModule{&(*device), info};

    std::vector<vk::raii::DescriptorSetLayout> descriptorSetLayouts;

    for (auto &bindingMap : descriptorMap) {
        std::vector<vk::DescriptorSetLayoutBinding> descriptorSetLayoutBindings;

        for (const auto &[binding, tensors] : bindingMap) {
            descriptorSetLayoutBindings.push_back(vk::DescriptorSetLayoutBinding{
                binding,                        // binding
                vk::DescriptorType::eTensorARM, // descriptor type
                uint32_t(tensors.size()),       // descriptor count
                vk::ShaderStageFlagBits::eAll,  // flags
            });
        }

        std::vector<vk::DescriptorBindingFlags> descriptorBindingFlags(descriptorSetLayoutBindings.size(),
                                                                       vk::DescriptorBindingFlagBits::eUpdateAfterBind);

        const vk::DescriptorSetLayoutBindingFlagsCreateInfo descriptorSetBindingFlagsCreateInfo{
            static_cast<uint32_t>(descriptorBindingFlags.size()), // binding count
            descriptorBindingFlags.data(),                        // binding flags
        };

        const vk::DescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo{
            vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool, // flags
            static_cast<uint32_t>(descriptorSetLayoutBindings.size()),   // binding count
            descriptorSetLayoutBindings.data(),                          // bindings
            &descriptorSetBindingFlagsCreateInfo,                        // next
        };

        descriptorSetLayouts.push_back(vk::raii::DescriptorSetLayout(&(*device), descriptorSetLayoutCreateInfo));
    }

    std::vector<vk::DescriptorSetLayout> vkDescriptorSetLayouts;
    std::transform(descriptorSetLayouts.begin(), descriptorSetLayouts.end(), std::back_inserter(vkDescriptorSetLayouts),
                   [](const auto &layout) { return *layout; });

    const vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo{
        {},                                                   // flags
        static_cast<uint32_t>(vkDescriptorSetLayouts.size()), // descriptor set layout count
        vkDescriptorSetLayouts.data()                         // descriptor set layouts
    };

    vk::raii::PipelineLayout pipelineLayout{&(*device), pipelineLayoutCreateInfo};

    vk::PipelineCreationFeedback creationFeedback{};

    const vk::PipelineCreationFeedbackCreateInfo feedbackCreateInfo{&creationFeedback, 0, nullptr, nullptr};

    const vk::DataGraphPipelineShaderModuleCreateInfoARM shaderModuleCreateInfo{
        *shaderModule,       // shader module
        "Graph Pipeline",    // name
        nullptr,             // specialization info
        0,                   // constant count
        nullptr,             // constants
        &feedbackCreateInfo, // next
    };

    const vk::DataGraphPipelineCreateInfoARM graphPipelineCreateInfo{
        {},                                          // flags
        *pipelineLayout,                             // pipeline layout
        uint32_t(graphPipelineResourceInfos.size()), // resource info count
        graphPipelineResourceInfos.data(),           // resource infos
        &shaderModuleCreateInfo,                     // next
    };

    vk::raii::Pipeline pipeline{&(*device), nullptr, nullptr, graphPipelineCreateInfo};

    ASSERT_TRUE(creationFeedback.flags & vk::PipelineCreationFeedbackFlagBits::eValid);
    ASSERT_GT(creationFeedback.duration, 0);
}
