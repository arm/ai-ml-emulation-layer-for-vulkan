/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include "vulkan_test_utils.hpp"
#include "mlel/pipeline.hpp"
#include "mlel/tensor.hpp"
#include "mlel/utils.hpp"
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <spirv-tools/libspirv.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlsdk::el::tests {

using namespace utilities;

#define STR(str) #str
#define TOSTRING(str) STR(str)

namespace {
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
} // namespace

std::string fileToString(const std::string &filename) {
    const std::filesystem::path path = std::filesystem::path(TOSTRING(SHADER_SOURCE_DIR)) / filename;
    std::ifstream ifs{path};
    if (!ifs) {
        throw std::runtime_error(std::string("Failed to open ") + filename);
    }

    std::string str(std::istreambuf_iterator<char>{ifs}, {});
    return str;
}

std::vector<uint32_t> compileGlsl(const std::string &text) {
    auto spirv = utils::glslToSpirv(text);
    spvtools::SpirvTools tools{SPV_ENV_VULKAN_1_3};
    tools.SetMessageConsumer(sprivMessageConsumer);
    if (!tools.Validate(spirv)) {
        throw std::runtime_error("Failed to validate compiled test shader for Vulkan 1.3");
    }
    return spirv;
}

std::vector<uint32_t> assembleSpirv(const std::string &text) {
    spvtools::SpirvTools tools{SPV_ENV_VULKAN_1_3};

    if (!tools.IsValid()) {
        throw std::runtime_error("Failed to instantiate SPIR-V tools");
    }

    tools.SetMessageConsumer(sprivMessageConsumer);

    std::vector<uint32_t> spirvModule;

    if (!tools.Assemble(text, &spirvModule)) {
        throw std::runtime_error("Failed to assemble SPIR-V program");
    }

    if (!tools.Validate(spirvModule)) {
        throw std::runtime_error("Failed to validate SPIR-V program");
    }

    return spirvModule;
}

std::shared_ptr<Device> createDevice() {
    std::vector<const char *> layers = {"VK_LAYER_ML_Graph_Emulation", "VK_LAYER_ML_Tensor_Emulation"};
    std::vector<const char *> extensions = {
        VK_ARM_DATA_GRAPH_EXTENSION_NAME,
        VK_ARM_DATA_GRAPH_INSTRUCTION_SET_TOSA_EXTENSION_NAME,
        VK_ARM_DATA_GRAPH_OPTICAL_FLOW_EXTENSION_NAME,
        VK_ARM_TENSORS_EXTENSION_NAME,
    };

    auto *const envValidation = std::getenv("VMEL_VALIDATION");
    if (envValidation && !std::string(envValidation).empty() && std::string(envValidation) != "0") {
        layers.emplace_back("VK_LAYER_KHRONOS_validation");
    }

    // Enable base features
    vk::PhysicalDeviceFeatures baseFeatures = {};
    baseFeatures.shaderInt64 = VK_TRUE;
    baseFeatures.shaderFloat64 = VK_TRUE;

    // Create the features2 wrapper
    vk::PhysicalDeviceFeatures2 features2 = {};
    features2.setFeatures(baseFeatures);

    auto context = std::make_shared<vk::raii::Context>();
    auto instance = std::make_shared<Instance>(context, layers);
    auto physicalDevice = std::make_shared<PhysicalDevice>(instance, extensions);

    return std::make_shared<Device>(physicalDevice, extensions, &features2);
}

ProfilingGraph makeMaxPoolProfilingGraph(std::shared_ptr<Device> &device) {
    auto inputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 16, 16, 3}});
    auto outputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 8, 8, 3}});
    GraphPipeline::DescriptorMap descriptorMap = {
        {
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
    auto graphPipeline = std::make_shared<GraphPipeline>(device, descriptorMap, GraphConstants{}, spirv, true);
    return {descriptorMap, graphPipeline};
}

void submitGraphWithoutFence(const std::shared_ptr<Device> &device, const ProfilingGraph &graph, bool waitDevice) {
    auto [descriptorPool, descriptorSets] = graph.pipeline->createDescriptorSets(graph.descriptorMap);
    auto commandBuffer = graph.pipeline->createCommandBuffer();

    const vk::CommandBufferBeginInfo commandBufferBeginInfo{
        vk::CommandBufferUsageFlagBits::eOneTimeSubmit, // flags
    };
    commandBuffer.begin(commandBufferBeginInfo);
    graph.pipeline->dispatch(commandBuffer, descriptorSets);
    commandBuffer.end();

    vk::raii::Queue queue(&(*device), device->getPhysicalDevice()->getComputeFamilyIndex(), 0);
    const VkCommandBuffer vkCommandBuffer = *commandBuffer;
    const VkSubmitInfo submitInfo{
        VK_STRUCTURE_TYPE_SUBMIT_INFO, // sType
        nullptr,                       // pNext
        0,                             // waitSemaphoreCount
        nullptr,                       // pWaitSemaphores
        nullptr,                       // pWaitDstStageMask
        1,                             // commandBufferCount
        &vkCommandBuffer,              // pCommandBuffers
        0,                             // signalSemaphoreCount
        nullptr,                       // pSignalSemaphores
    };

    const auto &vkDevice = &(*device);
    ASSERT_EQ(vkDevice.getDispatcher()->vkQueueSubmit(*queue, 1, &submitInfo, VK_NULL_HANDLE), VK_SUCCESS);
    if (waitDevice) {
        ASSERT_EQ(vkDevice.getDispatcher()->vkDeviceWaitIdle(*vkDevice), VK_SUCCESS);
    } else {
        ASSERT_EQ(vkDevice.getDispatcher()->vkQueueWaitIdle(*queue), VK_SUCCESS);
    }
}

} // namespace mlsdk::el::tests
