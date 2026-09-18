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
#include <cstdint>
#include <memory>
#include <vector>

namespace mlsdk::el::tests {

using namespace utilities;

namespace {

TEST_F(MLEmulationLayerGraphForVulkan, GetDataGraphPipelineAvailablePropertiesARM) {
    const auto &vkDevice = &(*device);

    vk::DataGraphPipelineInfoARM info;
    const auto result = vkDevice.getDataGraphPipelineAvailablePropertiesARM(info);
    ASSERT_FALSE(result.empty());
    ASSERT_EQ(result[0], vk::DataGraphPipelinePropertyARM{});
}

TEST_F(MLEmulationLayerGraphForVulkan, GetDataGraphPipelinePropertiesARM) {
    const auto &vkDevice = &(*device);

    vk::DataGraphPipelinePropertyQueryResultARM queryResult;
    auto result = vkDevice.getDataGraphPipelinePropertiesARM(nullptr, 1, &queryResult);
    ASSERT_EQ(result, vk::Result::eSuccess);
    std::vector<char> data(queryResult.dataSize);
    queryResult.pData = data.data();
    queryResult.dataSize = static_cast<uint32_t>(data.size());
    result = vkDevice.getDataGraphPipelinePropertiesARM(nullptr, 1, &queryResult);
    ASSERT_EQ(result, vk::Result::eSuccess);
    ASSERT_EQ(queryResult.isText, VK_TRUE);
    ASSERT_EQ(queryResult.dataSize, data.size());
}

TEST_F(MLEmulationLayerGraphForVulkan, GetQueueFamilyDataGraphProcessingEnginePropertiesARM) {
    const auto &physicalDevice = device->getPhysicalDevice();
    const auto &vkPhysicalDevice = &(*physicalDevice);

    vk::PhysicalDeviceQueueFamilyDataGraphProcessingEngineInfoARM info;
    const auto result = vkPhysicalDevice.getQueueFamilyDataGraphProcessingEnginePropertiesARM(info);
    ASSERT_FALSE(result.foreignSemaphoreHandleTypes);
}

TEST_F(MLEmulationLayerGraphForVulkan, GetQueueFamilyDataGraphPropertiesARM) {
    const auto &physicalDevice = device->getPhysicalDevice();
    const auto &vkPhysicalDevice = &(*physicalDevice);

    uint32_t queueFamilyIndex = physicalDevice->getComputeFamilyIndex();
    const auto result = vkPhysicalDevice.getQueueFamilyDataGraphPropertiesARM(queueFamilyIndex);
    ASSERT_FALSE(result.empty());
}

TEST_F(MLEmulationLayerGraphForVulkan, GetQueueFamilyDataGraphPropertiesARMTwoCallPattern) {
    const auto &physicalDevice = device->getPhysicalDevice();
    const auto &vkPhysicalDevice = &(*physicalDevice);
    const uint32_t queueFamilyIndex = physicalDevice->getComputeFamilyIndex();

    // Phase 1: query the count with nullptr properties
    uint32_t count = 0;
    const VkResult firstResult = vkPhysicalDevice.getDispatcher()->vkGetPhysicalDeviceQueueFamilyDataGraphPropertiesARM(
        *vkPhysicalDevice, queueFamilyIndex, &count, nullptr);
    ASSERT_EQ(firstResult, VK_SUCCESS);
    ASSERT_GT(count, 0u);

    // Phase 2: retrieve the indicated count of elements
    std::vector<VkQueueFamilyDataGraphPropertiesARM> properties(count);
    for (auto &p : properties) {
        p = {};
        p.sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_DATA_GRAPH_PROPERTIES_ARM;
    }
    uint32_t retrieveCount = count;
    const VkResult secondResult =
        vkPhysicalDevice.getDispatcher()->vkGetPhysicalDeviceQueueFamilyDataGraphPropertiesARM(
            *vkPhysicalDevice, queueFamilyIndex, &retrieveCount, properties.data());
    ASSERT_EQ(secondResult, VK_SUCCESS);
    ASSERT_EQ(retrieveCount, count);
}

void findGraphOperationProperties(std::shared_ptr<Device> &device, VkPhysicalDeviceDataGraphOperationTypeARM operation,
                                  VkQueueFamilyDataGraphPropertiesARM &result) {
    const auto &physicalDevice = device->getPhysicalDevice();
    const auto &vkPhysicalDevice = &(*physicalDevice);
    const auto *dispatcher = vkPhysicalDevice.getDispatcher();
    const auto queueFamilyIndex = physicalDevice->getComputeFamilyIndex();
    uint32_t count = 0;
    const VkResult firstResult = dispatcher->vkGetPhysicalDeviceQueueFamilyDataGraphPropertiesARM(
        *vkPhysicalDevice, queueFamilyIndex, &count, nullptr);
    ASSERT_EQ(firstResult, VK_SUCCESS);
    ASSERT_GT(count, 0u);

    std::vector<VkQueueFamilyDataGraphPropertiesARM> properties(count);
    for (auto &property : properties) {
        property = {};
        property.sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_DATA_GRAPH_PROPERTIES_ARM;
    }

    uint32_t retrieveCount = count;
    const VkResult secondResult = dispatcher->vkGetPhysicalDeviceQueueFamilyDataGraphPropertiesARM(
        *vkPhysicalDevice, queueFamilyIndex, &retrieveCount, properties.data());
    ASSERT_EQ(secondResult, VK_SUCCESS);
    ASSERT_GT(retrieveCount, 0u);

    const auto found = std::find_if(properties.begin(), properties.end(), [&](const auto &property) {
        return property.operation.operationType == operation;
    });
    ASSERT_NE(found, properties.end());
    result = *found;
}

TEST_F(MLEmulationLayerGraphForVulkan, TosaEngineOperationProperties) {
    const auto &physicalDevice = device->getPhysicalDevice();
    const auto &vkPhysicalDevice = &(*physicalDevice);
    const auto *dispatcher = vkPhysicalDevice.getDispatcher();

    // Check whether the physical-device extension function is actually resolved
    if (dispatcher->vkGetPhysicalDeviceQueueFamilyDataGraphEngineOperationPropertiesARM == nullptr) {
        // Linux loaders can still route unknown physical-device entry points through
        // vk_layerGetPhysicalDeviceProcAddr. Darwin loaders may not enable that fallback,
        // so these tests skip when the runtime cannot hand back a callable pointer.
        GTEST_SKIP() << "vkGetPhysicalDeviceQueueFamilyDataGraphEngineOperationPropertiesARM "
                     << "not resolved by loader on this runtime";
    }

    const uint32_t queueFamilyIndex = physicalDevice->getComputeFamilyIndex();

    VkQueueFamilyDataGraphPropertiesARM operation{};
    ASSERT_NO_FATAL_FAILURE(findGraphOperationProperties(
        device, VK_PHYSICAL_DEVICE_DATA_GRAPH_OPERATION_TYPE_SPIRV_EXTENDED_INSTRUCTION_SET_ARM, operation));
    VkQueueFamilyDataGraphTOSAPropertiesARM tosaProperties = {};
    tosaProperties.sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_DATA_GRAPH_TOSA_PROPERTIES_ARM;

    const auto tosaResult = dispatcher->vkGetPhysicalDeviceQueueFamilyDataGraphEngineOperationPropertiesARM(
        *vkPhysicalDevice, queueFamilyIndex, &operation, reinterpret_cast<VkBaseOutStructure *>(&tosaProperties));
    ASSERT_EQ(tosaResult, VK_SUCCESS);
    ASSERT_EQ(tosaProperties.profileCount, 1u);
    ASSERT_NE(tosaProperties.pProfiles, nullptr);
    ASSERT_EQ(tosaProperties.pProfiles[0].qualityFlags, VK_DATA_GRAPH_TOSA_QUALITY_CONFORMANT_ARM);
    ASSERT_EQ(tosaProperties.extensionCount, 0u);
    ASSERT_EQ(tosaProperties.pExtensions, nullptr);
    ASSERT_EQ(tosaProperties.level, VK_DATA_GRAPH_TOSA_LEVEL_8K_ARM);
}

TEST_F(MLEmulationLayerGraphForVulkan, OpticalFlowEngineOperationProperties) {
    const auto &physicalDevice = device->getPhysicalDevice();
    const auto &vkPhysicalDevice = &(*physicalDevice);
    const auto *dispatcher = vkPhysicalDevice.getDispatcher();

    // Check whether the physical-device extension function is actually resolved
    if (dispatcher->vkGetPhysicalDeviceQueueFamilyDataGraphEngineOperationPropertiesARM == nullptr) {
        // Linux loaders can still route unknown physical-device entry points through
        // vk_layerGetPhysicalDeviceProcAddr. Darwin loaders may not enable that fallback,
        // so these tests skip when the runtime cannot hand back a callable pointer.
        GTEST_SKIP() << "vkGetPhysicalDeviceQueueFamilyDataGraphEngineOperationPropertiesARM "
                     << "not resolved by loader on this runtime";
    }

    const uint32_t queueFamilyIndex = physicalDevice->getComputeFamilyIndex();

    VkQueueFamilyDataGraphPropertiesARM operation{};
    ASSERT_NO_FATAL_FAILURE(
        findGraphOperationProperties(device, VK_PHYSICAL_DEVICE_DATA_GRAPH_OPERATION_TYPE_OPTICAL_FLOW_ARM, operation));
    VkQueueFamilyDataGraphOpticalFlowPropertiesARM ofProps{};
    ofProps.sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_DATA_GRAPH_OPTICAL_FLOW_PROPERTIES_ARM;

    const auto ofResult = dispatcher->vkGetPhysicalDeviceQueueFamilyDataGraphEngineOperationPropertiesARM(
        *vkPhysicalDevice, queueFamilyIndex, &operation, reinterpret_cast<VkBaseOutStructure *>(&ofProps));

    ASSERT_EQ(ofResult, VK_SUCCESS);
    ASSERT_GT(ofProps.supportedOutputGridSizes, 0u);
    ASSERT_GT(ofProps.supportedHintGridSizes, 0u);
    ASSERT_GT(ofProps.maxWidth, 0u);
    ASSERT_GT(ofProps.maxHeight, 0u);
}

TEST_F(MLEmulationLayerGraphForVulkan, OpticalFlowInputImageFormatsTwoCallPattern) {
    const auto &physicalDevice = device->getPhysicalDevice();
    const auto &vkPhysicalDevice = &(*physicalDevice);
    const auto *dispatcher = vkPhysicalDevice.getDispatcher();

    // Check whether the physical-device extension function is actually resolved
    if (dispatcher->vkGetPhysicalDeviceQueueFamilyDataGraphOpticalFlowImageFormatsARM == nullptr) {
        // Linux loaders can still route unknown physical-device entry points through
        // vk_layerGetPhysicalDeviceProcAddr. Darwin loaders may not enable that fallback,
        // so these tests skip when the runtime cannot hand back a callable pointer.
        GTEST_SKIP() << "vkGetPhysicalDeviceQueueFamilyDataGraphOpticalFlowImageFormatsARM "
                     << "not resolved by loader on this runtime";
    }

    const uint32_t queueFamilyIndex = physicalDevice->getComputeFamilyIndex();

    VkQueueFamilyDataGraphPropertiesARM operation{};
    ASSERT_NO_FATAL_FAILURE(
        findGraphOperationProperties(device, VK_PHYSICAL_DEVICE_DATA_GRAPH_OPERATION_TYPE_OPTICAL_FLOW_ARM, operation));

    VkDataGraphOpticalFlowImageFormatInfoARM formatInfo{};
    formatInfo.sType = VK_STRUCTURE_TYPE_DATA_GRAPH_OPTICAL_FLOW_IMAGE_FORMAT_INFO_ARM;

    // Phase 1: query input format count
    formatInfo.usage = VK_DATA_GRAPH_OPTICAL_FLOW_IMAGE_USAGE_INPUT_BIT_ARM;
    uint32_t inputFormatCount = 0;
    VkResult result = dispatcher->vkGetPhysicalDeviceQueueFamilyDataGraphOpticalFlowImageFormatsARM(
        *vkPhysicalDevice, queueFamilyIndex, &operation, &formatInfo, &inputFormatCount, nullptr);
    ASSERT_EQ(result, VK_SUCCESS);
    ASSERT_GT(inputFormatCount, 0u);

    // Phase 2: retrieve input formats
    std::vector<VkDataGraphOpticalFlowImageFormatPropertiesARM> inputFormats(inputFormatCount);
    for (auto &f : inputFormats) {
        f = {};
        f.sType = VK_STRUCTURE_TYPE_DATA_GRAPH_OPTICAL_FLOW_IMAGE_FORMAT_PROPERTIES_ARM;
    }
    uint32_t retrievedInputCount = inputFormatCount;
    result = dispatcher->vkGetPhysicalDeviceQueueFamilyDataGraphOpticalFlowImageFormatsARM(
        *vkPhysicalDevice, queueFamilyIndex, &operation, &formatInfo, &retrievedInputCount, inputFormats.data());
    ASSERT_EQ(result, VK_SUCCESS);
    ASSERT_EQ(retrievedInputCount, inputFormatCount);
    for (const auto &f : inputFormats) {
        ASSERT_NE(f.format, VK_FORMAT_UNDEFINED);
    }
}

TEST_F(MLEmulationLayerGraphForVulkan, OpticalFlowOutputImageFormatCount) {
    const auto &physicalDevice = device->getPhysicalDevice();
    const auto &vkPhysicalDevice = &(*physicalDevice);
    const auto *dispatcher = vkPhysicalDevice.getDispatcher();

    // Check whether the physical-device extension function is actually resolved
    if (dispatcher->vkGetPhysicalDeviceQueueFamilyDataGraphOpticalFlowImageFormatsARM == nullptr) {
        // Linux loaders can still route unknown physical-device entry points through
        // vk_layerGetPhysicalDeviceProcAddr. Darwin loaders may not enable that fallback,
        // so these tests skip when the runtime cannot hand back a callable pointer.
        GTEST_SKIP() << "vkGetPhysicalDeviceQueueFamilyDataGraphOpticalFlowImageFormatsARM "
                     << "not resolved by loader on this runtime";
    }

    const uint32_t queueFamilyIndex = physicalDevice->getComputeFamilyIndex();

    VkQueueFamilyDataGraphPropertiesARM operation{};
    ASSERT_NO_FATAL_FAILURE(
        findGraphOperationProperties(device, VK_PHYSICAL_DEVICE_DATA_GRAPH_OPERATION_TYPE_OPTICAL_FLOW_ARM, operation));

    VkDataGraphOpticalFlowImageFormatInfoARM formatInfo{};
    formatInfo.sType = VK_STRUCTURE_TYPE_DATA_GRAPH_OPTICAL_FLOW_IMAGE_FORMAT_INFO_ARM;

    // Query output (flow vector) formats
    formatInfo.usage = VK_DATA_GRAPH_OPTICAL_FLOW_IMAGE_USAGE_OUTPUT_BIT_ARM;
    uint32_t outputFormatCount = 0;
    const auto result = dispatcher->vkGetPhysicalDeviceQueueFamilyDataGraphOpticalFlowImageFormatsARM(
        *vkPhysicalDevice, queueFamilyIndex, &operation, &formatInfo, &outputFormatCount, nullptr);
    ASSERT_EQ(result, VK_SUCCESS);
    ASSERT_GT(outputFormatCount, 0u);
}

TEST_F(MLEmulationLayerGraphForVulkan, GetDataGraphPipelineSessionBindPointRequirementsARMTwoCallPattern) {

    auto inputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 16, 16, 3}});
    auto outputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 8, 8, 3}});
    const GraphPipeline::DescriptorMap descriptorMap = {{
        {0, {inputTensor, inputTensor}},
        {1, {outputTensor}},
    }};
    const auto spirv = assembleSpirv(fileToString("maxpool.spvasm"));

    // Build resource infos from descriptor map
    std::vector<vk::DataGraphPipelineResourceInfoARM> resourceInfos;
    uint32_t set = 0;
    for (const auto &bindingMap : descriptorMap) {
        for (const auto &[binding, tensors] : bindingMap) {
            for (uint32_t i = 0; i < tensors.size(); i++) {
                resourceInfos.emplace_back(set, binding, i, &tensors[i]->getTensorDescription());
            }
        }
        ++set;
    }

    // Shader module
    const vk::ShaderModuleCreateInfo shaderModuleCI{
        {},
        spirv.size() * sizeof(uint32_t),
        spirv.data(),
    };
    vk::raii::ShaderModule shaderModule{&(*device), shaderModuleCI};

    // Descriptor set layouts and pipeline layout
    std::vector<vk::raii::DescriptorSetLayout> dsLayouts;
    for (const auto &bindingMap : descriptorMap) {
        std::vector<vk::DescriptorSetLayoutBinding> bindings;
        bindings.reserve(bindingMap.size());
        for (const auto &[binding, tensors] : bindingMap) {
            bindings.emplace_back(binding, vk::DescriptorType::eTensorARM, uint32_t(tensors.size()),
                                  vk::ShaderStageFlagBits::eAll);
        }
        std::vector<vk::DescriptorBindingFlags> bindingFlags(bindings.size(),
                                                             vk::DescriptorBindingFlagBits::eUpdateAfterBind);
        const vk::DescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsCI{uint32_t(bindingFlags.size()),
                                                                           bindingFlags.data()};
        const vk::DescriptorSetLayoutCreateInfo dsCI{vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool,
                                                     uint32_t(bindings.size()), bindings.data(), &bindingFlagsCI};
        dsLayouts.emplace_back(&(*device), dsCI);
    }
    std::vector<vk::DescriptorSetLayout> vkDSLayouts;
    vkDSLayouts.reserve(dsLayouts.size());
    for (const auto &l : dsLayouts) {
        vkDSLayouts.push_back(*l);
    }
    const vk::PipelineLayoutCreateInfo plCI{{}, uint32_t(vkDSLayouts.size()), vkDSLayouts.data()};
    vk::raii::PipelineLayout pipelineLayout{&(*device), plCI};

    // Pipeline
    const vk::DataGraphPipelineShaderModuleCreateInfoARM smCI{*shaderModule, "Graph Pipeline", nullptr, 0, nullptr};
    const vk::DataGraphPipelineCreateInfoARM pipelineCI{
        {}, *pipelineLayout, uint32_t(resourceInfos.size()), resourceInfos.data(), &smCI};
    vk::raii::Pipeline pipeline{&(*device), nullptr, nullptr, pipelineCI};

    // Session
    const vk::DataGraphPipelineSessionCreateInfoARM sessionCI{{}, *pipeline};
    vk::raii::DataGraphPipelineSessionARM session{&(*device), sessionCI};

    const auto &vkDevice = &(*device);
    const VkDataGraphPipelineSessionBindPointRequirementsInfoARM requirementsInfo{
        VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_REQUIREMENTS_INFO_ARM,
        nullptr,
        *session,
    };

    // Phase 1: query the count with nullptr requirements
    uint32_t count = 0;
    const VkResult firstResult = vkDevice.getDispatcher()->vkGetDataGraphPipelineSessionBindPointRequirementsARM(
        *vkDevice, &requirementsInfo, &count, nullptr);
    ASSERT_EQ(firstResult, VK_SUCCESS);

    // Phase 2: retrieve the indicated count of requirements
    std::vector<VkDataGraphPipelineSessionBindPointRequirementARM> requirements(count);
    for (auto &r : requirements) {
        r = {};
        r.sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_REQUIREMENT_ARM;
    }
    uint32_t retrieveCount = count;
    const VkResult secondResult = vkDevice.getDispatcher()->vkGetDataGraphPipelineSessionBindPointRequirementsARM(
        *vkDevice, &requirementsInfo, &retrieveCount, requirements.data());
    ASSERT_EQ(secondResult, VK_SUCCESS);
    ASSERT_EQ(retrieveCount, count);
}

TEST_F(MLEmulationLayerGraphForVulkan, GetExternalTensorPropertiesARM) {
    const auto &physicalDevice = device->getPhysicalDevice();
    const auto &vkPhysicalDevice = &(*physicalDevice);

    const VkExternalMemoryHandleTypeFlagBits handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    const std::vector<int64_t> dimensions = {1, 32, 4};
    const VkTensorDescriptionARM tensorDesc = {VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM,
                                               nullptr,
                                               VK_TENSOR_TILING_LINEAR_ARM,
                                               VK_FORMAT_R8_UINT,
                                               1,
                                               dimensions.data(),
                                               nullptr,
                                               VK_TENSOR_USAGE_SHADER_BIT_ARM};
    const VkPhysicalDeviceExternalTensorInfoARM externalTensorInfoARM = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_TENSOR_INFO_ARM, nullptr, {}, &tensorDesc, handleType};
    const auto properties = vkPhysicalDevice.getExternalTensorPropertiesARM(externalTensorInfoARM);

    const VkPhysicalDeviceExternalBufferInfo externalBufferInfo = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO,
        nullptr,
        {},
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT};
    const auto bufferProperties = vkPhysicalDevice.getExternalBufferProperties(externalBufferInfo);

    ASSERT_EQ(properties.externalMemoryProperties, bufferProperties.externalMemoryProperties);
}

} // namespace
} // namespace mlsdk::el::tests
