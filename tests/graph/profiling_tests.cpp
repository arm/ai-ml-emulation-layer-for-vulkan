/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include "mlel/pipeline.hpp"
#include "mlel/tensor.hpp"
#include "mlel/utils.hpp"
#include "test_utils.hpp"
#include "vulkan_test_utils.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace mlsdk::el::tests {

using namespace utilities;

namespace {

ProfilingGraph makeRawSadProfilingGraph(std::shared_ptr<Device> &device) {
    const int64_t height = 5;
    const int64_t width = 5;
    std::vector<uint8_t> zeros(height * width, 0);
    std::vector<uint8_t> ones(height * width, 1);

    auto inputTemplate =
        std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 1, height, width}}, zeros);
    auto inputSearch =
        std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 1, height, width}}, ones);
    auto outputCost =
        std::make_shared<Tensor>(device, Shape{vk::Format::eR16Uint, std::vector<int64_t>{1, 1, height, width}});

    GraphPipeline::DescriptorMap descriptorMap = {
        {
            {0, {outputCost}},
            {1, {inputTemplate}},
            {2, {inputSearch}},
        },
    };

    const auto spirv = assembleSpirv(fileToString("me_raw_sad_sr0.spvasm"));
    auto graphPipeline = std::make_shared<GraphPipeline>(device, descriptorMap, GraphConstants{}, spirv, true);
    return {descriptorMap, graphPipeline};
}

void submitGraphCommandBufferTwiceWithoutIntermediateWait(const std::shared_ptr<Device> &device,
                                                          const ProfilingGraph &graph) {
    auto [descriptorPool, descriptorSets] = graph.pipeline->createDescriptorSets(graph.descriptorMap);
    auto commandBuffer = graph.pipeline->createCommandBuffer();

    const vk::CommandBufferBeginInfo commandBufferBeginInfo{
        vk::CommandBufferUsageFlagBits::eSimultaneousUse,
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
    ASSERT_EQ(vkDevice.getDispatcher()->vkQueueSubmit(*queue, 1, &submitInfo, VK_NULL_HANDLE), VK_SUCCESS);
    ASSERT_EQ(vkDevice.getDispatcher()->vkQueueWaitIdle(*queue), VK_SUCCESS);
}

std::string queryPipelineTextProperty(const std::shared_ptr<Device> &device, vk::Pipeline pipeline,
                                      vk::DataGraphPipelinePropertyARM property) {
    const auto &vkDevice = &(*device);
    vk::DataGraphPipelineInfoARM info{pipeline};
    vk::DataGraphPipelinePropertyQueryResultARM queryResult{property};
    auto result = vkDevice.getDataGraphPipelinePropertiesARM(&info, 1, &queryResult);
    EXPECT_EQ(result, vk::Result::eSuccess);
    if (queryResult.dataSize == 0) {
        return {};
    }

    std::vector<char> data(queryResult.dataSize);
    queryResult.pData = data.data();
    queryResult.dataSize = data.size();
    result = vkDevice.getDataGraphPipelinePropertiesARM(&info, 1, &queryResult);
    EXPECT_EQ(result, vk::Result::eSuccess);
    EXPECT_EQ(queryResult.isText, VK_TRUE);
    return std::string{data.data(), queryResult.dataSize};
}

std::optional<vk::DataGraphPipelinePropertyARM> getProfilingProperty(const std::shared_ptr<Device> &device,
                                                                     vk::Pipeline pipeline) {
    const auto &vkDevice = &(*device);
    const vk::DataGraphPipelineInfoARM info{pipeline};
    const auto properties = vkDevice.getDataGraphPipelineAvailablePropertiesARM(info);
    const auto it = std::find_if(properties.begin(), properties.end(), [](const auto property) {
        return property != vk::DataGraphPipelinePropertyARM::eCreationLog;
    });
    if (it == properties.end()) {
        return std::nullopt;
    }
    return *it;
}

void expectProfilePropertyContains(const std::shared_ptr<Device> &device, vk::Pipeline pipeline,
                                   const std::vector<std::string> &needles) {
    const auto property = getProfilingProperty(device, pipeline);
    ASSERT_TRUE(property.has_value());
    const auto json = queryPipelineTextProperty(device, pipeline, *property);
    for (const auto &needle : needles) {
        EXPECT_NE(json.find(needle), std::string::npos) << needle;
    }
}

void expectMaxPoolProfileProperty(const std::shared_ptr<Device> &device, const ProfilingGraph &graph) {
    expectProfilePropertyContains(device, *graph.pipeline->getPipeline(),
                                  {
                                      "\"samples\"",
                                      "\"by_operator\"",
                                      "\"pipeline_kind\": \"tosa\"",
                                      "\"operator_name\": \"MAX_POOL2D\"",
                                      "\"cycle_count_before\"",
                                      "\"cycle_count_after\"",
                                      "\"time_ms\"",
                                  });
}

void expectRawSadProfileProperty(const std::shared_ptr<Device> &device, const ProfilingGraph &graph) {
    expectProfilePropertyContains(device, *graph.pipeline->getPipeline(),
                                  {
                                      "\"samples\"",
                                      "\"by_operator\"",
                                      "\"pipeline_kind\": \"motion_engine\"",
                                      "\"operator_name\": \"RAW_SAD\"",
                                      "\"cycle_count_before\"",
                                      "\"cycle_count_after\"",
                                      "\"time_ms\"",
                                  });
}
TEST(MLEmulationLayerForVulkan, GraphProfilingQueryableProperty) {
    ScopedEnvironment enableProfiling{"VMEL_GRAPH_PROFILING", "1"};

    auto device = createDevice();
    if (!computeQueueSupportsTimestampQueries(device)) {
        GTEST_SKIP() << "Compute queue family does not support timestamp queries";
    }
    auto graph = makeMaxPoolProfilingGraph(device);
    graph.pipeline->dispatchSubmit();

    expectMaxPoolProfileProperty(device, graph);
}

TEST(MLEmulationLayerForVulkan, GraphProfilingCollectsFenceLessSubmitOnQueueWaitIdle) {
    ScopedEnvironment enableProfiling{"VMEL_GRAPH_PROFILING", "1"};

    auto device = createDevice();
    if (!computeQueueSupportsTimestampQueries(device)) {
        GTEST_SKIP() << "Compute queue family does not support timestamp queries";
    }
    auto graph = makeMaxPoolProfilingGraph(device);
    submitGraphWithoutFence(device, graph, false);

    expectMaxPoolProfileProperty(device, graph);
}

TEST(MLEmulationLayerForVulkan, GraphProfilingCollectsFenceLessSubmitOnDeviceWaitIdle) {
    ScopedEnvironment enableProfiling{"VMEL_GRAPH_PROFILING", "1"};

    auto device = createDevice();
    if (!computeQueueSupportsTimestampQueries(device)) {
        GTEST_SKIP() << "Compute queue family does not support timestamp queries";
    }
    auto graph = makeMaxPoolProfilingGraph(device);
    submitGraphWithoutFence(device, graph, true);

    expectMaxPoolProfileProperty(device, graph);
}

TEST(MLEmulationLayerForVulkan, GraphProfilingCollectsReusedCommandBufferSubmits) {
    ScopedEnvironment enableProfiling{"VMEL_GRAPH_PROFILING", "1"};

    auto device = createDevice();
    if (!computeQueueSupportsTimestampQueries(device)) {
        GTEST_SKIP() << "Compute queue family does not support timestamp queries";
    }
    auto graph = makeMaxPoolProfilingGraph(device);
    submitGraphCommandBufferTwiceWithoutIntermediateWait(device, graph);

    const auto property = getProfilingProperty(device, *graph.pipeline->getPipeline());
    ASSERT_TRUE(property.has_value());
    const auto json = queryPipelineTextProperty(device, *graph.pipeline->getPipeline(), *property);
    const auto parsed = nlohmann::json::parse(std::string{json.c_str()});
    const auto &samples = parsed.at("samples");
    ASSERT_EQ(samples.size(), 2);

    EXPECT_EQ(samples.at(0).at("submission"), 0);
    EXPECT_EQ(samples.at(1).at("submission"), 1);
    EXPECT_NE(samples.at(0).at("cycle_count_before"), samples.at(1).at("cycle_count_before"));
    EXPECT_NE(samples.at(0).at("cycle_count_after"), samples.at(1).at("cycle_count_after"));
    ASSERT_EQ(parsed.at("by_operator").size(), 1);
    EXPECT_EQ(parsed.at("by_operator").at(0).at("dispatch_count"), 2);
}

TEST(MLEmulationLayerForVulkan, GraphProfilingIncludesMotionEngineGraphOps) {
    ScopedEnvironment enableProfiling{"VMEL_GRAPH_PROFILING", "1"};

    auto device = createDevice();
    if (!computeQueueSupportsTimestampQueries(device)) {
        GTEST_SKIP() << "Compute queue family does not support timestamp queries";
    }
    auto graph = makeRawSadProfilingGraph(device);
    graph.pipeline->dispatchSubmit();

    expectRawSadProfileProperty(device, graph);
}

TEST(MLEmulationLayerForVulkan, GraphProfilingPropertyIsScopedToQueriedPipeline) {
    ScopedEnvironment enableProfiling{"VMEL_GRAPH_PROFILING", "1"};

    auto device = createDevice();
    auto maxPoolGraph = makeMaxPoolProfilingGraph(device);
    auto rawSadGraph = makeRawSadProfilingGraph(device);
    maxPoolGraph.pipeline->dispatchSubmit();

    const auto property = getProfilingProperty(device, *rawSadGraph.pipeline->getPipeline());
    ASSERT_TRUE(property.has_value());
    const auto json = queryPipelineTextProperty(device, *rawSadGraph.pipeline->getPipeline(), *property);
    EXPECT_NE(json.find("\"samples\": []"), std::string::npos);
    EXPECT_EQ(json.find("MAX_POOL2D"), std::string::npos);
}

TEST(MLEmulationLayerForVulkan, GetDataGraphPipelineAvailablePropertiesIncludesProfilingWhenEnvEnabled) {
    ScopedEnvironment enableProfiling{"VMEL_GRAPH_PROFILING", "1"};

    auto device = createDevice();
    auto graph = makeMaxPoolProfilingGraph(device);
    const auto &vkDevice = &(*device);

    const vk::DataGraphPipelineInfoARM info{*graph.pipeline->getPipeline()};
    const auto result = vkDevice.getDataGraphPipelineAvailablePropertiesARM(info);

    ASSERT_NE(std::find(result.begin(), result.end(), vk::DataGraphPipelinePropertyARM::eCreationLog), result.end());
    ASSERT_TRUE(std::any_of(result.begin(), result.end(), [](const auto property) {
        return property != vk::DataGraphPipelinePropertyARM::eCreationLog;
    }));
}

TEST(MLEmulationLayerForVulkan, GetDataGraphPipelineAvailablePropertiesOmitsProfilingWhenEnvDisabled) {
    ScopedEnvironment disableProfiling{"VMEL_GRAPH_PROFILING", "0"};

    auto device = createDevice();
    auto graph = makeMaxPoolProfilingGraph(device);
    const auto &vkDevice = &(*device);

    const vk::DataGraphPipelineInfoARM info{*graph.pipeline->getPipeline()};
    const auto result = vkDevice.getDataGraphPipelineAvailablePropertiesARM(info);

    ASSERT_NE(std::find(result.begin(), result.end(), vk::DataGraphPipelinePropertyARM::eCreationLog), result.end());
    ASSERT_TRUE(std::all_of(result.begin(), result.end(), [](const auto property) {
        return property == vk::DataGraphPipelinePropertyARM::eCreationLog;
    }));
}

TEST(MLEmulationLayerForVulkan, GetDataGraphPipelinePropertiesARMReturnsIncompleteForSmallProfilingBuffer) {
    ScopedEnvironment enableProfiling{"VMEL_GRAPH_PROFILING", "1"};

    auto device = createDevice();
    auto graph = makeMaxPoolProfilingGraph(device);
    graph.pipeline->dispatchSubmit();

    const auto &vkDevice = &(*device);
    const vk::DataGraphPipelineInfoARM info{*graph.pipeline->getPipeline()};
    const auto property = getProfilingProperty(device, *graph.pipeline->getPipeline());
    ASSERT_TRUE(property.has_value());
    vk::DataGraphPipelinePropertyQueryResultARM queryResult{*property};
    auto result = vkDevice.getDataGraphPipelinePropertiesARM(&info, 1, &queryResult);
    ASSERT_EQ(result, vk::Result::eSuccess);
    ASSERT_GT(queryResult.dataSize, 4);

    std::array<char, 4> data{};
    queryResult.pData = data.data();
    queryResult.dataSize = data.size();
    result = vkDevice.getDataGraphPipelinePropertiesARM(&info, 1, &queryResult);

    ASSERT_EQ(result, vk::Result::eIncomplete);
    ASSERT_EQ(queryResult.isText, VK_TRUE);
    ASSERT_EQ(queryResult.dataSize, data.size());
}

} // namespace
} // namespace mlsdk::el::tests
