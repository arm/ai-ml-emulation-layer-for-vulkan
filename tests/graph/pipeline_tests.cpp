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
#include <cstdint>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

namespace mlsdk::el::tests {

using namespace utilities;

namespace {

TEST_F(MLEmulationLayerGraphForVulkan, SamePipelineLayout) {

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

    ASSERT_TRUE(outputTensor->compare(reinterpret_cast<const int8_t *>(&ref[0]), sizeof(ref))) << "Output mismatch";

    ASSERT_TRUE(outputTensor1->compare(reinterpret_cast<const int8_t *>(&ref[0]), sizeof(ref))) << "Output mismatch";
}

TEST_F(MLEmulationLayerGraphForVulkan, UpdateAfterDispatch) {

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

    ASSERT_TRUE(outputTensor->compare(reinterpret_cast<const int8_t *>(&ref[0]), sizeof(ref))) << "Output mismatch";
}

TEST_F(MLEmulationLayerGraphForVulkan, SequentialDispatch) {

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

    ASSERT_TRUE(outputTensor->compare(reinterpret_cast<const int8_t *>(&ref[0]), sizeof(ref))) << "Output mismatch";

    ASSERT_TRUE(outputTensor1->compare(reinterpret_cast<const int8_t *>(&ref[0]), sizeof(ref))) << "Output mismatch";
}

TEST(MLEmulationLayerForVulkan, ConcatMultipleDispatchesPreserveInputLiveRange) {
    ScopedEnvironment memoryPlanner{"VMEL_MEMORY_PLANNER", "Interval"};
    auto device = createDevice();

    const std::vector<float> input{0, 1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<float> one(9, 1);
    const std::vector<float> zero(3, 0);
    auto inputTensor =
        std::make_shared<Tensor>(device, Shape{vk::Format::eR32Sfloat, {3, 3}},
                                 reinterpret_cast<const uint8_t *>(input.data()), input.size() * sizeof(float));
    auto oneTensor =
        std::make_shared<Tensor>(device, Shape{vk::Format::eR32Sfloat, {3, 3}},
                                 reinterpret_cast<const uint8_t *>(one.data()), one.size() * sizeof(float));
    auto zeroTensor =
        std::make_shared<Tensor>(device, Shape{vk::Format::eR32Sfloat, {3, 1}},
                                 reinterpret_cast<const uint8_t *>(zero.data()), zero.size() * sizeof(float));
    auto outputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR32Sfloat, {12}});
    const GraphPipeline::DescriptorMap descriptorMap = {{
        {0, {inputTensor}},
        {1, {oneTensor}},
        {2, {zeroTensor}},
        {3, {outputTensor}},
    }};

    const auto spirv = assembleSpirv(fileToString("concat_interval.spvasm"));
    auto graphPipeline = std::make_shared<GraphPipeline>(device, descriptorMap, GraphConstants{}, spirv);

    graphPipeline->dispatchSubmit();

    const std::vector<float> expected{
        1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0,
    };
    ASSERT_TRUE(outputTensor->compare(expected.data(), expected.size() * sizeof(float))) << "Output mismatch";
}

TEST_F(MLEmulationLayerGraphForVulkan, MultiSessionsInOneCommandBuffer) {

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
        ASSERT_TRUE(tensor->compare(reinterpret_cast<const int8_t *>(&ref[0]), sizeof(ref))) << "Output mismatch";
    }
}

TEST_F(MLEmulationLayerGraphForVulkan, MultiSessionsOneAtTheTime) {

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

    for (const auto &pipeline : {graphPipeline, graphPipeline2}) {
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

            ASSERT_TRUE(outputTensor->compare(reinterpret_cast<const int8_t *>(&ref[0]), sizeof(ref)))
                << "Output mismatch";
            // descriptor set and pool are destroyed
        }
    }
}

TEST_F(MLEmulationLayerGraphForVulkan, PipelineCreationFeedback) {

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

    for (const auto &bindingMap : descriptorMap) {
        for (const auto &[binding, tensors] : bindingMap) {
            for (uint32_t i = 0; i < tensors.size(); i++) {
                const auto &tensor = tensors[i];

                graphPipelineResourceInfos.emplace_back(set,                            // descriptor set
                                                        binding,                        // binding
                                                        i,                              // array element
                                                        &tensor->getTensorDescription() // next
                );
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

    for (const auto &bindingMap : descriptorMap) {
        std::vector<vk::DescriptorSetLayoutBinding> descriptorSetLayoutBindings;

        descriptorSetLayoutBindings.reserve(bindingMap.size());
        for (const auto &[binding, tensors] : bindingMap) {
            descriptorSetLayoutBindings.emplace_back(binding,                        // binding
                                                     vk::DescriptorType::eTensorARM, // descriptor type
                                                     uint32_t(tensors.size()),       // descriptor count
                                                     vk::ShaderStageFlagBits::eAll);
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

        descriptorSetLayouts.emplace_back(&(*device), descriptorSetLayoutCreateInfo);
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

TEST_F(MLEmulationLayerGraphForVulkan, SpecConstBoolDoesNotBlockLowering) {

    constexpr vk::Format kTensorFormat = vk::Format::eR16Uint;
    constexpr int64_t kTensorLength = 4;
    constexpr uint32_t kBindingInput = 0u;
    constexpr uint32_t kBindingShift = 1u;
    constexpr uint32_t kBindingOutput = 2u;
    constexpr uint32_t kSpecIdRound = 0u;

    // Create three tensors: 1D R16Sint of length 4 for inputs and output
    auto inputTensor = std::make_shared<Tensor>(device, Shape{kTensorFormat, std::vector<int64_t>{kTensorLength}});
    auto shiftTensor = std::make_shared<Tensor>(device, Shape{kTensorFormat, std::vector<int64_t>{kTensorLength}});
    auto outputTensor = std::make_shared<Tensor>(device, Shape{kTensorFormat, std::vector<int64_t>{kTensorLength}});

    const GraphPipeline::DescriptorMap descriptorMap = {{
        {kBindingInput, {inputTensor}},
        {kBindingShift, {shiftTensor}},
        {kBindingOutput, {outputTensor}},
    }};

    const auto spirv = assembleSpirv(fileToString("arshift_specbool.spvasm"));

    // Build resource infos for pipeline creation
    std::vector<vk::DataGraphPipelineResourceInfoARM> graphPipelineResourceInfos;
    constexpr size_t kResourceInfoCapacity = 3u; // input, shift, output
    graphPipelineResourceInfos.reserve(kResourceInfoCapacity);
    uint32_t set = 0;
    for (const auto &bindingMap : descriptorMap) {
        for (const auto &[binding, tensors] : bindingMap) {
            for (uint32_t i = 0; i < tensors.size(); i++) {
                graphPipelineResourceInfos.emplace_back(set,                                // descriptor set
                                                        binding,                            // binding
                                                        i,                                  // array element
                                                        &tensors[i]->getTensorDescription() // next
                );
            }
        }
        set++;
    }

    // Shader module
    const vk::ShaderModuleCreateInfo info{
        {},                                                     // flags
        static_cast<uint32_t>(spirv.size() * sizeof(uint32_t)), // code size
        spirv.data()                                            // code
    };
    vk::raii::ShaderModule shaderModule{&(*device), info};

    // Descriptor set layouts
    std::vector<vk::raii::DescriptorSetLayout> descriptorSetLayouts;
    descriptorSetLayouts.reserve(descriptorMap.size());
    for (const auto &bindingMap : descriptorMap) {
        std::vector<vk::DescriptorSetLayoutBinding> descriptorSetLayoutBindings;
        descriptorSetLayoutBindings.reserve(bindingMap.size());
        for (const auto &[binding, tensors] : bindingMap) {
            descriptorSetLayoutBindings.emplace_back(binding,                        // binding
                                                     vk::DescriptorType::eTensorARM, // descriptor type
                                                     uint32_t(tensors.size()),       // descriptor count
                                                     vk::ShaderStageFlagBits::eAll);
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

        descriptorSetLayouts.emplace_back(&(*device), descriptorSetLayoutCreateInfo);
    }

    std::vector<vk::DescriptorSetLayout> vkDescriptorSetLayouts;
    vkDescriptorSetLayouts.reserve(descriptorSetLayouts.size());
    std::transform(descriptorSetLayouts.begin(), descriptorSetLayouts.end(), std::back_inserter(vkDescriptorSetLayouts),
                   [](const auto &layout) { return *layout; });

    const vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo{
        {},                                                   // flags
        static_cast<uint32_t>(vkDescriptorSetLayouts.size()), // descriptor set layout count
        vkDescriptorSetLayouts.data()                         // descriptor set layouts
    };
    vk::raii::PipelineLayout pipelineLayout{&(*device), pipelineLayoutCreateInfo};

    // Specialization constant for SpecId 0 (bool round). Vulkan expects 4 bytes for bool specialization.
    constexpr uint32_t specRoundTrue = 1u;
    constexpr uint32_t kMapEntryCount = 1u;
    const vk::SpecializationMapEntry mapEntry{kSpecIdRound, /*offset*/ 0u, /*size*/ sizeof(uint32_t)};
    const vk::SpecializationInfo specInfo{
        kMapEntryCount,   // mapEntryCount
        &mapEntry,        // pMapEntries
        sizeof(uint32_t), // dataSize
        &specRoundTrue    // pData
    };

    vk::PipelineCreationFeedback creationFeedback{};
    const vk::PipelineCreationFeedbackCreateInfo feedbackCreateInfo{&creationFeedback, 0, nullptr, nullptr};

    const vk::DataGraphPipelineShaderModuleCreateInfoARM shaderModuleCreateInfo{
        *shaderModule,       // shader module
        "Graph Pipeline",    // name
        &specInfo,           // specialization info (non-null to trigger spec-const passes)
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

    // Creating the pipeline runs our optimizer passes including spec-const defaults and folding
    vk::raii::Pipeline pipeline{&(*device), nullptr, nullptr, graphPipelineCreateInfo};

    ASSERT_TRUE(creationFeedback.flags & vk::PipelineCreationFeedbackFlagBits::eValid);
    ASSERT_GT(creationFeedback.duration, 0);
}

TEST_F(MLEmulationLayerGraphForVulkan, BufferBarrierGraphRewrite) {
    vk::raii::Queue queue(&(*device), device->getPhysicalDevice()->getComputeFamilyIndex(), 0);

    vk::DeviceSize bufferSize = 100;
    auto inputBufferInfo = vk::BufferCreateInfo{{}, bufferSize, vk::BufferUsageFlagBits::eTransferSrc};
    vk::raii::Buffer inputBuffer{&(*device), inputBufferInfo};
    const auto memoryTypeIndices =
        device->getPhysicalDevice()->getMemoryTypeIndices(vk::MemoryPropertyFlagBits::eDeviceLocal, 0xffffffff);
    vk::raii::DeviceMemory inputMemory{&(*device), {bufferSize, memoryTypeIndices[0]}};
    inputBuffer.bindMemory(*inputMemory, 0);

    vk::raii::Buffer outputBuffer{&(*device),
                                  vk::BufferCreateInfo{{}, bufferSize, vk::BufferUsageFlagBits::eTransferDst}};
    vk::raii::DeviceMemory outputMemory{&(*device), {bufferSize, memoryTypeIndices[0]}};
    outputBuffer.bindMemory(*outputMemory, 0);

    const vk::CommandPoolCreateInfo commandPoolCreateInfo{{}, device->getPhysicalDevice()->getComputeFamilyIndex()};
    auto commandPool = vk::raii::CommandPool(&(*device), commandPoolCreateInfo);
    const vk::CommandBufferAllocateInfo commandBufferAllocInfo{*commandPool, vk::CommandBufferLevel::ePrimary, 1};
    vk::raii::CommandBuffers commandBuffers(&(*device), commandBufferAllocInfo);
    auto commandBuffer = std::move(commandBuffers.front());

    const vk::CommandBufferBeginInfo commandBufferBeginInfo{
        vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };
    commandBuffer.begin(commandBufferBeginInfo);

    vk::BufferCopy bufferCopy{0, 0, bufferSize};
    commandBuffer.copyBuffer(*inputBuffer, *outputBuffer, bufferCopy);

    vk::BufferMemoryBarrier2 bufferBarrier2{{}, {}, {}, {}, 0, 0, inputBuffer, 0, bufferSize};
    auto info = vk::DependencyInfo{{}, {}, bufferBarrier2, {}};
    commandBuffer.pipelineBarrier2(info);
    commandBuffer.end();

    vk::raii::Fence fence{&(*device), vk::FenceCreateInfo{}};
    vk::SubmitInfo submitInfo;
    submitInfo.setCommandBuffers(*commandBuffer);
    queue.submit(submitInfo, *fence);
    auto result = (&(*device)).waitForFences({*fence}, vk::True, uint64_t(-1));

    ASSERT_TRUE(result == vk::Result::eSuccess);
}

} // namespace
} // namespace mlsdk::el::tests
