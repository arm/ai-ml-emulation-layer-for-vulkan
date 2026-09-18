/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include "mlel/pipeline.hpp"
#include "mlel/tensor.hpp"
#include "mlel/utils.hpp"
#include "vulkan_test_utils.hpp"
#include <cstdint>
#include <memory>
#include <vector>

namespace mlsdk::el::tests {

using namespace utilities;

namespace {

TEST_F(MLEmulationLayerGraphForVulkan, BlockMatchMinSadCost) {
    const int64_t height = 5;
    const int64_t width = 5;
    // clang-format off
    std::vector<uint8_t> inputTemplateData = {
        0,  0,  0,  0,  0,
        0, 10, 10, 10,  0,
        0, 10, 50, 10,  0,
        0, 10, 10, 10,  0,
        0,  0,  0,  0,  0,
    };
    std::vector<uint8_t> inputSearchData = {
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
        0,  0, 10, 10, 10,
        0,  0, 10, 50, 10,
        0,  0, 10, 10, 10,
    };
    // clang-format on

    auto inputTemplate = std::make_shared<Tensor>(
        device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 1, height, width}}, inputTemplateData);
    auto inputSearch = std::make_shared<Tensor>(
        device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 1, height, width}}, inputSearchData);

    auto outputFlow =
        std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 1, height, width * 2}});
    auto outputCost =
        std::make_shared<Tensor>(device, Shape{vk::Format::eR16Uint, std::vector<int64_t>{1, 1, height, width}});

    const GraphPipeline::DescriptorMap descriptorMap = {
        {
            {0, {outputFlow}},
            {1, {outputCost}},
            {2, {inputTemplate}},
            {3, {inputSearch}},
        },
    };

    const auto spirv = assembleSpirv(fileToString("me_min_sad_cost_sr0.spvasm"));
    auto graphPipeline = std::make_shared<GraphPipeline>(device, descriptorMap, GraphConstants{}, spirv);

    graphPipeline->dispatchSubmit();

    const int8_t refFlow[1][1][5][10] = {{{
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    }}};

    ASSERT_TRUE(outputFlow->compare(&refFlow[0][0][0][0], sizeof(refFlow))) << "Output mismatch";

    const int32_t refCost[1][1][5][5] = {{{
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
    }}};

    ASSERT_TRUE(outputCost->compare(&refCost[0][0][0][0], sizeof(refCost))) << "Output mismatch";
}

TEST_F(MLEmulationLayerGraphForVulkan, BlockMatchMinSad) {
    const int64_t height = 5;
    const int64_t width = 5;
    // clang-format off
    std::vector<uint8_t> inputTemplateData = {
        0,  0,  0,  0,  0,
        0, 10, 10, 10,  0,
        0, 10, 50, 10,  0,
        0, 10, 10, 10,  0,
        0,  0,  0,  0,  0,
    };
    std::vector<uint8_t> inputSearchData = {
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
        0,  0, 10, 10, 10,
        0,  0, 10, 50, 10,
        0,  0, 10, 10, 10,
    };
    // clang-format on

    auto inputTemplate = std::make_shared<Tensor>(
        device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 1, height, width}}, inputTemplateData);
    auto inputSearch = std::make_shared<Tensor>(
        device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 1, height, width}}, inputSearchData);

    auto outputFlow =
        std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 1, height, width * 2}});

    const GraphPipeline::DescriptorMap descriptorMap = {
        {
            {0, {outputFlow}},
            {1, {inputTemplate}},
            {2, {inputSearch}},
        },
    };

    const auto spirvMinSAD = assembleSpirv(fileToString("me_min_sad_sr0.spvasm"));
    auto graphPipeline = std::make_shared<GraphPipeline>(device, descriptorMap, GraphConstants{}, spirvMinSAD);

    graphPipeline->dispatchSubmit();

    const int8_t ref[1][1][5][10] = {{{
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    }}};

    ASSERT_TRUE(outputFlow->compare(&ref[0][0][0][0], sizeof(ref))) << "Output mismatch";
}

TEST_F(MLEmulationLayerGraphForVulkan, BlockMatchRawSad) {
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

    const GraphPipeline::DescriptorMap descriptorMap = {
        {
            {0, {outputCost}},
            {1, {inputTemplate}},
            {2, {inputSearch}},
        },
    };

    const auto spirvRawSAD = assembleSpirv(fileToString("me_raw_sad_sr0.spvasm"));
    auto graphPipeline = std::make_shared<GraphPipeline>(device, descriptorMap, GraphConstants{}, spirvRawSAD);

    graphPipeline->dispatchSubmit();

    const uint16_t ref[1][1][5][5] = {{{
        {4, 6, 6, 6, 4},
        {6, 9, 9, 9, 6},
        {6, 9, 9, 9, 6},
        {6, 9, 9, 9, 6},
        {4, 6, 6, 6, 4},
    }}};

    ASSERT_TRUE(outputCost->compare(&ref[0][0][0][0], sizeof(ref))) << "Output mismatch";
}

} // namespace
} // namespace mlsdk::el::tests
