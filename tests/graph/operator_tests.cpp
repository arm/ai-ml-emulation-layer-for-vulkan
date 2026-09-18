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
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

namespace mlsdk::el::tests {

using namespace utilities;

namespace {

TEST_F(MLEmulationLayerGraphForVulkan, MaxPool2D) {

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

    ASSERT_TRUE(outputTensor->compare(reinterpret_cast<const int8_t *>(&ref[0]), sizeof(ref))) << "Output mismatch";
}

TEST_F(MLEmulationLayerGraphForVulkan, TwoLayerMaxPool2D) {

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

    ASSERT_TRUE(outputTensor->compare(reinterpret_cast<const int8_t *>(&ref[0]), sizeof(ref))) << "Output mismatch";
}

TEST_F(MLEmulationLayerGraphForVulkan, Concat) {

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

    ASSERT_TRUE(outputTensor->compare(&ref[0][0][0][0], sizeof(ref))) << "Output mismatch";
}

TEST_F(MLEmulationLayerGraphForVulkan, Slice) {

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

    ASSERT_TRUE(outputTensor->compare(reinterpret_cast<const int8_t *>(&ref[0][0][0][0]), sizeof(ref)))
        << "Output mismatch";
}

TEST_F(MLEmulationLayerGraphForVulkan, NOPOutputs) {

    auto inputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 8, 8, 3}});
    auto outputTensor0 = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 2, 2, 2}});
    auto outputTensor1 = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 8, 8, 3}});
    auto outputTensor2 = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 8, 8, 3}});

    const GraphPipeline::DescriptorMap descriptorMap = {
        {
            // set 0
            {
                0,             // binding
                {inputTensor}, // tensor
            },
            {
                1,               // binding
                {outputTensor0}, // tensor
            },
            {
                2,               // binding
                {outputTensor1}, // tensor
            },
            {
                3,               // binding
                {outputTensor2}, // tensor
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
    outputTensor0->print();

    std::cout << "OUTPUT_1 from Input" << std::endl;
    outputTensor1->print();

    std::cout << "CONST TENSOR" << std::endl;
    graphConstants[0].print();

    std::cout << "OUTPUT_2 from Const Tensor" << std::endl;
    outputTensor2->print();

    ASSERT_TRUE(outputTensor0->compare(reinterpret_cast<const int8_t *>(&ref[0][0][0][0]), sizeof(ref)))
        << "Output mismatch in OUTPUT_0";

    ASSERT_TRUE(outputTensor1->compare(reinterpret_cast<const int8_t *>(inputTensor->data()), inputTensor->size()))
        << "Output mismatch in OUTPUT_1";
    ASSERT_TRUE(
        outputTensor2->compare(reinterpret_cast<const int8_t *>(graphConstants[0].data()), graphConstants[0].size()))
        << "Output mismatch in OUTPUT_2";
}

TEST_F(MLEmulationLayerGraphForVulkan, GraphConstantARM) {

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

    ASSERT_TRUE(outputTensor->compare(&ref[0], sizeof(ref))) << "Output mismatch";
}

TEST_F(MLEmulationLayerGraphForVulkan, HigherRankConstant) {

    auto inputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 3, 3, 1}});
    std::fill(inputTensor->data(), inputTensor->data() + inputTensor->size(), 1);

    auto outputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 3, 3, 1}});
    const GraphPipeline::DescriptorMap descriptorMap = {
        {
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

    GraphConstants graphConstants;

    const auto spirv = assembleSpirv(fileToString("inlined-higher-rank-constant.spvasm"));
    auto graphPipeline = std::make_shared<GraphPipeline>(device, descriptorMap, graphConstants, spirv);

    graphPipeline->dispatchSubmit();

    std::cout << "INPUT" << std::endl;
    inputTensor->print();

    std::cout << "OUTPUT" << std::endl;
    outputTensor->print();

    const uint8_t ref[1][3][3][1] = {
        {
            {{24}, {33}, {20}},
            {{27}, {36}, {21}},
            {{12}, {15}, {8}},
        },
    };

    ASSERT_TRUE(outputTensor->compare(&ref[0][0][0][0], sizeof(ref))) << "Output mismatch";
}

TEST_F(MLEmulationLayerGraphForVulkan, Maximum) {

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

    ASSERT_TRUE(outputTensor->compare(&ref[0][0][0][0], sizeof(ref))) << "Output mismatch";
}

TEST_F(MLEmulationLayerGraphForVulkan, FFT2D) {

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

TEST_F(MLEmulationLayerGraphForVulkan, SIN) {

    auto inputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR32Sfloat, std::vector<int64_t>{1}});
    auto outputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR32Sfloat, std::vector<int64_t>{1}});
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

    float32 value = 0.000111731846f;
    *(reinterpret_cast<float32 *>(inputTensor->data())) = value;

    const auto spirv = assembleSpirv(fileToString("sin.spvasm"));
    auto graphPipeline = std::make_shared<GraphPipeline>(device, descriptorMap, GraphConstants{}, spirv);

    graphPipeline->dispatchSubmit();

    std::cout << "INPUT" << std::endl;
    std::cout << std::setprecision(9) << *(reinterpret_cast<float32 *>(inputTensor->data()));
    inputTensor->print();
    std::cout << std::endl;

    std::cout << "OUTPUT" << std::endl;
    std::cout << std::setprecision(9) << *(reinterpret_cast<float32 *>(outputTensor->data()));
    outputTensor->print();
    std::cout << std::endl;

    float64 reference = 0.00011173184588586903;
    float64 errorBound = 7.715463482888105e-08;
    float64 refMin = reference - errorBound;
    float64 refMax = reference + errorBound;

    float64 output = *(reinterpret_cast<float32 *>(outputTensor->data()));

    ASSERT_GE(output, refMin);
    ASSERT_LE(output, refMax);
}

} // namespace
} // namespace mlsdk::el::tests
