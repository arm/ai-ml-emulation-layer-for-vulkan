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
#include <cstring>
#include <memory>
#include <sstream>
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

std::vector<uint32_t> makeRawSadModule(bool cost16) {
    std::ostringstream module;
    module << R"(
OpCapability Shader
OpCapability Int8
OpCapability Int16
OpCapability VulkanMemoryModel
OpCapability GraphARM
OpCapability TensorsARM
OpExtension "SPV_ARM_graph"
OpExtension "SPV_ARM_tensors"
%me = OpExtInstImport "Arm.MotionEngine.100"
OpMemoryModel Logical Vulkan
OpDecorate %output DescriptorSet 0
OpDecorate %output Binding 0
OpDecorate %input0 DescriptorSet 0
OpDecorate %input0 Binding 1
OpDecorate %input1 DescriptorSet 0
OpDecorate %input1 Binding 2
%u32 = OpTypeInt 32 0
%u8 = OpTypeInt 8 0
%u16 = OpTypeInt 16 0
%c0 = OpConstant %u32 0
%c1 = OpConstant %u32 1
%c2 = OpConstant %u32 2
%c3 = OpConstant %u32 3
%c4 = OpConstant %u32 4
%shape_type = OpTypeArray %u32 %c1
%shape = OpConstantComposite %shape_type %c2
%pair = OpTypeTensorARM %u32 %c1 %shape
%zeros = OpConstantComposite %pair %c0 %c0
%ones = OpConstantComposite %pair %c1 %c1
%search = OpConstantComposite %pair %c1 %c2
%input_type = OpTypeTensorARM %u8 %c4
%input_ptr = OpTypePointer UniformConstant %input_type
%input0 = OpVariable %input_ptr UniformConstant
%input1 = OpVariable %input_ptr UniformConstant
)";
    if (cost16) {
        module << "%cost_type = OpTypeTensorARM %u16 %c4\n"
               << "%output_ptr = OpTypePointer UniformConstant %cost_type\n";
    }
    const auto *costType = cost16 ? "%cost_type" : "%input_type";
    module << "%kernel = OpConstantComposite %pair %c1 " << (cost16 ? "%c3" : "%c1") << "\n"
           << "%output = OpVariable " << (cost16 ? "%output_ptr" : "%input_ptr") << " UniformConstant\n"
           << "%graph_type = OpTypeGraphARM 2 %input_type %input_type " << costType << "\n"
           << "OpGraphEntryPointARM %graph \"raw_sad\" %input0 %input1 %output\n"
           << "%graph = OpGraphARM %graph_type\n"
           << "%in0 = OpGraphInputARM %input_type %c0\n"
           << "%in1 = OpGraphInputARM %input_type %c1\n"
           << "%cost = OpExtInst " << costType << " %me RAW_SAD %kernel %search %ones %ones %zeros %zeros %in0 %in1\n"
           << "OpGraphSetOutputARM %cost %c0\nOpGraphEndARM\n";
    return assembleSpirv(module.str());
}

template <typename Cost>
void runRawSadStorageCase(std::shared_ptr<Device> &device, vk::Format inputFormat, vk::Format costFormat,
                          const std::vector<Cost> &expected) {
    const auto inputTemplate =
        std::make_shared<Tensor>(device, Shape{inputFormat, {1, 1, 1, 5}}, std::vector<uint8_t>(5, 0));
    const auto inputSearch =
        std::make_shared<Tensor>(device, Shape{inputFormat, {1, 1, 1, 5}}, std::vector<uint8_t>{0, 127, 128, 255, 1});
    // Two search positions produce two cost channels; inspect every output byte.
    const auto output = std::make_shared<Tensor>(device, Shape{costFormat, {1, 2, 1, 5}});
    const GraphPipeline::DescriptorMap descriptors = {{{0, {output}}, {1, {inputTemplate}}, {2, {inputSearch}}}};
    const auto module = makeRawSadModule(sizeof(Cost) == 2);
    GraphPipeline pipeline(device, descriptors, GraphConstants{}, module);
    pipeline.dispatchSubmit();
    ASSERT_EQ(output->size(), expected.size() * sizeof(Cost));
    std::vector<Cost> actual(expected.size());
    std::memcpy(actual.data(), output->data(), output->size());
    EXPECT_EQ(actual, expected);
}

TEST_F(MLEmulationLayerGraphForVulkan, BlockMatchRawSadUint8) {
    runRawSadStorageCase<uint8_t>(device, vk::Format::eR8Uint, vk::Format::eR8Uint,
                                  {0, 127, 128, 255, 1, 127, 128, 255, 1, 0});
}

TEST_F(MLEmulationLayerGraphForVulkan, BlockMatchRawSadSint8) {
    runRawSadStorageCase<uint8_t>(device, vk::Format::eR8Sint, vk::Format::eR8Sint,
                                  {0, 127, 128, 255, 1, 127, 128, 255, 1, 0});
}

TEST_F(MLEmulationLayerGraphForVulkan, BlockMatchRawSadUint16) {
    runRawSadStorageCase<uint16_t>(device, vk::Format::eR8Uint, vk::Format::eR16Uint,
                                   {255, 510, 384, 256, 1, 510, 384, 256, 1, 0});
}

TEST_F(MLEmulationLayerGraphForVulkan, BlockMatchRawSadSint16) {
    runRawSadStorageCase<uint16_t>(device, vk::Format::eR8Sint, vk::Format::eR16Sint,
                                   {255, 510, 384, 256, 1, 510, 384, 256, 1, 0});
}

} // namespace
} // namespace mlsdk::el::tests
