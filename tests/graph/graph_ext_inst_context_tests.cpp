/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include <gtest/gtest.h>

#include "graph_ext_inst_context.hpp"
#include "graph_pass_ext_inst.hpp"
#include "source/opt/build_module.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

TEST(GraphExtInstContext, ReplicatedPatternExpansionTwoToFour) {
    std::vector<int32_t> values{1, 2};
    ASSERT_TRUE(spvtools::opt::tryExpandReplicatedPattern(values, 4));
    ASSERT_EQ(values, (std::vector<int32_t>{1, 2, 1, 2}));
}

TEST(GraphExtInstContext, ReplicatedPatternExpansionRejectsNonDivisibleCounts) { // cppcheck-suppress syntaxError
    std::vector<int32_t> values{1, 2};
    ASSERT_FALSE(spvtools::opt::tryExpandReplicatedPattern(values, 3));
    ASSERT_EQ(values, (std::vector<int32_t>{1, 2}));
}

#ifdef VK_ARM_tensors2
TEST(GraphExtInstContext, RetainsAndValidatesScaledTensorMetadata) {
    const std::string source = R"(
OpCapability Shader
OpCapability GraphARM
OpCapability TensorsARM
OpCapability TensorsBlockScaledARM
OpCapability Float8EXT
OpCapability Float8UnsignedE8M0EXT
OpExtension "SPV_ARM_graph"
OpExtension "SPV_ARM_tensors"
OpExtension "SPV_ARM_tensors_block_scaled"
OpExtension "SPV_EXT_float8"
OpExtension "SPV_EXT_ocp_microscaling_types"
%10 = OpExtInstImport "TOSA.1x.1"
OpMemoryModel Logical GLSL450
OpDecorate %21 DescriptorSet 0
OpDecorate %21 Binding 0
OpDecorate %22 DescriptorSet 0
OpDecorate %22 Binding 1
%1 = OpTypeFloat 8 Float8E4M3EXT
%2 = OpTypeFloat 8 Float8UnsignedE8M0EXT
%3 = OpTypeInt 32 0
%4 = OpConstant %3 1
%5 = OpConstant %3 32
%6 = OpTypeArray %3 %4
%7 = OpConstantComposite %6 %5
%8 = OpTypeTensorScaledBlockARM %1 %2 %7
%11 = OpConstant %3 64
%12 = OpConstantComposite %6 %11
%9 = OpTypeTensorARM %8 %4 %12
%23 = OpTypePointer UniformConstant %9
%21 = OpVariable %23 UniformConstant
%22 = OpVariable %23 UniformConstant
%13 = OpConstant %3 0
%14 = OpTypeGraphARM 1 %9 %9
OpGraphEntryPointARM %15 "main" %21 %22
%15 = OpGraphARM %14
%16 = OpGraphInputARM %9 %13
%20 = OpExtInst %9 %10 ABS %16
OpGraphSetOutputARM %20 %13
OpGraphEndARM
)";
    const std::string blockConstant = "%5 = OpConstant %3 32";
    const auto position = source.find(blockConstant);
    ASSERT_NE(position, std::string::npos);
    auto loader = std::make_shared<vk::detail::DispatchLoaderDynamic>();
    loader->vkGetPhysicalDeviceProperties = [](VkPhysicalDevice, VkPhysicalDeviceProperties *properties) {
        *properties = {};
    };

    for (const auto blockSize : {32u, 0u}) {
        SCOPED_TRACE(blockSize);
        auto text = source;
        text.replace(position, blockConstant.size(), "%5 = OpConstant %3 " + std::to_string(blockSize));
        auto ir = spvtools::BuildModule(
            SPV_ENV_UNIVERSAL_1_6,
            [](spv_message_level_t, const char *, const spv_position_t &, const char *message) {
                ADD_FAILURE() << message;
            },
            text, SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
        ASSERT_NE(ir, nullptr);
        const auto *definition = ir->get_def_use_mgr()->GetDef(20);
        ASSERT_NE(definition, nullptr);
        mlsdk::el::compute::graph_op::GraphPipeline pipeline(loader, VK_NULL_HANDLE, VK_NULL_HANDLE, nullptr);
        spvtools::opt::GraphExtInstContext context(*ir, pipeline);
        if (blockSize == 0) {
            EXPECT_THROW(context.getTensor(*definition), std::invalid_argument);
            continue;
        }
        const auto tensor = context.getTensor(*definition);
        ASSERT_NE(tensor, nullptr);
        EXPECT_EQ(tensor->getFormat(), VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E4M3_ARM);
        const auto &blockScaling = tensor->getBlockScaling();
        ASSERT_TRUE(blockScaling);
        EXPECT_EQ(blockScaling->scaleFormat, VK_FORMAT_R8_UFLOAT_FPENCODING_FLOAT8UNSIGNEDE8M0_ARM);
        EXPECT_EQ(tensor->getDimensions(), (std::vector<int64_t>{64}));
        EXPECT_EQ(blockScaling->blockShape, (std::vector<uint32_t>{32}));

        loader->vkGetPhysicalDeviceQueueFamilyProperties = [](VkPhysicalDevice, uint32_t *count,
                                                              VkQueueFamilyProperties *properties) {
            if (properties) {
                properties[0] = {};
                properties[0].queueFlags = VK_QUEUE_COMPUTE_BIT;
            }
            *count = 1;
        };
        loader->vkCreateTensorARM = [](VkDevice, const VkTensorCreateInfoARM *, const VkAllocationCallbacks *,
                                       VkTensorARM *handle) {
            *handle = VK_NULL_HANDLE;
            return VK_SUCCESS;
        };
        loader->vkCreateTensorViewARM = [](VkDevice, const VkTensorViewCreateInfoARM *, const VkAllocationCallbacks *,
                                           VkTensorViewARM *handle) {
            *handle = VK_NULL_HANDLE;
            return VK_SUCCESS;
        };
        const int64_t dimension = 64;
        const VkTensorDescriptionARM ordinaryDescription{VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM,
                                                         nullptr,
                                                         VK_TENSOR_TILING_LINEAR_ARM,
                                                         VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E4M3_ARM,
                                                         1,
                                                         &dimension,
                                                         nullptr,
                                                         VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM};
        VkDataGraphPipelineResourceInfoARM resource{};
        resource.sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_RESOURCE_INFO_ARM;
        resource.pNext = &ordinaryDescription;
        pipeline.makeDescriptorSetBinding(resource);
        spvtools::opt::GraphPassExtInst pass(pipeline, {});
        try {
            pass.Run(ir.get());
            FAIL() << "Expected a mismatched graph resource to be rejected";
        } catch (const std::invalid_argument &error) {
            EXPECT_STREQ(error.what(), "Block-scaled graph resource does not match its SPIR-V tensor type");
        }
    }
}
#endif

} // namespace
