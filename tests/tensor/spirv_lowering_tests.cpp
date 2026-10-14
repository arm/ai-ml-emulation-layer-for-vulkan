/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include "spirv_tensor_lowering.hpp"
#include "tensor_processor.hpp"
#include "test_utils.hpp"

#include <gtest/gtest.h>
#include <spirv-tools/libspirv.hpp>

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlsdk::el::tests {
namespace {

using namespace layer;

const std::string plainShader = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpName %value "SPV_ARM_tensors"
OpDecorate %value SpecId 7
%void = OpTypeVoid
%function = OpTypeFunction %void
%uint = OpTypeInt 32 0
%value = OpSpecConstant %uint 19
%main = OpFunction %void None %function
%entry = OpLabel
OpNop
OpReturn
OpFunctionEnd
)";

std::vector<uint32_t> assemble(const std::string &source) {
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
    std::vector<uint32_t> spirv;
    if (!tools.Assemble(source, &spirv) || !tools.Validate(spirv)) {
        throw std::runtime_error("Invalid lowering test fixture");
    }
    return spirv;
}

spvtools::MessageConsumer captureDiagnostic(std::string &diagnostic) {
    return [&](spv_message_level_t, const char *, const spv_position_t &, const char *message) {
        diagnostic += std::string(message) + "\n";
    };
}

std::vector<uint32_t> tensorShader(const std::string &access = {}) {
    return assemble(R"(
OpCapability Shader
OpCapability TensorsARM
OpExtension "SPV_ARM_tensors"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %data
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %data DescriptorSet 0
OpDecorate %data Binding 0
%void = OpTypeVoid
%function = OpTypeFunction %void
%uint = OpTypeInt 32 0
%zero = OpConstant %uint 0
%one = OpConstant %uint 1
%tensor = OpTypeTensorARM %uint %one
%pointer = OpTypePointer UniformConstant %tensor
%data = OpVariable %pointer UniformConstant
%array = OpTypeArray %uint %one
%coords = OpConstantComposite %array %zero
%main = OpFunction %void None %function
%entry = OpLabel
%loaded = OpLoad %tensor %data
)" + access + R"(
OpReturn
OpFunctionEnd
)");
}

TEST(TensorSpirvLowering, PreservesNonTensorModuleExactly) {
    const auto input = assemble(plainShader);
    auto output = input;
    bool loweredTensorCompute = true;
    std::string diagnostic;
    EXPECT_TRUE(lowerTensorSpirv(output, loweredTensorCompute, captureDiagnostic(diagnostic)));
    EXPECT_FALSE(loweredTensorCompute);
    EXPECT_EQ(output, input); // Includes OpNop, names and specialization IDs/defaults.
    EXPECT_TRUE(diagnostic.empty());
    EXPECT_TRUE(spvtools::SpirvTools(SPV_ENV_VULKAN_1_3).Validate(output));
}

TEST(TensorSpirvLowering, RejectsTensorTypeAndEachAccess) {
    for (const auto &[access, name] : std::vector<std::pair<std::string, std::string>>{
             {"", "OpTypeTensorARM"},
             {"%value = OpTensorReadARM %uint %loaded %coords", "OpTensorReadARM"},
             {"OpTensorWriteARM %loaded %coords %one", "OpTensorWriteARM"},
             {"%size = OpTensorQuerySizeARM %uint %loaded %zero", "OpTensorQuerySizeARM"}}) {
        SCOPED_TRACE(name);
        auto spirv = tensorShader(access);
        bool loweredTensorCompute = true;
        std::string diagnostic;
        EXPECT_FALSE(lowerTensorSpirv(spirv, loweredTensorCompute, captureDiagnostic(diagnostic)));
        EXPECT_FALSE(loweredTensorCompute);
        EXPECT_TRUE(spirv.empty());
        EXPECT_NE(diagnostic.find("Unsupported TensorARM"), std::string::npos);
    }
}

TEST(TensorSpirvLowering, RejectsTensorDeclarationsWithoutAccesses) {
    for (const std::string capability :
         {"", "OpCapability TensorsARM\n",
          "OpCapability TensorsARM\nOpCapability StorageTensorArrayDynamicIndexingARM\n",
          "OpCapability TensorsARM\nOpCapability StorageTensorArrayNonUniformIndexingARM\n"}) {
        auto shader = plainShader;
        shader.insert(shader.find("OpMemoryModel"), capability + "OpExtension \"SPV_ARM_tensors\"\n");
        auto spirv = assemble(shader);
        bool loweredTensorCompute = true;
        std::string diagnostic;
        EXPECT_FALSE(lowerTensorSpirv(spirv, loweredTensorCompute, captureDiagnostic(diagnostic)));
        EXPECT_FALSE(loweredTensorCompute);
        EXPECT_TRUE(spirv.empty());
        EXPECT_NE(diagnostic.find("Unsupported TensorARM"), std::string::npos);
    }
}

TEST(TensorSpirvLowering, RejectsMalformedBinaries) {
    const auto valid = assemble(plainShader);
    std::vector<std::vector<uint32_t>> malformed{{}, {valid.begin(), valid.begin() + 4}};
    auto wrongMagic = valid;
    wrongMagic[0] = 0;
    malformed.push_back(wrongMagic);
    auto zeroWordCount = valid;
    zeroWordCount[5] &= 0xffff;
    malformed.push_back(zeroWordCount);
    auto truncatedInstruction = valid;
    truncatedInstruction.back() = (2u << 16) | (truncatedInstruction.back() & 0xffff);
    malformed.push_back(truncatedInstruction);
    auto invalidOpcode = valid;
    invalidOpcode.back() = (1u << 16) | 0xffff;
    malformed.push_back(invalidOpcode);
    auto malformedTensor = tensorShader();
    malformedTensor.push_back(0);
    malformed.push_back(malformedTensor);
    for (const auto &input : malformed) {
        auto output = input;
        bool loweredTensorCompute = true;
        std::string diagnostic;
        EXPECT_FALSE(lowerTensorSpirv(output, loweredTensorCompute, captureDiagnostic(diagnostic)));
        EXPECT_FALSE(loweredTensorCompute);
        EXPECT_TRUE(output.empty());
        EXPECT_FALSE(diagnostic.empty());
        EXPECT_EQ(diagnostic.find("Unsupported TensorARM"), std::string::npos);
        const TensorProcessor processor(input, TensorLoweringMode::DirectSpirv);
        EXPECT_FALSE(processor.isValidShader());
        EXPECT_TRUE(processor.getNewSpirv().empty());
    }
}

TEST(TensorSpirvLowering, DirectProcessorPreservesNonTensorShader) {
    const auto input = assemble(plainShader);
    const TensorProcessor processor(input, TensorLoweringMode::DirectSpirv);
    EXPECT_TRUE(processor.isValidShader());
    EXPECT_FALSE(processor.isTensorComputeShader());
    EXPECT_EQ(processor.getNewSpirv(), input);
}

// Environment-switch tests run in separate desktop processes to isolate their
// environment changes from other runtime tests.
TEST(TensorLoweringMode, UnsetUsesGlslToSpirv) {
    const ScopedEnvironment environment("VMEL_EXPERIMENTAL_TENSOR_LOWERING_MODE", std::nullopt);
    EXPECT_EQ(getTensorLoweringMode(), TensorLoweringMode::GlslToSpirv);
    const auto input = tensorShader();
    const TensorProcessor processor(input);
    ASSERT_TRUE(processor.isValidShader());
    ASSERT_TRUE(processor.isTensorComputeShader());
    const auto output = processor.getNewSpirv();
    EXPECT_NE(output, input);
    EXPECT_TRUE(spvtools::SpirvTools(SPV_ENV_VULKAN_1_3).Validate(output));
    auto directOutput = output;
    bool loweredTensorCompute = true;
    EXPECT_TRUE(lowerTensorSpirv(directOutput, loweredTensorCompute, {}));
    EXPECT_FALSE(loweredTensorCompute);
    EXPECT_EQ(directOutput, output);
}

TEST(TensorLoweringMode, ZeroUsesGlslToSpirv) {
    const ScopedEnvironment environment("VMEL_EXPERIMENTAL_TENSOR_LOWERING_MODE", "0");
    EXPECT_EQ(getTensorLoweringMode(), TensorLoweringMode::GlslToSpirv);
    const TensorProcessor processor(tensorShader());
    ASSERT_TRUE(processor.isValidShader());
    EXPECT_TRUE(spvtools::SpirvTools(SPV_ENV_VULKAN_1_3).Validate(processor.getNewSpirv()));
}

TEST(TensorLoweringMode, OneRejectsTensorWithoutFallback) {
    const ScopedEnvironment environment("VMEL_EXPERIMENTAL_TENSOR_LOWERING_MODE", "1");
    EXPECT_EQ(getTensorLoweringMode(), TensorLoweringMode::DirectSpirv);
    const TensorProcessor processor(tensorShader());
    EXPECT_FALSE(processor.isValidShader());
    EXPECT_TRUE(processor.getNewSpirv().empty());
}

} // namespace
} // namespace mlsdk::el::tests
