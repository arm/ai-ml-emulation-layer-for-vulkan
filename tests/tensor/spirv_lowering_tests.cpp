/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include "shaders/precompiled_shaders.hpp"
#include "spirv_tensor_lowering.hpp"
#include "tensor_processor.hpp"
#include "test_utils.hpp"

#include <gtest/gtest.h>
#include <spirv-tools/libspirv.hpp>
#include <spirv-tools/optimizer.hpp>

#include <algorithm>
#include <cstddef>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
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

std::vector<uint32_t> tensorShader(const std::string &access = {}, uint32_t rank = 1, bool specialised = false,
                                   const std::string &decorations = {}) {
    std::string zeros;
    for (uint32_t i = 0; i < rank; ++i) {
        zeros += " %zero";
    }
    return assemble(R"(
OpCapability Shader
OpCapability TensorsARM
OpExtension "SPV_ARM_tensors"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %data
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %data DescriptorSet 0
OpDecorate %data Binding 0
)" + decorations + std::string(specialised ? "OpDecorate %rank SpecId 9\n" : "") +
                    R"(
%void = OpTypeVoid
%function = OpTypeFunction %void
%uint = OpTypeInt 32 0
%zero = OpConstant %uint 0
%one = OpConstant %uint 1
)" + "%rank = " + (specialised ? "OpSpecConstant" : "OpConstant") +
                    " %uint " + std::to_string(rank) + R"(
%tensor = OpTypeTensorARM %uint %rank
%pointer = OpTypePointer UniformConstant %tensor
%data = OpVariable %pointer UniformConstant
%array = OpTypeArray %uint %rank
)" + (specialised ? "%coords = OpConstantNull %array" : "%coords = OpConstantComposite %array" + zeros) +
                    R"(
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

void expectLowered(std::vector<uint32_t> &spirv) {
    bool loweredTensorCompute = false;
    std::string diagnostic;
    ASSERT_TRUE(lowerTensorSpirv(spirv, loweredTensorCompute, captureDiagnostic(diagnostic))) << diagnostic;
    EXPECT_TRUE(loweredTensorCompute);
    EXPECT_TRUE(diagnostic.empty());
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
    tools.SetMessageConsumer(
        [](spv_message_level_t, const char *, const spv_position_t &position, const char *message) {
            ADD_FAILURE() << "Instruction " << position.index << ": " << message;
        });
    ASSERT_TRUE(tools.Validate(spirv));
    auto unchanged = spirv;
    ASSERT_TRUE(lowerTensorSpirv(unchanged, loweredTensorCompute, captureDiagnostic(diagnostic))) << diagnostic;
    EXPECT_FALSE(loweredTensorCompute);
    EXPECT_TRUE(diagnostic.empty());
    EXPECT_EQ(unchanged, spirv);
}

TEST(TensorSpirvLowering, LowersTensorTypeAndEachScalarAccess) {
    for (const auto &[access, name] : std::vector<std::pair<std::string, std::string>>{
             {"", "OpTypeTensorARM"},
             {"%value = OpTensorReadARM %uint %loaded %coords", "OpTensorReadARM"},
             {"OpTensorWriteARM %loaded %coords %one", "OpTensorWriteARM"},
             {"%size = OpTensorQuerySizeARM %uint %loaded %zero", "OpTensorQuerySizeARM"}}) {
        SCOPED_TRACE(name);
        auto spirv = tensorShader(access);
        ASSERT_NO_FATAL_FAILURE(expectLowered(spirv));
    }
}

TEST(TensorSpirvLowering, RemovesUnusedTensorDeclarations) {
    for (const std::string capability :
         {"", "OpCapability TensorsARM\n",
          "OpCapability TensorsARM\nOpCapability StorageTensorArrayDynamicIndexingARM\n",
          "OpCapability TensorsARM\nOpCapability StorageTensorArrayNonUniformIndexingARM\n"}) {
        auto shader = plainShader;
        shader.insert(shader.find("OpMemoryModel"), capability + "OpExtension \"SPV_ARM_tensors\"\n");
        auto spirv = assemble(shader);
        ASSERT_NO_FATAL_FAILURE(expectLowered(spirv));
    }
}

TEST(TensorSpirvLowering, KeepsTensorAccessRestrictionsOffDescriptorBuffers) {
    for (const auto &[decoration, access] : std::vector<std::pair<std::string, std::string>>{
             {"NonWritable", "%value = OpTensorReadARM %uint %loaded %coords"},
             {"NonReadable", "OpTensorWriteARM %loaded %coords %one"}}) {
        SCOPED_TRACE(decoration);
        auto spirv = tensorShader(access + "\n%size = OpTensorQuerySizeARM %uint %loaded %zero", 1, false,
                                  "OpDecorate %data " + decoration + "\n");
        ASSERT_NO_FATAL_FAILURE(expectLowered(spirv));
        std::string text;
        ASSERT_TRUE(spvtools::SpirvTools(SPV_ENV_VULKAN_1_3).Disassemble(spirv, &text));
        EXPECT_EQ(text.find("NonReadable"), std::string::npos);
        EXPECT_EQ(text.find("NonWritable"), std::string::npos);
        EXPECT_NE(text.find("DescriptorSet 0"), std::string::npos);
        EXPECT_NE(text.find("Binding 0"), std::string::npos);
    }
}

TEST(TensorSpirvLowering, RejectsUnsupportedTensorMemoryQualifiers) {
    for (const std::string decoration : {"Coherent", "Volatile"}) {
        SCOPED_TRACE(decoration);
        auto spirv = tensorShader("%value = OpTensorReadARM %uint %loaded %coords", 1, false,
                                  "OpDecorate %data " + decoration + "\n");
        bool loweredTensorCompute = true;
        std::string diagnostic;
        EXPECT_FALSE(lowerTensorSpirv(spirv, loweredTensorCompute, captureDiagnostic(diagnostic)));
        EXPECT_FALSE(loweredTensorCompute);
        EXPECT_TRUE(spirv.empty());
        EXPECT_NE(diagnostic.find("decoration"), std::string::npos);
    }
}

TEST(TensorSpirvLowering, LowersFixedRanksAndReusesAccessFunctions) {
    for (uint32_t rank = 1; rank <= 6; ++rank) {
        SCOPED_TRACE(rank);
        auto spirv = tensorShader(R"(
%a = OpTensorReadARM %uint %loaded %coords
%b = OpTensorReadARM %uint %loaded %coords
OpTensorWriteARM %loaded %coords %a
OpTensorWriteARM %loaded %coords %b
)",
                                  rank);
        ASSERT_NO_FATAL_FAILURE(expectLowered(spirv));
        size_t functions = 0;
        for (size_t offset = 5; offset < spirv.size(); offset += spirv[offset] >> 16) {
            functions += (spirv[offset] & 0xffff) == static_cast<uint32_t>(spv::OpFunction);
        }
        EXPECT_EQ(functions, 3u); // Entry point, shared read helper, shared write helper.
    }
}

TEST(TensorSpirvLowering, LowersGatherAndScatterVariants) {
    size_t variants = 0;
    for (const auto &[name, module] : compute::precompiledSpirvModules) {
        if (name.rfind("gather_", 0) != 0 && name.rfind("scatter_", 0) != 0) {
            continue;
        }
        SCOPED_TRACE(name);
        const auto &[words, size] = module;
        std::vector<uint32_t> spirv{words, words + size};
        ASSERT_NO_FATAL_FAILURE(expectLowered(spirv));
        ++variants;
    }
    EXPECT_EQ(variants, 16u);
}

std::vector<uint32_t> specializationInstructions(const std::vector<uint32_t> &spirv) {
    if (spirv.size() < 5) {
        throw std::runtime_error("Truncated SPIR-V header");
    }
    std::vector<uint32_t> result;
    for (size_t offset = 5; offset < spirv.size();) {
        const auto wordCount = spirv[offset] >> 16;
        // Check lengths before reading operands or copying instructions.
        if (wordCount == 0 || wordCount > spirv.size() - offset) {
            throw std::runtime_error("Invalid SPIR-V instruction length at word " + std::to_string(offset));
        }
        const auto opcode = static_cast<spv::Op>(spirv[offset] & 0xffff);
        if (opcode == spv::Op::OpDecorate && wordCount < 3) {
            throw std::runtime_error("Truncated OpDecorate at word " + std::to_string(offset));
        }
        if (opcode == spv::Op::OpSpecConstant || opcode == spv::Op::OpSpecConstantOp ||
            opcode == spv::Op::OpSpecConstantComposite || opcode == spv::Op::OpSpecConstantTrue ||
            opcode == spv::Op::OpSpecConstantFalse ||
            (opcode == spv::Op::OpDecorate && spirv[offset + 2] == static_cast<uint32_t>(spv::DecorationSpecId))) {
            result.insert(result.end(), spirv.data() + offset, spirv.data() + offset + wordCount);
        }
        offset += wordCount;
    }
    return result;
}

void expectSpecialized(const std::vector<uint32_t> &spirv,
                       const std::unordered_map<uint32_t, std::vector<uint32_t>> &values) {
    spvtools::Optimizer optimizer(SPV_ENV_VULKAN_1_3);
    optimizer.RegisterPass(spvtools::CreateSetSpecConstantDefaultValuePass(values));
    optimizer.RegisterPass(spvtools::CreateFreezeSpecConstantValuePass());
    optimizer.RegisterPass(spvtools::CreateFoldSpecConstantOpAndCompositePass());
    optimizer.RegisterPass(spvtools::CreateDeadBranchElimPass());
    std::vector<uint32_t> specialized;
    ASSERT_TRUE(optimizer.Run(spirv.data(), spirv.size(), &specialized));
    EXPECT_TRUE(spvtools::SpirvTools(SPV_ENV_VULKAN_1_3).Validate(specialized));
}

TEST(TensorSpirvLowering, PreservesSpecialisedRank) {
    const auto input = tensorShader(R"(
%value = OpTensorReadARM %uint %loaded %coords
OpTensorWriteARM %loaded %coords %value
)",
                                    1, true);
    auto spirv = input;
    ASSERT_NO_FATAL_FAILURE(expectLowered(spirv));
    EXPECT_EQ(specializationInstructions(spirv), specializationInstructions(input));
    for (uint32_t rank = 1; rank <= 6; ++rank) {
        SCOPED_TRACE(rank);
        ASSERT_NO_FATAL_FAILURE(expectSpecialized(spirv, {{9, {rank}}}));
    }
}

TEST(TensorSpirvLowering, LowersNumericScalarLayoutVariants) {
    std::map<std::string_view, size_t> operators{{"cast", 0},    {"clamp", 0},   {"negate", 0},
                                                 {"concat", 0},  {"slice", 0},   {"transpose", 0},
                                                 {"reshape", 0}, {"reverse", 0}, {"tile", 0}};
    for (const auto &[name, module] : compute::precompiledSpirvModules) {
        const auto it = operators.find(name.substr(0, name.find('_')));
        if (it == operators.end() || name.find("_bool") != std::string::npos ||
            name.rfind("transpose_conv2d_", 0) == 0) {
            continue;
        }
        SCOPED_TRACE(name);
        const auto &[words, size] = module;
        const std::vector<uint32_t> input(words, words + size);
        auto spirv = input;
        ASSERT_NO_FATAL_FAILURE(expectLowered(spirv));
        EXPECT_EQ(specializationInstructions(spirv), specializationInstructions(input));
        ++it->second;
    }
    for (const auto &[name, variants] : operators) {
        EXPECT_GT(variants, 0u) << name;
    }
}

TEST(TensorSpirvLowering, PreservesIndependentReshapeRanks) {
    const auto &[words, size] = compute::precompiledSpirvModules.at("reshape_float");
    std::vector<uint32_t> spirv{words, words + size};
    ASSERT_NO_FATAL_FAILURE(expectLowered(spirv));
    for (uint32_t inputRank = 1; inputRank <= 6; ++inputRank) {
        for (uint32_t outputRank = 1; outputRank <= 6; ++outputRank) {
            SCOPED_TRACE(std::to_string(inputRank) + " -> " + std::to_string(outputRank));
            ASSERT_NO_FATAL_FAILURE(expectSpecialized(spirv, {{0, {inputRank}}, {1, {outputRank}}}));
        }
    }
}

TEST(TensorSpirvLowering, RejectsMalformedRankOperands) {
    const auto valid = tensorShader();
    size_t tensorOffset = 0;
    uint32_t loadedTensor = 0;
    for (size_t offset = 5; offset < valid.size(); offset += valid[offset] >> 16) {
        const auto opcode = static_cast<spv::Op>(valid[offset] & 0xffff);
        if (opcode == spv::Op::OpTypeTensorARM) {
            tensorOffset = offset;
        } else if (opcode == spv::Op::OpLoad) {
            loadedTensor = valid[offset + 2];
        }
    }
    ASSERT_NE(tensorOffset, 0u);
    ASSERT_NE(loadedTensor, 0u);
    for (const auto &[rank, reason] : std::vector<std::pair<uint32_t, std::string>>{
             {valid[3], "Undefined TensorARM operand"}, {loadedTensor, "Expected an integer TensorARM operand"}}) {
        SCOPED_TRACE(reason);
        auto malformed = valid;
        malformed[3] = std::max(malformed[3], rank + 1);
        malformed[tensorOffset + 3] = rank;
        bool loweredTensorCompute = true;
        std::string diagnostic;
        EXPECT_FALSE(lowerTensorSpirv(malformed, loweredTensorCompute, captureDiagnostic(diagnostic)));
        EXPECT_FALSE(loweredTensorCompute);
        EXPECT_TRUE(malformed.empty());
        // Debug validation can diagnose malformed input before the lowering's own checks.
        EXPECT_FALSE(diagnostic.empty());
    }
}

TEST(TensorSpirvLowering, PreservesDescriptorLayoutAndUsesWideByteOffsets) {
    auto spirv = tensorShader(R"(
%value = OpTensorReadARM %uint %loaded %coords
OpTensorWriteARM %loaded %coords %value
)",
                              6);
    ASSERT_NO_FATAL_FAILURE(expectLowered(spirv));
    std::map<uint32_t, std::map<uint32_t, uint32_t>> memberOffsets;
    std::map<uint32_t, uint32_t> arrayStrides;
    uint32_t wideType = 0;
    uint32_t metadataArray = 0;
    size_t multiplies = 0;
    for (size_t offset = 5; offset < spirv.size(); offset += spirv[offset] >> 16) {
        const auto *words = spirv.data() + offset;
        const auto opcode = static_cast<spv::Op>(words[0] & 0xffff);
        if (opcode == spv::OpTypeInt && words[2] == 64 && words[3] == 0) {
            wideType = words[1];
        } else if (opcode == spv::OpMemberDecorate && words[3] == spv::DecorationOffset) {
            memberOffsets[words[1]][words[2]] = words[4];
        } else if (opcode == spv::OpTypeArray && words[2] == wideType) {
            metadataArray = words[1];
        } else if (opcode == spv::OpDecorate && words[2] == spv::DecorationArrayStride) {
            arrayStrides[words[1]] = words[3];
        } else if (opcode == spv::OpIMul || opcode == spv::OpIAdd) {
            EXPECT_NE(wideType, 0u);
            EXPECT_EQ(words[1], wideType);
            multiplies += opcode == spv::OpIMul;
        }
    }
    EXPECT_NE(metadataArray, 0u);
    EXPECT_EQ(arrayStrides[metadataArray], 16u);
    EXPECT_EQ(multiplies, 12u); // Six strides per access.
    const std::map<uint32_t, uint32_t> descriptorOffsets{{0, 0}, {1, 16}, {2, 112}};
    EXPECT_TRUE(std::any_of(memberOffsets.begin(), memberOffsets.end(),
                            [&](const auto &entry) { return entry.second == descriptorOffsets; }));
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
    auto missingMemoryModel = tensorShader();
    for (size_t offset = 5; offset < missingMemoryModel.size(); offset += missingMemoryModel[offset] >> 16) {
        if ((missingMemoryModel[offset] & 0xffff) == static_cast<uint32_t>(spv::OpMemoryModel)) {
            const auto instruction = missingMemoryModel.begin() + static_cast<std::ptrdiff_t>(offset);
            missingMemoryModel.erase(instruction, instruction + 3);
            break;
        }
    }
    malformed.push_back(missingMemoryModel);
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

TEST(TensorLoweringMode, OneUsesDirectLowering) {
    const ScopedEnvironment environment("VMEL_EXPERIMENTAL_TENSOR_LOWERING_MODE", "1");
    EXPECT_EQ(getTensorLoweringMode(), TensorLoweringMode::DirectSpirv);
    const TensorProcessor supported(tensorShader());
    ASSERT_TRUE(supported.isValidShader());
    EXPECT_TRUE(supported.isTensorComputeShader());
    auto spirv = supported.getNewSpirv();
    bool loweredTensorCompute = true;
    EXPECT_TRUE(lowerTensorSpirv(spirv, loweredTensorCompute, {}));
    EXPECT_FALSE(loweredTensorCompute);
    EXPECT_EQ(spirv, supported.getNewSpirv());
    const TensorProcessor processor(tensorShader("%values = OpTensorReadARM %array %loaded %coords"));
    EXPECT_FALSE(processor.isValidShader());
    EXPECT_TRUE(processor.getNewSpirv().empty());
}

} // namespace
} // namespace mlsdk::el::tests
