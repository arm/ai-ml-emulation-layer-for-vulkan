/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "graph_ext_inst_registry.hpp"
#include "mlel/pipeline.hpp"
#include "spirv-tools/libspirv.hpp"
#include "vulkan_test_utils.hpp"
#include <array>
#include <cstring>
#include <string>

namespace mlsdk::el::tests {
namespace {

void replace(std::string &text, const std::string &from, const std::string &to) {
    const auto offset = text.find(from);
    ASSERT_NE(offset, std::string::npos) << from;
    text.replace(offset, from.size(), to);
}

std::vector<uint32_t> assembleUnchecked(const std::string &text) {
    spvtools::SpirvTools tools(SPV_ENV_UNIVERSAL_1_6);
    std::vector<uint32_t> binary;
    if (!tools.Assemble(text, &binary, SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS)) {
        throw std::runtime_error("Cannot assemble test module");
    }
    return binary;
}

std::vector<uint32_t> withImport(const std::string &name) {
    auto binary = assembleUnchecked(fileToString("conv2d.spvasm"));
    for (size_t offset = 5; offset < binary.size();) {
        const auto words = binary[offset] >> 16;
        if ((binary[offset] & 0xffff) == 11) { // OpExtInstImport
            std::vector<uint32_t> encoded((name.size() + 4) / 4, 0);
            std::memcpy(encoded.data(), name.c_str(), name.size());
            binary.erase(binary.begin() + static_cast<std::ptrdiff_t>(offset + 2),
                         binary.begin() + static_cast<std::ptrdiff_t>(offset + words));
            binary.insert(binary.begin() + static_cast<std::ptrdiff_t>(offset + 2), encoded.begin(), encoded.end());
            binary[offset] = (static_cast<uint32_t>(encoded.size() + 2) << 16) | 11;
            return binary;
        }
        offset += words;
    }
    throw std::runtime_error("Test fixture is missing its import");
}

void expectPipelineResult(const std::shared_ptr<Device> &device, const std::vector<uint32_t> &binary, VkResult expected,
                          const std::string &diagnostic = {}) {
    const std::array<VkFormat, 4> formats{VK_FORMAT_R8_SINT, VK_FORMAT_R8_SINT, VK_FORMAT_R32_SINT, VK_FORMAT_R32_SINT};
    const std::array<std::vector<int64_t>, 4> dimensions{{{1, 8, 8, 3}, {3, 2, 2, 3}, {3}, {1, 4, 4, 3}}};
    std::array<VkTensorDescriptionARM, 4> descriptions{};
    std::array<VkDataGraphPipelineResourceInfoARM, 4> resources{};
    for (uint32_t i = 0; i < resources.size(); ++i) {
        descriptions[i].sType = VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM;
        descriptions[i].tiling = VK_TENSOR_TILING_LINEAR_ARM;
        descriptions[i].format = formats[i];
        descriptions[i].dimensionCount = static_cast<uint32_t>(dimensions[i].size());
        descriptions[i].pDimensions = dimensions[i].data();
        descriptions[i].usage = VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM;
        resources[i].sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_RESOURCE_INFO_ARM;
        resources[i].pNext = &descriptions[i];
        resources[i].binding = i;
    }
    VkShaderModuleCreateInfo shader{};
    shader.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader.codeSize = binary.size() * sizeof(uint32_t);
    shader.pCode = binary.data();
    VkDataGraphPipelineShaderModuleCreateInfoARM graphShader{};
    graphShader.sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SHADER_MODULE_CREATE_INFO_ARM;
    graphShader.pNext = &shader;
    graphShader.pName = "conv2d";
    VkDataGraphPipelineCreateInfoARM info{};
    info.sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_CREATE_INFO_ARM;
    info.pNext = &graphShader;
    info.resourceInfoCount = static_cast<uint32_t>(resources.size());
    info.pResourceInfos = resources.data();
    const auto &vkDevice = &(*device);
    const vk::raii::PipelineLayout layout(vkDevice, vk::PipelineLayoutCreateInfo{});
    info.layout = *layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    testing::internal::CaptureStdout();
    const auto result = vkDevice.getDispatcher()->vkCreateDataGraphPipelinesARM(
        *vkDevice, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline);
    const auto log = testing::internal::GetCapturedStdout();
    EXPECT_EQ(result, expected) << log;
    if (expected == VK_SUCCESS) {
        EXPECT_NE(pipeline, VK_NULL_HANDLE);
    } else {
        EXPECT_EQ(pipeline, VK_NULL_HANDLE);
        EXPECT_NE(log.find(diagnostic), std::string::npos) << log;
    }
    if (pipeline) {
        vkDevice.getDispatcher()->vkDestroyPipeline(*vkDevice, pipeline, nullptr);
    }
}

} // namespace

TEST_F(MLEmulationLayerGraphForVulkan, TosaRejectsUnknownImport) {
    expectPipelineResult(device, withImport("TOSA.999999.1"), VK_ERROR_UNKNOWN, "TOSA.999999.1");
}

TEST(GraphExtInstRegistry, ExactImportLookup) {
    for (const auto *name : {"TOSA.001000.1", "Arm.MotionEngine.100"}) {
        const auto *descriptor = spvtools::opt::findGraphExtInstSet(name);
        ASSERT_NE(descriptor, nullptr);
        EXPECT_EQ(descriptor->importName, name);
    }
    for (const auto *name : {"TOSA.999999.1", "TOSA.001000.2", "Arm.MotionEngine.101"}) {
        EXPECT_EQ(spvtools::opt::findGraphExtInstSet(name), nullptr) << name;
    }
}

TEST(GraphExtInstRegistry, MapsMultipleAndRepeatedImportsById) {
    auto text = fileToString("conv2d.spvasm");
    replace(text, "OpMemoryModel", R"(%100 = OpExtInstImport "TOSA.001000.1"
%101 = OpExtInstImport "Arm.MotionEngine.100"
%102 = OpExtInstImport "GLSL.std.450"
OpMemoryModel)");
    const auto binary = assembleUnchecked(text);
    const auto info = spvtools::opt::analyzeGraphModule(binary.data(), binary.size());
    ASSERT_EQ(info.extInstImports.size(), 3);
    EXPECT_EQ(info.extInstImports.at(45)->importName, "TOSA.001000.1");
    EXPECT_EQ(info.extInstImports.at(45), info.extInstImports.at(100));
    EXPECT_EQ(info.extInstImports.at(101)->importName, "Arm.MotionEngine.100");
    EXPECT_EQ(info.extInstImports.count(102), 0);
    EXPECT_TRUE(info.tosa);
    EXPECT_TRUE(info.motionEngine);
}

} // namespace mlsdk::el::tests
