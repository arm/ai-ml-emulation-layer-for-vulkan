/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "graph_ext_inst_registry.hpp"
#include "mlel/pipeline.hpp"
#include "spirv-tools/libspirv.hpp"
#include "storage_test_utils.hpp"
#include "vulkan_test_utils.hpp"
#include <cstring>
#include <string>
#include <utility>

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
                          const std::string &diagnostic = {}, const char *entryPoint = "conv2d",
                          const std::vector<std::pair<VkFormat, std::vector<int64_t>>> &tensors = {
                              {VK_FORMAT_R8_SINT, {1, 8, 8, 3}},
                              {VK_FORMAT_R8_SINT, {3, 2, 2, 3}},
                              {VK_FORMAT_R32_SINT, {3}},
                              {VK_FORMAT_R32_SINT, {1, 4, 4, 3}}}) {
    std::vector<VkTensorDescriptionARM> descriptions(tensors.size());
    std::vector<VkDataGraphPipelineResourceInfoARM> resources(tensors.size());
    for (uint32_t i = 0; i < resources.size(); ++i) {
        const auto &[format, dimensions] = tensors[i];
        descriptions[i].sType = VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM;
        descriptions[i].tiling = VK_TENSOR_TILING_LINEAR_ARM;
        descriptions[i].format = format;
        descriptions[i].dimensionCount = static_cast<uint32_t>(dimensions.size());
        descriptions[i].pDimensions = dimensions.data();
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
    graphShader.pName = entryPoint;
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
    for (const auto *name : {"TOSA.999999.1", "TOSA.001000.2", "TOSA.1x.2", "TOSA.001001.1", "Arm.MotionEngine.101"}) {
        EXPECT_EQ(spvtools::opt::findGraphExtInstSet(name), nullptr) << name;
    }
    const auto *descriptor = spvtools::opt::findGraphExtInstSet("TOSA.1x.1");
#if VMEL_ENABLE_TOSA_1X
    ASSERT_NE(descriptor, nullptr);
    EXPECT_EQ(descriptor->importName, "TOSA.1x.1");
#else
    EXPECT_EQ(descriptor, nullptr);
#endif
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

#if VMEL_ENABLE_TOSA_1X
// Together with the existing GATHER/SCATTER checks, the reused datatype,
// storage, operator and convolution tests cover 43 shared opcodes under 1.x.
// These cases cover the remaining 23 through pipeline creation only.
TEST_F(MLEmulationLayerGraphForVulkan, Tosa1xFloatDispatch) {
    const std::pair<const char *, bool> cases[]{
        {"CEIL %a", false},
        {"FLOOR %a", false},
        {"ERF %a", false},
        {"EXP %a", false},
        {"LOG %a", false},
        {"RECIPROCAL %a", false},
        {"RSQRT %a", false},
        {"SIGMOID %a", false},
        {"TANH %a", false},
        {"POW %a %b", true},
        {"REDUCE_PRODUCT %u0 %a", false},
        {"REDUCE_SUM %u0 %a", false},
    };
    for (const auto &[operation, binary] : cases) {
        SCOPED_TRACE(operation);
        const auto graph = makeStorageConversionGraph("%f32", "%f32", 1, 1, binary, operation, "", "TOSA.1x.1");
        const std::vector<std::pair<VkFormat, std::vector<int64_t>>> tensors(binary ? 3 : 2,
                                                                             {VK_FORMAT_R32_SFLOAT, {1}});
        expectPipelineResult(device, graph, VK_SUCCESS, "", "storage_conversion", tensors);
    }
}

TEST_F(MLEmulationLayerGraphForVulkan, Tosa1xBooleanDispatch) {
    const std::pair<const char *, bool> cases[]{
        {"LOGICAL_AND %a %b", true}, {"LOGICAL_OR %a %b", true},   {"LOGICAL_XOR %a %b", true},
        {"LOGICAL_NOT %a", false},   {"REDUCE_ALL %u0 %a", false}, {"REDUCE_ANY %u0 %a", false},
    };
    for (const auto &[operation, binary] : cases) {
        SCOPED_TRACE(operation);
        const auto graph = makeStorageConversionGraph("%bool", "%bool", 1, 1, binary, operation, "", "TOSA.1x.1");
        const std::vector<std::pair<VkFormat, std::vector<int64_t>>> tensors(binary ? 3 : 2,
                                                                             {VK_FORMAT_R8_BOOL_ARM, {1}});
        expectPipelineResult(device, graph, VK_SUCCESS, "", "storage_conversion", tensors);
    }
}

TEST_F(MLEmulationLayerGraphForVulkan, Tosa1xSelectDispatch) {
    const auto graph = makeStorageConversionGraph(
        "%bool", "%i32", 1, 1, false, "SELECT %a %yes %no",
        "%yes = OpConstantComposite %const_i32 %u2\n%no = OpConstantComposite %const_i32 %u5\n", "TOSA.1x.1");
    expectPipelineResult(device, graph, VK_SUCCESS, "", "storage_conversion",
                         {{VK_FORMAT_R8_BOOL_ARM, {1}}, {VK_FORMAT_R32_SINT, {1}}});
}

TEST_F(MLEmulationLayerGraphForVulkan, Tosa1xTableDispatch) {
    // TABLE requires the lookup length in its tensor type.
    const auto graph = makeStorageConversionGraph("%i8", "%i8", 1, 1, false, "TABLE %a %table", R"(
%table_size = OpConstant %i32 256
%table_shape = OpConstantComposite %shape_array %table_size
%table_type = OpTypeTensorARM %i8 %u1 %table_shape
%table = OpConstantNull %table_type
)",
                                                  "TOSA.1x.1");
    expectPipelineResult(device, graph, VK_SUCCESS, "", "storage_conversion",
                         {{VK_FORMAT_R8_SINT, {1}}, {VK_FORMAT_R8_SINT, {1}}});
}

TEST_F(MLEmulationLayerGraphForVulkan, Tosa1xConvolutionDispatch) {
    auto source = fileToString("conv2d.spvasm");
    replace(source, "TOSA.001000.1", "TOSA.1x.1");
    // With singleton dimensions, both weight layouts fit the CONV2D fixture.
    replace(source, "%uint_1 %uint_8 %uint_8 %uint_3", "%uint_1 %uint_1 %uint_1 %uint_1");
    replace(source, "%uint_3 %uint_2 %uint_2 %uint_3", "%uint_1 %uint_1 %uint_1 %uint_1");
    replace(source, "%uint_1 %uint_4 %uint_4 %uint_3", "%uint_1 %uint_1 %uint_1 %uint_1");
    replace(source, "%20 = OpConstantComposite %_arr_uint_uint_1 %uint_3",
            "%20 = OpConstantComposite %_arr_uint_uint_1 %uint_1");
    replace(source, "OpConstantCompositeReplicateEXT %35 %uint_2", "OpConstantCompositeReplicateEXT %35 %uint_1");
    for (const std::string operation : {"DEPTHWISE_CONV2D", "TRANSPOSE_CONV2D"}) {
        SCOPED_TRACE(operation);
        auto text = source;
        replace(text, "CONV2D", operation);
        if (operation == "TRANSPOSE_CONV2D") {
            replace(text, "%34 %37 %38 %uint_1", "%34 %37 %uint_1");
        }
        expectPipelineResult(device, assembleSpirv(text), VK_SUCCESS, "", "conv2d",
                             {{VK_FORMAT_R8_SINT, {1, 1, 1, 1}},
                              {VK_FORMAT_R8_SINT, {1, 1, 1, 1}},
                              {VK_FORMAT_R32_SINT, {1}},
                              {VK_FORMAT_R32_SINT, {1, 1, 1, 1}}});
    }
}

TEST_F(MLEmulationLayerGraphForVulkan, Tosa1xRfftDispatch) {
    auto source = fileToString("fft2d.spvasm");
    replace(source, "TOSA.001000.1", "TOSA.1x.1");
    // A singleton transform has the same input and output shape.
    replace(source, "%uint_4 = OpConstant %uint 4", "%uint_4 = OpConstant %uint 1");
    replace(source, "%uint_4096 = OpConstant %uint 4096", "%uint_4096 = OpConstant %uint 1");
    replace(source, "OpTypeGraphARM 2 %2 %2 %2 %2", "OpTypeGraphARM 1 %2 %2 %2");
    replace(source, "\"main\" %main_arg_0 %main_arg_1", "\"main\" %main_arg_0");
    replace(source, "%18 = OpGraphInputARM %2 %uint_1", "");
    replace(source, "FFT2D %true %false %17 %18", "RFFT2D %false %17");
    replace(source, "OpDecorate %main_res_0 Binding 2", "OpDecorate %main_res_0 Binding 1");
    replace(source, "OpDecorate %main_res_1 Binding 3", "OpDecorate %main_res_1 Binding 2");
    expectPipelineResult(
        device, assembleSpirv(source), VK_SUCCESS, "", "main",
        {{VK_FORMAT_R32_SFLOAT, {1, 1, 1}}, {VK_FORMAT_R32_SFLOAT, {1, 1, 1}}, {VK_FORMAT_R32_SFLOAT, {1, 1, 1}}});
}

TEST_F(MLEmulationLayerGraphForVulkan, Tosa1xRejectsNewOpcodes) {
    const auto argmin = makeStorageConversionGraph("%f32", "%i32", 2, 1, false, "ARGMIN %u0 %u1 %a", "", "TOSA.1x.1");
    expectPipelineResult(device, argmin, VK_ERROR_UNKNOWN, "Unsupported TOSA instruction 66", "storage_conversion",
                         {{VK_FORMAT_R32_SFLOAT, {1, 2}}, {VK_FORMAT_R32_SINT, {2}}});
    const auto matmul =
        makeStorageConversionGraph("%i8", "%i32", 3, 3, true, "MATMUL_T %a %b %zp8 %zp8", "", "TOSA.1x.1");
    expectPipelineResult(
        device, matmul, VK_ERROR_UNKNOWN, "Unsupported TOSA instruction 68", "storage_conversion",
        {{VK_FORMAT_R8_SINT, {1, 1, 1}}, {VK_FORMAT_R8_SINT, {1, 1, 1}}, {VK_FORMAT_R32_SINT, {1, 1, 1}}});
}

TEST_F(MLEmulationLayerGraphForVulkan, Tosa1xOptionalOperands) {
    auto text = fileToString("conv2d_1x.spvasm");
    // Pass the bias through CAST, GATHER and SCATTER with unchanged values.
    replace(text, "OpGraphEntryPointARM", R"(%array3 = OpTypeArray %uint %uint_3
%values_shape = OpConstantComposite %array3 %uint_1 %uint_3 %uint_1
%values_type = OpTypeTensorARM %uint %uint_3 %values_shape
%array2 = OpTypeArray %uint %uint_2
%indices_shape = OpConstantComposite %array2 %uint_1 %uint_3
%indices_type = OpTypeTensorARM %uint %uint_2 %indices_shape
%indices = OpConstantComposite %indices_type %uint_0 %uint_1 %uint_2
%values_reshape = OpConstantComposite %18 %uint_1 %uint_3 %uint_1
%bias_shape_type = OpTypeTensorARM %uint %uint_1 %42
%bias_shape = OpConstantComposite %bias_shape_type %uint_3
OpGraphEntryPointARM)");
    replace(text, "%44 = OpExtInst", R"(%cast = OpExtInst %18 %45 CAST %31
%values = OpExtInst %values_type %45 RESHAPE %cast %values_reshape
%gather = OpExtInst %values_type %45 GATHER %values %indices
%scatter = OpExtInst %values_type %45 SCATTER %values %indices %gather
%bias = OpExtInst %18 %45 RESHAPE %scatter %bias_shape
%44 = OpExtInst)");
    replace(text, "%28 %30 %31 %43 %43", "%28 %30 %bias %43 %43");
    expectPipelineResult(device, assembleUnchecked(text), VK_SUCCESS);

    // Even an explicit zero optional-operand mask is currently unsupported.
    for (const std::string operation : {"CAST %31", "GATHER %values %indices", "SCATTER %values %indices %gather"}) {
        SCOPED_TRACE(operation);
        auto optional = text;
        replace(optional, operation, operation + " %uint_0");
        expectPipelineResult(device, assembleUnchecked(optional), VK_ERROR_UNKNOWN,
                             "optional operands are not supported");
    }
}

TEST_F(MLEmulationLayerGraphForVulkan, TosaRejectsUnusedScaledTypeBeforeOptimizer) {
    auto text = fileToString("conv2d_1x.spvasm");
    replace(text, "OpGraphEntryPointARM", "%scaled = OpTypeTensorScaledBlockARM %2 %13 %7\n OpGraphEntryPointARM");
    expectPipelineResult(device, assembleUnchecked(text), VK_ERROR_UNKNOWN, "rejected before optimizer type analysis");
    replace(text, "TOSA.1x.1", "TOSA.001000.1");
    expectPipelineResult(device, assembleUnchecked(text), VK_ERROR_UNKNOWN, "rejected before optimizer type analysis");
}
#endif

} // namespace mlsdk::el::tests
