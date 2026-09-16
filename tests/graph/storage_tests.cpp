/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include "mlel/pipeline.hpp"
#include "mlel/tensor.hpp"
#include "mlel/utils.hpp"
#include "storage_test_utils.hpp"
#include "vulkan_test_utils.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace mlsdk::el::tests {

using namespace utilities;

namespace {

std::shared_ptr<Tensor> uint32BoundaryInput(std::shared_ptr<Device> &device) {
    return storageTestTensor(device, vk::Format::eR32Uint, {5},
                             std::vector<uint32_t>{0, 1, 0x01000001u, 0x80000001u, 0xffffffffu});
}

template <typename T>
void runUnsignedShaderCase(std::shared_ptr<Device> &device, const std::string &instruction,
                           const std::string &constants, std::vector<std::shared_ptr<Tensor>> inputs,
                           vk::Format outputFormat, const std::vector<int64_t> &outputDimensions,
                           const std::vector<T> &expected) {
    SCOPED_TRACE(instruction);
    auto output = std::make_shared<Tensor>(device, Shape{outputFormat, outputDimensions});
    const size_t inputCount = inputs.size();
    inputs.push_back(output);
    std::string annotations;
    std::string declarations = R"(
%u8 = OpTypeInt 8 0
%u16 = OpTypeInt 16 0
%u32 = OpTypeInt 32 0
%u64 = OpTypeInt 64 0
%bool = OpTypeBool
%false = OpConstantFalse %bool
%true = OpConstantTrue %bool
%c0 = OpConstant %u32 0
%c1 = OpConstant %u32 1
%c2 = OpConstant %u32 2
%c3 = OpConstant %u32 3
%c4 = OpConstant %u32 4
%a1 = OpTypeArray %u32 %c1
%a2 = OpTypeArray %u32 %c2
%a4 = OpTypeArray %u32 %c4
%s1 = OpConstantComposite %a1 %c1
%s2 = OpConstantComposite %a1 %c2
%s4 = OpConstantComposite %a1 %c4
%ct1 = OpTypeTensorARM %u32 %c1 %s1
%ct2 = OpTypeTensorARM %u32 %c1 %s2
%ct4 = OpTypeTensorARM %u32 %c1 %s4
%ct8 = OpTypeTensorARM %u8 %c1 %s1
%zero32 = OpConstantNull %ct1
%zero8 = OpConstantNull %ct8
%zero2 = OpConstantNull %ct2
%zero4 = OpConstantNull %ct4
%ones2 = OpConstantComposite %ct2 %c1 %c1
%twos2 = OpConstantComposite %ct2 %c2 %c2
%ones4 = OpConstantComposite %ct4 %c1 %c1 %c1 %c1
)";
    std::set<std::string> declaredTypes;
    std::string graphTypes = " = OpTypeGraphARM " + std::to_string(inputCount);
    std::string interfaces;
    std::string graphInputs;
    std::string outputType;
    GraphPipeline::DescriptorMap descriptors(1);
    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto &value = inputs[i];
        const auto width = Shape{value->getFormat(), value->getDimensions()}.getFormatSize() * 8;
        const auto rank = value->getDimensions().size();
        const auto type = "%t" + std::to_string(width) + "r" + std::to_string(rank);
        const auto variable = "%v" + std::to_string(i);
        annotations.append("OpDecorate ")
            .append(variable)
            .append(" DescriptorSet 0\nOpDecorate ")
            .append(variable)
            .append(" Binding ")
            .append(std::to_string(i))
            .append("\n");
        if (declaredTypes.insert(type).second) {
            declarations +=
                type + " = OpTypeTensorARM %u" + std::to_string(width) + " %c" + std::to_string(rank) + '\n';
            declarations.append(type).append("ptr = OpTypePointer UniformConstant ").append(type).append("\n");
        }
        declarations.append(variable).append(" = OpVariable ").append(type).append("ptr UniformConstant\n");
        graphTypes += ' ' + type;
        interfaces += ' ' + variable;
        if (i < inputCount) {
            graphInputs += "%in" + std::to_string(i) + " = OpGraphInputARM " + type + " %c" + std::to_string(i) + '\n';
        } else {
            outputType = type;
        }
        descriptors[0][static_cast<uint32_t>(i)] = {value};
    }
    const auto source = R"(
OpCapability Shader
OpCapability VulkanMemoryModel
OpCapability Int8
OpCapability Int16
OpCapability Int64
OpCapability TensorsARM
OpCapability GraphARM
OpExtension "SPV_ARM_tensors"
OpExtension "SPV_ARM_graph"
%tosa = OpExtInstImport "TOSA.001000.1"
OpMemoryModel Logical Vulkan
)" + annotations + declarations +
                        constants + "\n%graphType" + graphTypes + "\nOpGraphEntryPointARM %graph \"unsigned\"" +
                        interfaces + "\n%graph = OpGraphARM %graphType\n" + graphInputs + "%result = OpExtInst " +
                        outputType + " %tosa " + instruction + "\nOpGraphSetOutputARM %result %c0\nOpGraphEndARM\n";
    const auto spirv = assembleSpirv(source);
    const GraphConstants graphConstants;
    GraphPipeline pipeline{device, descriptors, graphConstants, spirv};
    pipeline.dispatchSubmit();
    ASSERT_EQ(output->size(), expected.size() * sizeof(expected[0]));
    for (size_t i = 0; i < expected.size(); ++i) {
        auto actual = expected[i];
        std::memcpy(&actual, output->data() + (i * sizeof(actual)), sizeof(actual));
        EXPECT_EQ(actual, expected[i]) << "element " << i;
    }
}

// UINT storage preserves bits; TOSA arithmetic interprets integers as signed unless explicitly overridden.
TEST_F(MLEmulationLayerGraphForVulkan, UnsignedStorageAbs) {
    const auto input = uint32BoundaryInput(device);
    runUnsignedShaderCase(device, "ABS %in0", "", {input}, vk::Format::eR32Uint, {5},
                          std::vector<uint32_t>{0, 1, 0x01000001u, 0x7fffffffu, 1});
}

TEST_F(MLEmulationLayerGraphForVulkan, UnsignedStorageClz) {
    const auto input = uint32BoundaryInput(device);
    runUnsignedShaderCase(device, "CLZ %in0", "", {input}, vk::Format::eR32Uint, {5},
                          std::vector<uint32_t>{32, 31, 7, 0, 0});
}

TEST_F(MLEmulationLayerGraphForVulkan, UnsignedStorageAdd) {
    const auto input = uint32BoundaryInput(device);
    const auto ones = storageTestTensor(device, vk::Format::eR32Uint, {5}, std::vector<uint32_t>(5, 1));
    runUnsignedShaderCase(device, "ADD %in0 %in1", "", {input, ones}, vk::Format::eR32Uint, {5},
                          std::vector<uint32_t>{1, 2, 0x01000002u, 0x80000002u, 0});
}

TEST_F(MLEmulationLayerGraphForVulkan, UnsignedStorageLogicalRightShift) {
    const auto input = uint32BoundaryInput(device);
    const auto ones = storageTestTensor(device, vk::Format::eR32Uint, {5}, std::vector<uint32_t>(5, 1));
    runUnsignedShaderCase(device, "LOGICAL_RIGHT_SHIFT %in0 %in1", "", {input, ones}, vk::Format::eR32Uint, {5},
                          std::vector<uint32_t>{0, 0, 0x00800000u, 0x40000000u, 0x7fffffffu});
}

TEST_F(MLEmulationLayerGraphForVulkan, UnsignedStorageNegate) {
    const auto input = uint32BoundaryInput(device);
    runUnsignedShaderCase(device, "NEGATE %in0 %zero32 %zero32", "", {input}, vk::Format::eR32Uint, {5},
                          std::vector<uint32_t>{0, 0xffffffffu, 0xfeffffffu, 0x7fffffffu, 1});
}

TEST_F(MLEmulationLayerGraphForVulkan, UnsignedStoragePadConstant) {
    const auto input = uint32BoundaryInput(device);
    runUnsignedShaderCase(device, "PAD %in0 %ones2 %padValue",
                          "%padBits = OpConstant %u32 4275878552\n%padValue = OpConstantComposite %ct1 %padBits",
                          {input}, vk::Format::eR32Uint, {7},
                          std::vector<uint32_t>{0xfedcba98u, 0, 1, 0x01000001u, 0x80000001u, 0xffffffffu, 0xfedcba98u});
}

TEST_F(MLEmulationLayerGraphForVulkan, UnsignedStorageAveragePool) {
    const auto poolInput =
        storageTestTensor(device, vk::Format::eR16Uint, {1, 2, 2, 1}, std::vector<uint16_t>(4, 65535));
    runUnsignedShaderCase(device, "AVG_POOL2D %twos2 %ones2 %zero4 %c1 %in0 %zero8 %zero8", "", {poolInput},
                          vk::Format::eR16Uint, {1, 1, 1, 1}, std::vector<uint16_t>{65535});
}

TEST_F(MLEmulationLayerGraphForVulkan, UnsignedStorageResizeSignExtension) {
    const auto poolInput =
        storageTestTensor(device, vk::Format::eR16Uint, {1, 2, 2, 1}, std::vector<uint16_t>(4, 65535));
    runUnsignedShaderCase(device, "RESIZE %c1 %in0 %ones4 %zero2 %zero2", "", {poolInput}, vk::Format::eR64Uint,
                          {1, 2, 2, 1}, std::vector<uint64_t>(4, 0xffffffffffffffffULL));
}

TEST_F(MLEmulationLayerGraphForVulkan, UnsignedStorageMatmul) {
    const auto lhs = storageTestTensor(device, vk::Format::eR16Uint, {1, 1, 2}, std::vector<uint16_t>(2, 65535));
    const auto rhs = storageTestTensor(device, vk::Format::eR16Uint, {1, 2, 1}, std::vector<uint16_t>(2, 65535));
    runUnsignedShaderCase(device, "MATMUL %in0 %in1 %zero8 %zero8", "", {lhs, rhs}, vk::Format::eR64Uint, {1, 1, 1},
                          std::vector<uint64_t>{2});
}

TEST_F(MLEmulationLayerGraphForVulkan, UnsignedStorageTableInterpolation) {
    const auto indices =
        storageTestTensor(device, vk::Format::eR16Uint, {4}, std::vector<uint16_t>{0x8000, 0xffff, 0, 0x7fff});
    std::vector<uint16_t> table(513);
    for (size_t i = 0; i < table.size(); ++i) {
        table[i] = static_cast<uint16_t>(static_cast<int>(i) - 256);
    }
    const auto tableTensor = storageTestTensor(device, vk::Format::eR16Uint, {513}, table);
    runUnsignedShaderCase(device, "TABLE %in0 %in1", "", {indices, tableTensor}, vk::Format::eR32Uint, {4},
                          std::vector<uint32_t>{0xffff8000u, 0xffffffffu, 0, 0x7fff});
}

template <typename T>
void expectStorageOperation(std::shared_ptr<Device> &device, const std::vector<std::shared_ptr<Tensor>> &inputs,
                            vk::Format outputFormat, const std::vector<int64_t> &outputShape,
                            const std::vector<uint32_t> &spirv, const std::vector<T> &expected) {
    auto output = std::make_shared<Tensor>(device, Shape{outputFormat, outputShape});
    GraphPipeline::DescriptorMap descriptors(1);
    for (uint32_t i = 0; i < inputs.size(); ++i) {
        descriptors[0][i] = {inputs[i]};
    }
    descriptors[0][static_cast<uint32_t>(inputs.size())] = {output};
    const GraphConstants constants;
    GraphPipeline pipeline(device, descriptors, constants, spirv);
    pipeline.dispatchSubmit();
    ASSERT_EQ(output->size(), expected.size() * sizeof(T));
    std::vector<T> actual(expected.size());
    std::memcpy(actual.data(), output->data(), output->size());
    EXPECT_EQ(std::memcmp(actual.data(), expected.data(), output->size()), 0)
        << "actual=" << testing::PrintToString(actual) << ", expected=" << testing::PrintToString(expected);
}

enum class StorageOperation { CastToFloat, Widen, Multiply, ArithmeticRightShift, Greater, Maximum };

struct IntegerStorageCase {
    StorageOperation operation;
    const char *name;
    int width;
    int bindings;
};

std::vector<IntegerStorageCase> integerStorageCases() {
    std::vector<IntegerStorageCase> cases;
    for (int width : {8, 16, 32}) {
        for (int bindings = 0; bindings < 8; ++bindings) {
            cases.push_back({StorageOperation::CastToFloat, "CastToFloat", width, bindings});
            if (width != 32) {
                cases.push_back({StorageOperation::Widen, "Widen", width, bindings});
            }
            cases.push_back({StorageOperation::Multiply, "Multiply", width, bindings});
            cases.push_back({StorageOperation::ArithmeticRightShift, "ArithmeticRightShift", width, bindings});
            if (width == 32) {
                cases.push_back({StorageOperation::Greater, "Greater", width, bindings});
                cases.push_back({StorageOperation::Maximum, "Maximum", width, bindings});
            }
        }
    }
    return cases;
}

vk::Format integerStorageFormat(int width, bool isUnsigned) {
    if (width == 8) {
        return isUnsigned ? vk::Format::eR8Uint : vk::Format::eR8Sint;
    }
    if (width == 16) {
        return isUnsigned ? vk::Format::eR16Uint : vk::Format::eR16Sint;
    }
    return isUnsigned ? vk::Format::eR32Uint : vk::Format::eR32Sint;
}

std::shared_ptr<Tensor> integerStorageInput(std::shared_ptr<Device> &device, int width, vk::Format format,
                                            const std::vector<int32_t> &values) {
    if (width == 8) {
        return storageTestTensor(device, format, {4}, std::vector<int8_t>(values.begin(), values.end()));
    }
    if (width == 16) {
        return storageTestTensor(device, format, {4}, std::vector<int16_t>(values.begin(), values.end()));
    }
    return storageTestTensor(device, format, {4}, values);
}

using TosaIntegerStorage = GraphTestWithParam<IntegerStorageCase>;

TEST_P(TosaIntegerStorage, PreservesSignedValues) {
    const auto &test = GetParam();
    const auto width = test.width;
    auto input =
        integerStorageInput(device, width, integerStorageFormat(width, (test.bindings & 1) != 0), {-128, -2, -1, 127});
    const auto inputType = "%i" + std::to_string(width);
    const auto out32 = (test.bindings & 4) ? vk::Format::eR32Uint : vk::Format::eR32Sint;
    const auto outputFormat = integerStorageFormat(width, (test.bindings & 4) != 0);
    if (test.operation == StorageOperation::CastToFloat) {
        expectStorageOperation(device, {input}, vk::Format::eR32Sfloat, {4},
                               makeStorageConversionGraph(inputType, "%f32", 1, 1, false, "CAST %a"),
                               std::vector<float>{-128, -2, -1, 127});
        return;
    }
    if (test.operation == StorageOperation::Widen) {
        expectStorageOperation(device, {input}, out32, {4},
                               makeStorageConversionGraph(inputType, "%i32", 1, 1, false, "CAST %a"),
                               std::vector<int32_t>{-128, -2, -1, 127});
        return;
    }
    auto second =
        integerStorageInput(device, width, integerStorageFormat(width, (test.bindings & 2) != 0), {1, 1, 1, 1});
    switch (test.operation) {
    case StorageOperation::Multiply:
        expectStorageOperation(device, {input, second}, out32, {4},
                               makeStorageConversionGraph(inputType, "%i32", 1, 1, true, "MUL %a %b %zp8"),
                               std::vector<int32_t>{-128, -2, -1, 127});
        break;
    case StorageOperation::ArithmeticRightShift: {
        const auto shift =
            makeStorageConversionGraph(inputType, inputType, 1, 1, true, "ARITHMETIC_RIGHT_SHIFT %false %a %b");
        if (width == 8) {
            expectStorageOperation(device, {input, second}, outputFormat, {4}, shift,
                                   std::vector<int8_t>{-64, -1, -1, 63});
        } else if (width == 16) {
            expectStorageOperation(device, {input, second}, outputFormat, {4}, shift,
                                   std::vector<int16_t>{-64, -1, -1, 63});
        } else {
            expectStorageOperation(device, {input, second}, outputFormat, {4}, shift,
                                   std::vector<int32_t>{-64, -1, -1, 63});
        }
        break;
    }
    case StorageOperation::Greater:
        expectStorageOperation(device, {input, second}, vk::Format::eR8BoolARM, {4},
                               makeStorageConversionGraph(inputType, "%bool", 1, 1, true, "GREATER %a %b"),
                               std::vector<uint8_t>{0, 0, 0, 1});
        break;
    case StorageOperation::Maximum:
        expectStorageOperation(device, {input, second}, outputFormat, {4},
                               makeStorageConversionGraph(inputType, inputType, 1, 1, true, "MAXIMUM %u1 %a %b"),
                               std::vector<int32_t>{1, 1, 1, 127});
        break;
    default:
        FAIL() << "Unexpected binary storage operation";
    }
}

INSTANTIATE_TEST_SUITE_P(TosaConversions, TosaIntegerStorage, testing::ValuesIn(integerStorageCases()),
                         [](const auto &info) {
                             const auto &test = info.param;
                             return std::string(test.name) + "_I" + std::to_string(test.width) + "_Input" +
                                    ((test.bindings & 1) ? "Uint" : "Sint") + "_Second" +
                                    ((test.bindings & 2) ? "Uint" : "Sint") + "_Output" +
                                    ((test.bindings & 4) ? "Uint" : "Sint");
                         });

TEST_F(MLEmulationLayerGraphForVulkan, TosaCastSignedSaturationInUnsignedStorage) {
    auto input =
        storageTestTensor(device, vk::Format::eR32Sfloat, {6}, std::vector<float>{-1000, -128, -1.5, 0.5, 126.5, 1000});
    expectStorageOperation(device, {input}, vk::Format::eR8Uint, {6},
                           makeStorageConversionGraph("%f32", "%i8", 1, 1, false, "CAST %a"),
                           std::vector<int8_t>{-128, -128, -2, 0, 126, 127});
}

struct Fp8Format {
    bool e4m3;
    const char *name;
    vk::Format format;
    const char *type;
};

constexpr std::array<Fp8Format, 2> fp8Formats{{
    {true, "E4M3", vk::Format::eR8SfloatFpencodingFloat8E4M3ARM, "%e4"},
    {false, "E5M2", vk::Format::eR8SfloatFpencodingFloat8E5M2ARM, "%e5"},
}};

// Every finite FP8 encoding is exactly representable in FP32. Keep the reference
// independent of the shader's bit conversion and exercise every finite encoding.
std::pair<std::vector<uint8_t>, std::vector<float>> finiteFp8Samples(bool e4m3) {
    std::vector<uint8_t> encodings;
    std::vector<float> decoded;
    for (unsigned bits = 0; bits < 256; ++bits) {
        const unsigned mantissaBits = e4m3 ? 3 : 2;
        const unsigned exponent = (bits & 0x7f) >> mantissaBits;
        const unsigned mantissa = bits & ((1u << mantissaBits) - 1);
        if ((e4m3 && exponent == 15 && mantissa == 7) || (!e4m3 && exponent == 31)) {
            continue;
        }
        const int bias = e4m3 ? 7 : 15;
        float value = exponent == 0
                          ? std::ldexp(float(mantissa), 1 - bias - int(mantissaBits))
                          : std::ldexp(1.0f + (float(mantissa) / float(1u << mantissaBits)), int(exponent) - bias);
        decoded.push_back((bits & 0x80) ? -value : value);
        encodings.push_back(uint8_t(bits));
    }
    return {encodings, decoded};
}

using TosaFp8Storage = GraphTestWithParam<Fp8Format>;

TEST_P(TosaFp8Storage, CastRoundingAndSpecialValues) {
    const auto &test = GetParam();
    const auto e4m3 = test.e4m3;
    const auto format = test.format;
    const auto *type = test.type;
    const float midpoint = e4m3 ? 1.0625f : 1.125f;
    const float minimum = std::ldexp(1.0f, e4m3 ? -9 : -16);
    const std::vector<float> values = {0.0f,
                                       -0.0f,
                                       midpoint,
                                       midpoint + std::ldexp(1.0f, -18),
                                       -midpoint - std::ldexp(1.0f, -18),
                                       minimum,
                                       minimum / 2,
                                       minimum * 1.5f,
                                       e4m3 ? 449.0f : 57345.0f,
                                       std::numeric_limits<float>::infinity(),
                                       -std::numeric_limits<float>::infinity(),
                                       std::numeric_limits<float>::quiet_NaN()};
    const uint8_t one = e4m3 ? 0x38 : 0x3c;
    const uint8_t overflow = e4m3 ? 0x7f : 0x7c;
    auto input = storageTestTensor(device, vk::Format::eR32Sfloat, {12}, values);
    expectStorageOperation(device, {input}, format, {12},
                           makeStorageConversionGraph("%f32", type, 1, 1, false, "CAST %a"),
                           std::vector<uint8_t>{0, 0x80, one, uint8_t(one + 1), uint8_t(one + 0x81), 1, 0, 2,
                                                uint8_t(e4m3 ? 0x7e : 0x7b), overflow, uint8_t(overflow | 0x80), 0x7f});
}

TEST_P(TosaFp8Storage, DecodeEveryFiniteEncoding) {
    const auto &test = GetParam();
    const auto [encodings, decoded] = finiteFp8Samples(test.e4m3);
    const std::vector<int64_t> shape{static_cast<int64_t>(encodings.size())};
    auto input = storageTestTensor(device, test.format, shape, encodings);
    expectStorageOperation(device, {input}, vk::Format::eR32Sfloat, shape,
                           makeStorageConversionGraph(test.type, "%f32", 1, 1, false, "CAST %a"), decoded);
}

TEST_P(TosaFp8Storage, EncodeEveryFiniteValue) {
    const auto &test = GetParam();
    const auto [encodings, decoded] = finiteFp8Samples(test.e4m3);
    const std::vector<int64_t> shape{static_cast<int64_t>(encodings.size())};
    auto input = storageTestTensor(device, vk::Format::eR32Sfloat, shape, decoded);
    expectStorageOperation(device, {input}, test.format, shape,
                           makeStorageConversionGraph("%f32", test.type, 1, 1, false, "CAST %a"), encodings);
}

INSTANTIATE_TEST_SUITE_P(TosaConversions, TosaFp8Storage, testing::ValuesIn(fp8Formats),
                         [](const auto &info) { return info.param.name; });

class TosaPoolingStorage : public GraphTestWithParam<bool> {
  protected:
    bool wide() const { return GetParam(); }
    std::string type() const { return wide() ? "%i16" : "%i8"; }
    vk::Format format() const { return wide() ? vk::Format::eR16Uint : vk::Format::eR8Uint; }
    std::string zeroPoints() const { return wide() ? "%zp16 %zp16" : "%zp8 %zp8"; }
    std::shared_ptr<Tensor> input(const std::vector<int64_t> &shape) {
        return wide() ? storageTestTensor(device, format(), shape, std::vector<int16_t>{-300, 2})
                      : storageTestTensor(device, format(), shape, std::vector<int8_t>{-100, 2});
    }
};

TEST_P(TosaPoolingStorage, MaxPool) {
    const auto source = input({1, 1, 2, 1});
    const auto graph =
        makeStorageConversionGraph(type(), type(), 4, 4, false, "MAX_POOL2D %kernel %stride %pad %u1 %a");
    if (wide()) {
        expectStorageOperation(device, {source}, format(), {1, 1, 1, 1}, graph, std::vector<int16_t>{2});
    } else {
        expectStorageOperation(device, {source}, format(), {1, 1, 1, 1}, graph, std::vector<int8_t>{2});
    }
}

TEST_P(TosaPoolingStorage, AveragePool) {
    const auto source = input({1, 1, 2, 1});
    const auto graph = makeStorageConversionGraph(type(), type(), 4, 4, false,
                                                  "AVG_POOL2D %kernel %stride %pad %u1 %a " + zeroPoints());
    if (wide()) {
        expectStorageOperation(device, {source}, format(), {1, 1, 1, 1}, graph, std::vector<int16_t>{-149});
    } else {
        expectStorageOperation(device, {source}, format(), {1, 1, 1, 1}, graph, std::vector<int8_t>{-49});
    }
}

TEST_P(TosaPoolingStorage, Argmax) {
    expectStorageOperation(device, {input({1, 1, 2, 1})}, vk::Format::eR32Uint, {1, 1, 1},
                           makeStorageConversionGraph(type(), "%i32", 4, 3, false, "ARGMAX %u2 %u1 %a"),
                           std::vector<int32_t>{1});
}

TEST_P(TosaPoolingStorage, Matmul) {
    const auto left = input({1, 1, 2});
    const auto right = wide() ? storageTestTensor(device, vk::Format::eR16Sint, {1, 2, 1}, std::vector<int16_t>{1, -1})
                              : storageTestTensor(device, vk::Format::eR8Sint, {1, 2, 1}, std::vector<int8_t>{1, -1});
    const auto graph =
        makeStorageConversionGraph(type(), wide() ? "%i64" : "%i32", 3, 3, true, "MATMUL %a %b " + zeroPoints());
    if (wide()) {
        expectStorageOperation(device, {left, right}, vk::Format::eR64Uint, {1, 1, 1}, graph,
                               std::vector<int64_t>{-302});
    } else {
        expectStorageOperation(device, {left, right}, vk::Format::eR32Uint, {1, 1, 1}, graph,
                               std::vector<int32_t>{-102});
    }
}

INSTANTIATE_TEST_SUITE_P(TosaConversions, TosaPoolingStorage, testing::Bool(),
                         [](const auto &info) { return info.param ? "I16" : "I8"; });

TEST_F(MLEmulationLayerGraphForVulkan, TosaIntegerStorageReduceMin) {
    auto input = storageTestTensor(device, vk::Format::eR32Uint, {4}, std::vector<int32_t>{-128, -2, -1, 127});
    expectStorageOperation(device, {input}, vk::Format::eR32Uint, {1},
                           makeStorageConversionGraph("%i32", "%i32", 1, 1, false, "REDUCE_MIN %u0 %u1 %a"),
                           std::vector<int32_t>{-128});
}

TEST_F(MLEmulationLayerGraphForVulkan, TosaIntegerStorageReduceMax) {
    auto input = storageTestTensor(device, vk::Format::eR32Uint, {4}, std::vector<int32_t>{-128, -2, -1, 127});
    expectStorageOperation(device, {input}, vk::Format::eR32Uint, {1},
                           makeStorageConversionGraph("%i32", "%i32", 1, 1, false, "REDUCE_MAX %u0 %u1 %a"),
                           std::vector<int32_t>{127});
}

TEST_F(MLEmulationLayerGraphForVulkan, TosaIntegerStorageNegate) {
    auto input = storageTestTensor(device, vk::Format::eR32Uint, {4}, std::vector<int32_t>{-128, -2, -1, 127});
    expectStorageOperation(device, {input}, vk::Format::eR32Uint, {4},
                           makeStorageConversionGraph("%i32", "%i32", 1, 1, false, "NEGATE %a %zp32 %zp32"),
                           std::vector<int32_t>{128, 2, 1, -127});
}

struct Fp8PadCase {
    Fp8Format format;
    const char *name;
    const char *literal;
    uint8_t bits;
};

// Exact hex literals avoid the bundled assembler's decimal E4M3 256-to-448 clamp.
constexpr std::array<Fp8PadCase, 7> fp8PadCases{{
    {fp8Formats[0], "E4M3_Subnormal", "0x1p-9", 1},
    {fp8Formats[0], "E4M3_256", "0x1p+8", 0x78},
    {fp8Formats[0], "E4M3_MaximumFinite", "0x1.cp+8", 0x7e},
    {fp8Formats[0], "E4M3_NegativeZero", "-0.0", 0x80},
    {fp8Formats[1], "E5M2_Subnormal", "0x1p-16", 1},
    {fp8Formats[1], "E5M2_MaximumFinite", "0x1.cp+15", 0x7b},
    {fp8Formats[1], "E5M2_NegativeZero", "-0.0", 0x80},
}};

using TosaFp8Pad = GraphTestWithParam<Fp8PadCase>;

TEST_P(TosaFp8Pad, PreservesConstantEncoding) {
    const auto &test = GetParam();
    const auto format = test.format.format;
    const auto *type = test.format.type;
    const auto *literal = test.literal;
    const auto bits = test.bits;
    auto input = storageTestTensor(device, format, {1}, std::vector<uint8_t>{0});
    std::ostringstream constants;
    constants << R"(
%padding_shape_type = OpTypeArray %i32 %u2
%padding_shape = OpConstantComposite %padding_shape_type %u1 %u2
%padding_type = OpTypeTensorARM %i32 %u2 %padding_shape
%padding_row = OpConstantComposite %const_pair %u1 %u1
%padding = OpConstantComposite %padding_type %padding_row
)"
              << "%pad_scalar = OpConstant " << type << " " << literal << "\n"
              << "%pad_type = OpTypeTensorARM " << type << " %u1 %shape1\n"
              << "%pad_value = OpConstantComposite %pad_type %pad_scalar\n";
    expectStorageOperation(
        device, {input}, format, {3},
        makeStorageConversionGraph(type, type, 1, 1, false, "PAD %a %padding %pad_value", constants.str()),
        std::vector<uint8_t>{bits, 0, bits});
}

INSTANTIATE_TEST_SUITE_P(TosaConversions, TosaFp8Pad, testing::ValuesIn(fp8PadCases),
                         [](const auto &info) { return info.param.name; });

TEST_F(MLEmulationLayerGraphForVulkan, TosaFloat16NegateFiniteValues) {
    const auto format = vk::Format::eR16Sfloat;
    auto input =
        storageTestTensor(device, format, {6}, std::vector<uint16_t>{0x3c00, 0xbc00, 0x3800, 0xb800, 0x7bff, 0xfbff});
    expectStorageOperation(device, {input}, format, {6},
                           makeStorageConversionGraph("%f16", "%f16", 1, 1, false, "NEGATE %a %zp32 %zp32"),
                           std::vector<uint16_t>{0xbc00, 0x3c00, 0xb800, 0x3800, 0xfbff, 0x7bff});
}

TEST_F(MLEmulationLayerGraphForVulkan, TosaBfloat16NegateFiniteValues) {
    const auto format = vk::Format::eR16SfloatFpencodingBfloat16ARM;
    auto input =
        storageTestTensor(device, format, {6}, std::vector<uint16_t>{0x3f80, 0xbf80, 0x3f00, 0xbf00, 0x7f7f, 0xff7f});
    expectStorageOperation(device, {input}, format, {6},
                           makeStorageConversionGraph("%bf16", "%bf16", 1, 1, false, "NEGATE %a %zp32 %zp32"),
                           std::vector<uint16_t>{0xbf80, 0x3f80, 0xbf00, 0x3f00, 0xff7f, 0x7f7f});
}

TEST_F(MLEmulationLayerGraphForVulkan, TosaBfloat16ClampConstants) {
    const auto format = vk::Format::eR16SfloatFpencodingBfloat16ARM;
    auto input = storageTestTensor(device, format, {3}, std::vector<uint16_t>{0x3e80, 0x3f40, 0x4000});
    const std::string constants = "%min = OpConstant %bf16 0.5\n%max = OpConstant %bf16 1.0\n";
    expectStorageOperation(
        device, {input}, format, {3},
        makeStorageConversionGraph("%bf16", "%bf16", 1, 1, false, "CLAMP %min %max %u1 %a", constants),
        std::vector<uint16_t>{0x3f00, 0x3f40, 0x3f80});
}

TEST_F(MLEmulationLayerGraphForVulkan, TosaFloat16ConstantStoragePreservesSubnormals) {
    auto input = storageTestTensor(device, vk::Format::eR16Sfloat, {1}, std::vector<uint16_t>{0});
    const std::string constants = R"(
%half_subnormal = OpConstant %f16 0x1p-24
%half_zero = OpConstantNull %f16
%half_type = OpTypeTensorARM %f16 %u1 %shape2
%half_constant = OpConstantComposite %half_type %half_subnormal %half_zero
)";
    expectStorageOperation(
        device, {input}, vk::Format::eR16Sfloat, {3},
        makeStorageConversionGraph("%f16", "%f16", 1, 1, false, "CONCAT %u0 %a %half_constant", constants),
        std::vector<uint16_t>{0, 1, 0});
}

} // namespace
} // namespace mlsdk::el::tests
