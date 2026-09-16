/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include "datatype_cases.hpp"
#include "mlel/pipeline.hpp"
#include "mlel/tensor.hpp"
#include "mlel/utils.hpp"
#include "storage_test_utils.hpp"
#include "vulkan_test_utils.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace mlsdk::el::tests {

using namespace utilities;

namespace {

// Run through the public graph API; expected results never use FormatInfo,
// decodeReducedFloat, float16 wrappers, or generated shader conversion helpers.
std::vector<uint32_t> runDatatypeGraph(std::shared_ptr<Device> &device, const datatype_test::Type &inputType,
                                       bool inputUint, const std::vector<int64_t> &inputShape,
                                       const std::vector<uint32_t> &inputWords, const datatype_test::Type &outputType,
                                       bool outputUint, const std::vector<int64_t> &outputShape,
                                       const std::string &operation, const std::string &constants = "",
                                       const std::vector<uint32_t> *secondWords = nullptr, bool secondUint = false) {
    auto input =
        storageTestTensor(device, inputType.format(inputUint), inputShape, datatype_test::pack(inputWords, inputType));
    auto output = std::make_shared<Tensor>(device, Shape{outputType.format(outputUint), outputShape});
    GraphPipeline::DescriptorMap descriptors = {{{0, {input}}}};
    if (secondWords) {
        descriptors[0][1] = {storageTestTensor(device, inputType.format(secondUint), inputShape,
                                               datatype_test::pack(*secondWords, inputType))};
    }
    descriptors[0][secondWords ? 2 : 1] = {output};
    const auto spirv =
        makeStorageConversionGraph(inputType.spirv, outputType.spirv, uint32_t(inputShape.size()),
                                   uint32_t(outputShape.size()), secondWords != nullptr, operation, constants);
    GraphPipeline pipeline(device, descriptors, GraphConstants{}, spirv);
    pipeline.dispatchSubmit();
    // Avoid a separate uncached device-memory read for each decoded value.
    std::vector<uint8_t> bytes(output->size());
    std::memcpy(bytes.data(), output->data(), bytes.size());
    return datatype_test::unpack(bytes.data(), bytes.size() / outputType.bytes(), outputType);
}

template <typename Case> using GeneratedDatatypeTest = GraphTestWithParam<Case>;

using GeneratedDatatypeCast = GeneratedDatatypeTest<datatype_test::CastCase>;

TEST_P(GeneratedDatatypeCast, NumericResults) {
    const auto &test = GetParam();
    SCOPED_TRACE(datatype_test::name(test));
    const auto input = datatype_test::castSamples(test);
    ASSERT_FALSE(input.empty());
    const std::vector<int64_t> shape{int64_t(input.size())};
    const auto actual = runDatatypeGraph(device, test.input, test.inputUint, shape, input, test.output, test.outputUint,
                                         shape, "CAST %a");
    ASSERT_EQ(actual.size(), input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        ASSERT_TRUE(datatype_test::castMatches(actual[i], input[i], test))
            << "element=" << i << " input bits=0x" << std::hex << input[i] << " output bits=0x" << actual[i] << std::dec
            << " input value=" << datatype_test::decode(input[i], test.input)
            << " output value=" << datatype_test::decode(actual[i], test.output);
    }
}

INSTANTIATE_TEST_SUITE_P(DatatypeCoverage, GeneratedDatatypeCast, testing::ValuesIn(datatype_test::castCases()),
                         [](const auto &info) { return datatype_test::name(info.param); });

TEST_F(MLEmulationLayerGraphForVulkan, TosaCastFloat16Overflow) {
    // Check both sides of the rounding boundary, as well as values that must
    // overflow under either RNE or RTZ. UINT bindings still use signed TOSA i32.
    const std::vector<int32_t> values{65504,  65519,  65520,  65521,  65535,  65536,  65537,  INT32_MAX,
                                      -65504, -65519, -65520, -65521, -65535, -65536, -65537, INT32_MIN};
    const std::vector<uint32_t> expected{0x7bff, 0x7bff, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00,
                                         0xfbff, 0xfbff, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00};
    const std::vector<int64_t> shape{int64_t(values.size())};
    for (const auto &test : datatype_test::castCases()) {
        if (std::string(test.output.name) != "F16" ||
            (std::string(test.input.name) != "I32" && std::string(test.input.name) != "F32")) {
            continue;
        }
        SCOPED_TRACE(datatype_test::name(test));
        std::vector<uint32_t> input;
        input.reserve(values.size());
        for (int32_t value : values) {
            input.push_back(test.input.integer() ? uint32_t(value) : datatype_test::encode(double(value), test.input));
        }
        const auto actual =
            runDatatypeGraph(device, test.input, test.inputUint, shape, input, test.output, false, shape, "CAST %a");
        ASSERT_EQ(actual.size(), expected.size());
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(actual[i], expected[i]) << "input=" << values[i];
        }
    }
}

using GeneratedDatatypeRescale = GeneratedDatatypeTest<datatype_test::RescaleCase>;

TEST_P(GeneratedDatatypeRescale, SignednessAndSaturation) {
    const auto &test = GetParam();
    SCOPED_TRACE(datatype_test::name(test));
    const auto &in = test.types.input;
    const auto &out = test.types.output;
    const auto input = datatype_test::samples(in, in.width == 8);
    const int64_t inputZp = datatype_test::zeroPoint(in, test.inputUnsigned, test.zeroPoints);
    const int64_t outputZp = datatype_test::zeroPoint(out, test.outputUnsigned, test.zeroPoints);
    const int64_t minimum = test.outputUnsigned ? 0 : -(int64_t{1} << (out.width - 1));
    const int64_t maximum = test.outputUnsigned ? int64_t(out.mask()) : (int64_t{1} << (out.width - 1)) - 1;
    std::vector<uint32_t> expected;
    for (uint32_t bits : input) {
        const auto value = test.inputUnsigned ? int64_t(bits) : int64_t(datatype_test::decode(bits, in));
        // Scale by exactly 1/4 with single-round semantics. This uses division
        // rather than reproducing the implementation's multiply/shift sequence.
        const auto rounded = datatype_test::floorDivide(value - inputZp + 2, 4) + outputZp;
        expected.push_back(uint32_t(uint64_t(std::clamp(rounded, minimum, maximum))) & out.mask());
    }
    const std::string mulType = test.scale32 ? "%i32" : "%i16";
    const std::string mulTensor = test.scale32 ? "%const_i32" : "%const_i16";
    std::ostringstream constants;
    constants << "%test_mul_scalar = OpConstant " << mulType << (test.scale32 ? " 1073741824\n" : " 16384\n")
              << "%test_mul = OpConstantComposite " << mulTensor << " %test_mul_scalar\n"
              << "%test_shift_scalar = OpConstant %i8 " << (test.scale32 ? "32\n" : "16\n")
              << "%test_shift = OpConstantComposite %const_i8 %test_shift_scalar\n"
              << "%test_izp_scalar = OpConstant " << in.spirv << " " << (uint32_t(uint64_t(inputZp)) & in.mask())
              << "\n"
              << "%test_izp = OpConstantComposite %const_i" << in.width << " %test_izp_scalar\n"
              << "%test_ozp_scalar = OpConstant " << out.spirv << " " << (uint32_t(uint64_t(outputZp)) & out.mask())
              << "\n"
              << "%test_ozp = OpConstantComposite %const_i" << out.width << " %test_ozp_scalar\n";
    const std::string operation = std::string("RESCALE ") + (test.scale32 ? "%true" : "%false") + " %u1 %false " +
                                  (test.inputUnsigned ? "%true" : "%false") + " " +
                                  (test.outputUnsigned ? "%true" : "%false") +
                                  " %a %test_mul %test_shift %test_izp %test_ozp";
    const std::vector<int64_t> shape{int64_t(input.size())};
    const auto actual = runDatatypeGraph(device, in, test.types.inputUint, shape, input, out, test.types.outputUint,
                                         shape, operation, constants.str());
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < input.size(); ++i) {
        ASSERT_EQ(actual[i], expected[i]) << "element=" << i << " input bits=" << input[i];
    }
}

INSTANTIATE_TEST_SUITE_P(DatatypeCoverage, GeneratedDatatypeRescale, testing::ValuesIn(datatype_test::rescaleCases()),
                         [](const auto &info) { return datatype_test::name(info.param); });

using GeneratedDatatypeLayout = GeneratedDatatypeTest<datatype_test::LayoutCase>;

TEST_P(GeneratedDatatypeLayout, PreservesEncodedValues) {
    using datatype_test::LayoutOp;
    const auto &test = GetParam();
    SCOPED_TRACE(datatype_test::name(test));
    auto input = datatype_test::samples(test.type, test.type.width <= 16);
    while (input.size() < 4) {
        input.push_back(uint32_t(input.size() % 2));
    }
    if (input.size() % 2) {
        input.push_back(0);
    }
    const auto count = int64_t(input.size());
    std::vector<int64_t> inputShape{count};
    std::vector<int64_t> outputShape{count};
    auto expected = input;
    std::string operation;
    std::string constants;
    switch (test.operation) {
    case LayoutOp::Reverse:
        operation = "REVERSE %u0 %a";
        std::reverse(expected.begin(), expected.end());
        break;
    case LayoutOp::Slice:
        operation = "SLICE %a %shape1 %test_size";
        outputShape = {count - 2};
        expected = std::vector<uint32_t>(input.begin() + 1, input.end() - 1);
        constants = "%test_size_scalar = OpConstant %i32 " + std::to_string(count - 2) +
                    "\n"
                    "%test_size = OpConstantComposite %shape_array %test_size_scalar\n";
        break;
    case LayoutOp::Tile:
    case LayoutOp::Concat:
        operation = test.operation == LayoutOp::Tile ? "TILE %a %shape2" : "CONCAT %u0 %a %a";
        outputShape = {count * 2};
        expected.insert(expected.end(), input.begin(), input.end());
        break;
    case LayoutOp::Transpose:
        inputShape = {2, count / 2};
        outputShape = {count / 2, 2};
        operation = "TRANSPOSE %test_perms %a";
        constants = "%test_perms_type = OpTypeArray %i32 %u2\n"
                    "%test_perms = OpConstantComposite %test_perms_type %u1 %u0\n";
        for (size_t i = 0; i < input.size(); ++i) {
            expected[((i % (input.size() / 2)) * 2) + (i / (input.size() / 2))] = input[i];
        }
        break;
    case LayoutOp::Reshape:
        outputShape = {2, count / 2};
        operation = "RESHAPE %a %test_shape";
        constants = "%test_half = OpConstant %i32 " + std::to_string(count / 2) +
                    "\n"
                    "%test_shape_type = OpTypeArray %i32 %u2\n"
                    "%test_shape = OpConstantComposite %test_shape_type %u2 %test_half\n";
        break;
    }
    const auto actual = runDatatypeGraph(device, test.type, test.inputUint, inputShape, input, test.type,
                                         test.outputUint, outputShape, operation, constants);
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        // TOSA's floating-point reference checker accepts any NaN encoding.
        // Preserve exact bits for all other values, including signed zero.
        const bool matches = test.type.floating() ? datatype_test::sameFloat(actual[i], expected[i], test.type)
                                                  : actual[i] == expected[i];
        ASSERT_TRUE(matches) << "element=" << i << " actual bits=0x" << std::hex << actual[i] << " expected bits=0x"
                             << expected[i] << std::dec;
    }
}

INSTANTIATE_TEST_SUITE_P(DatatypeCoverage, GeneratedDatatypeLayout, testing::ValuesIn(datatype_test::layoutCases()),
                         [](const auto &info) { return datatype_test::name(info.param); });

using GeneratedDatatypeInteger = GeneratedDatatypeTest<datatype_test::IntegerCase>;

TEST_P(GeneratedDatatypeInteger, WidthBoundariesOnUnsignedStorage) {
    using datatype_test::IntegerOp;
    const auto &test = GetParam();
    SCOPED_TRACE(datatype_test::name(test));
    const auto &type = test.type;
    const auto &op = test.operation;
    const auto values = datatype_test::samples(type, type.width == 8);
    const auto &numericOutputType = op == IntegerOp::Mul ? datatype_test::types()[3] : type;
    const auto &outputType = test.comparison() ? datatype_test::types()[0] : numericOutputType;
    std::vector<uint32_t> input;
    std::vector<uint32_t> second;
    for (uint32_t raw : values) {
        if (test.unary()) {
            // abs(INT_MIN) cannot be represented; TOSA leaves overflow unpredictable.
            if (op != IntegerOp::Abs || raw != type.sign()) {
                input.push_back(raw);
                second.push_back(0);
            }
            continue;
        }
        if (test.shift()) {
            for (uint32_t shift = 0; shift < type.width; ++shift) {
                input.push_back(raw);
                second.push_back(shift);
            }
            continue;
        }
        for (uint32_t other : values) {
            const auto rhs = int64_t(datatype_test::decode(other, type));
            if (op == IntegerOp::IntDiv && (rhs == 0 || (raw == type.sign() && rhs == -1))) {
                continue;
            }
            if (op == IntegerOp::Mul || op == IntegerOp::Add || op == IntegerOp::Sub) {
                const int64_t result = datatype_test::integerReference(test, raw, other);
                if (result < INT32_MIN || result > INT32_MAX) {
                    continue;
                }
            }
            input.push_back(raw);
            second.push_back(other);
        }
    }
    std::vector<uint32_t> expected;
    for (size_t i = 0; i < input.size(); ++i) {
        const int64_t result = datatype_test::integerReference(test, input[i], second[i]);
        expected.push_back(uint32_t(uint64_t(result)) & outputType.mask());
    }
    std::string operation = datatype_test::name(op);
    switch (op) {
    case IntegerOp::Mul:
        operation += " %a %b %zp8";
        break;
    case IntegerOp::Minimum:
    case IntegerOp::Maximum:
        operation += " %u1 %a %b";
        break;
    case IntegerOp::ArithmeticRightShift:
        operation += std::string(test.round ? " %true" : " %false") + " %a %b";
        break;
    default:
        operation += test.unary() ? " %a" : " %a %b";
        break;
    }
    const std::vector<int64_t> shape{int64_t(input.size())};
    const auto actual = runDatatypeGraph(device, type, test.inputUint, shape, input, outputType, test.outputUint, shape,
                                         operation, "", test.unary() ? nullptr : &second, test.secondUint);
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < input.size(); ++i) {
        ASSERT_EQ(actual[i], expected[i])
            << "element=" << i << " input bits=" << input[i] << " second bits=" << second[i];
    }
}

INSTANTIATE_TEST_SUITE_P(DatatypeCoverage, GeneratedDatatypeInteger, testing::ValuesIn(datatype_test::integerCases()),
                         [](const auto &info) { return datatype_test::name(info.param); });

} // namespace
} // namespace mlsdk::el::tests
