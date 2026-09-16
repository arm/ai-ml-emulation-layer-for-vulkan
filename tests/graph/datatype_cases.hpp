/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/*
 * Generated datatype coverage
 *
 * This header generates the parameters and independent numeric references
 * used by the DatatypeCoverage suites in tests/graph/datatype_tests.cpp. Each case assembles a
 * TOSA graph, dispatches it through the public Vulkan API, and compares the output.
 * The references do not use production format metadata or conversion helpers.
 *
 * The matrix follows the supported type combinations and constraints in
 * TOSA 1.0.1: https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html
 * An integer tensor's Vulkan SINT/UINT binding is varied independently from its
 * TOSA arithmetic interpretation. CAST covers each input/output binding pair.
 * RESCALE binds opposite to its arithmetic signedness; layout crosses bindings;
 * integer arithmetic focuses on UINT bindings. The existing TosaIntegerStorage
 * suite covers all eight binary-operation binding permutations.
 * Each of the 216 binding/attribute variants is a separately named GoogleTest
 * case. CTest groups each suite into one process by default, and the suite shares
 * one Vulkan device. Names identify the types, bindings and attributes; failures
 * also include the element and encoded operands. Exhaustive sample vectors stay
 * batched in each graph, preserving coverage without extra pipeline compilation.
 *
 * Suite      Cases
 * CAST          88
 * RESCALE       37
 * Layout        54
 * Integer       37
 *
 * - CAST: legal conversions among bool, i8/i16/i32, FP16, BF16, FP32, E4M3
 *   and E5M2; independent input/output bindings.
 * - RESCALE: i8/i16/i32, legal signed/unsigned arithmetic flags, independent
 *   bindings, scale16/scale32, zero and nonzero zero points.
 * - Layout: REVERSE, SLICE, TILE, TRANSPOSE, RESHAPE and CONCAT for all nine
 *   types; mixed SINT/UINT bindings for integers.
 * - Integer: bitwise operations, logical/arithmetic shifts and MUL for
 *   i8/i16/i32; ADD, SUB, INTDIV, MINIMUM, MAXIMUM, EQUAL, GREATER,
 *   GREATER_EQUAL, ABS and CLZ for i32.
 *
 * Input coverage and references
 *
 * - CAST enumerates every i8/i16, bool and FP8 encoding. FP16/BF16 to FP32 also
 *   enumerates every encoding, including signed zero, subnormals, infinities and
 *   NaNs. Other FP16/BF16 conversions use selected boundary values.
 * - FP32 to each reduced float checks the midpoint of every finite destination
 *   interval, its adjacent FP32 values, and both signs. Integer boundaries include
 *   each power of two and its neighbors, with both signs.
 * - The floating reference quantizes in FP64 using mathematical significands and
 *   exponent spacing. CAST accepts the permitted rounding choices and subnormal
 *   flushing for FP16/BF16/FP32, while checking FP8 subnormal preservation.
 *   Integer-to-float results allow 0.5 ULP, including either midpoint neighbor.
 *   NaN payloads are not required to survive a numeric conversion. NaN-to-integer
 *   results are deliberately excluded because they are unpredictable.
 * - RESCALE uses an exact scale of 1/4 and an independent division-based reference
 *   for single rounding, zero-point adjustment and saturation. Every i8 encoding
 *   is exercised; i16/i32 use boundaries across the full width. The generator
 *   excludes simultaneous unsigned input/output and the other prohibited flag
 *   combinations.
 * - Layout tests compare non-NaN output bits exactly, including signed zero and
 *   subnormals. Expected NaNs must remain NaNs; their sign and payload may change
 *   (TOSA 1.0.1 section 4.5.3). The tests enumerate all encodings up to 16 bits
 *   and sample 32-bit boundaries.
 * - Integer binary operations use all pairs of i8 encodings and the Cartesian
 *   product of boundary samples for wider types. Shifts use every legal amount
 *   for each input and both arithmetic rounding settings. References calculate
 *   in 64 bits and mask to the output width. Arithmetic results stay within the
 *   signed 32-bit range; division by zero, INT_MIN / -1 and abs(INT_MIN) are excluded.
 *
 * Running and extending
 *
 * Build mlel_vulkan_tests and use the usual layer/driver environment for the
 * repository. For example, from the repository root with a system Lavapipe ICD:
 *
 * VK_LAYER_PATH=build/graph:build/tensor:/usr/share/vulkan/explicit_layer.d \
 * LD_LIBRARY_PATH=build/graph:build/tensor:build/common \
 * VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json \
 * build/tests/mlel_vulkan_tests --gtest_filter='DatatypeCoverage/GeneratedDatatype*'
 *
 * Use --gtest_list_tests with the same filter to inspect the generated matrix.
 * To reproduce a failure, pass its complete printed name to --gtest_filter.
 *
 * This is datatype regression coverage, not full TOSA conformance or a code
 * coverage measurement. Further matrices are needed for convolution and matmul
 * accumulators, pooling and reductions, floating-point arithmetic, booleans beyond
 * CAST/layout, PAD and gather/scatter, packed i4 and i48, nonzero MUL shifts,
 * per-channel RESCALE and double rounding, constant operands, and broadcasting.
 * Graph operators always use precompiled shaders; missing variants fail.
 * External CTS/scenario-runner coverage is outside this matrix.
 */

#include <vulkan/vulkan.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace datatype_test {

struct Type {
    const char *name;
    const char *spirv;
    unsigned width;
    unsigned fraction;
    unsigned exponent;
    int bias;
    vk::Format signedFormat;
    vk::Format unsignedFormat;

    bool integer() const { return exponent == 0 && width != 1; }
    bool boolean() const { return width == 1; }
    bool floating() const { return exponent != 0; }
    bool fp8() const { return floating() && width == 8; }
    bool e4m3() const { return fp8() && fraction == 3; }
    unsigned bytes() const { return boolean() ? 1 : width / 8; }
    uint32_t mask() const { return uint32_t((uint64_t{1} << width) - 1); }
    uint32_t sign() const { return uint32_t{1} << (width - 1); }
    uint32_t fractionMask() const { return (uint32_t{1} << fraction) - 1; }
    uint32_t exponentMask() const { return ((uint32_t{1} << exponent) - 1) << fraction; }
    uint32_t maxFinite() const { return e4m3() ? 0x7e : exponentMask() - 1; }
    vk::Format format(bool useUint) const { return useUint ? unsignedFormat : signedFormat; }
};

inline const std::vector<Type> &types() {
    static const std::vector<Type> values = {
        {"Bool", "%bool", 1, 0, 0, 0, vk::Format::eR8BoolARM, vk::Format::eR8BoolARM},
        {"I8", "%i8", 8, 0, 0, 0, vk::Format::eR8Sint, vk::Format::eR8Uint},
        {"I16", "%i16", 16, 0, 0, 0, vk::Format::eR16Sint, vk::Format::eR16Uint},
        {"I32", "%i32", 32, 0, 0, 0, vk::Format::eR32Sint, vk::Format::eR32Uint},
        {"F16", "%f16", 16, 10, 5, 15, vk::Format::eR16Sfloat, vk::Format::eR16Sfloat},
        {"BF16", "%bf16", 16, 7, 8, 127, vk::Format::eR16SfloatFpencodingBfloat16ARM,
         vk::Format::eR16SfloatFpencodingBfloat16ARM},
        {"F32", "%f32", 32, 23, 8, 127, vk::Format::eR32Sfloat, vk::Format::eR32Sfloat},
        {"E4M3", "%e4", 8, 3, 4, 7, vk::Format::eR8SfloatFpencodingFloat8E4M3ARM,
         vk::Format::eR8SfloatFpencodingFloat8E4M3ARM},
        {"E5M2", "%e5", 8, 2, 5, 15, vk::Format::eR8SfloatFpencodingFloat8E5M2ARM,
         vk::Format::eR8SfloatFpencodingFloat8E5M2ARM},
    };
    return values;
}

inline bool isNaN(uint32_t bits, const Type &type) {
    if (!type.floating() || (bits & type.exponentMask()) != type.exponentMask()) {
        return false;
    }
    const uint32_t fraction = bits & type.fractionMask();
    // E4M3 reserves only the all-ones fraction for NaN; both signs are valid.
    return type.e4m3() ? fraction == type.fractionMask() : fraction != 0;
}

inline double decode(uint32_t bits, const Type &type) {
    if (type.boolean()) {
        return bits != 0 ? 1.0 : 0.0;
    }
    if (type.integer()) {
        return double((bits & type.sign()) ? int64_t(bits) - (int64_t{1} << type.width) : int64_t(bits));
    }
    const unsigned exponent = (bits & type.exponentMask()) >> type.fraction;
    const unsigned fraction = bits & type.fractionMask();
    double magnitude;
    if (isNaN(bits, type)) {
        magnitude = std::numeric_limits<double>::quiet_NaN();
    } else if (!type.e4m3() && exponent == (1u << type.exponent) - 1) {
        magnitude = std::numeric_limits<double>::infinity();
    } else {
        magnitude = std::ldexp(double(exponent ? (1u << type.fraction) + fraction : fraction),
                               (exponent ? int(exponent) : 1) - type.bias - int(type.fraction));
    }
    return (bits & type.sign()) ? -magnitude : magnitude;
}

inline double roundEven(double value) {
    const double lower = std::floor(value);
    const double fraction = value - lower;
    return lower + (fraction > 0.5 || (fraction == 0.5 && std::fmod(lower, 2.0) != 0.0) ? 1.0 : 0.0);
}

// Arithmetic reference: quantize to the destination's spacing using FP64.
// It does not reuse the shader's FP32 bit-shifting algorithm.
inline uint32_t encode(double value, const Type &type, bool nearest = true) {
    const uint32_t sign = std::signbit(value) ? type.sign() : 0;
    if (std::isnan(value)) {
        return sign | type.exponentMask() | type.fractionMask();
    }
    const uint32_t overflow = sign | (type.e4m3() ? 0x7f : type.exponentMask());
    if (std::isinf(value)) {
        return overflow;
    }
    const double magnitude = std::abs(value);
    if (magnitude == 0) {
        return sign;
    }
    const int exponent = std::max(std::ilogb(magnitude), 1 - type.bias);
    const double step = std::ldexp(1.0, exponent - int(type.fraction));
    const double significand = nearest ? roundEven(magnitude / step) : std::floor(magnitude / step);
    const double rounded = significand * step;
    if (rounded > decode(type.maxFinite(), type)) {
        return overflow;
    }
    if (rounded == 0) {
        return sign;
    }
    if (rounded < std::ldexp(1.0, 1 - type.bias)) {
        return sign | uint32_t(significand);
    }
    const int roundedExponent = std::ilogb(rounded);
    const auto fraction = uint32_t(std::ldexp(rounded, int(type.fraction) - roundedExponent)) - (1u << type.fraction);
    return sign | (uint32_t(roundedExponent + type.bias) << type.fraction) | fraction;
}

inline bool subnormal(uint32_t bits, const Type &type) {
    return type.floating() && (bits & type.exponentMask()) == 0 && (bits & type.fractionMask()) != 0;
}

inline bool sameFloat(uint32_t actual, uint32_t expected, const Type &type) {
    return actual == expected || (isNaN(actual, type) && isNaN(expected, type));
}

inline std::vector<uint8_t> pack(const std::vector<uint32_t> &words, const Type &type) {
    std::vector<uint8_t> bytes(words.size() * type.bytes());
    for (size_t i = 0; i < words.size(); ++i) {
        // Copy native-width objects, including on big-endian hosts.
        if (type.bytes() == 1) {
            bytes[i] = uint8_t(words[i]);
        } else if (type.bytes() == 2) {
            const auto word = uint16_t(words[i]);
            std::memcpy(bytes.data() + (i * 2), &word, 2);
        } else {
            std::memcpy(bytes.data() + (i * 4), &words[i], 4);
        }
    }
    return bytes;
}

inline std::vector<uint32_t> unpack(const uint8_t *bytes, size_t count, const Type &type) {
    std::vector<uint32_t> words(count);
    for (size_t i = 0; i < count; ++i) {
        if (type.bytes() == 1) {
            words[i] = bytes[i];
        } else if (type.bytes() == 2) {
            uint16_t word;
            std::memcpy(&word, bytes + (i * 2), 2);
            words[i] = word;
        } else {
            std::memcpy(&words[i], bytes + (i * 4), 4);
        }
    }
    return words;
}

inline std::vector<uint32_t> samples(const Type &type, bool exhaustive) {
    std::vector<uint32_t> words;
    if (type.width <= 16 && exhaustive) {
        for (uint32_t i = 0; i <= type.mask(); ++i) {
            words.push_back(i);
        }
        return words;
    }
    words = {0, 1, type.sign() - 1, type.sign(), type.sign() + 1, type.mask() - 1, type.mask()};
    if (type.floating()) {
        for (double x : {0.0, -0.0, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, -129.0, 127.5, 128.0, -32769.0, 32767.5, 32768.0,
                         -2147483648.0, 2147483648.0}) {
            words.push_back(encode(x, type));
        }
        for (auto bits : {type.fractionMask(), type.fractionMask() + 1, type.maxFinite(), type.exponentMask()}) {
            words.push_back(bits);
            words.push_back(bits | type.sign());
        }
    } else if (type.integer()) {
        for (unsigned bit = 1; bit < type.width; ++bit) {
            for (uint32_t value : {(1u << bit) - 1, 1u << bit, (1u << bit) + 1}) {
                words.push_back(value & type.mask());
                words.push_back(uint32_t(-int64_t(value)) & type.mask());
            }
        }
    }
    std::sort(words.begin(), words.end());
    words.erase(std::unique(words.begin(), words.end()), words.end());
    return words;
}

struct CastCase {
    Type input;
    Type output;
    bool inputUint;
    bool outputUint;
};

inline std::string bindingName(const Type &type, bool useUint) {
    if (!type.integer()) {
        return type.name;
    }
    return std::string(type.name) + (useUint ? "_UINT" : "_SINT");
}

inline std::string name(const CastCase &test) {
    return bindingName(test.input, test.inputUint) + "_To_" + bindingName(test.output, test.outputUint);
}

inline bool legalCast(const Type &in, const Type &out) {
    if (std::string(in.name) == out.name) {
        return false;
    }
    if (in.boolean() || out.boolean()) {
        return in.integer() || out.integer();
    }
    if (in.integer() || out.integer()) {
        return !in.fp8() && !out.fp8();
    }
    if (in.fp8() || out.fp8()) {
        return !(in.fp8() && out.fp8());
    }
    return in.width == 32 || out.width == 32; // No direct FP16 <-> BF16 CAST.
}

inline std::vector<CastCase> castCases() {
    std::vector<CastCase> cases;
    for (const auto &in : types()) {
        for (const auto &out : types()) {
            if (!legalCast(in, out)) {
                continue;
            }
            for (int inputUint = 0; inputUint <= int(in.integer()); ++inputUint) {
                for (int outputUint = 0; outputUint <= int(out.integer()); ++outputUint) {
                    cases.push_back({in, out, inputUint != 0, outputUint != 0});
                }
            }
        }
    }
    return cases;
}

inline std::vector<uint32_t> castSamples(const CastCase &test) {
    auto words = samples(test.input, test.input.integer() || test.input.boolean() || test.input.fp8() ||
                                         (test.input.width == 16 && test.output.width == 32 && test.output.floating()));
    if (test.input.integer() && test.input.width == 32 && std::string(test.output.name) == "F16") {
        // Cover every integer that rounds to finite FP16. In particular,
        // 8191 must round to 8192; truncation to 8188 exceeds the 0.5 ULP bound.
        constexpr int32_t overflowBoundary = 65520;
        words.reserve(words.size() + (size_t{2} * overflowBoundary) - 1);
        for (int32_t value = 1 - overflowBoundary; value < overflowBoundary; ++value) {
            words.push_back(uint32_t(value));
        }
    }
    // Every positive finite destination spacing: below, exactly at, and above
    // the midpoint, with both signs. FP32 represents these midpoints exactly.
    if (test.input.floating() && test.input.width == 32 && test.output.floating() && test.output.width < 32) {
        for (uint32_t bits = 0; bits <= test.output.maxFinite(); ++bits) {
            const double lower = decode(bits, test.output);
            const double upper = bits == test.output.maxFinite()
                                     ? lower + std::ldexp(1.0, std::ilogb(lower) - int(test.output.fraction))
                                     : decode(bits + 1, test.output);
            const auto midpoint = float((lower + upper) / 2);
            for (float value : {std::nextafter(midpoint, 0.0f), midpoint,
                                std::nextafter(midpoint, std::numeric_limits<float>::infinity())}) {
                words.push_back(encode(double(value), test.input));
                words.push_back(encode(-double(value), test.input));
            }
        }
    }
    // NaN -> integer is explicitly unpredictable; never invent an oracle for it.
    if (test.output.integer()) {
        words.erase(std::remove_if(words.begin(), words.end(), [&](uint32_t word) { return isNaN(word, test.input); }),
                    words.end());
    }
    return words;
}

inline bool castMatches(uint32_t actual, uint32_t input, const CastCase &test) {
    const double value = decode(input, test.input);
    if (test.output.boolean()) {
        return actual == uint32_t(value != 0);
    }
    if (test.output.integer()) {
        const double low = -double(uint64_t{1} << (test.output.width - 1));
        const double high = -low - 1;
        const int64_t expected = test.input.floating()
                                     ? int64_t(std::clamp(roundEven(std::clamp(value, low, high)), low, high))
                                     : int64_t(value);
        return actual == (uint32_t(uint64_t(expected)) & test.output.mask());
    }
    const double actualValue = decode(actual, test.output);
    if (test.input.integer() && value != 0 && std::isfinite(actualValue)) {
        // TOSA permits 0.5 ULP for integer-to-float CAST, including either
        // neighbor at a midpoint. Requiring ties-to-even bits is too strict.
        const int exponent = std::max(std::ilogb(std::abs(value)), 1 - test.output.bias);
        const double halfUlp = std::ldexp(1.0, exponent - int(test.output.fraction) - 1);
        return std::abs(actualValue - value) <= halfUlp;
    }
    for (bool nearest : {true, false}) {
        if (!nearest && !test.input.floating()) {
            continue;
        }
        const auto expected = encode(value, test.output, nearest);
        if (sameFloat(actual, expected, test.output)) {
            return true;
        }
        // TOSA permits flushing FP16/BF16/FP32 subnormals, but not FP8 subnormals.
        if (!test.output.fp8() && subnormal(expected, test.output) && actual == (expected & test.output.sign())) {
            return true;
        }
    }
    return test.input.floating() && !test.input.fp8() && subnormal(input, test.input) &&
           actual == ((input & test.input.sign()) ? test.output.sign() : 0);
}

struct RescaleCase {
    CastCase types;
    bool inputUnsigned;
    bool outputUnsigned;
    bool scale32;
    bool zeroPoints;
};

inline std::string name(const RescaleCase &test) {
    return name(test.types) + (test.inputUnsigned ? "_UnsignedInput" : "_SignedInput") +
           (test.outputUnsigned ? "_UnsignedOutput" : "_SignedOutput") + (test.scale32 ? "_Scale32" : "_Scale16") +
           (test.zeroPoints ? "_Offset" : "_Zero");
}

inline std::vector<RescaleCase> rescaleCases() {
    std::vector<RescaleCase> cases;
    for (const auto &in : types()) {
        for (const auto &out : types()) {
            if (!in.integer() || !out.integer()) {
                continue;
            }
            for (bool iu : {false, true}) {
                for (bool ou : {false, true}) {
                    // TOSA 1.0.1 RESCALE ERROR_IF constraints, not implementation limits.
                    if ((iu && ou) || ((iu || ou) && (in.width == 32 || out.width == 32))) {
                        continue;
                    }
                    // Bind opposite to the arithmetic signedness to expose accidental
                    // coupling. CAST and the existing storage tests cover the full
                    // binding Cartesian product; repeating it here is expensive.
                    const bool offset = in.width == 8 || out.width == 8 || iu || ou;
                    for (bool scale32 : {false, true}) {
                        cases.push_back({{in, out, !iu, !ou}, iu, ou, scale32, offset});
                    }
                    // Keep zero-point-free i8 controls without multiplying every
                    // type/scale/binding combination by this attribute.
                    if (in.width == 8 && out.width == 8) {
                        cases.push_back({{in, out, !iu, !ou}, iu, ou, true, false});
                    }
                }
            }
        }
    }
    return cases;
}

inline int64_t zeroPoint(const Type &type, bool isUnsigned, bool offset) {
    if (!offset) {
        return 0;
    }
    if (type.width == 8) {
        return isUnsigned ? 128 : -3;
    }
    if (type.width == 16 && isUnsigned) {
        return 32768;
    }
    return 0;
}

inline int64_t floorDivide(int64_t value, int64_t divisor) { return (value / divisor) - (value % divisor < 0 ? 1 : 0); }

enum class LayoutOp { Reverse, Slice, Tile, Transpose, Reshape, Concat };

enum class IntegerOp {
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    BitwiseNot,
    LogicalLeftShift,
    LogicalRightShift,
    ArithmeticRightShift,
    Mul,
    Add,
    Sub,
    IntDiv,
    Minimum,
    Maximum,
    Equal,
    Greater,
    GreaterEqual,
    Abs,
    Clz
};

inline const char *name(LayoutOp operation) {
    switch (operation) {
    case LayoutOp::Reverse:
        return "REVERSE";
    case LayoutOp::Slice:
        return "SLICE";
    case LayoutOp::Tile:
        return "TILE";
    case LayoutOp::Transpose:
        return "TRANSPOSE";
    case LayoutOp::Reshape:
        return "RESHAPE";
    case LayoutOp::Concat:
        return "CONCAT";
    }
    throw std::logic_error("Unknown datatype test operation");
}

inline const char *name(IntegerOp operation) {
    switch (operation) {
    case IntegerOp::BitwiseAnd:
        return "BITWISE_AND";
    case IntegerOp::BitwiseOr:
        return "BITWISE_OR";
    case IntegerOp::BitwiseXor:
        return "BITWISE_XOR";
    case IntegerOp::BitwiseNot:
        return "BITWISE_NOT";
    case IntegerOp::LogicalLeftShift:
        return "LOGICAL_LEFT_SHIFT";
    case IntegerOp::LogicalRightShift:
        return "LOGICAL_RIGHT_SHIFT";
    case IntegerOp::ArithmeticRightShift:
        return "ARITHMETIC_RIGHT_SHIFT";
    case IntegerOp::Mul:
        return "MUL";
    case IntegerOp::Add:
        return "ADD";
    case IntegerOp::Sub:
        return "SUB";
    case IntegerOp::IntDiv:
        return "INTDIV";
    case IntegerOp::Minimum:
        return "MINIMUM";
    case IntegerOp::Maximum:
        return "MAXIMUM";
    case IntegerOp::Equal:
        return "EQUAL";
    case IntegerOp::Greater:
        return "GREATER";
    case IntegerOp::GreaterEqual:
        return "GREATER_EQUAL";
    case IntegerOp::Abs:
        return "ABS";
    case IntegerOp::Clz:
        return "CLZ";
    }
    throw std::logic_error("Unknown datatype test operation");
}

struct LayoutCase {
    Type type;
    bool inputUint;
    bool outputUint;
    LayoutOp operation;
};

inline std::string name(const LayoutCase &test) {
    return std::string(name(test.operation)) + "_" + bindingName(test.type, test.inputUint) + "_To_" +
           bindingName(test.type, test.outputUint);
}

inline std::vector<LayoutCase> layoutCases() {
    std::vector<LayoutCase> cases;
    for (const auto &type : types()) {
        bool inputUint = false;
        for (auto op : {LayoutOp::Reverse, LayoutOp::Slice, LayoutOp::Tile, LayoutOp::Transpose, LayoutOp::Reshape,
                        LayoutOp::Concat}) {
            // Cross SINT/UINT bindings to check that copying preserves the bits.
            cases.push_back({type, type.integer() && inputUint, type.integer() && !inputUint, op});
            inputUint = !inputUint;
        }
    }
    return cases;
}

struct IntegerCase {
    Type type;
    IntegerOp operation;
    bool inputUint;
    bool secondUint;
    bool outputUint;
    bool round;

    bool unary() const {
        return operation == IntegerOp::BitwiseNot || operation == IntegerOp::Abs || operation == IntegerOp::Clz;
    }
    bool comparison() const {
        return operation == IntegerOp::Equal || operation == IntegerOp::Greater || operation == IntegerOp::GreaterEqual;
    }
    bool shift() const {
        return operation == IntegerOp::LogicalLeftShift || operation == IntegerOp::LogicalRightShift ||
               operation == IntegerOp::ArithmeticRightShift;
    }
    bool supportsNarrow() const {
        return shift() || operation == IntegerOp::Mul || operation == IntegerOp::BitwiseAnd ||
               operation == IntegerOp::BitwiseOr || operation == IntegerOp::BitwiseXor ||
               operation == IntegerOp::BitwiseNot;
    }
};

inline std::string name(const IntegerCase &test) {
    std::string result = name(test.operation);
    result += "_";
    result += bindingName(test.type, test.inputUint);
    if (!test.unary()) {
        result += test.secondUint ? "_SecondUINT" : "_SecondSINT";
    }
    if (test.comparison()) {
        result += "_Bool";
    } else {
        result += test.outputUint ? "_OutUINT" : "_OutSINT";
    }
    if (test.operation == IntegerOp::ArithmeticRightShift) {
        result += test.round ? "_Round" : "_Truncate";
    }
    return result;
}

inline int64_t integerReference(const IntegerCase &test, uint32_t raw, uint32_t other) {
    const auto &type = test.type;
    const auto value = int64_t(decode(raw, type));
    const auto rhs = int64_t(decode(other, type));
    switch (test.operation) {
    case IntegerOp::BitwiseAnd:
        return raw & other;
    case IntegerOp::BitwiseOr:
        return raw | other;
    case IntegerOp::BitwiseXor:
        return raw ^ other;
    case IntegerOp::BitwiseNot:
        return (~raw) & type.mask();
    case IntegerOp::LogicalLeftShift:
        return int64_t((uint64_t(raw) << other) & type.mask());
    case IntegerOp::LogicalRightShift:
        return raw >> other;
    case IntegerOp::ArithmeticRightShift: {
        const int64_t divisor = int64_t{1} << other;
        return floorDivide(value + (test.round && other ? divisor / 2 : 0), divisor);
    }
    case IntegerOp::Mul:
        return value * rhs;
    case IntegerOp::Add:
        return value + rhs;
    case IntegerOp::Sub:
        return value - rhs;
    case IntegerOp::IntDiv:
        return value / rhs;
    case IntegerOp::Minimum:
        return std::min(value, rhs);
    case IntegerOp::Maximum:
        return std::max(value, rhs);
    case IntegerOp::Equal:
        return value == rhs;
    case IntegerOp::Greater:
        return value > rhs;
    case IntegerOp::GreaterEqual:
        return value >= rhs;
    case IntegerOp::Abs:
        return std::abs(value);
    case IntegerOp::Clz: {
        int64_t result = 0;
        for (uint32_t bit = type.sign(); bit && !(raw & bit); bit >>= 1) {
            ++result;
        }
        return result;
    }
    }
    throw std::logic_error("Missing integer test reference");
}

inline std::vector<IntegerCase> integerCases() {
    std::vector<IntegerCase> cases;
    for (const auto &type : types()) {
        if (!type.integer()) {
            continue;
        }
        for (auto operation :
             {IntegerOp::BitwiseAnd, IntegerOp::BitwiseOr, IntegerOp::BitwiseXor, IntegerOp::BitwiseNot,
              IntegerOp::LogicalLeftShift, IntegerOp::LogicalRightShift, IntegerOp::ArithmeticRightShift,
              IntegerOp::Mul, IntegerOp::Add, IntegerOp::Sub, IntegerOp::IntDiv, IntegerOp::Minimum, IntegerOp::Maximum,
              IntegerOp::Equal, IntegerOp::Greater, IntegerOp::GreaterEqual, IntegerOp::Abs, IntegerOp::Clz}) {
            IntegerCase test{type, operation, true, true, true, false};
            if (!test.supportsNarrow() && type.width != 32) {
                continue;
            }
            // Full-width arithmetic on UINT bindings is the regression target.
            // TosaIntegerStorage already exercises all eight binding permutations.
            for (bool round : {false, true}) {
                if (round && operation != IntegerOp::ArithmeticRightShift) {
                    continue;
                }
                cases.push_back({type, operation, true, true, true, round});
            }
        }
    }
    return cases;
}

} // namespace datatype_test
