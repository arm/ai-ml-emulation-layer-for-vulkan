/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include <gtest/gtest.h>

#include "mlel/float.hpp"
#include "mlel/log.hpp"
#include "mlel/utils.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <vulkan/vulkan.h>

using namespace mlsdk::el::log;
using namespace mlsdk::el::utils;

namespace {

TEST(MLEmulationLayerLog, DefaultLogLevelHighSeverity) {
    Log testLog("VMEL_TEST_SEVERITY", "TestLog");

    testLog(Severity::Debug) << "Serverity is Debug(3)\n";
}

TEST(MLEmulationLayerLog, DefaultLogLevelLowSeverity) { // cppcheck-suppress syntaxError
    Log testLog("VMEL_TEST_SEVERITY", "TestLog");

    testLog(Severity::Error) << "Serverity is Error(0)\n";
}

TEST(MLEmulationLayerLog, StdFunctions) {
    Log testLog("VMEL_TEST_SEVERITY", "TestLog");

    testLog(Severity::Error) << "Serverity is Error(0)" << std::endl;
    testLog(Severity::Info) << "Serverity is Info(2)" << std::endl;
    testLog(Severity::Error) << "Serverity is Error(0)" << std::endl;
}

TEST(MLEmulationLayerLog, Vectors) {
    Log testLog("VMEL_TEST_SEVERITY", "TestLog");
    const std::vector<std::string> strVector{"Hello", "World", "!"};
    const std::vector<int> intVector{1, 2, 3, 4, 5};

    testLog(Severity::Error) << strVector << "\n";
    testLog(Severity::Error) << intVector << "\n";
}

TEST(MLEmulationLayerLog, LineNumbers) {
    Log testLog("VMEL_TEST_SEVERITY", "TestLog");
    const std::string str("Hello world\nThis is line 2\nFinal line");

    testLog(Severity::Error) << StringLineNumber(str) << std::endl;
}

TEST(MLEmulationLayerLog, HexDump) {
    Log testLog("VMEL_TEST_SEVERITY", "TestLog");
    const uint8_t testchar[]{"Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor "
                             "incididunt ut labore et dolore magna "
                             "aliqua."};
    const auto *charPointer{testchar};

    testLog(Severity::Error) << HexDump(charPointer, sizeof(testchar));
}

template <typename T> void checkFloat(T v) {
    float8_e4m3 f8{v};
    float16 f16{v};
    float32 f32{v};
    float64 f64{v};

    EXPECT_NEAR(double(f8), double(v), 0.5);
    ASSERT_EQ(double(f16), v);
    ASSERT_EQ(double(f32), v);
    ASSERT_EQ(double(f64), v);

    v += 1;
    f8 = v;
    f16 = v;
    f32 = v;
    f64 = v;

    EXPECT_NEAR(double(f8), double(v), 0.5);
    ASSERT_EQ(double(f16), v);
    ASSERT_EQ(double(f32), v);
    ASSERT_EQ(double(f64), v);
}

TEST(MLEmulationLayerFloat, Formats) {
    ASSERT_EQ(sizeof(float8_e4m3), 1);
    ASSERT_EQ(sizeof(float16), 2);
    ASSERT_EQ(sizeof(float32), 4);
    ASSERT_EQ(sizeof(float64), 8);

    checkFloat(float(-10.5));
    checkFloat(float(-0.5));
    checkFloat(float(0));
    checkFloat(float(0.5));
    checkFloat(float(10.0));

    checkFloat(int8_t(1));
    checkFloat(uint8_t(2));
    checkFloat(int16_t(3));
    checkFloat(uint16_t(4));
    checkFloat(int32_t(5));
    checkFloat(uint32_t(6));
    checkFloat(int64_t(7));
    checkFloat(uint64_t(8));

    checkFloat(float8_e4m3(10.5));
    checkFloat(float16(11.5));
    checkFloat(float32(12.5));
    checkFloat(float64(13.5));

    auto f = float(float16(10.5));
    ASSERT_EQ(double(f), 10.5);

    float16 f16;

    f16 = f16 + 10;
    ASSERT_EQ(double(f16), 10);

    f16 += 10;
    ASSERT_EQ(double(f16), 20);

    f16 = f16 - 5;
    ASSERT_EQ(double(f16), 15);

    f16 -= 5;
    ASSERT_EQ(double(f16), 10);

    f16 = f16 * 2;
    ASSERT_EQ(double(f16), 20);

    f16 *= 2;
    ASSERT_EQ(double(f16), 40);

    f16 = f16 / 4;
    ASSERT_EQ(double(f16), 10);

    f16 /= 4;
    ASSERT_EQ(double(f16), 2.5);

    ASSERT_TRUE(f16 < 5);
    ASSERT_FALSE(5 < f16);

    ASSERT_TRUE(f16 <= 5);
    ASSERT_FALSE(5 <= f16);

    ASSERT_FALSE(f16 > 5);
    ASSERT_TRUE(5 > f16);

    ASSERT_FALSE(f16 >= 5);
    ASSERT_TRUE(5 >= f16);

    ASSERT_FALSE(f16 == 5);
    ASSERT_TRUE(f16 == 2.5);

    ASSERT_TRUE(f16 != 5);
    ASSERT_FALSE(f16 != 2.5);

    uint32_t overflow = 0U << 31 | 250 << 23 | 1 << 22;
    void *overflowPtr = &overflow;
    f16 = *reinterpret_cast<float *>(overflowPtr);
    ASSERT_FALSE(f16.isnan());
    ASSERT_TRUE(f16.isinf());
    ASSERT_FALSE(std::isnan(f16));
    ASSERT_TRUE(std::isinf(f16));
    ASSERT_FALSE(std::isnormal(float(f16)));

    uint32_t nan = 0xffffffff;
    void *nanPtr = &nan;
    f16 = *reinterpret_cast<float *>(nanPtr);
    ASSERT_TRUE(f16.isnan());
    ASSERT_FALSE(f16.isinf());
    ASSERT_TRUE(std::isnan(f16));
    ASSERT_FALSE(std::isinf(f16));
    ASSERT_FALSE(std::isnormal(float(f16)));

    uint32_t pinf = 0U << 31 | 0xffU << 23 | 0;
    void *pinfPtr = &pinf;
    f16 = *reinterpret_cast<float *>(pinfPtr);
    ASSERT_FALSE(f16.isnan());
    ASSERT_TRUE(f16.isinf());
    ASSERT_FALSE(std::isnan(f16));
    ASSERT_TRUE(std::isinf(f16));
    ASSERT_FALSE(std::isnormal(float(f16)));

    uint32_t ninf = 1U << 31 | 0xffU << 23 | 0;
    void *ninfPtr = &ninf;
    f16 = *reinterpret_cast<float *>(ninfPtr);
    ASSERT_FALSE(f16.isnan());
    ASSERT_TRUE(f16.isinf());
    ASSERT_FALSE(std::isnan(f16));
    ASSERT_TRUE(std::isinf(f16));
    ASSERT_FALSE(std::isnormal(float(f16)));
}

struct ExpectedFormatInfo {
    VkFormat format;
    bool isInteger;
    bool isSigned;
    std::string lowest;
    std::string max;
    std::string_view glslType;
    uint64_t typeId;
    std::string_view compType;
};

template <typename T> constexpr int legacyCharType() {
    if constexpr (std::numeric_limits<T>::is_integer) {
        if constexpr (std::numeric_limits<T>::digits == 1) {
            return 'b';
        } else if constexpr (std::numeric_limits<T>::is_signed) {
            return 'i';
        } else {
            return 'u';
        }
    } else {
        return 'f';
    }
}

template <typename T>
ExpectedFormatInfo legacyFormat(VkFormat format, std::string_view glslType, std::string_view literalSuffix = "",
                                std::string_view compType = "") {
    constexpr auto typeId = uint64_t(legacyCharType<T>()) << 8 | ('0' + sizeof(T));
    return {
        format,
        std::numeric_limits<T>::is_integer,
        std::numeric_limits<T>::is_signed,
        std::to_string(std::numeric_limits<T>::lowest()) + std::string(literalSuffix),
        std::to_string(std::numeric_limits<T>::max()) + std::string(literalSuffix),
        glslType,
        typeId,
        compType.empty() ? glslType : compType,
    };
}

uint64_t parseTypeId(std::string_view typeId) { return std::stoull(std::string(typeId), nullptr, 0); }

void expectFormatInfo(const ExpectedFormatInfo &expected) {
    const auto *actual = getFormatInfo(expected.format);

    ASSERT_NE(actual, nullptr);
    EXPECT_EQ(actual->isInteger, expected.isInteger);
    EXPECT_EQ(actual->isSigned, expected.isSigned);
    EXPECT_EQ(actual->lowest, expected.lowest);
    EXPECT_EQ(actual->max, expected.max);
    EXPECT_EQ(actual->glslType, expected.glslType);
    EXPECT_EQ(parseTypeId(actual->typeId), expected.typeId);
    EXPECT_EQ(actual->compType, expected.compType);
}

TEST(MLEmulationLayerUtils, MakeFormatSupportedFormatsMatchLegacyFormatImplementation) {
    const std::vector<ExpectedFormatInfo> expectedFormats = {
        legacyFormat<int8_t>(VK_FORMAT_R8_SINT, "int8_t"),
        legacyFormat<uint8_t>(VK_FORMAT_R8_UINT, "uint8_t", "u"),
        legacyFormat<uint8_t>(VK_FORMAT_S8_UINT, "uint8_t", "u"),
        legacyFormat<bool>(VK_FORMAT_R8_BOOL_ARM, "bool"),
        legacyFormat<int16_t>(VK_FORMAT_R16_SINT, "int16_t"),
        legacyFormat<uint16_t>(VK_FORMAT_R16_UINT, "uint16_t", "u"),
        legacyFormat<float16>(VK_FORMAT_R16_SFLOAT, "float16_t"),
        {VK_FORMAT_R16_SFLOAT_FPENCODING_BFLOAT16_ARM, false, true, "-3.3895313892515355e+38", "3.3895313892515355e+38",
         "bfloat16_t", 0x6642, "float"},
        {VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E5M2_ARM, false, true, "-57344", "57344", "float8_e5m2_t", 0x664D,
         "float16_t"},
        {VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E4M3_ARM, false, true, "-448", "448", "float8_e4m3_t", 0x664E,
         "float16_t"},
        legacyFormat<int32_t>(VK_FORMAT_R32_SINT, "int"),
        legacyFormat<uint32_t>(VK_FORMAT_R32_UINT, "uint32_t", "u"),
        legacyFormat<float>(VK_FORMAT_R32_SFLOAT, "float"),
        legacyFormat<int64_t>(VK_FORMAT_R64_SINT, "int64_t", "ll"),
        legacyFormat<uint64_t>(VK_FORMAT_R64_UINT, "uint64_t", "ull"),
        legacyFormat<double>(VK_FORMAT_R64_SFLOAT, "double", "ll"),
    };

    for (const auto &expected : expectedFormats) {
        SCOPED_TRACE(std::to_string(expected.format));
        expectFormatInfo(expected);
    }
}

TEST(MLEmulationLayerUtils, MakeFormatThrowsForUnsupportedFormat) {
    EXPECT_THROW(static_cast<void>(getFormatInfo(VK_FORMAT_UNDEFINED)), std::runtime_error);
}

} // namespace
