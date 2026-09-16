/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "datatype_cases.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>

namespace {

using namespace datatype_test;

TEST(DatatypeReference, IntegerToFloatAcceptsBothMidpointNeighbors) {
    struct Midpoint {
        size_t outputType;
        int32_t input;
        uint32_t lower;
        uint32_t upper;
    };
    // Exercise both tie parities in FP16, BF16, and FP32. Either neighbor of
    // a midpoint is valid, including 4095 -> 4094 and 4095 -> 4096.
    constexpr std::array<Midpoint, 6> midpoints{{
        {4, 2049, 0x6800, 0x6801},
        {4, 4095, 0x6bff, 0x6c00},
        {5, 257, 0x4380, 0x4381},
        {5, 511, 0x43ff, 0x4400},
        {6, 16777217, 0x4b800000, 0x4b800001},
        {6, 33554431, 0x4bffffff, 0x4c000000},
    }};
    for (const auto &midpoint : midpoints) {
        for (bool inputUint : {false, true}) {
            const CastCase test{types()[3], types()[midpoint.outputType], inputUint, false};
            SCOPED_TRACE(name(test));
            SCOPED_TRACE(midpoint.input);
            for (int32_t sign : {1, -1}) {
                const auto input = uint32_t(sign * midpoint.input);
                const uint32_t signBit = sign < 0 ? test.output.sign() : 0;
                EXPECT_TRUE(castMatches(midpoint.lower | signBit, input, test));
                EXPECT_TRUE(castMatches(midpoint.upper | signBit, input, test));
                EXPECT_FALSE(castMatches((midpoint.lower - 1) | signBit, input, test));
                EXPECT_FALSE(castMatches((midpoint.upper + 1) | signBit, input, test));
            }
        }
    }
}

TEST(DatatypeReference, IntegerToFloatRejectsExcessError) {
    for (bool inputUint : {false, true}) {
        const CastCase test{types()[3], types()[4], inputUint, false};
        SCOPED_TRACE(name(test));
        // Truncating 8191 to 8188 is 0.75 ULP away, beyond TOSA's 0.5 ULP bound.
        EXPECT_FALSE(castMatches(0x6fff, 8191, test));
        EXPECT_TRUE(castMatches(0x7000, 8191, test));
        EXPECT_FALSE(castMatches(0xefff, uint32_t(-8191), test));
        EXPECT_TRUE(castMatches(0xf000, uint32_t(-8191), test));
        EXPECT_TRUE(castMatches(0, 0, test));
        EXPECT_FALSE(castMatches(1, 0, test));
        EXPECT_FALSE(castMatches(0xfc00, INT32_MAX, test));
        EXPECT_FALSE(castMatches(0x7bff, INT32_MAX, test));
        EXPECT_TRUE(castMatches(0x7c00, INT32_MAX, test));
        EXPECT_TRUE(castMatches(0xfc00, uint32_t(INT32_MIN), test));
        EXPECT_FALSE(castMatches(0x7e00, 4095, test));
    }
}

} // namespace
