/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include <gtest/gtest.h>

#include "../graph/spirv_pass.hpp"

#include <cstdint>
#include <vector>

namespace {

TEST(MLEmulationLayerSpirvPass, ReplicatedPatternExpansionTwoToFour) {
    std::vector<int32_t> values{1, 2};
    ASSERT_TRUE(spvtools::opt::tryExpandReplicatedPattern(values, 4));
    ASSERT_EQ(values, (std::vector<int32_t>{1, 2, 1, 2}));
}

TEST(MLEmulationLayerSpirvPass, ReplicatedPatternExpansionRejectsNonDivisibleCounts) { // cppcheck-suppress syntaxError
    std::vector<int32_t> values{1, 2};
    ASSERT_FALSE(spvtools::opt::tryExpandReplicatedPattern(values, 3));
    ASSERT_EQ(values, (std::vector<int32_t>{1, 2}));
}

} // namespace
