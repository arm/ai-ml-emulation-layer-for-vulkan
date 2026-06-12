/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include <gtest/gtest.h>

#include "interval_memory_planner.hpp"

#include <vector>

namespace {

namespace planner = mlsdk::el::compute::graph_op::details;

TEST(IntervalMemoryPlanner, NonOverlappingTensorsReuseMemory) {
    std::vector<planner::LiveInterval> intervals{
        {0, 0, 0, 64, 0},
        {1, 1, 1, 64, 1},
    };
    const std::vector<planner::UnseenAllocation> unseen;
    const auto plan = planner::allocateIntervals(intervals, unseen, 16);

    ASSERT_EQ(plan.offsets.at(0), 0u);
    ASSERT_EQ(plan.offsets.at(1), plan.offsets.at(0));
    ASSERT_EQ(plan.memorySize, 64u);
}

TEST(IntervalMemoryPlanner, OverlappingTensorsDoNotAlias) { // cppcheck-suppress syntaxError
    std::vector<planner::LiveInterval> intervals{
        {0, 0, 2, 64, 0},
        {1, 1, 3, 64, 1},
    };
    const std::vector<planner::UnseenAllocation> unseen;
    const auto plan = planner::allocateIntervals(intervals, unseen, 16);

    ASSERT_NE(plan.offsets.at(0), plan.offsets.at(1));
    ASSERT_EQ(plan.memorySize, 128u);
}

TEST(IntervalMemoryPlanner, SplitsOversizedFreeBlocks) {
    std::vector<planner::LiveInterval> intervals{
        {0, 0, 0, 128, 0},
        {1, 1, 2, 64, 1},
        {2, 1, 2, 64, 2},
    };
    const std::vector<planner::UnseenAllocation> unseen;
    const auto plan = planner::allocateIntervals(intervals, unseen, 16);

    ASSERT_EQ(plan.offsets.at(0), 0u);
    ASSERT_EQ(plan.offsets.at(1), 0u);
    ASSERT_EQ(plan.offsets.at(2), 64u);
    ASSERT_EQ(plan.memorySize, 128u);
}

TEST(IntervalMemoryPlanner, DoesNotCoalesceAdjacentFreeBlocks) {
    std::vector<planner::LiveInterval> intervals{
        {0, 0, 0, 64, 0},
        {1, 0, 0, 64, 1},
        {2, 1, 1, 128, 2},
    };
    const std::vector<planner::UnseenAllocation> unseen;
    const auto plan = planner::allocateIntervals(intervals, unseen, 16);

    ASSERT_EQ(plan.offsets.at(0), 0u);
    ASSERT_EQ(plan.offsets.at(1), 64u);
    ASSERT_EQ(plan.offsets.at(2), 128u);
    ASSERT_EQ(plan.memorySize, 256u);
}

TEST(IntervalMemoryPlanner, UnseenTensorsReceiveUniqueStorage) {
    std::vector<planner::LiveInterval> intervals{
        {0, 0, 0, 64, 0},
    };
    const std::vector<planner::UnseenAllocation> unseen{
        {1, 32},
        {2, 32},
    };
    const auto plan = planner::allocateIntervals(intervals, unseen, 16);

    ASSERT_EQ(plan.offsets.at(0), 0u);
    ASSERT_EQ(plan.offsets.at(1), 64u);
    ASSERT_EQ(plan.offsets.at(2), 96u);
    ASSERT_NE(plan.offsets.at(1), plan.offsets.at(2));
    ASSERT_EQ(plan.memorySize, 128u);
}

TEST(IntervalMemoryPlanner, ExternalTensorsAreNotIncludedInSessionAllocation) {
    constexpr size_t sessionTensor = 0;
    constexpr size_t unseenSessionTensor = 1;
    constexpr size_t externalTensor = 2;

    std::vector<planner::LiveInterval> intervals{
        {sessionTensor, 0, 0, 64, 0},
    };
    const std::vector<planner::UnseenAllocation> unseen{
        {unseenSessionTensor, 64},
    };
    const auto plan = planner::allocateIntervals(intervals, unseen, 16);

    ASSERT_NE(plan.offsets.find(sessionTensor), plan.offsets.end());
    ASSERT_NE(plan.offsets.find(unseenSessionTensor), plan.offsets.end());
    ASSERT_EQ(plan.offsets.find(externalTensor), plan.offsets.end());
    ASSERT_EQ(plan.memorySize, 128u);
}

} // namespace
