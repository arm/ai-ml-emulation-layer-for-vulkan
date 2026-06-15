/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "interval_memory_planner.hpp"

#include "mlel/utils.hpp"

#include <algorithm>
#include <queue>
#include <set>
#include <vector>

using namespace mlsdk::el::utils;

namespace mlsdk::el::compute::graph_op::details {
namespace {

struct ActiveAllocation {
    uint32_t last;
    VkDeviceSize offset;
    VkDeviceSize size;
    size_t order;
};

struct ActiveAllocationEndsFirst {
    bool operator()(const ActiveAllocation &left, const ActiveAllocation &right) const {
        if (left.last != right.last) {
            return left.last > right.last;
        }

        return left.order > right.order;
    }
};

struct FreeBlock {
    VkDeviceSize offset;
    VkDeviceSize size;
    size_t order;
};

struct FreeBlockSmallestFitFirst {
    bool operator()(const FreeBlock &left, const FreeBlock &right) const {
        if (left.size != right.size) {
            return left.size < right.size;
        }

        return left.order < right.order;
    }
};

using FreeBlocks = std::multiset<FreeBlock, FreeBlockSmallestFitFirst>;

VkDeviceSize allocateFromFreeBlocks(FreeBlocks &freeBlocks, const VkDeviceSize size) {
    auto freeBlock = freeBlocks.lower_bound({0, size, 0});
    if (freeBlock == freeBlocks.end()) {
        return VK_WHOLE_SIZE;
    }

    const auto offset = freeBlock->offset;
    if (freeBlock->size > size) {
        freeBlocks.insert({freeBlock->offset + size, freeBlock->size - size, freeBlock->order});
    }
    freeBlocks.erase(freeBlock);

    return offset;
}

} // namespace

AllocationPlan allocateIntervals(std::vector<LiveInterval> intervals, const std::vector<UnseenAllocation> &unseen,
                                 const VkDeviceSize alignment) {
    std::stable_sort(intervals.begin(), intervals.end(), [](const auto &left, const auto &right) {
        if (left.first != right.first) {
            return left.first < right.first;
        }

        if (left.size != right.size) {
            return left.size > right.size;
        }

        return left.order < right.order;
    });

    AllocationPlan plan;
    std::priority_queue<ActiveAllocation, std::vector<ActiveAllocation>, ActiveAllocationEndsFirst> activeAllocations;
    FreeBlocks freeBlocks;
    size_t freeBlockOrder = 0;

    for (const auto &interval : intervals) {
        while (!activeAllocations.empty() && activeAllocations.top().last < interval.first) {
            const auto &expired = activeAllocations.top();
            freeBlocks.insert({expired.offset, expired.size, freeBlockOrder++});
            activeAllocations.pop();
        }

        auto offset = allocateFromFreeBlocks(freeBlocks, interval.size);
        if (offset == VK_WHOLE_SIZE) {
            offset = roundUp(plan.memorySize, alignment);
            plan.memorySize = offset + interval.size;
        }

        plan.offsets[interval.id] = offset;
        activeAllocations.push({interval.last, offset, interval.size, interval.order});
    }

    for (const auto &allocation : unseen) {
        const auto offset = roundUp(plan.memorySize, alignment);
        plan.offsets[allocation.id] = offset;
        plan.memorySize = offset + allocation.size;
    }

    plan.memorySize = roundUp(plan.memorySize, alignment);

    return plan;
}

} // namespace mlsdk::el::compute::graph_op::details
