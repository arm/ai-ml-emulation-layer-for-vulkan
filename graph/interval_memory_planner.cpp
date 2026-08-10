/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "interval_memory_planner.hpp"

#include "graph_log.hpp"
#include "mlel/utils.hpp"

#include <algorithm>
#include <map>
#include <queue>
#include <set>
#include <utility>
#include <vector>

using namespace mlsdk::el::log;
using namespace mlsdk::el::utils;

namespace mlsdk::el::compute::graph_op {
namespace {

using LiveRange = std::pair<uint32_t, uint32_t>;
using LiveRanges = std::map<std::shared_ptr<TensorDescriptor>, LiveRange>;
using SessionTensors = std::set<std::shared_ptr<TensorDescriptor>>;
using TensorIds = std::map<size_t, std::shared_ptr<TensorDescriptor>>;
using TensorOrder = std::map<std::shared_ptr<TensorDescriptor>, size_t>;
using UnseenAllocations = std::vector<details::UnseenAllocation>;

VkDeviceSize alignedSize(const std::shared_ptr<TensorDescriptor> &tensor, const VkDeviceSize alignment) {
    return roundUp(tensor->getMemoryRequirementsSize(), alignment);
}

void extendTensorLiveRange(const std::shared_ptr<TensorDescriptor> &tensor, const uint32_t executionIndex,
                           const SessionTensors &sessionTensors, LiveRanges &liveRanges) {
    if (sessionTensors.find(tensor) == sessionTensors.end()) {
        return;
    }

    auto [it, inserted] = liveRanges.emplace(tensor, std::make_pair(executionIndex, executionIndex));
    if (!inserted) {
        it->second.first = std::min(it->second.first, executionIndex);
        it->second.second = std::max(it->second.second, executionIndex);
    }
}

void extendPipelineTensorLiveRanges(const ComputePipelineBase &pipeline, const uint32_t executionIndex,
                                    const SessionTensors &sessionTensors, LiveRanges &liveRanges) {

    const auto extend = [&](const auto &tensor) {
        extendTensorLiveRange(tensor, executionIndex, sessionTensors, liveRanges);
    };
    const auto &pipelineLayout = pipeline.getComputePipelineLayout();
    if (pipelineLayout) {
        for (const auto &descriptor : pipelineLayout->getDescriptorMap()) {
            extend(descriptor.tensor);
        }
    }

    const auto &parents = pipeline.getParents();
    std::for_each(parents.begin(), parents.end(), [&](const auto &tensor) { extend(tensor->getTensor()); });

    const auto &descendants = pipeline.getDescendants();
    std::for_each(descendants.begin(), descendants.end(), [&](const auto &tensor) { extend(tensor->getTensor()); });
}

std::vector<details::LiveInterval> createLiveIntervals(const std::shared_ptr<GraphPipeline> &graphPipeline,
                                                       const VkDeviceSize alignment, TensorIds &tensorIds,
                                                       UnseenAllocations &unseenTensors) {
    const auto &tensors = graphPipeline->getTensors();
    const SessionTensors sessionTensors(tensors.begin(), tensors.end());
    TensorOrder tensorOrder;
    LiveRanges liveRanges;

    for (size_t i = 0; i < tensors.size(); ++i) {
        tensorOrder[tensors[i]] = i;
        tensorIds[i] = tensors[i];
    }

    uint32_t executionIndex = 0;
    extendPipelineTensorLiveRanges(graphPipeline->getInputs(), executionIndex++, sessionTensors, liveRanges);

    for (const auto &pipeline : graphPipeline->getPipelines()) {
        extendPipelineTensorLiveRanges(*pipeline, executionIndex++, sessionTensors, liveRanges);
    }

    extendPipelineTensorLiveRanges(graphPipeline->getOutputs(), executionIndex, sessionTensors, liveRanges);

    std::vector<details::LiveInterval> intervals;
    intervals.reserve(liveRanges.size());
    for (const auto &[tensor, interval] : liveRanges) {
        const auto tensorOrderIt = tensorOrder.find(tensor);
        if (tensorOrderIt == tensorOrder.end()) {
            continue;
        }

        const auto order = tensorOrderIt->second;
        intervals.push_back({order, interval.first, interval.second, alignedSize(tensor, alignment), order});
    }

    unseenTensors.clear();
    for (const auto &tensor : tensors) {
        if (liveRanges.find(tensor) == liveRanges.end()) {
            unseenTensors.push_back({tensorOrder.at(tensor), alignedSize(tensor, alignment)});
        }
    }

    return intervals;
}

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

namespace details {
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
} // namespace details

/*******************************************************************************
 * IntervalMemoryPlanner
 *******************************************************************************/

IntervalMemoryPlanner::IntervalMemoryPlanner(const std::shared_ptr<GraphPipeline> &_graphPipeline)
    : MemoryPlanner(_graphPipeline) {
    const auto alignment = std::get<0>(memoryRequirements);

    TensorIds tensorIds;
    UnseenAllocations unseenTensors;
    auto intervals = createLiveIntervals(graphPipeline, alignment, tensorIds, unseenTensors);
    const auto allocationPlan = details::allocateIntervals(std::move(intervals), unseenTensors, alignment);

    memorySize = allocationPlan.memorySize;
    for (const auto &[id, offset] : allocationPlan.offsets) {
        tensorOffsets[tensorIds.at(id)] = offset;
    }

    graphLog(Severity::Info) << "Memory usage after interval allocation: " << memorySize << std::endl;
}

VkMemoryRequirements IntervalMemoryPlanner::getGraphPipelineSessionMemoryRequirements() const {
    const auto [alignment, memoryTypeBits] = memoryRequirements;

    VkMemoryRequirements requirements = {
        memorySize,
        alignment,
        memoryTypeBits,
    };

    return requirements;
}

void IntervalMemoryPlanner::bindGraphPipelineSessionMemory(VkDeviceMemory memory, VkDeviceSize offset,
                                                           const ComputeDescriptorSetMap &descriptorSetsMapping) {
    std::set<VkTensorARM> tensorSet;
    for ([[maybe_unused]] const auto &[_, descriptorSet] : descriptorSetsMapping) {
        for (const auto &tensor : descriptorSet->getTensors()) {
            auto *const tensorARM = tensor->getVkTensorARM();
            if (tensorSet.find(tensorARM) != tensorSet.end()) {
                continue;
            }

            const auto tensorOffset = tensorOffsets.find(tensor->getTensorDescriptor());
            if (tensorOffset == tensorOffsets.end()) {
                continue;
            }

            // To avoid duplicates
            tensorSet.insert(tensorARM);

            (void)tensor->bindTensorMemory(memory, offset + tensorOffset->second);
        }
    }
}

} // namespace mlsdk::el::compute::graph_op
