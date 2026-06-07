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
        extendTensorLiveRange(tensor->getTensor(), executionIndex, sessionTensors, liveRanges);
    };

    const auto &parents = pipeline.getParents();
    std::for_each(parents.begin(), parents.end(), extend);

    const auto &descendants = pipeline.getDescendants();
    std::for_each(descendants.begin(), descendants.end(), extend);
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

} // namespace

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
