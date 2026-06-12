/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "memory_planner.hpp"

#include <map>
#include <memory>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace mlsdk::el::compute::graph_op {

namespace details {

struct LiveInterval {
    size_t id;
    uint32_t first;
    uint32_t last;
    VkDeviceSize size;
    size_t order;
};

struct UnseenAllocation {
    size_t id;
    VkDeviceSize size;
};

struct AllocationPlan {
    VkDeviceSize memorySize{0};
    std::map<size_t, VkDeviceSize> offsets;
};

AllocationPlan allocateIntervals(std::vector<LiveInterval> intervals, const std::vector<UnseenAllocation> &unseen,
                                 VkDeviceSize alignment);

} // namespace details

/*******************************************************************************
 * IntervalMemoryPlanner
 *******************************************************************************/

class IntervalMemoryPlanner : public MemoryPlanner {
  public:
    explicit IntervalMemoryPlanner(const std::shared_ptr<GraphPipeline> &_graphPipeline);
    ~IntervalMemoryPlanner() override = default;

    VkMemoryRequirements getGraphPipelineSessionMemoryRequirements() const override;
    void bindGraphPipelineSessionMemory(VkDeviceMemory memory, VkDeviceSize offset,
                                        const ComputeDescriptorSetMap &descriptorSets) override;

  private:
    VkDeviceSize memorySize{0};
    std::map<std::shared_ptr<TensorDescriptor>, VkDeviceSize> tensorOffsets;
};

} // namespace mlsdk::el::compute::graph_op
