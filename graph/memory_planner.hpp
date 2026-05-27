/*
 * SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "compute_graph_op.hpp"
#include "tensor.hpp"

#include <map>
#include <memory>
#include <set>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace mlsdk::el::compute::graph_op {

/*******************************************************************************
 * MemoryPlanner
 *******************************************************************************/

class MemoryPlanner {
  public:
    explicit MemoryPlanner(const std::shared_ptr<GraphPipeline> &_graphPipeline);

    virtual ~MemoryPlanner() = default;

    virtual VkMemoryRequirements getGraphPipelineSessionMemoryRequirements() const = 0;
    virtual void bindGraphPipelineSessionMemory(VkDeviceMemory memory, VkDeviceSize offset,
                                                const ComputeDescriptorSetMap &descriptorSets) = 0;

  protected:
    std::tuple<VkDeviceSize, uint32_t> getGraphPipelineSessionMemoryRequirementsPartial() const;

    std::shared_ptr<GraphPipeline> graphPipeline;
    std::tuple<VkDeviceSize, uint32_t> memoryRequirements;
};

/*******************************************************************************
 * LinearMemoryPlanner
 *******************************************************************************/

class LinearMemoryPlanner : public MemoryPlanner {
  public:
    using MemoryPlanner::MemoryPlanner;
    ~LinearMemoryPlanner() override = default;

    VkMemoryRequirements getGraphPipelineSessionMemoryRequirements() const override;
    void bindGraphPipelineSessionMemory(VkDeviceMemory memory, VkDeviceSize offset,
                                        const ComputeDescriptorSetMap &descriptorSets) override;
};

using Tensors = std::vector<std::shared_ptr<TensorDescriptor>>;

/*******************************************************************************
 * BestFitMemoryPlanner
 *******************************************************************************/

class BestFitMemoryPlanner : public MemoryPlanner {
  public:
    explicit BestFitMemoryPlanner(const std::shared_ptr<GraphPipeline> &_graphPipeline);
    ~BestFitMemoryPlanner() override = default;

    VkMemoryRequirements getGraphPipelineSessionMemoryRequirements() const override;
    void bindGraphPipelineSessionMemory(VkDeviceMemory memory, VkDeviceSize offset,
                                        const ComputeDescriptorSetMap &descriptorSets) override;

  private:
    VkDeviceSize memorySize{0};
    std::map<std::shared_ptr<TensorDescriptor>, VkDeviceSize> tensorOffsets;

    using SafeSet = std::set<std::shared_ptr<TensorDescriptor>>;
    using SafeToReuseMap = std::map<std::shared_ptr<TensorDescriptor>, SafeSet>;
    using AlternativesMap = std::map<std::shared_ptr<TensorDescriptor>, Tensors>;
    using OccupationMap = std::map<std::shared_ptr<TensorDescriptor>, std::shared_ptr<Tensors>>;

    void bestFitAllocation(const Tensors &tensors, const SafeToReuseMap &safeToReuse,
                           const AlternativesMap &allAlternatives);
    void allocate(const std::shared_ptr<TensorDescriptor> &tensor, VkDeviceSize memoryAddress);
    std::shared_ptr<TensorDescriptor> findAlternativeTensor(const std::shared_ptr<TensorDescriptor> &tensor,
                                                            const OccupationMap &tensorOccupation,
                                                            const AlternativesMap &allAlternatives,
                                                            const SafeToReuseMap &safeToReuse) const;
    bool isAllocated(const std::shared_ptr<TensorDescriptor> &tensor) const;
    bool isSafeToReuse(const std::shared_ptr<Tensors> &occupationList, const std::shared_ptr<TensorDescriptor> &tensor,
                       const SafeToReuseMap &safeToReuse) const;
    SafeToReuseMap liveTensorAnalysis(const Tensors &tensors) const;
    Tensors createInitialTensorOrder() const;
    AlternativesMap createAllAlternatives(const Tensors &tensors, const SafeToReuseMap &safeToReuse) const;
    std::vector<ComputePipelineBase *> getTopologicalOrder() const;
};

} // namespace mlsdk::el::compute::graph_op
