/*
 * SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "compute.hpp"
#include "tensor.hpp"

#include <map>
#include <memory>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace mlsdk::el::compute {

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

  protected:
    void bestFitAllocation();

    VkDeviceSize memorySize;

    const Tensors tensors;
    const std::map<std::shared_ptr<TensorDescriptor>, Tensors> safeToReuse;
    const std::map<std::shared_ptr<TensorDescriptor>, Tensors> allAlternatives;
    std::map<std::shared_ptr<TensorDescriptor>, VkDeviceSize> tensorOffsets;

  private:
    void allocate(const std::shared_ptr<TensorDescriptor> &tensor, VkDeviceSize memoryAddress);
    std::shared_ptr<TensorDescriptor> findAlternativeTensor(
        const std::shared_ptr<TensorDescriptor> &tensor,
        const std::map<std::shared_ptr<TensorDescriptor>, std::shared_ptr<Tensors>> &tensorOccupation);
    bool isAllocated(const std::shared_ptr<TensorDescriptor> &tensor) const;
    bool isSafeToReuse(const std::shared_ptr<Tensors> &occupationList,
                       const std::shared_ptr<TensorDescriptor> &tensor) const;
    std::map<std::shared_ptr<TensorDescriptor>, Tensors> liveTensorAnalysis() const;
    Tensors createInitialTensorOrder() const;
    std::map<std::shared_ptr<TensorDescriptor>, Tensors> createAllAlternatives() const;
    std::vector<ComputePipelineBase *> getTopologicalOrder() const;
};

} // namespace mlsdk::el::compute
