/*
 * SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "memory_planner.hpp"
#include "graph_log.hpp"
#include "mlel/utils.hpp"

#include <cmath>
#include <deque>
#include <numeric>
#include <set>
#include <stdexcept>

using namespace mlsdk::el::utils;
using namespace mlsdk::el::log;

namespace mlsdk::el::compute {

/*******************************************************************************
 * MemoryPlanner
 *******************************************************************************/

MemoryPlanner::MemoryPlanner(const std::shared_ptr<GraphPipeline> &_graphPipeline)
    : graphPipeline{_graphPipeline}, memoryRequirements{getGraphPipelineSessionMemoryRequirementsPartial()} {}

std::tuple<VkDeviceSize, uint32_t> MemoryPlanner::getGraphPipelineSessionMemoryRequirementsPartial() const {
    const auto &tensorSet = graphPipeline->getTensors();

    VkDeviceSize alignment = 1;
    uint32_t memoryTypeBits = 0xffffffff;
    for (auto const &tensor : tensorSet) {
        const auto &tensorMemReqs = tensor->getMemoryRequirements();
        alignment = std::max(alignment, tensorMemReqs.alignment);
        memoryTypeBits &= tensorMemReqs.memoryTypeBits;
    }

    return std::make_tuple(alignment, memoryTypeBits);
}

/*******************************************************************************
 * LinearMemoryPlanner
 *******************************************************************************/

VkMemoryRequirements LinearMemoryPlanner::getGraphPipelineSessionMemoryRequirements() const {
    const auto &tensorSet = graphPipeline->getTensors();
    const auto [_alignment, memoryTypeBits] = memoryRequirements;

    VkDeviceSize size = std::accumulate(tensorSet.begin(), tensorSet.end(), VkDeviceSize(0),
                                        [alignment = _alignment](VkDeviceSize sum, const auto &tensorDescriptor) {
                                            const auto reqsSize = tensorDescriptor->getMemoryRequirementsSize();
                                            VkDeviceSize offset = roundUp(sum, alignment);
                                            offset = roundUp(offset + reqsSize, alignment);
                                            return offset;
                                        });

    graphLog(Severity::Info) << "Memory usage after linear allocation: " << size << std::endl;

    VkMemoryRequirements requirements = {
        size,
        _alignment,
        memoryTypeBits,
    };

    return requirements;
}

void LinearMemoryPlanner::bindGraphPipelineSessionMemory(VkDeviceMemory memory, VkDeviceSize offset,
                                                         const ComputeDescriptorSetMap &descriptorSetsMapping) {
    const auto [alignment, memoryTypeBits] = memoryRequirements;

    std::set<VkTensorARM> tensorSet;
    for ([[maybe_unused]] const auto &[_, descriptorSet] : descriptorSetsMapping) {
        const auto &tensorARM = descriptorSet->getVkTensorARM();
        if (tensorSet.find(tensorARM) != tensorSet.end()) {
            continue;
        }

        // To avoid duplicates
        tensorSet.insert(tensorARM);

        offset = roundUp(offset, alignment);
        offset = roundUp(offset + descriptorSet->getTensor()->bindTensorMemory(memory, offset), alignment);
    }
}

/*******************************************************************************
 * BestFitMemoryPlanner
 *******************************************************************************/

BestFitMemoryPlanner::BestFitMemoryPlanner(const std::shared_ptr<GraphPipeline> &_graphPipeline)
    : MemoryPlanner(_graphPipeline), tensors{createInitialTensorOrder()}, safeToReuse{liveTensorAnalysis()},
      allAlternatives{createAllAlternatives()} {
    graphLog(Severity::Debug) << "Number of tensors: " << tensors.size() << std::endl;
    bestFitAllocation();

    graphLog(Severity::Info) << "Memory usage after best-fit allocation: " << memorySize << std::endl;
}

VkMemoryRequirements BestFitMemoryPlanner::getGraphPipelineSessionMemoryRequirements() const {
    const auto [alignment, memoryTypeBits] = memoryRequirements;

    VkMemoryRequirements requirements = {
        memorySize,
        alignment,
        memoryTypeBits,
    };

    return requirements;
}

void BestFitMemoryPlanner::bindGraphPipelineSessionMemory(VkDeviceMemory memory, VkDeviceSize offset,
                                                          const ComputeDescriptorSetMap &descriptorSetsMapping) {
    std::set<VkTensorARM> tensorSet;
    for ([[maybe_unused]] const auto &[_, descriptorSet] : descriptorSetsMapping) {
        const auto &tensorARM = descriptorSet->getVkTensorARM();
        if (tensorSet.find(tensorARM) != tensorSet.end()) {
            continue;
        }

        // To avoid duplicates
        tensorSet.insert(tensorARM);

        const auto &tensor = descriptorSet->getTensor();
        const auto newOffset = offset + tensorOffsets.at(tensor->getTensorDescriptor());
        (void)tensor->bindTensorMemory(memory, newOffset);
    }
}

/*
 * Live-Tensor Analysis creates sets of safe tensors. Tensor A can use the same memory space
 * as Tensor B as long as Tensor B is part of the safeToReuse-set for Tensor A. The sets are
 * created by traversing the graph as described below.
 *     1. Ensure the nodes are organized topologically. For each node (pipeline), repeat step 2-4.
 *     2. Add the virtual tensors in the carryOn-set to the carryOn-sets of all the node's descendants.
 *     3. For each virtual tensor in the node's carryOn-set:
 *        3.1 Count how many unique virtual tensor references have been received at the node.
 *        3.2 If the number is the same as the tensor's getReferenceCounter, goto step 4.
 *     4. For all descendant tensors of the node, if the tensor is not an input/output tensor,
 *        place the tensor in the descendant tensor's safeToReuse-set, and vice versa.
 */
std::map<std::shared_ptr<TensorDescriptor>, Tensors> BestFitMemoryPlanner::liveTensorAnalysis() const {
    const auto &topological = getTopologicalOrder();
    std::map<ComputePipelineBase *, std::set<std::shared_ptr<VirtualTensor>>> carryOnSet;
    std::map<ComputePipelineBase *, std::vector<std::shared_ptr<VirtualTensor>>> carryOn;
    std::map<std::shared_ptr<TensorDescriptor>, Tensors> _safeToReuse;
    for (const auto &tensor : tensors) {
        _safeToReuse.emplace(tensor, Tensors());
    }

    const auto &input = graphPipeline->getInputs();
    const auto &output = graphPipeline->getOutputs();
    const auto &inputTensor = input.getDescendants();
    const auto &outputTensor = output.getParents();
    std::set<std::shared_ptr<TensorDescriptor>> inputOutput;

    for (const auto &virtualTensor : inputTensor) {
        inputOutput.insert(virtualTensor->getTensor());
    }
    for (const auto &virtualTensor : outputTensor) {
        inputOutput.insert(virtualTensor->getTensor());
    }

    for (const auto &pipeline : topological) {
        const auto &parents = pipeline->getParents();
        const auto &descendants = pipeline->getDescendants();

        // Pass on the virtual tensor references along the edges of the graph
        for (const auto &descendant : descendants) {
            const auto descendantPipeline = descendant->getDescendantPipeline();

            auto carryOnInsert = [&](const auto &tensor) {
                auto [iterator, inserted] = carryOnSet[descendantPipeline].insert(tensor);
                if (inserted) {
                    carryOn[descendantPipeline].push_back(tensor);
                }
            };

            for (const auto &tensor : carryOn[pipeline]) {
                carryOnInsert(tensor);
            }

            for (const auto &parent : parents) {
                carryOnInsert(parent);
            }
        }

        std::map<std::shared_ptr<TensorDescriptor>, uint64_t> tensorCounter;

        for (const auto &virtualTensor : carryOn[pipeline]) {
            const auto &tensor = virtualTensor->getTensor();

            if (inputOutput.find(tensor) == inputOutput.end()) {
                tensorCounter[tensor] += 1;

                // When all virtual tensor references are received, the tensor can be added to safeToReuse,
                // as long as it is not an input/output tensor
                if (tensorCounter[tensor] == tensor->getReferenceCounter()) {
                    for (const auto &descendant : descendants) {
                        const auto &descendantTensor = descendant->getTensor();

                        if (inputOutput.find(descendantTensor) != inputOutput.end()) {
                            continue;
                        }

                        _safeToReuse[descendantTensor].push_back(tensor);
                        _safeToReuse[tensor].push_back(descendantTensor);
                    }
                }
            }
        }
    }

    return _safeToReuse;
}

/*
 * Best-fit allocation will try to put each tensor in a space as small as possible, as long as it fits.
 * It does so by iterating over all tensors in the order specified by tensorOrder, and executing the
 * following steps:
 *     1. Check all the alternatives given by function createAllAlternatives. If there is an allocated
 *        tensor among the alternatives, goto step 2, otherwise step 3B if no alternatives are left.
 *     2. Make sure all tensors allocated in the same space are in safeToReuse. If yes, goto step 3A.
 *        Otherwise check the next alternative tensor.
 *     3A Allocate the tensor in the alternative tensor's space.
 *     3B Allocate the tensor at the end of the memory and update memory size.
 */
void BestFitMemoryPlanner::bestFitAllocation() {
    // tensorOccupation is used to track which tensor has already been occupied on the same tensor space

    std::map<std::shared_ptr<TensorDescriptor>, std::shared_ptr<Tensors>> tensorOccupation;
    const auto [alignment, memoryTypeBits] = memoryRequirements;

    memorySize = 0;
    for (const auto &tensor : tensors) {
        const auto alternativeTensor = findAlternativeTensor(tensor, tensorOccupation);

        // If no alternative was found, use a new occupation list
        // Otherwise, use the occupation list from the alternative and retrieve the memory address of the alternative
        if (alternativeTensor != nullptr) {
            tensorOccupation[tensor] = tensorOccupation[alternativeTensor];
            allocate(tensor, tensorOffsets[alternativeTensor]);
        } else {
            tensorOccupation[tensor] = std::make_shared<Tensors>();
            allocate(tensor, memorySize);
            memorySize = roundUp(memorySize, alignment);
        }

        tensorOccupation[tensor]->emplace_back(tensor);
    }
}

Tensors BestFitMemoryPlanner::createInitialTensorOrder() const {
    Tensors tensorSet = graphPipeline->getTensors();

    // Sort tensors by size so that the biggest tensor comes first
    // During testing, this has shown to be a good starting point
    stable_sort(tensorSet.begin(), tensorSet.end(), [](const auto &a, const auto &b) {
        return a->getMemoryRequirementsSize() > b->getMemoryRequirementsSize();
    });

    return tensorSet;
}

/*
 * An alternative for a tensor is another tensor that is safe to reuse, and is large enough to hold it.
 * This function creates a map where each tensor maps to a vector with all the alternatives. The vector
 * of alternatives is sorted so that the closest tensor in size comes first, in order to find the best
 * fitting match.
 */
std::map<std::shared_ptr<TensorDescriptor>, Tensors> BestFitMemoryPlanner::createAllAlternatives() const {
    auto _allAlternatives = std::map<std::shared_ptr<TensorDescriptor>, Tensors>();

    for (const auto &tensor : tensors) {
        auto &alternatives = _allAlternatives[tensor];
        const auto tensorSize = tensor->getMemoryRequirementsSize();

        for (auto const &safeTensor : safeToReuse.at(tensor)) {
            const auto safeTensorSize = safeTensor->getMemoryRequirementsSize();

            if (safeTensorSize >= tensorSize) {
                alternatives.emplace_back(safeTensor);
            }
        }

        stable_sort(alternatives.begin(), alternatives.end(), [](const auto &a, const auto &b) {
            return a->getMemoryRequirementsSize() < b->getMemoryRequirementsSize();
        });
    }

    return _allAlternatives;
}

/*
 * Our models are represented as directed acyclic graphs (DAG). This means no cycles are present.
 * Topological order means that the graph nodes can be sorted such that each parent always comes
 * before its descendants. This function produces such a vector.
 */
std::vector<ComputePipelineBase *> BestFitMemoryPlanner::getTopologicalOrder() const {
    auto input = &graphPipeline->getInputs();

    std::set<ComputePipelineBase *> processed;
    std::deque<ComputePipelineBase *> workingList;
    workingList.push_back(input);

    std::vector<ComputePipelineBase *> topologicalOrder;

    while (workingList.size() != 0) {
        ComputePipelineBase *current = workingList.front();
        workingList.pop_front();

        auto &descendants = current->getDescendants();

        for (const auto &descendant : descendants) {
            const auto pipeline = descendant->getDescendantPipeline();

            if (processed.count(pipeline) == 0) {
                workingList.push_back(pipeline);
                processed.insert(pipeline);
            }
        }

        topologicalOrder.push_back(current);
    }

    return topologicalOrder;
}

/*
 * Allocates a tensor at the given memory address. If the tensor is allocated at the end it increases the memory size.
 */
void BestFitMemoryPlanner::allocate(const std::shared_ptr<TensorDescriptor> &tensor, VkDeviceSize memoryAddress) {
    tensorOffsets[tensor] = memoryAddress;
    VkDeviceSize tensorSize = tensor->getMemoryRequirementsSize();

    if (memoryAddress + tensorSize > memorySize) {
        memorySize = memoryAddress + tensorSize;
    }
}

std::shared_ptr<TensorDescriptor> BestFitMemoryPlanner::findAlternativeTensor(
    const std::shared_ptr<TensorDescriptor> &tensor,
    const std::map<std::shared_ptr<TensorDescriptor>, std::shared_ptr<Tensors>> &tensorOccupation) {
    for (const auto &alternativeTensor : allAlternatives.at(tensor)) {
        if (isAllocated(alternativeTensor) && isSafeToReuse(tensorOccupation.at(alternativeTensor), tensor)) {
            return alternativeTensor;
        }
    }

    return nullptr;
}

bool BestFitMemoryPlanner::isAllocated(const std::shared_ptr<TensorDescriptor> &tensor) const {
    return tensorOffsets.find(tensor) != tensorOffsets.end();
}

/*
 * Loop through the occupation list and see if there is any tensor that is not safe for the tensor.
 * Returns true if all tensors are safe to reuse for tensor, false otherwise.
 */
bool BestFitMemoryPlanner::isSafeToReuse(const std::shared_ptr<Tensors> &occupationList,
                                         const std::shared_ptr<TensorDescriptor> &tensor) const {
    for (const auto &allocatedTensor : *occupationList) {
        const auto &safeTensors = safeToReuse.at(tensor);
        if (std::find(safeTensors.begin(), safeTensors.end(), allocatedTensor) == safeTensors.end()) {
            return false;
        }
    }

    return true;
}

} // namespace mlsdk::el::compute
