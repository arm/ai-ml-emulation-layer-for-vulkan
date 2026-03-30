/*
 * SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include <vulkan/vulkan.hpp>

#include <memory>
#include <optional>
#include <vector>

namespace mlsdk::el::log {
class Log;
}

namespace mlsdk::el::compute {

/*******************************************************************************
 * Tensor
 *******************************************************************************/

class TensorDescriptor;

class Tensor {
  public:
    Tensor(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
           std::shared_ptr<TensorDescriptor> _tensorDescriptor, VkTensorARM _tensorARM, VkTensorViewARM _tensorViewARM);

    ~Tensor();

    std::shared_ptr<TensorDescriptor> getTensorDescriptor() const;
    VkTensorARM getVkTensorARM() const;
    VkTensorViewARM getVkTensorViewARM() const;

    VkDeviceSize bindTensorMemory(VkDeviceMemory deviceMemory, VkDeviceSize offset) const;

  private:
    std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> loader;
    VkDevice device;
    std::shared_ptr<TensorDescriptor> tensorDescriptor;
    VkTensorARM tensorARM;
    VkTensorViewARM tensorViewARM;
};

/*******************************************************************************
 * TensorDescriptor
 *******************************************************************************/

class ComputePipelineBase;

class TensorDescriptor {
  public:
    TensorDescriptor(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader,
                     VkPhysicalDevice _physicalDevice, VkDevice _device, const VkFormat _format,
                     const std::vector<int64_t> &_dimensions = {}, const std::vector<int64_t> &_strides = {});

    TensorDescriptor(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader,
                     VkPhysicalDevice _physicalDevice, VkDevice _device,
                     const VkTensorDescriptionARM &_tensorDescription);

    // Function is static because the created tensor takes ownership of the supplied tensor description.
    static std::shared_ptr<Tensor> makeTensor(const std::shared_ptr<TensorDescriptor> &_this);
    VkDeviceMemory createInitializeDeviceMemory(const void *data);

    VkFormat getFormat() const;
    const std::vector<int64_t> &getDimensions() const;
    uint32_t getRank() const;
    size_t getShapeSize() const;
    size_t getSize() const;

    uint64_t getReferenceCounter() const;
    void incrementReferenceCounter();
    ComputePipelineBase *getPipeline() const;
    void setPipeline(ComputePipelineBase *pipeline);

    VkMemoryRequirements getMemoryRequirements();
    VkDeviceSize getMemoryRequirementsSize();
    VkTensorDescriptionARM getTensorDescription() const;

  private:
    std::vector<int64_t> createStrides(const VkTensorDescriptionARM &tensorDescription) const;
    VkTensorARM createTensorARM(const VkTensorDescriptionARM &tensorDescription) const;
    VkTensorViewARM createTensorViewARM(const VkTensorARM tensor, const VkFormat _format) const;
    std::vector<VkQueueFamilyProperties> enumerateQueueFamilyProperties() const;
    uint32_t getComputeFamilyIndex() const;

    VkDeviceMemory allocateDeviceMemory(const size_t size, const uint32_t memoryTypeBits) const;
    std::vector<uint32_t> getMemoryTypeIndices(const VkMemoryPropertyFlags memoryPropertyFlags,
                                               const uint32_t memoryTypeBits) const;

    std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> loader;
    VkPhysicalDevice physicalDevice;
    VkDevice device;

    VkFormat format;
    std::vector<int64_t> dimensions;
    std::vector<int64_t> strides;

    std::optional<VkMemoryRequirements> vkMemoryRequirements;

    uint64_t referenceCounter{};
    ComputePipelineBase *pipeline{nullptr};
};

mlsdk::el::log::Log &operator<<(mlsdk::el::log::Log &os, const Tensor &tensor);
mlsdk::el::log::Log &operator<<(mlsdk::el::log::Log &os, const TensorDescriptor &tensor);

/*******************************************************************************
 * VirtualTensor
 *******************************************************************************/

class VirtualTensor {
  public:
    VirtualTensor(const std::shared_ptr<TensorDescriptor> &_tensor, ComputePipelineBase *_parent,
                  ComputePipelineBase *_descendant);

    bool getVisited() const;
    void setVisited(const bool _visited);

    std::shared_ptr<TensorDescriptor> getTensor() const;
    ComputePipelineBase *getParentPipeline() const;
    ComputePipelineBase *getDescendantPipeline() const;

  private:
    std::shared_ptr<TensorDescriptor> tensor;
    ComputePipelineBase *parent;
    ComputePipelineBase *descendant;
    bool visited{false};
};

} // namespace mlsdk::el::compute
