/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

#include "mlel/vulkan_layer.hpp"
#include "tensor_description.hpp"

#include <unordered_map>
#include <vector>

#include <vulkan/vulkan.hpp>

namespace mlsdk::el::layer {
class TensorCopyPipeline;
class TensorCommandBuffer;

// Backing storage for one physical tensor layout.
struct TensorPlane {
    TensorLayout layout;
    VkBuffer buffer = VK_NULL_HANDLE;
};

// A tensor's backing storage. Ordinary tensors have one entry.
struct TensorStorage {
    TensorStorage() : planes(1) {}
    explicit TensorStorage(const VkTensorCreateInfoARM &createInfo);
    void destroy(const Device &dev, const VkAllocationCallbacks *pAllocator);

    std::vector<TensorPlane> planes;
};

class TensorARM {
  public:
    TensorARM() = default;
    virtual ~TensorARM() = default;
    TensorARM(const TensorARM &) = delete;
    TensorARM &operator=(const TensorARM &) = delete;

    VkBuffer getTensorBuffer() const { return m_storage.planes.front().buffer; };
    const TensorLayout &getTensorInfo() const { return m_info; };
    const std::vector<TensorPlane> &getPlanes() const { return m_storage.planes; }

    VkResult create(const Device &dev, const VkTensorCreateInfoARM &createInfo, const VkAllocationCallbacks *allocator);
    void destroy(const Device &dev, const VkAllocationCallbacks *pAllocator);
    void getMemoryRequirements(const Device &dev, VkMemoryRequirements *requirements) const;
    static void getDeviceTensorMemoryRequirements(const Device &dev, const VkTensorCreateInfoARM &createInfo,
                                                  VkMemoryRequirements2 *requirements);
    VkResult bindTensorMemory(const Device &dev, VkDeviceMemory memory, VkDeviceSize offset) const;
    void updateAliasedTensorInfo(const Device &dev, VkImage image);
    void copyToTensor(TensorCommandBuffer &cmd, const TensorARM &dstTensor);
    VkResult getOpaqueCaptureDescriptorDataEXT(const Device &dev, void *pData) const;

  private:
    TensorLayout m_info;
    TensorStorage m_storage;
};

class TensorCopyPipeline {
  public:
    TensorCopyPipeline(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader,
                       VkDevice _device, const TensorPlane &srcPlane, const TensorPlane &dstPlane);
    TensorCopyPipeline(const TensorCopyPipeline &) = delete;
    TensorCopyPipeline &operator=(const TensorCopyPipeline &) = delete;
    virtual ~TensorCopyPipeline();
    void cmdBindAndDispatchCopy(VkCommandBuffer cmd, uint32_t regionCount);

  private:
    struct PushConstant {
        int64_t dimensions[TensorLayout::MAX_DIMENSIONS];
        uint64_t srcStrides[TensorLayout::MAX_DIMENSIONS];
        uint64_t dstStrides[TensorLayout::MAX_DIMENSIONS];
    };

    VkDescriptorPool createDescriptorPool() const;
    VkDescriptorSetLayout createDescriptorSetLayout() const;
    VkDescriptorSet createDescriptorSet(const TensorPlane &srcPlane, const TensorPlane &dstPlane) const;
    VkPipelineLayout createPipelineLayout() const;
    VkShaderModule createShaderModule(const TensorPlane &srcPlane) const;
    VkPipeline createPipeline(const TensorPlane &srcPlane) const;
    PushConstant createPushConstant(const TensorPlane &srcPlane, const TensorPlane &dstPlane) const;

    std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> loader;
    VkDevice device;
    PushConstant pushConstant;
    VkShaderModule shaderModule;
    VkDescriptorPool descriptorPool;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline pipeline;
    VkDescriptorSet descriptorSet;

    static const uint32_t warp1D = 128;
    static const std::string glsl;
    static inline std::mutex cacheMutex;
    static inline std::unordered_map<std::string, std::vector<uint32_t>> spirvCache;
};

class TensorCommandBuffer : public CommandBuffer {
  public:
    using CommandBuffer::CommandBuffer;

    void resetRecording() override;
    void retainCopyPipeline(std::unique_ptr<TensorCopyPipeline> pipeline);

  private:
    std::vector<std::unique_ptr<TensorCopyPipeline>> m_copyPipelines;
};

} // namespace mlsdk::el::layer
