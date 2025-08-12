/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

#include "mlel/vulkan_layer.hpp"

#include <unordered_map>
#include <vulkan/vulkan.hpp>

namespace mlsdk::el::layer {

class TensorCopyPipeline;

class TensorARM {
  public:
    static constexpr size_t TENSOR_MAX_DIMENSIONS = 6;

    class TensorInfo {
      public:
        TensorInfo() = default;

        explicit TensorInfo(const VkTensorCreateInfoARM &createInfo);
        std::vector<int64_t> dimensions;
        std::vector<int64_t> strides;
        size_t size;
        size_t elementSize;
        VkBufferUsageFlags usage;
        VkBufferCreateFlags flags;
        VkFormat format;
        bool isOptimalTilingAliasing;
    };

    TensorARM() = default;
    virtual ~TensorARM() = default;
    TensorARM(const TensorARM &) = delete;
    TensorARM &operator=(const TensorARM &) = delete;

    VkBuffer getTensorBuffer() const { return m_tensorBuffer; };
    TensorInfo getTensorInfo() const { return m_info; };

    VkResult create(const Device &dev, const VkTensorCreateInfoARM &createInfo, const VkAllocationCallbacks *allocator);
    void destroy(const Device &dev, const VkAllocationCallbacks *pAllocator);
    void getMemoryRequirements(const Device &dev, VkMemoryRequirements *requirements) const;
    static void getDeviceTensorMemoryRequirements(const Device &dev, const VkTensorCreateInfoARM &createInfo,
                                                  VkMemoryRequirements2 *requirements);
    VkResult bindTensorMemory(const Device &dev, VkDeviceMemory memory, VkDeviceSize offset);
    void updateAliasedTensorInfo(const Device &dev, VkImage image);
    void copyToTensor(CommandBuffer &cmd, const TensorARM &dstTensor);
    VkResult getOpaqueCaptureDescriptorDataEXT(const Device &dev, void *pData);

  private:
    VkBuffer m_tensorBuffer = {};
    TensorInfo m_info;
    std::shared_ptr<TensorCopyPipeline> m_copy_pipeline;
};

class TensorCopyPipeline {
  public:
    TensorCopyPipeline(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader,
                       VkDevice _device, const TensorARM &srcTensor, const TensorARM &dstTensor);
    virtual ~TensorCopyPipeline();
    void cmdBindAndDispatchCopy(VkCommandBuffer cmd, uint32_t regionCount);

  private:
    struct PushConstant {
        int64_t dimensions[TensorARM::TENSOR_MAX_DIMENSIONS];
        uint64_t srcStrides[TensorARM::TENSOR_MAX_DIMENSIONS];
        uint64_t dstStrides[TensorARM::TENSOR_MAX_DIMENSIONS];
    };

    VkDescriptorPool createDescriptorPool() const;
    VkDescriptorSetLayout createDescriptorSetLayout() const;
    VkDescriptorSet createDescriptorSet(const TensorARM &srcTensor, const TensorARM &dstTensor) const;
    VkPipelineLayout createPipelineLayout() const;
    VkShaderModule createShaderModule(const TensorARM &srcTensor) const;
    VkPipeline createPipeline(const TensorARM &srcTensor) const;
    PushConstant createPushConstant(const TensorARM &srcTensor, const TensorARM &dstTensor) const;

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

} // namespace mlsdk::el::layer
