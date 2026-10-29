/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

#include "mlel/vulkan_layer.hpp"
#include "tensor_descriptor_layout.hpp"
#include <vulkan/vulkan.hpp>

namespace mlsdk::el::layer {

class TensorDescriptor {
  public:
    using DescriptorBuffer = TensorDescriptorLayout;

    TensorDescriptor() = default;
    virtual ~TensorDescriptor() = default;
    TensorDescriptor(const TensorDescriptor &) = delete;
    TensorDescriptor &operator=(const TensorDescriptor &) = delete;

    VkResult create(const Device &dev, const VkTensorViewCreateInfoARM *createInfo,
                    const VkAllocationCallbacks *allocator);
    void destroy(const Device &dev, const VkAllocationCallbacks *pAllocator);
    VkBuffer getTensorDescriptorBuffer(const Device &dev, VkTensorARM tensorHandle);

  private:
    uint32_t findMemoryType(const Device &dev, uint32_t typeFilter, VkMemoryPropertyFlags properties) const;
    VkDeviceMemory m_memory = VK_NULL_HANDLE;
    VkBuffer m_buffer = VK_NULL_HANDLE;
};
} // namespace mlsdk::el::layer
