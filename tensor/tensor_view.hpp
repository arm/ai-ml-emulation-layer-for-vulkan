/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

#include "mlel/vulkan_layer.hpp"
#include "tensor_arm.hpp"
#include "tensor_descriptor.hpp"
#include <vulkan/vulkan.hpp>

namespace mlsdk::el::layer {
class TensorViewARM {
  public:
    TensorViewARM() = default;
    TensorViewARM(const TensorViewARM &) = delete;
    TensorViewARM &operator=(const TensorViewARM &) = delete;

    VkResult create(const Device &dev, const VkTensorViewCreateInfoARM *createInfo,
                    const VkAllocationCallbacks *allocator);
    void destroy(const Device &dev, const VkAllocationCallbacks *pAllocator);
    VkBuffer getDescriptorBuffer(const Device &dev) const;
    VkBuffer getTensorBuffer() const;
    VkResult getOpaqueCaptureDescriptorDataEXT(const Device &dev, void *pData);

  private:
    TensorDescriptor *m_descriptor;
    VkTensorARM m_tensor;
};
} // namespace mlsdk::el::layer
