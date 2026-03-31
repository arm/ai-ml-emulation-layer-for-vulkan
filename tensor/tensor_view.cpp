/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include "tensor_view.hpp"

namespace mlsdk::el::layer {

VkResult TensorViewARM::create(const Device &dev, const VkTensorViewCreateInfoARM *createInfo,
                               const VkAllocationCallbacks *allocator) {
    m_descriptor = allocateObject<TensorDescriptor>(allocator);
    m_tensor = createInfo->tensor;
    return m_descriptor->create(dev, createInfo, allocator);
}

void TensorViewARM::destroy(const Device &dev, const VkAllocationCallbacks *pAllocator) {
    m_descriptor->destroy(dev, pAllocator);
    destroyObject(pAllocator, m_descriptor);
}

VkBuffer TensorViewARM::getDescriptorBuffer(const Device &dev) const {
    return m_descriptor->getTensorDescriptorBuffer(dev, m_tensor);
}

VkBuffer TensorViewARM::getTensorBuffer() const { return reinterpret_cast<TensorARM *>(m_tensor)->getTensorBuffer(); }

VkResult TensorViewARM::getOpaqueCaptureDescriptorDataEXT(const Device &dev, void *pData) const {
    const VkBufferCaptureDescriptorDataInfoEXT info = {
        VK_STRUCTURE_TYPE_BUFFER_CAPTURE_DESCRIPTOR_DATA_INFO_EXT,
        nullptr,
        getDescriptorBuffer(dev),
    };
    return dev.loader->vkGetBufferOpaqueCaptureDescriptorDataEXT(dev.device, &info, pData);
}
} // namespace mlsdk::el::layer
