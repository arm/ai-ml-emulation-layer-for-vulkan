/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include "tensor_descriptor.hpp"

namespace mlsdk::el::layer {

VkResult TensorDescriptor::create(const Device &dev, const VkTensorViewCreateInfoARM *createInfo,
                                  const VkAllocationCallbacks *allocator) {
    VkResult result;
    // create buffer
    auto pCaptureDescriptorInfo = findType<VkOpaqueCaptureDescriptorDataCreateInfoEXT>(
        createInfo->pNext, VK_STRUCTURE_TYPE_OPAQUE_CAPTURE_DESCRIPTOR_DATA_CREATE_INFO_EXT);
    VkBufferCreateFlags flags = (createInfo->flags & VK_TENSOR_VIEW_CREATE_DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT_ARM)
                                    ? VK_BUFFER_CREATE_DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT_EXT
                                    : 0;
    const VkBufferCreateInfo bufferCreateInfo = {
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        pCaptureDescriptorInfo,
        flags,
        sizeof(DescriptorBuffer),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_SHARING_MODE_EXCLUSIVE,
        0,
        nullptr,
    };
    result = dev.loader->vkCreateBuffer(dev.device, &bufferCreateInfo, allocator, &m_buffer);
    if (result != VK_SUCCESS) {
        return result;
    }
    // allocate memory
    VkMemoryRequirements memoryRequirements;
    dev.loader->vkGetBufferMemoryRequirements(dev.device, m_buffer, &memoryRequirements);
    uint32_t memoryTypeIndex =
        findMemoryType(dev, memoryRequirements.memoryTypeBits,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    const VkMemoryAllocateInfo allocInfo = {
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        nullptr,
        memoryRequirements.size,
        memoryTypeIndex,
    };
    result = dev.loader->vkAllocateMemory(dev.device, &allocInfo, allocator, &m_memory);
    if (result != VK_SUCCESS) {
        dev.loader->vkDestroyBuffer(dev.device, m_buffer, allocator);
        return result;
    }
    result = dev.loader->vkBindBufferMemory(dev.device, m_buffer, m_memory, 0);
    return result;
}

VkBuffer TensorDescriptor::getTensorDescriptorBuffer(const Device &dev, VkTensorARM tensorHandle) {
    auto tensor = reinterpret_cast<TensorARM *>(tensorHandle);
    const auto info = tensor->getTensorInfo();

    DescriptorBuffer *descriptor;
    VkResult result =
        dev.loader->vkMapMemory(dev.device, m_memory, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void **>(&descriptor));
    if (result != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }

    const VkBufferDeviceAddressInfo addressInfo = {
        VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        nullptr,
        tensor->getTensorBuffer(),
    };

    descriptor->address = dev.loader->vkGetBufferDeviceAddress(dev.device, &addressInfo);
    std::copy(info.dimensions.begin(), info.dimensions.end(), descriptor->dimensions);
    std::copy(info.strides.begin(), info.strides.end(), descriptor->strides);

    dev.loader->vkUnmapMemory(dev.device, m_memory);

    return m_buffer;
}

uint32_t TensorDescriptor::findMemoryType(const Device &dev, uint32_t typeFilter,
                                          VkMemoryPropertyFlags properties) const {
    VkPhysicalDeviceMemoryProperties memoryProperties;
    dev.loader->vkGetPhysicalDeviceMemoryProperties(dev.physicalDevice->physicalDevice, &memoryProperties);

    uint32_t memoryTypeIndex = 0;
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
        auto propertyFlags = memoryProperties.memoryTypes[i].propertyFlags;
        if (((1U << i) & typeFilter) && (propertyFlags & properties)) {
            memoryTypeIndex = i;
            break;
        }
    }
    return memoryTypeIndex;
}

void TensorDescriptor::destroy(const Device &dev, const VkAllocationCallbacks *pAllocator) {
    if (m_buffer != VK_NULL_HANDLE) {
        dev.loader->vkDestroyBuffer(dev.device, m_buffer, pAllocator);
        m_buffer = VK_NULL_HANDLE;
    }
    if (m_memory != VK_NULL_HANDLE) {
        dev.loader->vkFreeMemory(dev.device, m_memory, pAllocator);
        m_memory = VK_NULL_HANDLE;
    }
}

} // namespace mlsdk::el::layer
