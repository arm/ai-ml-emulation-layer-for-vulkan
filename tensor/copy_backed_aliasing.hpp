/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

#include "tensor_descriptor_layout.hpp"

#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_format_traits.hpp>

#include <limits>

namespace mlsdk::el::layer::copy_backed_aliasing {

// activate only after tensor actions insert the refresh/commit copies, gating off the feature until fully implemented
// in later patches
inline constexpr bool enabled = false;

struct FormatMapping {
    uint32_t components = 0;
    uint32_t componentBytes = 0;
};

// return component metadata only for formats supported by copy-backed aliasing
inline FormatMapping getFormatMapping(VkFormat format) {
    // initial set of formats for which the copy path maps each image texel directly to
    // adjacent scalar tensor elements, with no repacking or conversion
    static constexpr VkFormat supportedFamilies[][4] = {
        {VK_FORMAT_R8_UNORM, VK_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8B8_UNORM, VK_FORMAT_R8G8B8A8_UNORM},
        {VK_FORMAT_R8_SNORM, VK_FORMAT_R8G8_SNORM, VK_FORMAT_R8G8B8_SNORM, VK_FORMAT_R8G8B8A8_SNORM},
        {VK_FORMAT_R8_UINT, VK_FORMAT_R8G8_UINT, VK_FORMAT_R8G8B8_UINT, VK_FORMAT_R8G8B8A8_UINT},
        {VK_FORMAT_R8_SINT, VK_FORMAT_R8G8_SINT, VK_FORMAT_R8G8B8_SINT, VK_FORMAT_R8G8B8A8_SINT},
        {VK_FORMAT_R16_UNORM, VK_FORMAT_R16G16_UNORM, VK_FORMAT_R16G16B16_UNORM, VK_FORMAT_R16G16B16A16_UNORM},
        {VK_FORMAT_R16_SNORM, VK_FORMAT_R16G16_SNORM, VK_FORMAT_R16G16B16_SNORM, VK_FORMAT_R16G16B16A16_SNORM},
        {VK_FORMAT_R16_UINT, VK_FORMAT_R16G16_UINT, VK_FORMAT_R16G16B16_UINT, VK_FORMAT_R16G16B16A16_UINT},
        {VK_FORMAT_R16_SINT, VK_FORMAT_R16G16_SINT, VK_FORMAT_R16G16B16_SINT, VK_FORMAT_R16G16B16A16_SINT},
        {VK_FORMAT_R16_SFLOAT, VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R16G16B16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT},
        {VK_FORMAT_R32_UINT, VK_FORMAT_R32G32_UINT, VK_FORMAT_R32G32B32_UINT, VK_FORMAT_R32G32B32A32_UINT},
        {VK_FORMAT_R32_SINT, VK_FORMAT_R32G32_SINT, VK_FORMAT_R32G32B32_SINT, VK_FORMAT_R32G32B32A32_SINT},
        {VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R32G32B32_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT},
    };
    for (const auto &family : supportedFamilies) {
        for (const auto supported : family) {
            if (supported == format) {
                const auto vkFormat = vk::Format(format);
                return {vk::componentCount(vkFormat), vk::componentBits(vkFormat, 0) / 8u};
            }
        }
    }
    return {};
}

inline bool isSupportedImage(const VkImageCreateInfo &image) {
    return image.pNext == nullptr && // no extension-specific create behavior
           (image.flags & ~VkImageCreateFlags(VK_IMAGE_CREATE_ALIAS_BIT)) == 0 && // only the alias flag is allowed
           image.imageType == VK_IMAGE_TYPE_2D &&                                 // the copy path handles 2D images
           getFormatMapping(image.format).components != 0 &&                      // supported byte layout
           image.extent.width != 0 &&                                             // non-empty width
           image.extent.height != 0 &&                                            // non-empty height
           image.extent.depth == 1 &&                                             // a single 2D slice
           image.mipLevels == 1 &&                                                // no mip chain to copy
           image.arrayLayers == 1 &&                                              // no array layers to copy
           image.samples == VK_SAMPLE_COUNT_1_BIT &&                              // no multisampling
           image.tiling == VK_IMAGE_TILING_OPTIMAL &&                             // the copy-backed case
           (image.usage & VK_IMAGE_USAGE_TENSOR_ALIASING_BIT_ARM) != 0 &&         // image requests tensor aliasing
           (image.usage & VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT) == 0 && // transfer usage excludes transient images
           image.sharingMode == VK_SHARING_MODE_EXCLUSIVE;                 // no concurrent queue sharing
}

inline bool isSupportedTensor(const VkTensorCreateInfoARM &tensor) {
    const auto &desc = *tensor.pDescription;
    const auto format = getFormatMapping(desc.format);
    if (tensor.pNext != nullptr || desc.pNext != nullptr || tensor.flags != 0 ||
        tensor.sharingMode != VK_SHARING_MODE_EXCLUSIVE || desc.tiling != VK_TENSOR_TILING_OPTIMAL_ARM ||
        desc.pStrides != nullptr || (desc.usage & VK_TENSOR_USAGE_IMAGE_ALIASING_BIT_ARM) == 0 ||
        format.components != 1 || desc.dimensionCount < 3 || desc.dimensionCount > maxTensorRank) {
        return false;
    }
    // TensorInfo stores sizes and strides in signed 64-bit values; reject an unrepresentable packed size first.
    uint64_t size = format.componentBytes;
    for (uint32_t i = 0; i < desc.dimensionCount; ++i) {
        if (desc.pDimensions[i] <= 0 ||
            static_cast<uint64_t>(desc.pDimensions[i]) > static_cast<uint64_t>(INT64_MAX) / size) {
            return false;
        }
        size *= static_cast<uint64_t>(desc.pDimensions[i]);
        if (i + 3 < desc.dimensionCount && desc.pDimensions[i] != 1) {
            return false;
        }
    }
    const auto components = desc.pDimensions[desc.dimensionCount - 1];
    return components <= 4 && size <= std::numeric_limits<size_t>::max();
}

// preserve an image's optimal tiling and add the downstream copy usages
inline VkImageCreateInfo translateImageCreateInfo(const VkImageCreateInfo &image) {
    auto translated = image;
    translated.usage = (image.usage & ~VkImageUsageFlags(VK_IMAGE_USAGE_TENSOR_ALIASING_BIT_ARM)) |
                       VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    return translated;
}

// match the packed component byte width and HWC shape of supported resources
inline bool isCompatibleAlias(const VkImageCreateInfo &image, const VkTensorCreateInfoARM &tensor) {
    if (!isSupportedImage(image) || !isSupportedTensor(tensor)) {
        return false;
    }
    const auto mapping = getFormatMapping(image.format);
    const auto &desc = *tensor.pDescription;
    const auto *shape = desc.pDimensions + desc.dimensionCount - 3;
    return getFormatMapping(desc.format).componentBytes == mapping.componentBytes && shape[0] == image.extent.height &&
           shape[1] == image.extent.width && shape[2] == mapping.components;
}

// derive logical requirements from the buffer requirements; also fill any dedicated-allocation output
inline void getLogicalTensorMemoryRequirements(const VkMemoryRequirements &bufferRequirements,
                                               const VkPhysicalDeviceMemoryProperties &memoryProperties,
                                               VkMemoryRequirements2 &logicalRequirements) {
    logicalRequirements.memoryRequirements = bufferRequirements;
    // the private buffer uses a separate allocation, so rebuild the application's allowed memory-type mask
    // which is represented by memoryTypeBits
    logicalRequirements.memoryRequirements.memoryTypeBits = 0;
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
        const auto &type = memoryProperties.memoryTypes[i];
        if ((type.propertyFlags & (VK_MEMORY_PROPERTY_PROTECTED_BIT | VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT)) == 0 &&
            (memoryProperties.memoryHeaps[type.heapIndex].flags & VK_MEMORY_HEAP_TILE_MEMORY_BIT_QCOM) == 0) {
            logicalRequirements.memoryRequirements.memoryTypeBits |= 1u << i;
        }
    }
    for (auto *next = static_cast<VkBaseOutStructure *>(logicalRequirements.pNext); next; next = next->pNext) {
        if (next->sType == VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS) {
            auto *dedicated = reinterpret_cast<VkMemoryDedicatedRequirements *>(next);
            dedicated->prefersDedicatedAllocation = VK_FALSE;
            dedicated->requiresDedicatedAllocation = VK_FALSE;
        }
    }
}

// check allocation bounds, offset alignment and memory type without overflowing the range calculation
inline bool isValidLogicalBinding(const VkMemoryRequirements &requirements, VkDeviceSize allocationSize,
                                  uint32_t memoryTypeIndex, VkDeviceSize offset) {
    return memoryTypeIndex < VK_MAX_MEMORY_TYPES && (requirements.memoryTypeBits & (1u << memoryTypeIndex)) != 0 &&
           requirements.alignment != 0 && offset % requirements.alignment == 0 && offset < allocationSize &&
           requirements.size <= allocationSize - offset;
}

} // namespace mlsdk::el::layer::copy_backed_aliasing
