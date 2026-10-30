/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tensor_description.hpp"

#include <stdexcept>
#include <vulkan/vulkan_format_traits.hpp>

namespace mlsdk::el::layer {
namespace tensor_description_detail {

void updateAliasedStrides(const std::size_t rank, std::vector<int64_t> &strides,
                          const VkSubresourceLayout &imageLayout) {
    if (rank == 4) {
        // alias to 3D image
        strides[0] = static_cast<int64_t>(imageLayout.depthPitch);
        strides[1] = static_cast<int64_t>(imageLayout.rowPitch);
    } else if (rank == 3) {
        // alias to 2D image
        strides[0] = static_cast<int64_t>(imageLayout.rowPitch);
    }
}

} // namespace tensor_description_detail

VkBufferCreateFlags TensorLayout::convertToBufferCreateFlags(VkTensorCreateFlagsARM tensorFlags) {
    VkBufferCreateFlags bufferFlags = {};
    if (tensorFlags & VK_TENSOR_CREATE_PROTECTED_BIT_ARM) {
        bufferFlags |= VK_BUFFER_CREATE_PROTECTED_BIT;
    }
    if (tensorFlags & VK_TENSOR_CREATE_DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT_ARM) {
        bufferFlags |= VK_BUFFER_CREATE_DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT_EXT;
    }
    return bufferFlags;
}

VkBufferUsageFlags TensorLayout::convertToBufferUsageFlags(VkTensorUsageFlagsARM tensorUsage) {
    VkBufferUsageFlags bufferUsage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
#ifdef EXPERIMENTAL_MOLTEN_VK_SUPPORT
    bufferUsage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
#endif
    if (tensorUsage & VK_TENSOR_USAGE_SHADER_BIT_ARM) {
        bufferUsage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    }
    if (tensorUsage & VK_TENSOR_USAGE_TRANSFER_SRC_BIT_ARM) {
        bufferUsage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    }
    if (tensorUsage & VK_TENSOR_USAGE_TRANSFER_DST_BIT_ARM) {
        bufferUsage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    }
    if (tensorUsage & VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM) {
        bufferUsage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    }
    return bufferUsage;
}

TensorLayout::TensorLayout(const VkTensorCreateInfoARM &createInfo) {
    const VkTensorDescriptionARM &desc = *createInfo.pDescription;
    format = desc.format;
    if (desc.dimensionCount <= 0 || desc.dimensionCount > TensorLayout::MAX_DIMENSIONS) {
        throw std::runtime_error(std::string("Tensor dimension count not supported: ") +
                                 std::to_string(desc.dimensionCount));
    }
    usesImageAliasing = (desc.usage & VK_TENSOR_USAGE_IMAGE_ALIASING_BIT_ARM) != 0;

    usage = convertToBufferUsageFlags(desc.usage);
    flags = convertToBufferCreateFlags(createInfo.flags);

    uint32_t dimensionCount = desc.dimensionCount;
    dimensions = std::vector<int64_t>{desc.pDimensions, desc.pDimensions + dimensionCount};
    elementSize = vk::blockSize(vk::Format(desc.format));
    if (desc.pStrides == nullptr) {
        strides.resize(dimensionCount);
        strides.back() = static_cast<int64_t>(elementSize);
        for (size_t i = dimensionCount - 1; i > 0; --i) {
            strides[i - 1] = strides[i] * static_cast<int64_t>(dimensions[i]);
        }
    } else {
        strides = std::vector<int64_t>{desc.pStrides, desc.pStrides + dimensionCount};
    }
    size = static_cast<size_t>(dimensions[0] * strides[0]);
}

} // namespace mlsdk::el::layer
