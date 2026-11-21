/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tensor_description.hpp"

#include "mlel/utils.hpp"

#include <stdexcept>

namespace mlsdk::el::layer {
void TensorLayout::updateAliasedImageLayout(const VkSubresourceLayout &imageLayout) {
    if (!usesImageAliasing) {
        return;
    }
    const auto rank = dimensions.size();
    if (rank < 2) {
        throw std::runtime_error("A tensor needs at least two dimensions to alias with an image.");
    }
    // The innermost axes are components, width, height and depth. Any extra leading axes
    // are singleton for a compatible image alias; their strides do not affect addressing.
    for (size_t i = rank - 2; i > 0; --i) {
        strides[i - 1] = strides[i] * dimensions[i];
        if (dimensions[i - 1] == 1) {
            continue;
        }
        if (i == rank - 2) {
            strides[i - 1] = static_cast<int64_t>(imageLayout.rowPitch);
        } else if (i == rank - 3) {
            // A non-singleton depth axis requires a 3D image, for which depthPitch is defined.
            strides[i - 1] = static_cast<int64_t>(imageLayout.depthPitch);
        }
    }

    // Include every addressed element, without requiring padding after the last row or slice.
    size = elementSize;
    for (size_t i = 0; i < rank; ++i) {
        size += static_cast<size_t>(dimensions[i] - 1) * static_cast<size_t>(strides[i]);
    }
}

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

TensorLayout::TensorLayout(const VkTensorDescriptionARM &desc, VkTensorCreateFlagsARM tensorFlags)
    : format(desc.format) {
    if (desc.dimensionCount <= 0 || desc.dimensionCount > TensorLayout::MAX_DIMENSIONS) {
        throw std::runtime_error(std::string("Tensor dimension count not supported: ") +
                                 std::to_string(desc.dimensionCount));
    }
    usesImageAliasing =
        desc.tiling == VK_TENSOR_TILING_OPTIMAL_ARM && (desc.usage & VK_TENSOR_USAGE_IMAGE_ALIASING_BIT_ARM) != 0;

    usage = convertToBufferUsageFlags(desc.usage);
    flags = convertToBufferCreateFlags(tensorFlags);

    uint32_t dimensionCount = desc.dimensionCount;
    dimensions = std::vector<int64_t>{desc.pDimensions, desc.pDimensions + dimensionCount};
    const auto block = utils::getFormatStorageBlock(desc.format);
    elementSize = block.byteSize;
    if (dimensions.back() <= 0 || dimensions.back() % block.elementCount != 0) {
        throw std::invalid_argument("Innermost tensor dimension does not contain whole format blocks");
    }
    if (desc.pStrides == nullptr) {
        strides.resize(dimensionCount);
        strides.back() = static_cast<int64_t>(elementSize);
        for (size_t i = dimensionCount - 1; i > 0; --i) {
            strides[i - 1] =
                strides[i] * (i + 1 == dimensionCount ? dimensions[i] / block.elementCount : dimensions[i]);
        }
    } else {
        strides = std::vector<int64_t>{desc.pStrides, desc.pStrides + dimensionCount};
    }
    size = static_cast<size_t>(dimensions[0] / (dimensionCount == 1 ? block.elementCount : 1) * strides[0]);
}

} // namespace mlsdk::el::layer
