/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace mlsdk::el::layer::tensor_arm_detail {

inline bool usesImageAliasing(const VkTensorDescriptionARM &description) {
    return (description.usage & VK_TENSOR_USAGE_IMAGE_ALIASING_BIT_ARM) != 0;
}

inline void updateAliasedStrides(const std::size_t rank, std::vector<int64_t> &strides,
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

} // namespace mlsdk::el::layer::tensor_arm_detail
