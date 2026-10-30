/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "tensor_descriptor_layout.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>

namespace mlsdk::el::layer {

struct TensorLayout {
    static constexpr size_t MAX_DIMENSIONS = maxTensorRank;

    TensorLayout() = default;
    explicit TensorLayout(const VkTensorCreateInfoARM &createInfo);

    std::vector<int64_t> dimensions;
    std::vector<int64_t> strides;
    size_t size = 0;
    size_t elementSize = 0;
    VkBufferUsageFlags usage = 0;
    VkBufferCreateFlags flags = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
    bool usesImageAliasing = false;

    static VkBufferCreateFlags convertToBufferCreateFlags(VkTensorCreateFlagsARM flags);
    static VkBufferUsageFlags convertToBufferUsageFlags(VkTensorUsageFlagsARM usage);
};

namespace tensor_description_detail {
void updateAliasedStrides(std::size_t rank, std::vector<int64_t> &strides, const VkSubresourceLayout &imageLayout);
} // namespace tensor_description_detail

} // namespace mlsdk::el::layer
