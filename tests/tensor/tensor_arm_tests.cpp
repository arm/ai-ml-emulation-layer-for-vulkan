/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include <gtest/gtest.h>

#include "tensor_arm_detail.hpp"

#include <vector>

namespace {

TEST(TensorARM, LinearImageAliasingUsesImageRowPitch) {
    const std::vector<int64_t> dimensions{4, 8, 1};
    std::vector<int64_t> strides{8, 1, 1};
    const VkTensorDescriptionARM description{
        VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM,
        nullptr,
        VK_TENSOR_TILING_LINEAR_ARM,
        VK_FORMAT_R8_UNORM,
        static_cast<uint32_t>(dimensions.size()),
        dimensions.data(),
        strides.data(),
        VK_TENSOR_USAGE_IMAGE_ALIASING_BIT_ARM,
    };
    const VkSubresourceLayout imageLayout{
        0, 64, 16, 64, 64,
    };

    ASSERT_TRUE(mlsdk::el::layer::tensor_arm_detail::usesImageAliasing(description));

    mlsdk::el::layer::tensor_arm_detail::updateAliasedStrides(dimensions.size(), strides, imageLayout);

    EXPECT_EQ(strides[0], static_cast<int64_t>(imageLayout.rowPitch));
}

} // namespace
