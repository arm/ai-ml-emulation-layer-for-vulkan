/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include <gtest/gtest.h>

#include "descriptor_binding.hpp"
#include "tensor_descriptor_layout.hpp"

#include <vector>

namespace {

TEST(TensorDescriptors, NarrowOffsetsPreserveDescriptorLayout) {
    using mlsdk::el::layer::TensorDescriptorLayout;
    EXPECT_EQ(offsetof(TensorDescriptorLayout, address), 0u);
    EXPECT_EQ(offsetof(TensorDescriptorLayout, narrowOffsets), 8u);
    EXPECT_EQ(offsetof(TensorDescriptorLayout, dimensions), 16u);
    EXPECT_EQ(offsetof(TensorDescriptorLayout, strides), 112u);
    EXPECT_EQ(sizeof(TensorDescriptorLayout), 208u);
}

TEST(TensorDescriptors, NarrowOffsetsRequireBoundedProductsAndSums) {
    using mlsdk::el::layer::supports32BitTensorOffsets;
    constexpr int64_t limit = int64_t(std::numeric_limits<uint32_t>::max()) - 3;
    EXPECT_TRUE(supports32BitTensorOffsets({2, 3}, {32, 4}));
    // Contiguous FP32 storage just above 2 GiB remains eligible.
    EXPECT_TRUE(supports32BitTensorOffsets({1, 536870913}, {2147483652, 4}));
    EXPECT_TRUE(supports32BitTensorOffsets({6, 7}, {0, 0}));
    EXPECT_TRUE(supports32BitTensorOffsets({2, 2}, {limit - 1, 1}));
    EXPECT_FALSE(supports32BitTensorOffsets({2, 2}, {limit, 1}));
    EXPECT_TRUE(supports32BitTensorOffsets({3}, {limit / 2}));
    EXPECT_FALSE(supports32BitTensorOffsets({3}, {limit / 2 + 1}));
    // Four byte-sized elements at the upper boundary cannot wrap the uint offset or coordinate.
    EXPECT_TRUE(supports32BitTensorOffsets({limit}, {1}));
    EXPECT_EQ(uint32_t(limit - 1) + 3u, std::numeric_limits<uint32_t>::max() - 1u);
    EXPECT_TRUE(supports32BitTensorOffsets({1}, {limit}));
    EXPECT_FALSE(supports32BitTensorOffsets({1}, {limit + 1}));
    EXPECT_FALSE(supports32BitTensorOffsets({limit + 1}, {0}));
    EXPECT_FALSE(supports32BitTensorOffsets({limit}, {std::numeric_limits<int64_t>::max()}));
    EXPECT_FALSE(supports32BitTensorOffsets({2}, {-4}));
    EXPECT_FALSE(supports32BitTensorOffsets({0}, {4}));
    EXPECT_FALSE(supports32BitTensorOffsets({-1}, {4}));
    EXPECT_FALSE(supports32BitTensorOffsets({}, {}));
    EXPECT_FALSE(supports32BitTensorOffsets({2}, {4, 4}));
    EXPECT_TRUE(supports32BitTensorOffsets(std::vector<int64_t>(6, 1), std::vector<int64_t>(6, 4)));
    EXPECT_FALSE(supports32BitTensorOffsets(std::vector<int64_t>(7, 1), std::vector<int64_t>(7, 4)));
}

TEST(TensorDescriptors, NonTensorBindingFlagsAreNotDuplicated) {
    const std::vector<VkDescriptorSetLayoutBinding> bindings{
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    const std::vector<VkDescriptorBindingFlags> flags{
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
        0,
    };
    const VkDescriptorSetLayoutBindingFlagsCreateInfo bindingInfo{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        nullptr,
        static_cast<uint32_t>(flags.size()),
        flags.data(),
    };

    const auto substitutedBindings = mlsdk::el::layer::descriptor_binding::substituteTensorBinding(
        static_cast<uint32_t>(bindings.size()), bindings.data());
    const auto substitutedFlags = mlsdk::el::layer::descriptor_binding::substituteTensorBindingFlags(
        static_cast<uint32_t>(bindings.size()), bindings.data(), bindingInfo, true);

    EXPECT_EQ(substitutedBindings.size(), bindings.size());
    EXPECT_EQ(substitutedFlags, flags);
}

TEST(TensorDescriptors, TensorBindingFlagsFollowAppendedBufferAlias) { // cppcheck-suppress syntaxError
    const std::vector<VkDescriptorSetLayoutBinding> bindings{
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_TENSOR_ARM, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    const std::vector<VkDescriptorBindingFlags> flags{
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
    };
    const VkDescriptorSetLayoutBindingFlagsCreateInfo bindingInfo{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        nullptr,
        static_cast<uint32_t>(flags.size()),
        flags.data(),
    };

    const auto substitutedBindings = mlsdk::el::layer::descriptor_binding::substituteTensorBinding(
        static_cast<uint32_t>(bindings.size()), bindings.data());
    const auto substitutedFlags = mlsdk::el::layer::descriptor_binding::substituteTensorBindingFlags(
        static_cast<uint32_t>(bindings.size()), bindings.data(), bindingInfo, false);
    const auto expectedTensorFlags = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;

#ifdef EXPERIMENTAL_MOLTEN_VK_SUPPORT
    ASSERT_EQ(substitutedBindings.size(), 3u);
    ASSERT_EQ(substitutedFlags.size(), 3u);
    EXPECT_EQ(substitutedBindings.back().descriptorType, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    EXPECT_EQ(substitutedFlags.back(), expectedTensorFlags);
#else
    ASSERT_EQ(substitutedBindings.size(), 2u);
    ASSERT_EQ(substitutedFlags.size(), 2u);
#endif
    EXPECT_EQ(substitutedFlags[0], flags[0]);
    EXPECT_EQ(substitutedFlags[1], expectedTensorFlags);
    EXPECT_EQ(flags[1], VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT);
}

} // namespace
