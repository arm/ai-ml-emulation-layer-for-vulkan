/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include <gtest/gtest.h>

#include "descriptor_binding.hpp"

#include <vector>

namespace {

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
