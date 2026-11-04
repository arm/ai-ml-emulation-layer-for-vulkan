/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include <gtest/gtest.h>

#include "copy_backed_aliasing.hpp"
#include "tensor_arm.hpp"

#include <array>
#include <limits>
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

    const VkTensorCreateInfoARM createInfo{
        VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_ARM, nullptr, 0, &description, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
    const mlsdk::el::layer::TensorStorage storage(createInfo);
    ASSERT_EQ(storage.planes.size(), 1u);
    ASSERT_TRUE(storage.planes.front().layout.usesImageAliasing);
    EXPECT_EQ(storage.planes.front().layout.dimensions, dimensions);

    mlsdk::el::layer::tensor_description_detail::updateAliasedStrides(dimensions.size(), strides, imageLayout);

    EXPECT_EQ(strides[0], static_cast<int64_t>(imageLayout.rowPitch));
}

VkImageCreateInfo makeCopyBackedImageInfo() {
    VkImageCreateInfo image{};
    image.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image.flags = VK_IMAGE_CREATE_ALIAS_BIT;
    image.imageType = VK_IMAGE_TYPE_2D;
    image.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    image.extent = {7, 4, 1};
    image.mipLevels = 1;
    image.arrayLayers = 1;
    image.samples = VK_SAMPLE_COUNT_1_BIT;
    image.tiling = VK_IMAGE_TILING_OPTIMAL;
    image.usage = VK_IMAGE_USAGE_TENSOR_ALIASING_BIT_ARM | VK_IMAGE_USAGE_SAMPLED_BIT;
    image.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    return image;
}

TEST(TensorARM, CopyBackedImageCreationPreservesOptimalTiling) {
    const auto image = makeCopyBackedImageInfo();
    const auto translated = mlsdk::el::layer::copy_backed_aliasing::translateImageCreateInfo(image);

    EXPECT_EQ(translated.tiling, VK_IMAGE_TILING_OPTIMAL);
    EXPECT_EQ(translated.format, image.format);
    EXPECT_EQ(translated.flags, image.flags);
    EXPECT_EQ(translated.extent.width, image.extent.width);
    EXPECT_EQ(translated.extent.height, image.extent.height);
    EXPECT_EQ(translated.usage,
              VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    EXPECT_EQ(image.usage, VK_IMAGE_USAGE_TENSOR_ALIASING_BIT_ARM | VK_IMAGE_USAGE_SAMPLED_BIT);
}

TEST(TensorARM, CopyBackedAliasRequiresMatchingComponentWidthAndShape) {
    using mlsdk::el::layer::copy_backed_aliasing::isCompatibleAlias;
    auto image = makeCopyBackedImageInfo();
    const auto isCompatibleTensor = [&image](VkFormat format, const auto &dimensions) {
        const VkTensorDescriptionARM description{
            VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM,
            nullptr,
            VK_TENSOR_TILING_OPTIMAL_ARM,
            format,
            static_cast<uint32_t>(dimensions.size()),
            dimensions.data(),
            nullptr,
            VK_TENSOR_USAGE_IMAGE_ALIASING_BIT_ARM | VK_TENSOR_USAGE_SHADER_BIT_ARM,
        };
        const VkTensorCreateInfoARM tensor{
            VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_ARM, nullptr, 0, &description, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr,
        };
        return isCompatibleAlias(image, tensor);
    };

    EXPECT_TRUE(isCompatibleTensor(VK_FORMAT_R16_SFLOAT, std::array<int64_t, 3>{4, 7, 4}));
    EXPECT_TRUE(isCompatibleTensor(VK_FORMAT_R16_UINT, std::array<int64_t, 3>{4, 7, 4}));
    EXPECT_FALSE(isCompatibleTensor(VK_FORMAT_R8_SINT, std::array<int64_t, 3>{4, 7, 4}));
    EXPECT_FALSE(isCompatibleTensor(VK_FORMAT_R16_SFLOAT, std::array<int64_t, 3>{4, 8, 4}));
    EXPECT_FALSE(isCompatibleTensor(VK_FORMAT_R16_SFLOAT, std::array<int64_t, 3>{4, 7, 3}));
    EXPECT_TRUE(isCompatibleTensor(VK_FORMAT_R16_SFLOAT, std::array<int64_t, 4>{1, 4, 7, 4}));
    EXPECT_FALSE(isCompatibleTensor(VK_FORMAT_R16_SFLOAT, std::array<int64_t, 4>{2, 4, 7, 4}));

    image.format = VK_FORMAT_R8G8B8A8_SNORM;
    EXPECT_TRUE(isCompatibleTensor(VK_FORMAT_R8_SINT, std::array<int64_t, 3>{4, 7, 4}));
}

TEST(TensorARM, CopyBackedLogicalRequirementsAreIndependentOfPrivateMemoryType) {
    const VkMemoryRequirements bufferRequirements{256, 64, 1u << 1};
    VkPhysicalDeviceMemoryProperties properties{};
    properties.memoryHeapCount = 2;
    properties.memoryHeaps[0] = {1024, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
    properties.memoryHeaps[1].size = 1024;
    properties.memoryHeaps[1].flags = VK_MEMORY_HEAP_TILE_MEMORY_BIT_QCOM;
    properties.memoryTypeCount = 5;
    properties.memoryTypes[0] = {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0};
    properties.memoryTypes[1] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0};
    properties.memoryTypes[2] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT, 0};
    properties.memoryTypes[3] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_PROTECTED_BIT, 0};
    properties.memoryTypes[4] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 1};
    VkMemoryDedicatedRequirements dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS, nullptr, VK_TRUE, VK_TRUE};
    VkMemoryRequirements2 logical{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, &dedicated, {}};

    mlsdk::el::layer::copy_backed_aliasing::getLogicalTensorMemoryRequirements(bufferRequirements, properties, logical);

    EXPECT_EQ(logical.memoryRequirements.size, bufferRequirements.size);
    EXPECT_EQ(logical.memoryRequirements.alignment, bufferRequirements.alignment);
    EXPECT_EQ(logical.memoryRequirements.memoryTypeBits, (1u << 0) | (1u << 1));
    EXPECT_FALSE(dedicated.requiresDedicatedAllocation);
    EXPECT_FALSE(dedicated.prefersDedicatedAllocation);
    EXPECT_EQ(bufferRequirements.memoryTypeBits, 1u << 1);
}

TEST(TensorARM, CopyBackedLogicalBindingChecksAlignmentTypeAndBounds) {
    using mlsdk::el::layer::copy_backed_aliasing::isValidLogicalBinding;
    const VkMemoryRequirements requirements{112, 16, 1u << 1};

    EXPECT_TRUE(isValidLogicalBinding(requirements, 256, 1, 0));
    EXPECT_TRUE(isValidLogicalBinding(requirements, 256, 1, 144)); // Fits exactly at a nonzero offset.
    EXPECT_FALSE(isValidLogicalBinding(requirements, 256, 1, 1));  // Misaligned, but otherwise fits.
    EXPECT_FALSE(isValidLogicalBinding(requirements, 256, 1, 160));
    EXPECT_FALSE(isValidLogicalBinding(requirements, 256, 0, 0));
    EXPECT_FALSE(isValidLogicalBinding(requirements, 256, 32, 0));
    EXPECT_FALSE(isValidLogicalBinding(requirements, 256, 1, 512));
    const auto maxSize = std::numeric_limits<VkDeviceSize>::max();
    EXPECT_FALSE(isValidLogicalBinding(requirements, maxSize, 1, maxSize - 15));
}

} // namespace
