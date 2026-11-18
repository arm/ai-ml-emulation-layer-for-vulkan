/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include <gtest/gtest.h>

#include "copy_backed_aliasing.hpp"
#include "tensor_arm.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <type_traits>
#include <vector>

namespace {

#ifdef VK_ARM_tensors2
TEST(TensorARM, BlockScaledLayoutsDeriveOmittedStrides) {
    const int64_t dimensions[][3] = {{2, 3, 64}, {2, 3, 2}};
    const int64_t strides[][3] = {{1024, 256, 1}, {32, 8, 1}};
    const VkTensorPlaneARM types[] = {VK_TENSOR_PLANE_VALUES_ARM, VK_TENSOR_PLANE_SCALES_ARM};
    VkTensorDescriptionARM planes[] = {
        {VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM, nullptr, VK_TENSOR_TILING_LINEAR_ARM,
         VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E4M3_ARM, 3, dimensions[0], nullptr,
         VK_TENSOR_USAGE_TRANSFER_SRC_BIT_ARM},
        {VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM, nullptr, VK_TENSOR_TILING_LINEAR_ARM,
         VK_FORMAT_R8_UFLOAT_FPENCODING_FLOAT8UNSIGNEDE8M0_ARM, 3, dimensions[1], nullptr,
         VK_TENSOR_USAGE_TRANSFER_SRC_BIT_ARM},
    };
    const VkBlockScaledTensorDescriptionARM scaled{VK_STRUCTURE_TYPE_BLOCK_SCALED_TENSOR_DESCRIPTION_ARM,
                                                   nullptr,
                                                   VK_TENSOR_BLOCK_SCALING_SCHEME_PLANAR_OCP_MICROSCALING_ARM,
                                                   2,
                                                   types,
                                                   planes};
    const VkTensorCreateInfoARM create{
        VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_ARM, &scaled, 0, nullptr, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
    for (uint32_t supplied = 0; supplied < 2; ++supplied) {
        SCOPED_TRACE(supplied);
        planes[0].pStrides = supplied == 0 ? strides[0] : nullptr;
        planes[1].pStrides = supplied == 1 ? strides[1] : nullptr;
        const mlsdk::el::layer::TensorStorage storage(create);
        ASSERT_EQ(storage.planes.size(), 2u);
        EXPECT_EQ(storage.planes[0].layout.strides, (std::vector<int64_t>{1024, 256, 1}));
        EXPECT_EQ(storage.planes[1].layout.strides, (std::vector<int64_t>{32, 8, 1}));
        EXPECT_EQ(storage.planes[0].layout.size, 2048u);
        EXPECT_EQ(storage.planes[1].layout.size, 64u);
    }
}
#endif

TEST(TensorARM, ImageAliasingUsesImagePitchesAndAllowsLeadingSingletonDimensions) {
    struct Case {
        std::vector<int64_t> dimensions;
        VkSubresourceLayout imageLayout;
        std::vector<int64_t> strides;
        size_t size;
    };
    const Case cases[]{
        // 1D image: no row or depth pitch is needed, even with extra tensor axes.
        {{80, 2}, {0, 320, 0, 0, 0}, {4, 2}, 320},
        {{1, 1, 1, 80, 2}, {0, 320, 0, 0, 0}, {320, 320, 320, 4, 2}, 320},
        // 2D image: depthPitch is undefined; all extra leading axes are singleton.
        {{1, 48, 80, 2}, {0, 24576, 512, 0, 0}, {24576, 512, 4, 2}, 24384},
        {{1, 1, 1, 48, 80, 2}, {0, 24576, 512, 0, 0}, {24576, 24576, 24576, 512, 4, 2}, 24384},
        // 3D image, including a singleton height that must not be collapsed.
        {{1, 1, 3, 48, 80, 2}, {0, 98304, 512, 0, 32768}, {98304, 98304, 32768, 512, 4, 2}, 89920},
        {{1, 1, 3, 1, 80, 2}, {0, 3072, 512, 0, 1024}, {3072, 3072, 1024, 320, 4, 2}, 2368},
    };
    for (const auto &test : cases) {
        SCOPED_TRACE(testing::PrintToString(test.dimensions));
        const vk::TensorDescriptionARM description{vk::TensorTilingARM::eOptimal,
                                                   vk::Format::eR16Uint,
                                                   static_cast<uint32_t>(test.dimensions.size()),
                                                   test.dimensions.data(),
                                                   nullptr,
                                                   vk::TensorUsageFlagBitsARM::eImageAliasing};
        mlsdk::el::layer::TensorLayout layout(description, 0);
        layout.updateAliasedImageLayout(test.imageLayout);
        EXPECT_EQ(layout.strides, test.strides);
        EXPECT_EQ(layout.size, test.size);
    }
}

TEST(TensorARM, LinearTensorKeepsApplicationStrides) {
    const int64_t dimensions[]{4, 8, 1};
    for (const auto &strides : {std::vector<int64_t>{}, std::vector<int64_t>{32, 1, 1}}) {
        const vk::TensorDescriptionARM description{vk::TensorTilingARM::eLinear,
                                                   vk::Format::eR8Unorm,
                                                   3,
                                                   dimensions,
                                                   strides.empty() ? nullptr : strides.data(),
                                                   vk::TensorUsageFlagBitsARM::eShader};
        mlsdk::el::layer::TensorLayout layout(description, 0);
        const auto original = layout;
        layout.updateAliasedImageLayout({0, 64, 16, 0, 0});
        EXPECT_EQ(layout.strides, original.strides);
        EXPECT_EQ(layout.size, original.size);
    }
}

// Exercise buffer growth with a fixed image pitch, independently of a driver's bounds handling.
class TensorImageBuffer : public testing::Test {
  protected:
    template <typename Handle> static Handle makeHandle(uintptr_t value) {
        if constexpr (std::is_pointer_v<Handle>) {
            return reinterpret_cast<Handle>(value);
        } else {
            return static_cast<Handle>(value);
        }
    }
    struct Driver {
        std::map<VkBuffer, VkDeviceSize> buffers;
        uintptr_t nextBuffer = 0;
        VkDeviceSize rowPitch = 512;
        VkResult createResult = VK_SUCCESS;
        VkResult bindResult = VK_SUCCESS;
        VkBuffer lastBoundBuffer = VK_NULL_HANDLE;

        static Driver &get(VkDevice dev) { return *reinterpret_cast<Driver *>(dev); }
        static VKAPI_ATTR VkResult VKAPI_CALL create(VkDevice dev, const VkBufferCreateInfo *info,
                                                     const VkAllocationCallbacks *, VkBuffer *buffer) {
            auto &driver = get(dev);
            if (driver.createResult != VK_SUCCESS) {
                return driver.createResult;
            }
            *buffer = makeHandle<VkBuffer>(++driver.nextBuffer);
            driver.buffers.emplace(*buffer, info->size);
            return VK_SUCCESS;
        }
        static VKAPI_ATTR void VKAPI_CALL destroy(VkDevice dev, VkBuffer buffer, const VkAllocationCallbacks *) {
            get(dev).buffers.erase(buffer);
        }
        static VKAPI_ATTR void VKAPI_CALL requirements(VkDevice dev, VkBuffer buffer, VkMemoryRequirements *req) {
            *req = {get(dev).buffers.at(buffer), 256, 1};
        }
        static VKAPI_ATTR VkResult VKAPI_CALL bind(VkDevice dev, VkBuffer buffer, VkDeviceMemory memory,
                                                   VkDeviceSize offset) {
            auto &driver = get(dev);
            driver.lastBoundBuffer = buffer;
            EXPECT_EQ(memory, makeHandle<VkDeviceMemory>(1));
            EXPECT_EQ(offset, 256u);
            return driver.bindResult;
        }
        static VKAPI_ATTR void VKAPI_CALL imageLayout(VkDevice dev, VkImage, const VkImageSubresource *,
                                                      VkSubresourceLayout *layout) {
            *layout = {0, 48 * get(dev).rowPitch, get(dev).rowPitch, 0, 0};
        }
    } driver;
    static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL instanceProc(VkInstance, const char *) { return nullptr; }
    static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL deviceProc(VkDevice, const char *) { return nullptr; }
    std::unique_ptr<mlsdk::el::layer::Device> device;
    mlsdk::el::layer::TensorARM tensor;

    void SetUp() override {
        using namespace mlsdk::el::layer;
        auto instance = std::make_shared<Instance>(VK_NULL_HANDLE, instanceProc, nullptr, nullptr, nullptr);
        auto physical = std::make_shared<PhysicalDevice>(instance, VK_NULL_HANDLE);
        device =
            std::make_unique<Device>(physical, reinterpret_cast<VkDevice>(&driver), instanceProc, deviceProc, nullptr);
        device->loader->vkCreateBuffer = Driver::create;
        device->loader->vkDestroyBuffer = Driver::destroy;
        device->loader->vkGetBufferMemoryRequirements = Driver::requirements;
        device->loader->vkBindBufferMemory = Driver::bind;
        device->loader->vkGetImageSubresourceLayout = Driver::imageLayout;
        const int64_t dimensions[]{1, 48, 80, 2};
        const vk::TensorDescriptionARM description{vk::TensorTilingARM::eOptimal,
                                                   vk::Format::eR16Uint,
                                                   4,
                                                   dimensions,
                                                   nullptr,
                                                   vk::TensorUsageFlagBitsARM::eImageAliasing |
                                                       vk::TensorUsageFlagBitsARM::eShader};
        const vk::TensorCreateInfoARM info{{}, &description, vk::SharingMode::eExclusive};
        ASSERT_EQ(tensor.create(*device, info, nullptr), VK_SUCCESS);
        ASSERT_EQ(tensor.bindTensorMemory(*device, makeHandle<VkDeviceMemory>(1), 256, std::nullopt), VK_SUCCESS);
    }
    void TearDown() override {
        tensor.destroy(*device, nullptr);
        EXPECT_TRUE(driver.buffers.empty());
    }
    VkResult discoverAlias(VkDeviceSize available = 24576) {
        return tensor.updateAliasedTensorInfo(*device, VK_NULL_HANDLE, makeHandle<VkDeviceMemory>(1), 256, available);
    }
};

TEST_F(TensorImageBuffer, GrowsBufferOverSameMemoryAndKeepsRequirements) {
    VkMemoryRequirements before{};
    ASSERT_TRUE(tensor.getMemoryRequirements(*device, &before, std::nullopt));
    const auto originalBuffer = tensor.getPlanes().front().buffer;
    ASSERT_EQ(discoverAlias(), VK_SUCCESS);
    const auto &plane = tensor.getPlanes().front();
    EXPECT_NE(plane.buffer, originalBuffer);
    EXPECT_EQ(before.size, 15360u); // Public requirements remain those of the packed tensor.
    EXPECT_GE(driver.buffers.at(plane.buffer), 24384u);
    EXPECT_EQ(driver.lastBoundBuffer, plane.buffer);
    EXPECT_EQ(driver.buffers.count(originalBuffer), 0u);
    EXPECT_EQ(driver.buffers.size(), 1u);
    VkMemoryRequirements after{};
    ASSERT_TRUE(tensor.getMemoryRequirements(*device, &after, std::nullopt));
    EXPECT_EQ(vk::MemoryRequirements(after), vk::MemoryRequirements(before));
    const auto resizedBuffer = plane.buffer;
    ASSERT_EQ(discoverAlias(), VK_SUCCESS);
    EXPECT_EQ(tensor.getPlanes().front().buffer, resizedBuffer);
    EXPECT_EQ(driver.buffers.size(), 1u);
}

TEST_F(TensorImageBuffer, FailedGrowthLeavesOriginalBufferAndLayoutIntact) {
    const auto original = tensor.getPlanes().front();
    driver.createResult = VK_ERROR_OUT_OF_HOST_MEMORY;
    EXPECT_EQ(discoverAlias(), VK_ERROR_OUT_OF_HOST_MEMORY);
    driver.createResult = VK_SUCCESS;
    EXPECT_EQ(discoverAlias(15360), VK_ERROR_OUT_OF_DEVICE_MEMORY);
    driver.bindResult = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    EXPECT_EQ(discoverAlias(), VK_ERROR_OUT_OF_DEVICE_MEMORY);
    EXPECT_EQ(tensor.getPlanes().front().buffer, original.buffer);
    EXPECT_EQ(tensor.getPlanes().front().layout.strides, original.layout.strides);
    EXPECT_EQ(tensor.getPlanes().front().layout.size, original.layout.size);
    EXPECT_EQ(driver.buffers.size(), 1u);
}

TEST_F(TensorImageBuffer, PackedImageLayoutReusesExistingBuffer) {
    const auto original = tensor.getPlanes().front();
    driver.rowPitch = 320;
    ASSERT_EQ(discoverAlias(), VK_SUCCESS);
    EXPECT_EQ(tensor.getPlanes().front().buffer, original.buffer);
    EXPECT_EQ(tensor.getPlanes().front().layout.strides, original.layout.strides);
    EXPECT_EQ(driver.buffers.size(), 1u);
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
