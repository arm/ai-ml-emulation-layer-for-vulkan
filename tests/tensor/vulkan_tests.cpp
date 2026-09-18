/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include "mlel/pipeline.hpp"
#include "mlel/tensor.hpp"
#include "mlel/utils.hpp"
#include "vulkan_test_utils.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

namespace mlsdk::el::tests {

using namespace utilities;

namespace {

#ifndef EXPERIMENTAL_MOLTEN_VK_SUPPORT
vk::raii::ShaderModule createShaderModule(const vk::raii::Device &device, const std::vector<uint32_t> &code) {
    const vk::ShaderModuleCreateInfo info{
        {},                                                    // flags
        static_cast<uint32_t>(code.size() * sizeof(uint32_t)), // code size
        code.data()                                            // code
    };

    return {device, info};
}

// FIXME: Temporarily disabled in Darwin due to not being able to pass SSBO's to functions
TEST_F(MLEmulationLayerGraphForVulkan, CreateTensorComputeShader) {

    const auto spirvModule = mlsdk::el::utils::glslToSpirv(fileToString("tensor_all_access.comp"));
    [[maybe_unused]] const auto shaderModule = createShaderModule((&(*device)), spirvModule);
}

// FIXME: Temporarily disabled in Darwin due to not being able to pass SSBO's to functions
TEST_F(MLEmulationLayerGraphForVulkan, TensorArray) {

    const auto spirvModule = mlsdk::el::utils::glslToSpirv(fileToString("tensor_array.comp"));
    [[maybe_unused]] const auto shaderModule = createShaderModule((&(*device)), spirvModule);
    std::vector<std::shared_ptr<Tensor>> inputTensors;
    std::vector<std::shared_ptr<Tensor>> outputTensors;
    for ([[maybe_unused]] auto _ : {1, 2, 3, 4}) {
        auto inputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{4}});
        inputTensors.push_back(std::move(inputTensor));
    }
    for ([[maybe_unused]] auto _ : {1, 2, 3, 4}) {
        auto outputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{4}});
        outputTensors.push_back(std::move(outputTensor));
    }

    const TensorComputePipeline::DescriptorMap descriptorMap = {
        {
            // set 0
            {
                0,                                                                    // binding
                {inputTensors[0], inputTensors[1], inputTensors[2], inputTensors[3]}, // tensor
            },
            {
                1,                                                                        // binding
                {outputTensors[0], outputTensors[1], outputTensors[2], outputTensors[3]}, // tensor
            },
        },
    };
    auto computePipeline = std::make_shared<TensorComputePipeline>(device, descriptorMap, spirvModule);

    uint8_t start = 17;
    for (auto &inputTensor : inputTensors) {
        std::iota(inputTensor->data(), inputTensor->data() + inputTensor->size(), start);
        start += uint8_t(inputTensor->size());
    }
    for (auto &outputTensor : outputTensors) {
        std::fill(outputTensor->data(), outputTensor->data() + outputTensor->size(), 0xFF);
    }

    computePipeline->dispatchSubmit(16, 1, 1);

    for (auto &inputTensor : inputTensors) {
        std::cout << "INPUT" << std::endl;
        inputTensor->print();
    }
    for (auto &outputTensor : outputTensors) {
        std::cout << "OUTPUT" << std::endl;
        outputTensor->print();
    }

    for (auto i : {0U, 1U, 2U, 3U}) {
        ASSERT_TRUE(outputTensors[i]->compare(inputTensors[i]->data(), inputTensors[i]->size())) << "Output mismatch";
    }
}

// FIXME: Temporarily disabled in Darwin due to not being able to pass SSBO's to functions
TEST_F(MLEmulationLayerGraphForVulkan, CreateTensorComputePipeline) {

    const auto spirv = mlsdk::el::utils::glslToSpirv(fileToString("tensor.comp"));
    auto inputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 2, 2, 2}});
    auto outputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, std::vector<int64_t>{1, 2, 2, 2}});
    const TensorComputePipeline::DescriptorMap descriptorMap = {
        {
            // set 0
            {
                0,             // binding
                {inputTensor}, // tensor
            },
            {
                1,              // binding
                {outputTensor}, // tensor
            },
        },
    };
    auto computePipeline = std::make_shared<TensorComputePipeline>(device, descriptorMap, spirv);

    std::iota(inputTensor->data(), inputTensor->data() + inputTensor->size(), uint8_t{});
    computePipeline->dispatchSubmit(16, 16, 3);

    std::cout << "INPUT" << std::endl;
    inputTensor->print();

    std::cout << "OUTPUT" << std::endl;
    outputTensor->print();

    const uint8_t ref[1][2][2][2] = {
        // batch
        {// height
         {
             // width
             {0x00, 0x01}, // channel
             {0x02, 0x03}, // channel
         },
         {
             // width
             {0x04, 0x05}, // channel
             {0x06, 0x07}, // channel
         }},
    };

    ASSERT_TRUE(outputTensor->compare(&ref[0][0][0][0], sizeof(ref))) << "Output mismatch";
}
#endif

TEST_F(MLEmulationLayerGraphForVulkan, LinearTensorImageAliasingUsesImageRowPitch) {
    constexpr uint32_t width = 7;
    constexpr uint32_t height = 4;
    const std::vector<int64_t> dimensions{height, width, 1};

    vk::raii::DeviceMemory aliasedMemory{nullptr};

    const vk::TensorDescriptionARM tensorDescription{
        vk::TensorTilingARM::eLinear,
        vk::Format::eR8Uint,
        static_cast<uint32_t>(dimensions.size()),
        dimensions.data(),
        nullptr,
        vk::TensorUsageFlagBitsARM::eShader | vk::TensorUsageFlagBitsARM::eImageAliasing,
    };
    const vk::TensorCreateInfoARM tensorCreateInfo{
        {}, &tensorDescription, vk::SharingMode::eExclusive, 0, nullptr,
    };
    vk::raii::TensorARM destinationTensor{&(*device), tensorCreateInfo};

    vk::ImageCreateInfo imageCreateInfo{};
    imageCreateInfo.setImageType(vk::ImageType::e2D)
        .setFormat(vk::Format::eR8Uint)
        .setExtent(vk::Extent3D{width, height, 1})
        .setMipLevels(1)
        .setArrayLayers(1)
        .setSamples(vk::SampleCountFlagBits::e1)
        .setTiling(vk::ImageTiling::eLinear)
        .setUsage(vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTensorAliasingARM)
        .setSharingMode(vk::SharingMode::eExclusive)
        .setInitialLayout(vk::ImageLayout::eUndefined);
    vk::raii::Image image{&(*device), imageCreateInfo};

    const vk::TensorMemoryRequirementsInfoARM tensorRequirementsInfo{*destinationTensor};
    const auto tensorRequirements = (&(*device)).getTensorMemoryRequirementsARM(tensorRequirementsInfo);
    const auto imageRequirements = image.getMemoryRequirements();
    const auto memoryTypeBits = tensorRequirements.memoryRequirements.memoryTypeBits & imageRequirements.memoryTypeBits;
    const auto memoryTypeIndices = device->getPhysicalDevice()->getMemoryTypeIndices(
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, memoryTypeBits);
    ASSERT_FALSE(memoryTypeIndices.empty());

    const vk::MemoryAllocateFlagsInfo allocateFlags{vk::MemoryAllocateFlagBits::eDeviceAddress};
    const vk::MemoryAllocateInfo allocateInfo{
        std::max(tensorRequirements.memoryRequirements.size, imageRequirements.size),
        memoryTypeIndices.front(),
        &allocateFlags,
    };
    aliasedMemory = vk::raii::DeviceMemory{&(*device), allocateInfo};
    image.bindMemory(*aliasedMemory, 0);
    const vk::BindTensorMemoryInfoARM bindTensorInfo{*destinationTensor, *aliasedMemory, 0};
    (&(*device)).bindTensorMemoryARM(bindTensorInfo);

    const vk::ImageSubresource subresource{vk::ImageAspectFlagBits::eColor, 0, 0};
    const auto imageLayout = image.getSubresourceLayout(subresource);
    if (imageLayout.rowPitch == width) {
        GTEST_SKIP() << "The Vulkan driver did not pad the linear image row pitch";
    }

    auto *mappedMemory = static_cast<uint8_t *>(aliasedMemory.mapMemory(0, VK_WHOLE_SIZE));
    std::fill(mappedMemory, mappedMemory + allocateInfo.allocationSize, uint8_t{0});
    aliasedMemory.unmapMemory();

    const vk::TensorViewCreateInfoARM tensorViewCreateInfo{
        {},
        *destinationTensor,
        vk::Format::eR8Uint,
    };
    vk::raii::TensorViewARM tensorView{&(*device), tensorViewCreateInfo};

    auto placeholderTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Uint, dimensions});
    const TensorComputePipeline::DescriptorMap descriptorMap = {{{0, {placeholderTensor}}}};
    const auto spirv = mlsdk::el::utils::glslToSpirv(fileToString("tensor_image_alias.comp"));
    TensorComputePipeline computePipeline{device, descriptorMap, spirv};
    auto [descriptorPool, descriptorSets] = computePipeline.createDescriptorSets(descriptorMap);

    const vk::WriteDescriptorSetTensorARM tensorWriteInfo{
        1,
        &(*tensorView),
    };
    const vk::WriteDescriptorSet descriptorWrite{
        *descriptorSets.front(), 0, 0, 1, vk::DescriptorType::eTensorARM, nullptr, nullptr, nullptr, &tensorWriteInfo,
    };
    (&(*device)).updateDescriptorSets({descriptorWrite}, {});
    computePipeline.dispatchSubmit(descriptorSets, width, height, 1);

    mappedMemory = static_cast<uint8_t *>(aliasedMemory.mapMemory(0, VK_WHOLE_SIZE));
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const auto expected = static_cast<uint8_t>(((y + 1) * 16) + x);
            EXPECT_EQ(mappedMemory[imageLayout.offset + (y * imageLayout.rowPitch) + x], expected)
                << "Mismatch at image coordinate (" << x << ", " << y << ")";
        }
    }
    aliasedMemory.unmapMemory();
}

TEST_F(MLEmulationLayerGraphForVulkan, CopyLargeNonPackedTensor) {
    const std::vector<int64_t> dimensions{1, 1920, 1080, 3};
    const std::vector<int64_t> strides = {12441600, 6480, 6, 2};

    const std::vector<uint8_t> data;
    const auto useForCopy = true;
    auto inputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, dimensions}, data, useForCopy);
    auto outputTensor =
        std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, dimensions, strides}, data, useForCopy);
    const auto elementCount = inputTensor->size();
    std::vector<uint8_t> expected(elementCount);
    for (size_t i = 0; i < elementCount; ++i) {
        expected[i] = -128 + static_cast<int>(i % 256);
    }
    std::memcpy(inputTensor->data(), expected.data(), expected.size());
    // command pool
    const vk::CommandPoolCreateInfo commandPoolCreateInfo{
        {},                                                  // flags
        device->getPhysicalDevice()->getComputeFamilyIndex() // queue family index
    };
    auto commandPool = vk::raii::CommandPool(&(*device), commandPoolCreateInfo);

    // command buffer
    const vk::CommandBufferAllocateInfo commandBufferAllocInfo{
        *commandPool,                     // command pool
        vk::CommandBufferLevel::ePrimary, // command buffer level
        1                                 // command buffer count
    };
    vk::raii::CommandBuffers commandBuffers(&(*device), commandBufferAllocInfo);
    auto commandBuffer = std::move(commandBuffers.front());

    // command copy tensor
    const vk::CommandBufferBeginInfo commandBufferBeginInfo{
        vk::CommandBufferUsageFlagBits::eOneTimeSubmit, // flags
    };
    commandBuffer.begin(commandBufferBeginInfo);
    const vk::TensorCopyARM region{static_cast<uint32_t>(dimensions.size())};
    const vk::CopyTensorInfoARM copyInfo{
        &(*inputTensor),  // srcTensor
        &(*outputTensor), // dstTensor
        1,                // regionCount
        &region           // pRegions
    };
    commandBuffer.copyTensorARM(copyInfo);
    commandBuffer.end();

    // submit
    vk::raii::Queue queue(&(*device), device->getPhysicalDevice()->getComputeFamilyIndex(), 0);
    vk::raii::Fence fence(&(*device), vk::FenceCreateInfo());
    const vk::SubmitInfo submitInfo{
        0,                 // wait semaphore count
        nullptr,           // wait semaphore
        nullptr,           // pipeline stage flags
        1,                 // command buffer count
        &(*commandBuffer), // command buffers
        0,                 // signal semaphore count
    };
    auto begin = std::chrono::steady_clock::now();

    queue.submit({1, &submitInfo}, *fence);
    while (vk::Result::eTimeout == (&(*device)).waitForFences({*fence}, vk::True, uint64_t(-1))) {
        // Wait again
    }

    auto end = std::chrono::steady_clock::now();

    std::cout << "Runtime " << std::dec << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << " ms" << std::endl;

    // Mapped device memory can be uncached on the CPU. Read it in bulk before
    // checking every logical element against the original host-side input.
    std::vector<uint8_t> actual(outputTensor->size());
    std::memcpy(actual.data(), outputTensor->data(), actual.size());
    // This layout stores each byte at twice its packed offset.
    for (size_t i = 0; i < elementCount; ++i) {
        ASSERT_EQ(actual[2 * i], expected[i]) << "element " << i;
    }
}

} // namespace
} // namespace mlsdk::el::tests
