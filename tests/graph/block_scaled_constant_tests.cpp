/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compute_graph_op.hpp"
#include "tensor.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <memory>
#include <tuple>
#include <vector>

namespace mlsdk::el::compute {
namespace {

#ifdef VK_ARM_tensors2
class BlockScaledConstants : public testing::Test {
  protected:
    std::shared_ptr<vk::detail::DispatchLoaderDynamic> loader = std::make_shared<vk::detail::DispatchLoaderDynamic>();
    std::array<std::array<uint8_t, 256>, 2> memory{};
    std::array<VkDeviceMemory, 2> bindings{};
    size_t allocationCount = 0;
    size_t createdTensorCount = 0;
    size_t destroyedTensorCount = 0;
    std::array<size_t, 2> freedMemoryCounts{};

    static BlockScaledConstants &state(VkDevice device) { return *reinterpret_cast<BlockScaledConstants *>(device); }

    static void VKAPI_CALL properties(VkPhysicalDevice, VkPhysicalDeviceProperties *properties) { *properties = {}; }
    static void VKAPI_CALL queues(VkPhysicalDevice, uint32_t *count, VkQueueFamilyProperties *properties) {
        if (properties) {
            properties[0] = {};
            properties[0].queueFlags = VK_QUEUE_COMPUTE_BIT;
        }
        *count = 1;
    }
    static void VKAPI_CALL memoryProperties(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties *properties) {
        *properties = {};
        properties->memoryTypeCount = 1;
        properties->memoryTypes[0].propertyFlags =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        properties->memoryHeapCount = 1;
        properties->memoryHeaps[0].size = 4096;
    }
    static VkResult VKAPI_CALL createTensor(VkDevice device, const VkTensorCreateInfoARM *info,
                                            const VkAllocationCallbacks *, VkTensorARM *tensor) {
        if (info->pDescription || !info->pNext) {
            return VK_ERROR_UNKNOWN;
        }
        const auto *scaled = static_cast<const VkBlockScaledTensorDescriptionARM *>(info->pNext);
        EXPECT_EQ(scaled->sType, VK_STRUCTURE_TYPE_BLOCK_SCALED_TENSOR_DESCRIPTION_ARM);
        EXPECT_EQ(scaled->planeCount, 2u);
        EXPECT_EQ(scaled->pPlaneTypes[0], VK_TENSOR_PLANE_VALUES_ARM);
        EXPECT_EQ(scaled->pPlaneTypes[1], VK_TENSOR_PLANE_SCALES_ARM);
        EXPECT_EQ(scaled->pPlaneDescriptions[0].pDimensions[1], 32);
        EXPECT_EQ(scaled->pPlaneDescriptions[1].pDimensions[1], 1);
        EXPECT_EQ(scaled->pPlaneDescriptions[1].format, VK_FORMAT_R8_UFLOAT_FPENCODING_FLOAT8UNSIGNEDE8M0_ARM);
        *tensor = reinterpret_cast<VkTensorARM>(device);
        ++state(device).createdTensorCount;
        return VK_SUCCESS;
    }
    static void VKAPI_CALL destroyTensor(VkDevice device, VkTensorARM tensor, const VkAllocationCallbacks *) {
        EXPECT_EQ(tensor, reinterpret_cast<VkTensorARM>(device));
        ++state(device).destroyedTensorCount;
    }
    static VkResult VKAPI_CALL createView(VkDevice, const VkTensorViewCreateInfoARM *info,
                                          const VkAllocationCallbacks *, VkTensorViewARM *view) {
        *view = reinterpret_cast<VkTensorViewARM>(info->tensor);
        return VK_SUCCESS;
    }
    static void VKAPI_CALL destroyView(VkDevice, VkTensorViewARM, const VkAllocationCallbacks *) {}
    static void VKAPI_CALL memoryRequirements(VkDevice, const VkTensorMemoryRequirementsInfoARM *info,
                                              VkMemoryRequirements2 *requirements) {
        const auto *plane = static_cast<const VkTensorPlaneInfoARM *>(info->pNext);
        ASSERT_NE(plane, nullptr);
        EXPECT_EQ(plane->sType, VK_STRUCTURE_TYPE_TENSOR_PLANE_INFO_ARM);
        EXPECT_LT(plane->planeIndex, 2u);
        // Device allocation padding must not change the source offset of the scales.
        requirements->memoryRequirements = {256, 64, 1};
    }
    static VkResult VKAPI_CALL allocate(VkDevice device, const VkMemoryAllocateInfo *info,
                                        const VkAllocationCallbacks *, VkDeviceMemory *memory) {
        auto &self = state(device);
        if (self.allocationCount == self.memory.size() || info->allocationSize > self.memory[0].size()) {
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }
        *memory = reinterpret_cast<VkDeviceMemory>(self.memory[self.allocationCount++].data());
        return VK_SUCCESS;
    }
    static void VKAPI_CALL free(VkDevice device, VkDeviceMemory allocation, const VkAllocationCallbacks *) {
        auto &self = state(device);
        for (size_t index = 0; index < self.memory.size(); ++index) {
            if (allocation == reinterpret_cast<VkDeviceMemory>(self.memory[index].data())) {
                ++self.freedMemoryCounts[index];
                return;
            }
        }
        ADD_FAILURE() << "Unexpected device memory handle";
    }
    static VkResult VKAPI_CALL map(VkDevice, VkDeviceMemory memory, VkDeviceSize, VkDeviceSize, VkMemoryMapFlags,
                                   void **data) {
        *data = reinterpret_cast<void *>(memory);
        return VK_SUCCESS;
    }
    static void VKAPI_CALL unmap(VkDevice, VkDeviceMemory) {}
    static VkResult VKAPI_CALL bind(VkDevice device, uint32_t count, const VkBindTensorMemoryInfoARM *infos) {
        if (count != 1) {
            return VK_ERROR_UNKNOWN;
        }
        const auto *plane = static_cast<const VkTensorPlaneInfoARM *>(infos[0].pNext);
        if (!plane || plane->planeIndex >= 2) {
            return VK_ERROR_UNKNOWN;
        }
        EXPECT_EQ(plane->sType, VK_STRUCTURE_TYPE_TENSOR_PLANE_INFO_ARM);
        EXPECT_EQ(infos[0].memoryOffset, 0u);
        state(device).bindings[plane->planeIndex] = infos[0].memory;
        return VK_SUCCESS;
    }

    void SetUp() override {
        loader->vkGetPhysicalDeviceProperties = properties;
        loader->vkGetPhysicalDeviceQueueFamilyProperties = queues;
        loader->vkGetPhysicalDeviceMemoryProperties = memoryProperties;
        loader->vkCreateTensorARM = createTensor;
        loader->vkDestroyTensorARM = destroyTensor;
        loader->vkCreateTensorViewARM = createView;
        loader->vkDestroyTensorViewARM = destroyView;
        loader->vkGetTensorMemoryRequirementsARM = memoryRequirements;
        loader->vkAllocateMemory = allocate;
        loader->vkFreeMemory = free;
        loader->vkMapMemory = map;
        loader->vkUnmapMemory = unmap;
        loader->vkBindTensorMemoryARM = bind;
    }
};

TEST_F(BlockScaledConstants, InitializesBothPlanesFromOnePayload) {
    const std::array<int64_t, 2> valueShape{2, 32};
    const std::array<int64_t, 2> scaleShape{2, 1};
    const std::array planeTypes{VK_TENSOR_PLANE_VALUES_ARM, VK_TENSOR_PLANE_SCALES_ARM};
    for (const auto &[format, valueRowStride, scaleRowStride] :
         {std::tuple{VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E4M3_ARM, 0, 0},
          std::tuple{VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E5M2_ARM, 0, 0},
          std::tuple{VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E4M3_ARM, 64, 0},
          std::tuple{VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E4M3_ARM, 0, 2}}) {
        SCOPED_TRACE(format);
        SCOPED_TRACE(valueRowStride);
        SCOPED_TRACE(scaleRowStride);
        for (auto &allocation : memory) {
            allocation.fill(0);
        }
        bindings.fill(VK_NULL_HANDLE);
        allocationCount = 0;
        createdTensorCount = 0;
        destroyedTensorCount = 0;
        freedMemoryCounts.fill(0);

        const auto valueMemory = reinterpret_cast<VkDeviceMemory>(memory[0].data());
        const auto scaleMemory = reinterpret_cast<VkDeviceMemory>(memory[1].data());

        const std::array<int64_t, 2> valueStrides{valueRowStride, 1};
        const std::array<int64_t, 2> scaleStrides{scaleRowStride, 1};
        const auto scaleStride =
            static_cast<size_t>(scaleRowStride ? scaleRowStride : (valueRowStride ? valueRowStride / 32 : 1));
        const auto valueSize = 64 * scaleStride;
        std::vector<uint8_t> payload(valueSize + 2 * scaleStride);
        std::fill_n(payload.begin(), valueSize, 0x38);
        payload[valueSize] = 0x7f;
        payload[valueSize + scaleStride] = 0x80;
        const std::array<VkTensorDescriptionARM, 2> planes{{
            {VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM, nullptr, VK_TENSOR_TILING_LINEAR_ARM, format, 2,
             valueShape.data(), valueRowStride ? valueStrides.data() : nullptr, VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM},
            {VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM, nullptr, VK_TENSOR_TILING_LINEAR_ARM,
             VK_FORMAT_R8_UFLOAT_FPENCODING_FLOAT8UNSIGNEDE8M0_ARM, 2, scaleShape.data(),
             scaleRowStride ? scaleStrides.data() : nullptr, VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM},
        }};
        const VkBlockScaledTensorDescriptionARM description{VK_STRUCTURE_TYPE_BLOCK_SCALED_TENSOR_DESCRIPTION_ARM,
                                                            nullptr,
                                                            VK_TENSOR_BLOCK_SCALING_SCHEME_PLANAR_OCP_MICROSCALING_ARM,
                                                            2,
                                                            planeTypes.data(),
                                                            planes.data()};
        {
            graph_op::GraphPipeline pipeline(loader, reinterpret_cast<VkPhysicalDevice>(this),
                                             reinterpret_cast<VkDevice>(this), nullptr);
            const VkDataGraphPipelineConstantARM constant{VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_CONSTANT_ARM,
                                                          &description, 17, payload.data()};
            pipeline.makeConstTensor(constant);
            const auto descriptor = pipeline.getConstTensor(17);
            EXPECT_EQ(descriptor->getFormat(), format);
            EXPECT_EQ(descriptor->getDimensions(), (std::vector<int64_t>{2, 32}));
            ASSERT_TRUE(descriptor->getBlockScaling());
            EXPECT_EQ(descriptor->getBlockScaling()->blockShape, (std::vector<uint32_t>{32}));
            EXPECT_EQ(descriptor->getBlockScaling()->scaleFormat, planes[1].format);
            ASSERT_EQ(allocationCount, 2u);
            EXPECT_EQ(bindings[0], valueMemory);
            EXPECT_EQ(bindings[1], scaleMemory);
            EXPECT_TRUE(std::equal(payload.data(), payload.data() + valueSize, memory[0].begin()));
            EXPECT_EQ(memory[0][valueSize], 0);
            EXPECT_EQ(memory[1][0], payload[valueSize]);
            EXPECT_EQ(memory[1][scaleStride], payload[valueSize + scaleStride]);
            EXPECT_EQ(memory[1][2 * scaleStride], 0);
        }
        EXPECT_EQ(createdTensorCount, 1u);
        EXPECT_EQ(destroyedTensorCount, 1u);
        EXPECT_EQ(freedMemoryCounts[0], 1u);
        EXPECT_EQ(freedMemoryCounts[1], 1u);
    }
}

#endif

} // namespace
} // namespace mlsdk::el::compute
