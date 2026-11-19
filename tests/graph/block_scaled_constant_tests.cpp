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
    struct Allocation {
        std::vector<uint8_t> bytes;
        bool freed = false;
    };

    std::shared_ptr<vk::detail::DispatchLoaderDynamic> loader = std::make_shared<vk::detail::DispatchLoaderDynamic>();
    std::vector<std::unique_ptr<Allocation>> allocations;
    std::array<VkDeviceMemory, 2> bindings{};
    const VkBlockScaledTensorDescriptionARM *constantDescription = nullptr;
    uint32_t tensorCount = 0;
    uint32_t destroyedTensors = 0;

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
        auto &self = state(device);
        EXPECT_EQ(info->pDescription, nullptr);
        EXPECT_EQ(info->pNext, self.constantDescription);
        const auto *scaled = static_cast<const VkBlockScaledTensorDescriptionARM *>(info->pNext);
        if (!scaled) {
            return VK_ERROR_UNKNOWN;
        }
        EXPECT_EQ(scaled->sType, VK_STRUCTURE_TYPE_BLOCK_SCALED_TENSOR_DESCRIPTION_ARM);
        EXPECT_EQ(scaled->planeCount, 2u);
        EXPECT_EQ(scaled->pPlaneTypes[0], VK_TENSOR_PLANE_VALUES_ARM);
        EXPECT_EQ(scaled->pPlaneTypes[1], VK_TENSOR_PLANE_SCALES_ARM);
        EXPECT_EQ(scaled->pPlaneDescriptions[0].pDimensions[1], 32);
        EXPECT_EQ(scaled->pPlaneDescriptions[1].pDimensions[1], 1);
        ++self.tensorCount;
        *tensor = reinterpret_cast<VkTensorARM>(device);
        return VK_SUCCESS;
    }
    static void VKAPI_CALL destroyTensor(VkDevice device, VkTensorARM, const VkAllocationCallbacks *) {
        ++state(device).destroyedTensors;
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
        auto &allocation = state(device).allocations.emplace_back(std::make_unique<Allocation>());
        allocation->bytes.resize(static_cast<size_t>(info->allocationSize));
        *memory = reinterpret_cast<VkDeviceMemory>(allocation.get());
        return VK_SUCCESS;
    }
    static void VKAPI_CALL free(VkDevice, VkDeviceMemory memory, const VkAllocationCallbacks *) {
        reinterpret_cast<Allocation *>(memory)->freed = true;
    }
    static VkResult VKAPI_CALL map(VkDevice, VkDeviceMemory memory, VkDeviceSize, VkDeviceSize, VkMemoryMapFlags,
                                   void **data) {
        *data = reinterpret_cast<Allocation *>(memory)->bytes.data();
        return VK_SUCCESS;
    }
    static void VKAPI_CALL unmap(VkDevice, VkDeviceMemory) {}
    static VkResult VKAPI_CALL bind(VkDevice device, uint32_t count, const VkBindTensorMemoryInfoARM *infos) {
        EXPECT_EQ(count, 1u);
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
        const std::array<int64_t, 2> valueStrides{valueRowStride, 1};
        const std::array<int64_t, 2> scaleStrides{scaleRowStride, 1};
        const auto scaleStride =
            static_cast<size_t>(scaleRowStride ? scaleRowStride : (valueRowStride ? valueRowStride / 32 : 1));
        const auto valueSize = 64 * scaleStride;
        std::vector<uint8_t> payload(valueSize + 2 * scaleStride);
        std::fill_n(payload.begin(), valueSize, 0x38);
        payload[valueSize] = 0x7f;               // E8M0 scale 1.
        payload[valueSize + scaleStride] = 0x80; // E8M0 scale 2.
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
        constantDescription = &description;
        {
            graph_op::GraphPipeline pipeline(loader, reinterpret_cast<VkPhysicalDevice>(this),
                                             reinterpret_cast<VkDevice>(this), nullptr);
            const VkDataGraphPipelineConstantARM constant{VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_CONSTANT_ARM,
                                                          &description, 17, payload.data()};
            pipeline.makeConstTensor(constant);
            const auto descriptor = pipeline.getConstTensor(17);
            EXPECT_EQ(descriptor->getFormat(), format);
            EXPECT_EQ(descriptor->getDimensions(), (std::vector<int64_t>{2, 32}));
            ASSERT_EQ(allocations.size(), 2u);
            EXPECT_EQ(bindings[0], reinterpret_cast<VkDeviceMemory>(allocations[0].get()));
            EXPECT_EQ(bindings[1], reinterpret_cast<VkDeviceMemory>(allocations[1].get()));
            EXPECT_TRUE(std::equal(payload.data(), payload.data() + valueSize, allocations[0]->bytes.begin()));
            EXPECT_EQ(allocations[0]->bytes[valueSize], 0);
            EXPECT_EQ(allocations[1]->bytes[0], payload[valueSize]);
            EXPECT_EQ(allocations[1]->bytes[scaleStride], payload[valueSize + scaleStride]);
            EXPECT_EQ(allocations[1]->bytes[2 * scaleStride], 0);
            EXPECT_EQ(tensorCount, 1u);
        }
        EXPECT_EQ(destroyedTensors, 1u);
        for (const auto &allocation : allocations) {
            EXPECT_TRUE(allocation->freed);
        }
        allocations.clear();
        tensorCount = 0;
        destroyedTensors = 0;
    }
}
#endif

} // namespace
} // namespace mlsdk::el::compute
