/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

#include "tensor_view.hpp"

#include <algorithm>
#include <list>
#include <set>
#include <vulkan/vulkan.hpp>

namespace mlsdk::el {
namespace layer {
namespace descriptor_binding {

template <typename T> inline bool hasTensor(const T &obj) {
    return obj.descriptorType == VK_DESCRIPTOR_TYPE_TENSOR_ARM;
}

inline bool hasTensor(const VkDescriptorPoolSize &obj) { return obj.type == VK_DESCRIPTOR_TYPE_TENSOR_ARM; }

inline std::vector<VkDescriptorSetLayoutBinding>
substituteTensorBinding(uint32_t bindingCount, const VkDescriptorSetLayoutBinding *pBindings,
                        const VkDescriptorSetLayoutBindingFlagsCreateInfo *bindingInfo) {
    std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings{pBindings, pBindings + bindingCount};

    // Loop over bindings and replace tensors bindings with uniform buffer for tensor descriptor
    for (uint32_t i = 0; i < bindingCount; i++) {
        if (hasTensor(pBindings[i])) {
            // Change binding to uniform buffer
            descriptorSetLayoutBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;

            // Remove uniform update after bind
            if (bindingInfo) {
                const_cast<VkDescriptorBindingFlags *>(bindingInfo->pBindingFlags)[i] &=
                    static_cast<VkDescriptorBindingFlags>(~VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT);
            }
        }
    }

    return descriptorSetLayoutBindings;
}

inline std::tuple<std::vector<VkWriteDescriptorSet>, std::list<VkDescriptorBufferInfo>,
                  std::list<VkDescriptorImageInfo>>
substituteTensorWriteDescriptorSet(const Device &dev, uint32_t descriptorWriteCount,
                                   const VkWriteDescriptorSet *pDescriptorWrites) {
    std::vector<VkWriteDescriptorSet> writes;
    std::list<VkDescriptorBufferInfo> bufferInfos;
    std::list<VkDescriptorImageInfo> imageInfos;

    // Loop over write descriptors and replace tensor bindings with uniform buffer for tensor description
    for (uint32_t i = 0; i < descriptorWriteCount; i++) {
        if (!hasTensor(pDescriptorWrites[i])) {
            const auto &write = pDescriptorWrites[i];
            if (!write.pImageInfo || write.pImageInfo->imageLayout != VK_IMAGE_LAYOUT_TENSOR_ALIASING_ARM) {
                writes.emplace_back(write);
                continue;
            }
            // Image layout must be updated
            imageInfos.emplace_back(VkDescriptorImageInfo{
                write.pImageInfo->sampler,   // sampler
                write.pImageInfo->imageView, // imageView
                VK_IMAGE_LAYOUT_GENERAL,     // imageLayout
            });
            auto write_copy = write;
            write_copy.pImageInfo = &imageInfos.back();
            writes.emplace_back(write_copy);
            continue;
        }

        const auto tensorInfo = findType<VkWriteDescriptorSetTensorARM>(
            pDescriptorWrites[i].pNext, VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_TENSOR_ARM);
        if (tensorInfo == nullptr) {
            throw std::runtime_error("Write descriptor is missing tensor descriptor tensor info");
        }

        for (uint32_t j = 0; j < tensorInfo->tensorViewCount; j++) {
            const auto tensorViewARM = reinterpret_cast<TensorViewARM *>(tensorInfo->pTensorViews[j]);

            bufferInfos.emplace_back(VkDescriptorBufferInfo{
                tensorViewARM->getDescriptorBuffer(dev), // buffer
                0,                                       // offset,
                VK_WHOLE_SIZE,                           // range
            });

            writes.emplace_back(VkWriteDescriptorSet{
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,   // sType
                nullptr,                                  // pNext
                pDescriptorWrites[i].dstSet,              // dstSet
                pDescriptorWrites[i].dstBinding,          // dstBinding
                pDescriptorWrites[i].dstArrayElement + j, // dstArrayElement
                1,                                        // descriptorCount
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,        // descriptorType
                nullptr,                                  // pImageInfo
                &bufferInfos.back(),                      // pBufferInfo
                nullptr                                   // pTexelBufferView
            });
        }
    }

    return {std::move(writes), std::move(bufferInfos), std::move(imageInfos)};
}

inline std::vector<VkDescriptorPoolSize>
substituteTensorDescriptorPoolSizes(const std::vector<VkDescriptorPoolSize> &poolSizes) {
    std::vector<VkDescriptorPoolSize> descriptorPoolSizes;
    uint32_t tensorCount = 0;

    // Loop over the descriptor sizes and remove tensor descriptors
    for (const auto &poolSize : poolSizes) {
        if (hasTensor(poolSize)) {
            tensorCount += poolSize.descriptorCount;
            continue;
        }

        descriptorPoolSizes.emplace_back(poolSize);
    }

    // For each tensor descriptor increase the number of uniform descriptors
    if (tensorCount > 0) {
        auto desc = std::find_if(descriptorPoolSizes.begin(), descriptorPoolSizes.end(), [](const auto &poolSize) {
            return poolSize.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        });

        if (desc == descriptorPoolSizes.end()) {
            descriptorPoolSizes.emplace_back(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, tensorCount});
        } else {
            desc->descriptorCount += tensorCount;
        }
    }

    return descriptorPoolSizes;
}

} // namespace descriptor_binding
} // namespace layer
} // namespace mlsdk::el
