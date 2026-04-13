/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "pipeline_cache.hpp"

#include <vulkan/vulkan.hpp>

#include <memory>
#include <string_view>
#include <vector>

namespace mlsdk::el::compute::common {

enum class BlockMatchMode : uint32_t {
    MIN_SAD = 0,
    MIN_SAD_COST = 1,
    RAW_SAD = 2,
};

struct SpecializationConstantsView {
    const void *data = nullptr;
    uint32_t sizeBytes = 0;
    uint32_t entryCount = 0;
};

inline SpecializationConstantsView makeSpecializationConstantsView(const std::vector<uint32_t> &constants) {
    return {constants.data(), static_cast<uint32_t>(constants.size() * sizeof(uint32_t)),
            static_cast<uint32_t>(constants.size())};
}

inline SpecializationConstantsView makeSpecializationConstantsView(const void *data, const uint32_t sizeBytes) {
    return {data, sizeBytes, static_cast<uint32_t>(sizeBytes / sizeof(uint32_t))};
}

std::vector<VkSpecializationMapEntry> makeSpecializationMapEntries(uint32_t entryCount);

VkShaderModule createShaderModule(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &loader,
                                  VkDevice device, const SpirvBinary &code,
                                  const VkAllocationCallbacks *allocator = nullptr);

VkPipeline createComputePipeline(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &loader,
                                 VkDevice device, VkPipelineCache pipelineCache, VkShaderModule shaderModule,
                                 VkPipelineLayout pipelineLayout,
                                 const SpecializationConstantsView *specializationConstants,
                                 const VkAllocationCallbacks *allocator = nullptr);

} // namespace mlsdk::el::compute::common
