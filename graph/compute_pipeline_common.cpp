/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "compute_pipeline_common.hpp"

#include <stdexcept>
#include <string>

namespace mlsdk::el::compute::common {

std::vector<VkSpecializationMapEntry> makeSpecializationMapEntries(const uint32_t entryCount) {
    std::vector<VkSpecializationMapEntry> entries;
    entries.reserve(entryCount);

    for (uint32_t i = 0; i < entryCount; ++i) {
        entries.push_back({
            i,                                           // constantID
            static_cast<uint32_t>(i * sizeof(uint32_t)), // offset
            sizeof(uint32_t),                            // size
        });
    }

    return entries;
}

VkShaderModule createShaderModule(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &loader,
                                  const VkDevice device, const SpirvBinary &code,
                                  const VkAllocationCallbacks *allocator) {
    const VkShaderModuleCreateInfo shaderModuleCreateInfo = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,           // type
        nullptr,                                               // next
        0,                                                     // flags
        static_cast<uint32_t>(code.size() * sizeof(uint32_t)), // code size
        code.data(),                                           // code
    };

    VkShaderModule shaderModule;
    if (loader->vkCreateShaderModule(device, &shaderModuleCreateInfo, allocator, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module");
    }

    return shaderModule;
}

VkPipeline createComputePipeline(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &loader,
                                 const VkDevice device, const VkPipelineCache pipelineCache,
                                 const VkShaderModule shaderModule, const VkPipelineLayout pipelineLayout,
                                 const SpecializationConstantsView *specializationConstants,
                                 const VkAllocationCallbacks *allocator) {
    std::vector<VkSpecializationMapEntry> specEntries;
    const VkSpecializationInfo *specializationInfo = nullptr;
    VkSpecializationInfo specializationInfoStorage{};

    if (specializationConstants != nullptr && specializationConstants->entryCount > 0) {
        specEntries = makeSpecializationMapEntries(specializationConstants->entryCount);
        specializationInfoStorage = {
            static_cast<uint32_t>(specEntries.size()), // mapEntryCount
            specEntries.data(),                        // pMapEntries
            specializationConstants->sizeBytes,        // dataSize
            specializationConstants->data,             // pData
        };
        specializationInfo = &specializationInfoStorage;
    }

    const VkPipelineShaderStageCreateInfo pipelineShaderCreateInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, // type
        nullptr,                                             // next
        0,                                                   // flags
        VK_SHADER_STAGE_COMPUTE_BIT,                         // stage flag bits
        shaderModule,                                        // shader module
        "main",                                              // name
        specializationInfo,                                  // specialization info
    };

    const VkComputePipelineCreateInfo computePipelineCreateInfo = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, // type
        nullptr,                                        // next
        0,                                              // flags
        pipelineShaderCreateInfo,                       // create info
        pipelineLayout,                                 // pipeline layout
        nullptr,                                        // base pipeline handle
        0,                                              // base pipeline index
    };

    VkPipeline pipeline;
    if (loader->vkCreateComputePipelines(device, pipelineCache, 1, &computePipelineCreateInfo, allocator, &pipeline) !=
        VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute pipelines");
    }

    return pipeline;
}

} // namespace mlsdk::el::compute::common
