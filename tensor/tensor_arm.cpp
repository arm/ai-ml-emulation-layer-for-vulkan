/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */
#include "tensor_arm.hpp"

#include "mlel/utils.hpp"

#include <numeric>

using namespace mlsdk::el::utils;

namespace mlsdk::el::layer {

TensorStorage::TensorStorage(const VkTensorCreateInfoARM &createInfo) {
    planes.push_back({TensorLayout(*createInfo.pDescription, createInfo.flags)});
}

void TensorStorage::destroy(const Device &dev, const VkAllocationCallbacks *pAllocator) {
    for (const auto &plane : planes) {
        dev.loader->vkDestroyBuffer(dev.device, plane.buffer, pAllocator);
    }
}

VkResult TensorARM::create(const Device &dev, const VkTensorCreateInfoARM &createInfo,
                           const VkAllocationCallbacks *allocator) {
    m_storage = TensorStorage(createInfo);
    m_info = m_storage.planes.front().layout;
    const auto *pCaptureDescriptorInfo = findType<VkOpaqueCaptureDescriptorDataCreateInfoEXT>(
        createInfo.pNext, VK_STRUCTURE_TYPE_OPAQUE_CAPTURE_DESCRIPTOR_DATA_CREATE_INFO_EXT);
    for (auto &plane : m_storage.planes) {
        const VkBufferCreateInfo bufferInfo = {
            VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            pCaptureDescriptorInfo,
            plane.layout.flags,
            plane.layout.size,
            plane.layout.usage,
            createInfo.sharingMode,
            createInfo.queueFamilyIndexCount,
            createInfo.pQueueFamilyIndices,
        };
        const VkResult result = dev.loader->vkCreateBuffer(dev.device, &bufferInfo, allocator, &plane.buffer);
        if (result != VK_SUCCESS) {
            plane.buffer = VK_NULL_HANDLE;
            destroy(dev, allocator);
            return result;
        }
    }
    return VK_SUCCESS;
}

void TensorARM::destroy(const Device &dev, const VkAllocationCallbacks *pAllocator) {
    m_storage.destroy(dev, pAllocator);
}

void TensorARM::getMemoryRequirements(const Device &dev, VkMemoryRequirements *requirements) const {
    dev.loader->vkGetBufferMemoryRequirements(dev.device, getTensorBuffer(), requirements);
}

void TensorARM::getDeviceTensorMemoryRequirements(const Device &dev, const VkTensorCreateInfoARM &createInfo,
                                                  VkMemoryRequirements2 *requirements) {
    TensorLayout info(*createInfo.pDescription, createInfo.flags);
    const VkBufferCreateInfo bufferInfo = {
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        nullptr,
        info.flags,
        info.size,
        info.usage,
        createInfo.sharingMode,
        createInfo.queueFamilyIndexCount,
        createInfo.pQueueFamilyIndices,
    };
    if (dev.loader->vkGetDeviceBufferMemoryRequirements) {
        VkDeviceBufferMemoryRequirements req = {};
        req.sType = VK_STRUCTURE_TYPE_DEVICE_BUFFER_MEMORY_REQUIREMENTS;
        req.pCreateInfo = &bufferInfo;
        dev.loader->vkGetDeviceBufferMemoryRequirements(dev.device, &req, requirements);
    } else {
        VkBuffer buffer;
        VkResult result = dev.loader->vkCreateBuffer(dev.device, &bufferInfo, nullptr, &buffer);
        if (result == VK_SUCCESS) {
            VkMemoryRequirements req;
            dev.loader->vkGetBufferMemoryRequirements(dev.device, buffer, &req);
            requirements->memoryRequirements = req;
            dev.loader->vkDestroyBuffer(dev.device, buffer, nullptr);
        } else {
            throw std::runtime_error("Fail to get device tensor memory requirements.");
        }
    }
}

VkResult TensorARM::bindTensorMemory(const Device &dev, VkDeviceMemory memory, VkDeviceSize offset) const {
    return dev.loader->vkBindBufferMemory(dev.device, getTensorBuffer(), memory, offset);
}

void TensorARM::updateAliasedTensorInfo(const Device &dev, VkImage image) {
    auto &info = m_info;
    auto rank = info.dimensions.size();
    if (rank < 2 || rank > 4) {
        throw std::runtime_error("Tensor rank should be 2, 3 or 4 to alias with image.");
    }

    if (info.usesImageAliasing) {
        const VkImageSubresource imageSubresource = {
            VK_IMAGE_ASPECT_COLOR_BIT,
            0,
            0,
        };
        VkSubresourceLayout imageSubresourceLayout{};
        dev.loader->vkGetImageSubresourceLayout(dev.device, image, &imageSubresource, &imageSubresourceLayout);
        tensor_description_detail::updateAliasedStrides(rank, info.strides, imageSubresourceLayout);
        m_storage.planes.front().layout.strides = info.strides;
    }
}

void TensorARM::copyToTensor(TensorCommandBuffer &cmd, const TensorARM &dstTensor) {
    if (m_info.dimensions != dstTensor.m_info.dimensions || m_info.elementSize != dstTensor.m_info.elementSize) {
        throw std::runtime_error("Src tensor and dst tensor should have same dimensions and element size.");
    }

    const auto &srcPlane = m_storage.planes.front();
    const auto &dstPlane = dstTensor.m_storage.planes.front();
    if (srcPlane.layout.strides == dstPlane.layout.strides) {
        const VkBufferCopy copyInfo{0, 0, srcPlane.layout.size};
        cmd.loader->vkCmdCopyBuffer(cmd.commandBuffer, srcPlane.buffer, dstPlane.buffer, 1, &copyInfo);
    } else {
        const auto regionCount = utils::getElementCount(srcPlane.layout.dimensions);
        assert(regionCount < std::numeric_limits<uint32_t>::max());
        auto pipeline = std::make_unique<TensorCopyPipeline>(cmd.loader, cmd.device->device, srcPlane, dstPlane);
        cmd.recordSecondaryCommandBuffer([&](VkCommandBuffer secondary) {
            pipeline->cmdBindAndDispatchCopy(secondary, static_cast<uint32_t>(regionCount));
        });
        cmd.retainCopyPipeline(std::move(pipeline));
    }
}

VkResult TensorARM::getOpaqueCaptureDescriptorDataEXT(const Device &dev, void *pData) const {
    const VkBufferCaptureDescriptorDataInfoEXT info = {
        VK_STRUCTURE_TYPE_BUFFER_CAPTURE_DESCRIPTOR_DATA_INFO_EXT,
        nullptr,
        getTensorBuffer(),
    };
    return dev.loader->vkGetBufferOpaqueCaptureDescriptorDataEXT(dev.device, &info, pData);
}

TensorCopyPipeline::TensorCopyPipeline(
    const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
    const TensorPlane &srcPlane, const TensorPlane &dstPlane)
    : loader{_loader}, device{_device}, pushConstant{createPushConstant(srcPlane, dstPlane)},
      shaderModule{createShaderModule(srcPlane)}, descriptorPool{createDescriptorPool()},
      descriptorSetLayout{createDescriptorSetLayout()}, pipelineLayout{createPipelineLayout()},
      pipeline{createPipeline(srcPlane)}, descriptorSet{createDescriptorSet(srcPlane, dstPlane)} {}

TensorCopyPipeline::~TensorCopyPipeline() {
    loader->vkFreeDescriptorSets(device, descriptorPool, 1, &descriptorSet);
    loader->vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    loader->vkDestroyPipeline(device, pipeline, nullptr);
    loader->vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    loader->vkDestroyShaderModule(device, shaderModule, nullptr);
    loader->vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
}

TensorCopyPipeline::PushConstant TensorCopyPipeline::createPushConstant(const TensorPlane &srcPlane,
                                                                        const TensorPlane &dstPlane) const {
    PushConstant constant;
    const auto &dimensions = srcPlane.layout.dimensions;
    const auto &srcStrides = srcPlane.layout.strides;
    const auto &dstStrides = dstPlane.layout.strides;
    std::copy(dimensions.begin(), dimensions.end(), &constant.dimensions[0]);
    std::copy(srcStrides.begin(), srcStrides.end(), &constant.srcStrides[0]);
    std::copy(dstStrides.begin(), dstStrides.end(), &constant.dstStrides[0]);
    return constant;
}

VkShaderModule TensorCopyPipeline::createShaderModule(const TensorPlane &srcPlane) const {
    const auto srcType = std::string(getFormatInfo(srcPlane.layout.format)->glslType);
    const std::lock_guard lock(cacheMutex);
    auto &spirv = spirvCache[srcType];
    if (spirv.empty()) {
        std::string tmp = glsl;
        replaceAll(tmp, "%warpX%", std::to_string(warp1D));
        replaceAll(tmp, "%type%", srcType);
        replaceAll(tmp, "%type_size%", std::to_string(srcPlane.layout.elementSize));
        spirv = glslToSpirv(tmp);
    }

    const VkShaderModuleCreateInfo shaderModuleCreateInfo = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,            // type
        nullptr,                                                // next
        0,                                                      // flags
        static_cast<uint32_t>(spirv.size() * sizeof(uint32_t)), // code size
        spirv.data(),                                           // code
    };

    VkShaderModule vkShaderModule;
    if (loader->vkCreateShaderModule(device, &shaderModuleCreateInfo, nullptr, &vkShaderModule) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module");
    }

    return vkShaderModule;
}

VkDescriptorPool TensorCopyPipeline::createDescriptorPool() const {
    const VkDescriptorPoolSize poolSize = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // type
        2,                                 // descriptor count
    };

    const VkDescriptorPoolCreateInfo createInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,     // type
        nullptr,                                           // next
        VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT, // flags
        1,                                                 // max sets
        1,                                                 // pool size count
        &poolSize,                                         // pool size
    };

    VkDescriptorPool pool;
    if (loader->vkCreateDescriptorPool(device, &createInfo, nullptr, &pool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor pool");
    }

    return pool;
}

VkDescriptorSetLayout TensorCopyPipeline::createDescriptorSetLayout() const {
    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        {
            0,                                 // binding
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // type
            1,                                 // descriptor count
            VK_SHADER_STAGE_COMPUTE_BIT,       // stage flags
            nullptr,                           // VkSampler
        },
        {
            1,                                 // binding
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // type
            1,                                 // descriptor count
            VK_SHADER_STAGE_COMPUTE_BIT,       // stage flags
            nullptr,                           // VkSampler
        },
    };

    const VkDescriptorSetLayoutCreateInfo createInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, // type
        nullptr,                                             // next
        0,                                                   // flags
        uint32_t(bindings.size()),                           // binding count
        bindings.data(),                                     // bindings
    };

    VkDescriptorSetLayout layout;
    if (loader->vkCreateDescriptorSetLayout(device, &createInfo, nullptr, &layout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout");
    }

    return layout;
}

VkPipelineLayout TensorCopyPipeline::createPipelineLayout() const {
    const VkPushConstantRange pushConstantRange = {
        VK_SHADER_STAGE_COMPUTE_BIT, // flags
        0,                           // offset
        sizeof(pushConstant)         // size
    };

    const VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, // type
        nullptr,                                       // next
        0,                                             // flags
        1,                                             // layout count
        &descriptorSetLayout,                          // layout
        1,                                             // push constant count
        &pushConstantRange,                            // push constants
    };

    VkPipelineLayout layout;
    if (loader->vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &layout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout");
    }

    return layout;
}

VkPipeline TensorCopyPipeline::createPipeline(const TensorPlane &srcPlane) const {
    const auto rank = static_cast<uint32_t>(srcPlane.layout.dimensions.size());
    const VkSpecializationMapEntry entry = {
        0,               // constantID
        0,               // offset
        sizeof(uint32_t) // size
    };

    const VkSpecializationInfo specInfo = {
        1,                // mapEntryCount
        &entry,           // pMapEntries
        sizeof(uint32_t), // dataSize
        &rank             // pData
    };

    const VkPipelineShaderStageCreateInfo pipelineShaderCreateInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, // type
        nullptr,                                             // next
        0,                                                   // flags
        VK_SHADER_STAGE_COMPUTE_BIT,                         // stage flag bits
        shaderModule,                                        // shader module
        "main",                                              // name
        &specInfo                                            // specialization info
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

    VkPipeline vkPipeline;
    if (loader->vkCreateComputePipelines(device, nullptr, 1, &computePipelineCreateInfo, nullptr, &vkPipeline) !=
        VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute pipelines");
    }

    return vkPipeline;
}

VkDescriptorSet TensorCopyPipeline::createDescriptorSet(const TensorPlane &srcPlane,
                                                        const TensorPlane &dstPlane) const {
    const VkDescriptorSetAllocateInfo allocateInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, // type
        nullptr,                                        // next
        descriptorPool,                                 // descriptor pool
        1,                                              // descriptor set layout count
        &descriptorSetLayout,                           // descriptor set layouts
    };

    VkDescriptorSet vkDescriptorSet;
    if (loader->vkAllocateDescriptorSets(device, &allocateInfo, &vkDescriptorSet) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor set");
    }

    struct VkDescriptorBufferInfo srcBufferInfo = {
        srcPlane.buffer, // buffer
        0,               // offset
        VK_WHOLE_SIZE,   // range
    };

    struct VkDescriptorBufferInfo dstBufferInfo = {
        dstPlane.buffer, // buffer
        0,               // offset
        VK_WHOLE_SIZE,   // range
    };

    const std::vector<VkWriteDescriptorSet> write = {
        {
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, // type
            nullptr,                                // next
            vkDescriptorSet,                        // set
            0,                                      // binding
            0,                                      // array index
            1,                                      // descriptor count
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,      // descriptor type
            nullptr,                                // image info
            &srcBufferInfo,                         // buffer info
            nullptr,                                // texel buffer views
        },
        {
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, // type
            nullptr,                                // next
            vkDescriptorSet,                        // set
            1,                                      // binding
            0,                                      // array index
            1,                                      // descriptor count
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,      // descriptor type
            nullptr,                                // image info
            &dstBufferInfo,                         // buffer info
            nullptr,                                // texel buffer views
        },
    };

    loader->vkUpdateDescriptorSets(device, uint32_t(write.size()), write.data(), 0, nullptr);

    return vkDescriptorSet;
}

void TensorCopyPipeline::cmdBindAndDispatchCopy(VkCommandBuffer cmd, uint32_t regionCount) {
    loader->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    loader->vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstant),
                               &pushConstant);
    loader->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0,
                                    nullptr);
    loader->vkCmdDispatch(cmd, divideRoundUp(regionCount, warp1D), 1, 1);
}

void TensorCommandBuffer::resetRecording() {
    CommandBuffer::resetRecording();
    m_copyPipelines.clear();
}

void TensorCommandBuffer::retainCopyPipeline(std::unique_ptr<TensorCopyPipeline> pipeline) {
    m_copyPipelines.push_back(std::move(pipeline));
}

const std::string TensorCopyPipeline::glsl =
#include "shaders/copy.comp"
    ;

} // namespace mlsdk::el::layer
