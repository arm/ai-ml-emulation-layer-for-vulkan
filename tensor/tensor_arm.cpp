/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */
#include "tensor_arm.hpp"

#include "mlel/utils.hpp"

#include <numeric>
#include <regex>
#include <vulkan/vulkan_format_traits.hpp>

using namespace mlsdk::el::utils;

namespace mlsdk::el::layer {

TensorARM::TensorInfo::TensorInfo(const VkTensorCreateInfoARM &createInfo) {
    const VkTensorDescriptionARM &desc = *createInfo.pDescription;
    format = desc.format;
    if (desc.dimensionCount <= 0 || desc.dimensionCount > TENSOR_MAX_DIMENSIONS)
        throw std::runtime_error(std::string("Tensor dimension count not supported: ") +
                                 std::to_string(desc.dimensionCount));
    isOptimalTilingAliasing =
        (desc.usage & VK_TENSOR_USAGE_IMAGE_ALIASING_BIT_ARM) && desc.tiling == VK_TENSOR_TILING_OPTIMAL_ARM;
    usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
#ifdef EXPERIMENTAL_MOLTEN_VK_SUPPORT
    usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
#endif
    if (desc.usage & VK_TENSOR_USAGE_SHADER_BIT_ARM) {
        usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    }
    if (desc.usage & VK_TENSOR_USAGE_TRANSFER_SRC_BIT_ARM) {
        usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    }
    if (desc.usage & VK_TENSOR_USAGE_TRANSFER_DST_BIT_ARM) {
        usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    }
    if (desc.usage & VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM) {
        usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    }

    flags = (createInfo.flags & VK_TENSOR_CREATE_DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT_ARM)
                ? VK_BUFFER_CREATE_DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT_EXT
                : 0;

    uint32_t dimensionCount = desc.dimensionCount;
    dimensions = std::vector<int64_t>{desc.pDimensions, desc.pDimensions + dimensionCount};
    elementSize = vk::blockSize(vk::Format(desc.format));
    if (desc.pStrides == nullptr) {
        // calculate strides
        strides.resize(dimensionCount);
        strides.back() = static_cast<int64_t>(elementSize);
        for (size_t i = dimensionCount - 1; i > 0; --i) {
            strides[i - 1] = strides[i] * static_cast<int64_t>(dimensions[i]);
        }
    } else {
        strides = std::vector<int64_t>{desc.pStrides, desc.pStrides + dimensionCount};
    }
    size = static_cast<size_t>(dimensions[0] * strides[0]);
}

VkResult TensorARM::create(const Device &dev, const VkTensorCreateInfoARM &createInfo,
                           const VkAllocationCallbacks *allocator) {
    m_info = TensorInfo(createInfo);
    auto pCaptureDescriptorInfo = findType<VkOpaqueCaptureDescriptorDataCreateInfoEXT>(
        createInfo.pNext, VK_STRUCTURE_TYPE_OPAQUE_CAPTURE_DESCRIPTOR_DATA_CREATE_INFO_EXT);
    const VkBufferCreateInfo bufferInfo = {
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, // type
        pCaptureDescriptorInfo,               // next
        m_info.flags,                         // flags
        m_info.size,                          // size
        m_info.usage,                         // usage flags
        createInfo.sharingMode,               // sharing
        createInfo.queueFamilyIndexCount,     // queue family index count
        createInfo.pQueueFamilyIndices,       // queue family index

    };
    return dev.loader->vkCreateBuffer(dev.device, &bufferInfo, allocator, &m_tensorBuffer);
}

void TensorARM::destroy(const Device &dev, const VkAllocationCallbacks *pAllocator) {
    dev.loader->vkDestroyBuffer(dev.device, m_tensorBuffer, pAllocator);
}

void TensorARM::getMemoryRequirements(const Device &dev, VkMemoryRequirements *requirements) const {
    dev.loader->vkGetBufferMemoryRequirements(dev.device, m_tensorBuffer, requirements);
}

void TensorARM::getDeviceTensorMemoryRequirements(const Device &dev, const VkTensorCreateInfoARM &createInfo,
                                                  VkMemoryRequirements2 *requirements) {
    TensorInfo info(createInfo);
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

VkResult TensorARM::bindTensorMemory(const Device &dev, VkDeviceMemory memory, VkDeviceSize offset) {
    return dev.loader->vkBindBufferMemory(dev.device, m_tensorBuffer, memory, offset);
}

void TensorARM::updateAliasedTensorInfo(const Device &dev, VkImage image) {
    auto rank = m_info.dimensions.size();
    if (rank < 2 || rank > 4) {
        throw std::runtime_error("Tensor rank should be 2, 3 or 4 to alias with image.");
    }

    if (m_info.isOptimalTilingAliasing) {
        const VkImageSubresource imageSubresource = {
            VK_IMAGE_ASPECT_COLOR_BIT,
            0,
            0,
        };
        VkSubresourceLayout imageSubresourceLayout{};
        dev.loader->vkGetImageSubresourceLayout(dev.device, image, &imageSubresource, &imageSubresourceLayout);
        if (rank == 4) {
            // alias to 3D image
            m_info.strides[0] = static_cast<int64_t>(imageSubresourceLayout.depthPitch);
            m_info.strides[1] = static_cast<int64_t>(imageSubresourceLayout.rowPitch);
        } else if (rank == 3) {
            // alias to 2D image
            m_info.strides[0] = static_cast<int64_t>(imageSubresourceLayout.rowPitch);
        }
    }
}

void TensorARM::copyToTensor(CommandBuffer &cmd, const TensorARM &dstTensor) {
    const auto &srcDimensions = m_info.dimensions;
    const auto &srcStrides = m_info.strides;
    const auto srcElementSize = m_info.elementSize;
    const auto &dstDimensions = dstTensor.m_info.dimensions;
    const auto &dstStrides = dstTensor.m_info.strides;
    const auto dstElementSize = dstTensor.m_info.elementSize;

    if (srcDimensions.size() != dstDimensions.size() || srcElementSize != dstElementSize ||
        !std::equal(srcDimensions.begin(), srcDimensions.end(), dstDimensions.begin())) {
        throw std::runtime_error("Src tensor and dst tensor should have same dimensions and element size.");
    }

    if (std::equal(srcStrides.begin(), srcStrides.end(), dstStrides.begin())) {
        VkBufferCopy copyInfo = {0, 0, m_info.size};
        cmd.loader->vkCmdCopyBuffer(cmd.commandBuffer, getTensorBuffer(), dstTensor.getTensorBuffer(), 1, &copyInfo);
    } else {
        const auto regionCount = static_cast<uint32_t>(std::abs(
            std::accumulate(srcDimensions.begin(), srcDimensions.end(), int64_t(1), std::multiplies<int64_t>())));
        m_copy_pipeline = std::make_shared<TensorCopyPipeline>(cmd.loader, cmd.device->device, *this, dstTensor);
        cmd.beginSecondaryCommandBuffer();
        assert(regionCount < std::numeric_limits<uint32_t>::max());
        m_copy_pipeline->cmdBindAndDispatchCopy(cmd.secondaryCommandBuffer, regionCount);
        cmd.endAndSubmitSecondaryCommandBuffer();
    }
}

VkResult TensorARM::getOpaqueCaptureDescriptorDataEXT(const Device &dev, void *pData) {
    const VkBufferCaptureDescriptorDataInfoEXT info = {
        VK_STRUCTURE_TYPE_BUFFER_CAPTURE_DESCRIPTOR_DATA_INFO_EXT,
        nullptr,
        m_tensorBuffer,
    };
    return dev.loader->vkGetBufferOpaqueCaptureDescriptorDataEXT(dev.device, &info, pData);
}

TensorCopyPipeline::TensorCopyPipeline(
    const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
    const TensorARM &srcTensor, const TensorARM &dstTensor)
    : loader{_loader}, device{_device}, pushConstant{createPushConstant(srcTensor, dstTensor)},
      shaderModule{createShaderModule(srcTensor)}, descriptorPool{createDescriptorPool()},
      descriptorSetLayout{createDescriptorSetLayout()}, pipelineLayout{createPipelineLayout()},
      pipeline{createPipeline(srcTensor)}, descriptorSet{createDescriptorSet(srcTensor, dstTensor)} {}

TensorCopyPipeline::~TensorCopyPipeline() {
    loader->vkFreeDescriptorSets(device, descriptorPool, 1, &descriptorSet);
    loader->vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    loader->vkDestroyPipeline(device, pipeline, nullptr);
    loader->vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    loader->vkDestroyShaderModule(device, shaderModule, nullptr);
    loader->vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
}

TensorCopyPipeline::PushConstant TensorCopyPipeline::createPushConstant(const TensorARM &srcTensor,
                                                                        const TensorARM &dstTensor) const {
    PushConstant constant;
    const auto &dimensions = srcTensor.getTensorInfo().dimensions;
    const auto &srcStrides = srcTensor.getTensorInfo().strides;
    const auto &dstStrides = dstTensor.getTensorInfo().strides;
    std::copy(dimensions.begin(), dimensions.end(), constant.dimensions);
    std::copy(srcStrides.begin(), srcStrides.end(), constant.srcStrides);
    std::copy(dstStrides.begin(), dstStrides.end(), constant.dstStrides);
    return constant;
}

VkShaderModule TensorCopyPipeline::createShaderModule(const TensorARM &srcTensor) const {
    const auto srcType = makeFormat((srcTensor.getTensorInfo().format))->glslType();
    const std::lock_guard lock(cacheMutex);
    auto &spirv = spirvCache[srcType];
    if (spirv.empty()) {
        std::string tmp = glsl;
        tmp = std::regex_replace(tmp, std::regex{"%warpX%"}, std::to_string(warp1D));
        tmp = std::regex_replace(tmp, std::regex{"%type%"}, srcType);
        tmp = std::regex_replace(tmp, std::regex{"%type_size%"},
                                 std::to_string(vk::blockSize(vk::Format(srcTensor.getTensorInfo().format))));
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

VkPipeline TensorCopyPipeline::createPipeline(const TensorARM &srcTensor) const {
    uint32_t rank = uint32_t(srcTensor.getTensorInfo().dimensions.size());
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

VkDescriptorSet TensorCopyPipeline::createDescriptorSet(const TensorARM &srcTensor, const TensorARM &dstTensor) const {
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
        srcTensor.getTensorBuffer(), // buffer
        0,                           // offset
        VK_WHOLE_SIZE,               // range
    };

    struct VkDescriptorBufferInfo dstBufferInfo = {
        dstTensor.getTensorBuffer(), // buffer
        0,                           // offset
        VK_WHOLE_SIZE,               // range
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

const std::string TensorCopyPipeline::glsl =
#include "shaders/copy.comp"
    ;

} // namespace mlsdk::el::layer
