/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "compute_optical_flow.hpp"
#include "compute_pipeline_common.hpp"

using namespace mlsdk::el::utils;

namespace mlsdk::el::compute::optical_flow {

/*******************************************************************************
 * ComputePipeline
 *******************************************************************************/

ComputePipeline::ComputePipeline(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &loader,
                                 const VkDevice device, const std::shared_ptr<PipelineCache> &pipelineCache,
                                 const SpirvBinary spirv, const DescriptorConfigs &descriptorConfigs,
                                 const SpecConstants &specConstants, uint32_t pushConstantsSize,
                                 const ScheduleHelper &schedule, const std::string &debugName)
    : loader_(loader), device_(device), pipelineCache_(pipelineCache), spirv_(spirv),
      descriptorConfigs_(descriptorConfigs), specConstants_(specConstants), pushConstantsSize_(pushConstantsSize),
      scheduler_(schedule), debugName_(debugName) {}

ComputePipeline::~ComputePipeline() {
    if (device_) {
        if (vkPipeline_) {
            loader_->vkDestroyPipeline(device_, vkPipeline_, pAllocator_);
        }
        if (pipelineLayout_) {
            loader_->vkDestroyPipelineLayout(device_, pipelineLayout_, pAllocator_);
        }
        if (descriptorSetLayout_) {
            loader_->vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, pAllocator_);
        }
        if (descriptorPool_) {
            loader_->vkDestroyDescriptorPool(device_, descriptorPool_, pAllocator_);
        }
        for (auto *sampler : samplers_) {
            loader_->vkDestroySampler(device_, sampler, pAllocator_);
        }
    }
};

const std::string &ComputePipeline::getDebugName() const { return debugName_; }

void ComputePipeline::makePipeline() {
    // Shader module
    const VkShaderModule shaderModule = common::createShaderModule(loader_, device_, spirv_, pAllocator_);

    // Create descriptor pool
    // TODO: maybe use descriptor buffers so we can use session memory instead of seperate allocations?
    const std::vector<VkDescriptorPoolSize> poolSizes = {
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 5}, {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 5},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 5}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5},
        {VK_DESCRIPTOR_TYPE_SAMPLER, 5},
    };

    const VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        nullptr,
        0, // flags
        3, // maxSets
        static_cast<uint32_t>(poolSizes.size()),
        poolSizes.data(),
    };
    VkResult res = loader_->vkCreateDescriptorPool(device_, &descriptorPoolCreateInfo, pAllocator_, &descriptorPool_);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor pool");
    }

    // Create descriptor set layout
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.reserve(descriptorConfigs_.size());
    for (const auto &dconf : descriptorConfigs_) {
        bindings.emplace_back(VkDescriptorSetLayoutBinding{
            dconf.index,
            dconf.type,
            /* descriptorCount */ 1,
            VK_SHADER_STAGE_COMPUTE_BIT,
            nullptr,
        });
    }

    const VkDescriptorSetLayoutCreateInfo dsetLayoutCreateInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        nullptr,
        0,
        static_cast<uint32_t>(bindings.size()),
        bindings.data(),
    };
    res = loader_->vkCreateDescriptorSetLayout(device_, &dsetLayoutCreateInfo, pAllocator_, &descriptorSetLayout_);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout");
    }

    // Allocate descriptor sets
    const VkDescriptorSetAllocateInfo dsetAllocInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descriptorPool_, 1, &descriptorSetLayout_,
    };
    res = loader_->vkAllocateDescriptorSets(device_, &dsetAllocInfo, &descriptorSet_);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor sets");
    }

    // Create pipeline layout
    const VkPushConstantRange pushConstantRange = {
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        pushConstantsSize_,
    };

    const VkPipelineLayoutCreateInfo pipeLayoutCreateInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,         // sType
        nullptr,                                               // pNext
        0,                                                     // flags
        1,                                                     // setLayoutCount
        &descriptorSetLayout_,                                 // pSetLayouts
        pushConstantsSize_ > 0 ? 1u : 0u,                      // pushConstantRangeCount
        pushConstantsSize_ > 0 ? &pushConstantRange : nullptr, // pPushConstantRanges
    };
    res = loader_->vkCreatePipelineLayout(device_, &pipeLayoutCreateInfo, pAllocator_, &pipelineLayout_);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout");
    }

    // Create compute pipeline
    const auto specializationConstants =
        common::makeSpecializationConstantsView(specConstants_.pointer, specConstants_.sizeBytes);
    const auto *specialization = specConstants_.sizeBytes > 0 ? &specializationConstants : nullptr;
    vkPipeline_ = common::createComputePipeline(loader_, device_, pipelineCache_->getPipelineCache(), shaderModule,
                                                pipelineLayout_, specialization, pAllocator_);

    setDebugUtilsObjectName(loader_, device_, VK_OBJECT_TYPE_PIPELINE, reinterpret_cast<uint64_t>(vkPipeline_),
                            debugName_);

    // clean up
    loader_->vkDestroyShaderModule(device_, shaderModule, nullptr);
}

void ComputePipeline::setInputStorage(VkCommandBuffer cmdBuf, uint32_t binding, const std::shared_ptr<Image> &image,
                                      VkSampler sampler) {
    image->makeBarrier(cmdBuf, Image::BarrierState::ShaderRead);

    if (image->isBufferLoad()) {
        setBuffer(binding, image);
    } else {
        setCombinedImageSampler(binding, image, sampler);
    }
}

void ComputePipeline::setOutputStorage(VkCommandBuffer cmdBuf, uint32_t binding, const std::shared_ptr<Image> &image) {
    image->makeBarrier(cmdBuf, Image::BarrierState::ShaderWrite);

    if (image->isBufferStore()) {
        setBuffer(binding, image);
    } else {
        setOutputImage(binding, image);
    }
}

void ComputePipeline::setCombinedImageSampler(uint32_t binding, const std::shared_ptr<Image> &image,
                                              VkSampler sampler) {
    setImage(binding, image, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, sampler);
}

void ComputePipeline::setOutputImage(uint32_t binding, const std::shared_ptr<Image> &image) {
    setImage(binding, image, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_NULL_HANDLE);
}

void ComputePipeline::setImage(uint32_t binding, const std::shared_ptr<Image> &image, VkDescriptorType descriptorType,
                               VkSampler sampler) {
    const VkDescriptorImageInfo imageInfo = {sampler, image->getImageView(), image->getImageLayout()};

    const VkWriteDescriptorSet writeDescriptorSet = {
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        nullptr,
        descriptorSet_,
        binding,
        0, // dstArrayElement
        1, // descriptorCount
        descriptorType,
        &imageInfo, // pImageInfo
        nullptr,
        nullptr, // pTexelBufferView
    };
    loader_->vkUpdateDescriptorSets(device_, 1, &writeDescriptorSet, 0, nullptr);
}

void ComputePipeline::setBuffer(uint32_t binding, const std::shared_ptr<Image> &image) {
    assert(image->isInternal());
    const VkDescriptorBufferInfo bufferInfo = {image->getBuffer(), 0, VK_WHOLE_SIZE};
    const VkWriteDescriptorSet writeDescriptorSet = {
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        nullptr,
        descriptorSet_,
        binding,
        0, // dstArrayElement
        1, // descriptorCount
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        nullptr, // pImageInfo
        &bufferInfo,
        nullptr, // pTexelBufferView
    };
    loader_->vkUpdateDescriptorSets(device_, 1, &writeDescriptorSet, 0, nullptr);
}

VkSampler ComputePipeline::createSampler(VkFilter filter, VkSamplerAddressMode addressMode,
                                         bool unnormalizedCoordinates = true) {
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.minFilter = filter;
    samplerInfo.magFilter = filter;
    samplerInfo.addressModeU = addressMode;
    samplerInfo.addressModeV = addressMode;
    samplerInfo.addressModeW = addressMode;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.unnormalizedCoordinates = unnormalizedCoordinates;

    VkSampler sampler;
    VkResult res = loader_->vkCreateSampler(device_, &samplerInfo, nullptr, &sampler);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("Failed to create sampler");
    }
    samplers_.push_back(sampler);
    return sampler;
}

void ComputePipeline::bindPipeline(VkCommandBuffer cmdBuf) {
    loader_->vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, vkPipeline_);
    loader_->vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_, 0, 1, &descriptorSet_, 0,
                                     nullptr);
}

void ComputePipeline::dispatchPipeline(VkCommandBuffer cmdBuf) {
    loader_->vkCmdDispatch(cmdBuf, scheduler_.groupCountX, scheduler_.groupCountY, scheduler_.groupCountZ);
}

/*******************************************************************************
 * RGBToY
 *******************************************************************************/

RGBToY::RGBToY(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &loader,
               const VkDevice device, const std::shared_ptr<PipelineCache> &pipelineCache,
               std::shared_ptr<Image> srcRGBImage, const std::shared_ptr<Image> &dstDownsampledImage,
               std::shared_ptr<Image> dstFullImage, bool outputFullRes, float downsampleScale,
               const std::string &debugName)
    : ComputePipeline(loader, device, pipelineCache,
                      createSpirv(pipelineCache, outputFullRes, dstDownsampledImage->isImageStore()),
                      descriptorConfigs_, {&specConstants_, sizeof(specConstants_)}, 0,
                      {dstDownsampledImage->width(), dstDownsampledImage->height()}, debugName),
      srcImage_(std::move(srcRGBImage)), dstYDownsampled_(dstDownsampledImage), dstYFull_(std::move(dstFullImage)),
      outputFull_(outputFullRes), specConstants_{makeSpecConstants(downsampleScale)},
      linearSampler_{createSampler(VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)} {
    assert(!outputFull_ || dstYFull_);
    assert(!outputFull_ || dstYDownsampled_->isImageStore() == dstYFull_->isImageStore());
}

SpirvBinary RGBToY::createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache, bool outputFull, bool imageStore) {
    return pipelineCache->lookup(makeShaderName(outputFull, imageStore), {});
}

std::string RGBToY::makeShaderName(bool outputFull, bool imageStore) {
    std::string shaderName{shaderBaseName};
    if (outputFull) {
        shaderName += "_full";
    }
    shaderName += imageStore ? "_img" : "_buf";

    return shaderName;
}

RGBToY::SpecConstants RGBToY::makeSpecConstants(float downsampleScale) const {
    const VkBool32 isLumaInput = srcImage_->componentCount() == 1;

    SpecConstants specConstants = {
        32,
        8,
        isLumaInput,
        downsampleScale,
        downsampleScale,
        dstYDownsampled_->width(),
        dstYDownsampled_->height(),
        dstYDownsampled_->stride(),
        outputFull_ ? dstYFull_->width() : 1,
        outputFull_ ? dstYFull_->height() : 1,
        outputFull_ ? dstYFull_->stride() : 1,
    };
    return specConstants;
}

void RGBToY::setInput(std::shared_ptr<Image> src) {
    assert(Image::isCompatible(srcImage_, src));
    srcImage_ = std::move(src);
}

void RGBToY::bindAndDispatch(VkCommandBuffer cmdBuf) {
    setInputStorage(cmdBuf, 0, srcImage_, linearSampler_);
    setOutputStorage(cmdBuf, dstYDownsampled_->isImageStore() ? 1 : 2, dstYDownsampled_);
    if (outputFull_) {
        setOutputStorage(cmdBuf, 3, dstYFull_);
    }

    bindPipeline(cmdBuf);
    dispatchPipeline(cmdBuf);
}

/*******************************************************************************
 * Downsample
 *******************************************************************************/

Downsample::Downsample(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &loader,
                       const VkDevice device, const std::shared_ptr<PipelineCache> &pipelineCache,
                       std::shared_ptr<Image> src, const std::shared_ptr<Image> &dst, const std::string &debugName)
    : ComputePipeline(loader, device, pipelineCache, createSpirv(pipelineCache), descriptorConfigs_,
                      {&specConstants_, sizeof(specConstants_)}, 0, {dst->width(), dst->height()}, debugName),
      srcImage_(std::move(src)), dstImage_(dst), specConstants_{makeSpecConstants()},
      linearSampler_{createSampler(VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)} {}

SpirvBinary Downsample::createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache) {
    return pipelineCache->lookup(shaderName, {});
}

Downsample::SpecConstants Downsample::makeSpecConstants() const {
    assert(dstImage_->isImageStore());

    SpecConstants specConstants = {
        32,
        8,
        srcImage_->width() != dstImage_->width() * 2,
        srcImage_->height() != dstImage_->height() * 2,
        dstImage_->width(),
        dstImage_->height(),
    };
    return specConstants;
}

void Downsample::bindAndDispatch(VkCommandBuffer cmdBuf) {
    setInputStorage(cmdBuf, 0, srcImage_, linearSampler_);
    setOutputStorage(cmdBuf, 1, dstImage_);

    bindPipeline(cmdBuf);
    dispatchPipeline(cmdBuf);
}

/*******************************************************************************
 * MVProcessAndWarp
 *******************************************************************************/

MVProcessAndWarp::MVProcessAndWarp(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &loader,
                                   const VkDevice device, const std::shared_ptr<PipelineCache> &pipelineCache,
                                   std::shared_ptr<Image> srcImage, std::shared_ptr<Image> srcFlow,
                                   const std::shared_ptr<Image> &dstImage, std::shared_ptr<Image> dstFlow,
                                   const std::string &debugName)
    : ComputePipeline(loader, device, pipelineCache, createSpirv(pipelineCache), descriptorConfigs_,
                      {&specConstants_, sizeof(specConstants_)}, 0, {dstImage->width(), dstImage->height()}, debugName),
      srcSearch_(std::move(srcImage)), srcFlow_(std::move(srcFlow)), dstWarped_(dstImage),
      // Output flow and specialization state.
      dstFlow_(std::move(dstFlow)), specConstants_{makeSpecConstants()},
      linearSampler_{createSampler(VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)} {}

SpirvBinary MVProcessAndWarp::createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache) {
    return pipelineCache->lookup(shaderName, {});
}

MVProcessAndWarp::SpecConstants MVProcessAndWarp::makeSpecConstants() const {
    assert(dstWarped_->isBufferStore());
    assert(dstFlow_->isBufferStore());

    SpecConstants specConstants = {
        32, 8, 2.0f, 2.0f, 0.5f, 0.5f, dstFlow_->width(), dstFlow_->height(), dstWarped_->stride(), dstFlow_->stride(),
    };
    return specConstants;
}

void MVProcessAndWarp::bindAndDispatch(VkCommandBuffer cmdBuf) {
    setInputStorage(cmdBuf, 0, srcSearch_, linearSampler_);
    setInputStorage(cmdBuf, 1, srcFlow_, linearSampler_);
    setOutputStorage(cmdBuf, 2, dstWarped_);
    setOutputStorage(cmdBuf, 3, dstFlow_);

    bindPipeline(cmdBuf);
    dispatchPipeline(cmdBuf);
}

/*******************************************************************************
 * DenseWarp
 *******************************************************************************/

DenseWarp::DenseWarp(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &loader,
                     const VkDevice device, const std::shared_ptr<PipelineCache> &pipelineCache,
                     std::shared_ptr<Image> srcImage, std::shared_ptr<Image> srcFlow,
                     const std::shared_ptr<Image> &dstImage, float inputFlowScale, const std::string &debugName)
    : ComputePipeline(loader, device, pipelineCache, createSpirv(pipelineCache), descriptorConfigs_,
                      {&specConstants_, sizeof(specConstants_)}, 0, {dstImage->width(), dstImage->height()}, debugName),
      srcSearch_(std::move(srcImage)), srcFlow_(std::move(srcFlow)),
      // Output image and specialization state.
      dstWarped_(dstImage), specConstants_{makeSpecConstants(inputFlowScale)},
      linearSampler_{createSampler(VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)},
      nearestSampler_{createSampler(VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)} {}

SpirvBinary DenseWarp::createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache) {
    return pipelineCache->lookup(shaderName, {});
}

DenseWarp::SpecConstants DenseWarp::makeSpecConstants(float inputFlowScale) const {
    assert(dstWarped_->isImageStore());

    SpecConstants specConstants = {
        32, 8, inputFlowScale, dstWarped_->width(), dstWarped_->height(),
    };
    return specConstants;
}

void DenseWarp::setInputFlow(std::shared_ptr<Image> srcFlow) {
    assert(Image::isCompatible(srcFlow, srcFlow_));
    srcFlow_ = std::move(srcFlow);
}

void DenseWarp::bindAndDispatch(VkCommandBuffer cmdBuf) {
    setInputStorage(cmdBuf, 0, srcSearch_, linearSampler_);
    setInputStorage(cmdBuf, 1, srcFlow_, nearestSampler_);
    setOutputStorage(cmdBuf, 2, dstWarped_);

    bindPipeline(cmdBuf);
    dispatchPipeline(cmdBuf);
}

/*******************************************************************************
 * MedianFilter
 *******************************************************************************/

MedianFilter::MedianFilter(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &loader,
                           const VkDevice device, const std::shared_ptr<PipelineCache> &pipelineCache,
                           std::shared_ptr<Image> srcImage, const std::shared_ptr<Image> &dstImage,
                           float outputFlowScale, const std::string &debugName)
    : ComputePipeline(loader, device, pipelineCache, createSpirv(pipelineCache), descriptorConfigs_,
                      {&specConstants_, sizeof(specConstants_)}, 0, {dstImage->width(), dstImage->height()}, debugName),
      srcFlow_(std::move(srcImage)), dstFlow_(dstImage), specConstants_{makeSpecConstants(outputFlowScale)},
      nearestSampler_{createSampler(VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)} {}

SpirvBinary MedianFilter::createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache) {
    return pipelineCache->lookup(shaderName, {});
}

MedianFilter::SpecConstants MedianFilter::makeSpecConstants(float outputFlowScale) const {
    assert(dstFlow_->isImageStore());

    SpecConstants specConstants = {
        32, 8, outputFlowScale, dstFlow_->width(), dstFlow_->height(),
    };
    return specConstants;
}

void MedianFilter::setOutput(std::shared_ptr<Image> dstImage) {
    assert(Image::isCompatible(dstFlow_, dstImage));
    dstFlow_ = std::move(dstImage);
}

void MedianFilter::bindAndDispatch(VkCommandBuffer cmdBuf) {
    setInputStorage(cmdBuf, 0, srcFlow_, nearestSampler_);
    setOutputStorage(cmdBuf, 1, dstFlow_);

    bindPipeline(cmdBuf);
    dispatchPipeline(cmdBuf);
}

/*******************************************************************************
 * BilateralFilter
 *******************************************************************************/

BilateralFilter::BilateralFilter(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &loader,
                                 const VkDevice device, const std::shared_ptr<PipelineCache> &pipelineCache,
                                 std::shared_ptr<Image> srcImage, std::shared_ptr<Image> srcFlow,
                                 const std::shared_ptr<Image> &dstFlow, float outputFlowScale,
                                 const std::string &debugName)
    : ComputePipeline(loader, device, pipelineCache, createSpirv(pipelineCache, dstFlow->isImageStore()),
                      descriptorConfigs_, {&specConstants_, sizeof(specConstants_)}, 0,
                      {dstFlow->width(), dstFlow->height()}, debugName),
      srcTemplate_(std::move(srcImage)), srcFlow_(std::move(srcFlow)),
      // Output flow and specialization state.
      dstFlow_(dstFlow), specConstants_{makeSpecConstants(outputFlowScale)},
      nearestSampler_{createSampler(VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)} {}

SpirvBinary BilateralFilter::createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache, bool imageStore) {
    return pipelineCache->lookup(makeShaderName(imageStore), {});
}

std::string BilateralFilter::makeShaderName(bool imageStore) {
    std::string shaderName{shaderBaseName};
    shaderName += imageStore ? "_img" : "_buf";
    return shaderName;
}

BilateralFilter::SpecConstants BilateralFilter::makeSpecConstants(float outputFlowScale) const {
    assert(dstFlow_->isImageStore() || dstFlow_->isBufferStore());
    SpecConstants specConstants = {
        32, 8, outputFlowScale, dstFlow_->width(), dstFlow_->height(), dstFlow_->stride(),
    };
    return specConstants;
}

void BilateralFilter::setOutput(std::shared_ptr<Image> dstFlow) {
    assert(Image::isCompatible(dstFlow, dstFlow_));
    dstFlow_ = std::move(dstFlow);
}

void BilateralFilter::bindAndDispatch(VkCommandBuffer cmdBuf) {
    setInputStorage(cmdBuf, 0, srcTemplate_, nearestSampler_);
    setInputStorage(cmdBuf, 1, srcFlow_, nearestSampler_);
    setOutputStorage(cmdBuf, dstFlow_->isImageStore() ? 2 : 3, dstFlow_);

    bindPipeline(cmdBuf);
    dispatchPipeline(cmdBuf);
}

/*******************************************************************************
 * SubpixelME
 *******************************************************************************/

SubpixelME::SubpixelME(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &loader,
                       const VkDevice device, const std::shared_ptr<PipelineCache> &pipelineCache,
                       std::shared_ptr<Image> srcImageSearch, std::shared_ptr<Image> srcImageTemplate,
                       std::shared_ptr<Image> srcFlow, std::shared_ptr<Image> prevLevelFlow,
                       const std::shared_ptr<Image> &dstFlow, bool doAccumulate, const std::string &debugName)
    : ComputePipeline(loader, device, pipelineCache, createSpirv(pipelineCache, doAccumulate), descriptorConfigs_,
                      {&specConstants_, sizeof(specConstants_)}, 0, {dstFlow->width(), dstFlow->height()}, debugName),
      srcSearch_(std::move(srcImageSearch)), srcTemplate_(std::move(srcImageTemplate)), srcFlow_(std::move(srcFlow)),
      srcPrevLevelFlow_(std::move(prevLevelFlow)), dstFlow_(dstFlow), doAccumulate_(doAccumulate),
      specConstants_{makeSpecConstants()},
      nearestZeroPadSampler_{createSampler(VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER)},
      nearestRepeatPadSampler_{createSampler(VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)} {}

SpirvBinary SubpixelME::createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache, bool doAccumulate) {
    return pipelineCache->lookup(makeShaderName(doAccumulate), {});
}

std::string SubpixelME::makeShaderName(bool doAccumulate) {
    std::string shaderName{shaderBaseName};
    shaderName += doAccumulate ? "_acc_buf" : "_buf";
    return shaderName;
}

SubpixelME::SpecConstants SubpixelME::makeSpecConstants() const {
    assert(srcFlow_->isBufferLoad());
    assert(dstFlow_->isBufferStore());
    assert(!doAccumulate_ || srcPrevLevelFlow_->isBufferLoad());

    SpecConstants specConstants = {
        32,
        8,
        dstFlow_->width(),
        dstFlow_->height(),
        srcFlow_->stride(),
        doAccumulate_ ? srcPrevLevelFlow_->stride() : 1,
        dstFlow_->stride(),
    };
    return specConstants;
}

void SubpixelME::bindAndDispatch(VkCommandBuffer cmdBuf) {
    setInputStorage(cmdBuf, 0, srcSearch_, nearestZeroPadSampler_);
    setInputStorage(cmdBuf, 1, srcTemplate_, nearestRepeatPadSampler_);
    setInputStorage(cmdBuf, 2, srcFlow_, nearestZeroPadSampler_);
    if (doAccumulate_) {
        setInputStorage(cmdBuf, 3, srcPrevLevelFlow_, nearestZeroPadSampler_);
    }
    setOutputStorage(cmdBuf, 4, dstFlow_);

    bindPipeline(cmdBuf);
    dispatchPipeline(cmdBuf);
}

/*******************************************************************************
 * MVReplace
 *******************************************************************************/

MVReplace::MVReplace(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &loader,
                     const VkDevice device, const std::shared_ptr<PipelineCache> &pipelineCache,
                     std::shared_ptr<Image> mvInput, std::shared_ptr<Image> flowBlockMatch,
                     std::shared_ptr<Image> costAtInput, std::shared_ptr<Image> minCostBlockMatch,
                     const std::shared_ptr<Image> &dstFlow, std::shared_ptr<Image> dstCost, bool outputCost,
                     const std::string &debugName)
    : ComputePipeline(loader, device, pipelineCache, createSpirv(pipelineCache, outputCost), descriptorConfigs_,
                      {&specConstants_, sizeof(specConstants_)}, 0, {dstFlow->width(), dstFlow->height()}, debugName),
      srcInputMV_(std::move(mvInput)), srcBlockMatchFlow_(std::move(flowBlockMatch)),
      srcInputMVCost_(std::move(costAtInput)), srcBlockMatchCost_(std::move(minCostBlockMatch)), dstFlow_(dstFlow),
      dstCost_(std::move(dstCost)), outputCost_(outputCost), specConstants_{makeSpecConstants()},
      nearestSampler_{createSampler(VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)} {}

SpirvBinary MVReplace::createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache, bool outputCost) {
    return pipelineCache->lookup(makeShaderName(outputCost), {});
}

std::string MVReplace::makeShaderName(bool outputCost) {
    std::string shaderName{shaderBaseName};
    if (outputCost) {
        shaderName += "_cost";
    }
    shaderName += "_img";
    return shaderName;
}

MVReplace::SpecConstants MVReplace::makeSpecConstants() const {
    assert(srcInputMV_->isImageSample());
    assert(srcBlockMatchFlow_->isImageSample());
    assert(srcInputMVCost_->isImageSample());
    assert(srcBlockMatchCost_->isImageSample());
    assert(dstFlow_->isImageStore());
    assert(!outputCost_ || (dstCost_ && dstCost_->isImageStore()));

    SpecConstants specConstants = {
        32,
        8,
        dstFlow_->width(),
        dstFlow_->height(),
    };
    return specConstants;
}

void MVReplace::setInputMv(std::shared_ptr<Image> srcMV) {
    assert(Image::isCompatible(srcInputMV_, srcMV));
    srcInputMV_ = std::move(srcMV);
}

void MVReplace::setOutputFlow(std::shared_ptr<Image> dstFlow) {
    assert(Image::isCompatible(dstFlow, dstFlow_));
    dstFlow_ = std::move(dstFlow);
}

void MVReplace::setOutputCost(std::shared_ptr<Image> dstCost) {
    assert(Image::isCompatible(dstCost, dstCost_));
    dstCost_ = std::move(dstCost);
}

void MVReplace::bindAndDispatch(VkCommandBuffer cmdBuf) {
    setInputStorage(cmdBuf, 0, srcInputMV_, nearestSampler_);
    setInputStorage(cmdBuf, 1, srcBlockMatchFlow_, nearestSampler_);
    setInputStorage(cmdBuf, 2, srcInputMVCost_, nearestSampler_);
    setInputStorage(cmdBuf, 3, srcBlockMatchCost_, nearestSampler_);
    setOutputStorage(cmdBuf, 4, dstFlow_);

    if (outputCost_) {
        setOutputStorage(cmdBuf, 5, dstCost_);
    }

    bindPipeline(cmdBuf);
    dispatchPipeline(cmdBuf);
}

/*******************************************************************************
 * BlockMatch
 *******************************************************************************/

BlockMatch::BlockMatch(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &loader,
                       const VkDevice device, const std::shared_ptr<PipelineCache> &pipelineCache,
                       SearchType searchType, int32_t maxSearchRange, const std::shared_ptr<Image> &srcSearch,
                       std::shared_ptr<Image> srcTemplate, std::shared_ptr<Image> dstFlow,
                       std::shared_ptr<Image> dstCost, const std::string &debugName)
    : ComputePipeline(loader, device, pipelineCache,
                      createSpirv(pipelineCache, searchType, dstCost && dstCost->isImageStore()), descriptorConfigs_,
                      {&specConstants_, sizeof(specConstants_)}, sizeof(PushConstants),
                      {srcSearch->width(), srcSearch->height()}, debugName),
      srcSearch_(srcSearch), srcTemplate_(std::move(srcTemplate)), dstFlow_(std::move(dstFlow)),
      dstCost_(std::move(dstCost)), searchType_(searchType), maxSearchRange_(maxSearchRange),
      searchRangeLimit_(maxSearchRange), specConstants_{makeSpecConstants()},
      nearestSampler_{createSampler(VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER)} {
    // MIN_SAD* assumes maxSearchRange > 0
    assert(searchType_ == SearchType::RAW_SAD || maxSearchRange_ != 0);
    // RAW_SAD supports only in case maxSearchRange = 0
    assert(searchType_ != SearchType::RAW_SAD || maxSearchRange_ == 0);
}

SpirvBinary BlockMatch::createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache, SearchType searchType,
                                    bool costImageStore) {
    return pipelineCache->lookup(makeShaderName(searchType, costImageStore), {});
}

std::string BlockMatch::makeShaderName(SearchType searchType, bool costImageStore) {
    std::string shaderName{shaderBaseName};
    switch (searchType) {
    case SearchType::MIN_SAD:
        shaderName += "_flow";
        break;
    case SearchType::MIN_SAD_COST:
        shaderName += costImageStore ? "_flow_cost_img" : "_flow_cost_buf";
        break;
    case SearchType::RAW_SAD:
        shaderName += "_cost_buf";
        break;
    default:
        throw std::runtime_error("Unsupported BlockMatch search type");
    }
    return shaderName;
}

BlockMatch::SpecConstants BlockMatch::makeSpecConstants() const {
    assert(!hasFlowOutput() || (dstFlow_ && dstFlow_->isBufferStore()));
    assert(!hasCostOutput() || (dstCost_ && (dstCost_->isImageStore() || dstCost_->isBufferStore())));
    assert(searchType_ != SearchType::MIN_SAD || !dstCost_);
    assert(searchType_ != SearchType::RAW_SAD || (!dstFlow_ && dstCost_->isBufferStore()));

    SpecConstants specConstants = {
        32,
        8,
        static_cast<int32_t>(searchType_),
        hasFlowOutput() ? dstFlow_->width() : dstCost_->width(),
        hasFlowOutput() ? dstFlow_->height() : dstCost_->height(),
        hasFlowOutput() ? dstFlow_->stride() : 0,
        hasCostOutput() ? dstCost_->stride() : 0,
    };
    return specConstants;
}

bool BlockMatch::hasFlowOutput() const {
    return searchType_ == SearchType::MIN_SAD_COST || searchType_ == SearchType::MIN_SAD;
}

bool BlockMatch::hasCostOutput() const {
    return searchType_ == SearchType::MIN_SAD_COST || searchType_ == SearchType::RAW_SAD;
}

void BlockMatch::setOutputCost(std::shared_ptr<Image> dstImage) {
    assert(Image::isCompatible(dstCost_, dstImage));
    dstCost_ = std::move(dstImage);
}

void BlockMatch::setSearchRangeLimit(int newSearchRangeLimit) {
    if (newSearchRangeLimit < 0 || newSearchRangeLimit > maxSearchRange_) {
        throw std::runtime_error("Search range limit must be between 0 and maxSearchRange");
    }
    searchRangeLimit_ = newSearchRangeLimit;
}

void BlockMatch::bindAndDispatch(VkCommandBuffer cmdBuf) {
    setInputStorage(cmdBuf, 0, srcTemplate_, nearestSampler_);
    setInputStorage(cmdBuf, 1, srcSearch_, nearestSampler_);

    if (hasFlowOutput()) {
        setOutputStorage(cmdBuf, 2, dstFlow_);
    }
    if (hasCostOutput()) {
        setOutputStorage(cmdBuf, dstCost_->isImageStore() ? 3 : 4, dstCost_);
    }

    bindPipeline(cmdBuf);

    const int searchIndexLimit = (searchRangeLimit_ * 2 + 1) * (searchRangeLimit_ * 2 + 1);
    const PushConstants pushConstants{searchIndexLimit};
    setPushConstants(cmdBuf, pushConstants);

    dispatchPipeline(cmdBuf);
}

} // namespace mlsdk::el::compute::optical_flow
