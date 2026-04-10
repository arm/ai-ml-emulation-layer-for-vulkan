/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "optical_flow.hpp"
#include "compute_optical_flow.hpp"
#include "image.hpp"

#include <algorithm>
#include <cmath>
#include <vulkan/vulkan.hpp>

using namespace mlsdk::el::utils;
using SearchType = mlsdk::el::compute::common::BlockMatchMode;

namespace mlsdk::el::compute::optical_flow {

/*******************************************************************************
 * OpticalFlow
 *******************************************************************************/

OpticalFlow::OpticalFlow(std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> loader,
                         VkPhysicalDevice physicalDevice, VkDevice device,
                         const std::shared_ptr<PipelineCache> &pipelineCache)
    : loader_(loader), physicalDevice_(physicalDevice), device_(device), pipelineCache_(pipelineCache) {}

void OpticalFlow::init(const Config &config) {
    // Check configuration is supported
    [[maybe_unused]] auto isSupported = [](auto fmt, const auto &supported) -> bool {
        return supported.find(fmt) != supported.end();
    };
    assert(isSupported(config.imageFormat, Spec::supportedImageFormats));
    assert(isSupported(config.flowFormat, Spec::supportedFlowFormats));
    if (config.outputCost) {
        assert(isSupported(config.costFormat, Spec::supportedCostFormats));
    }
    assert(isSupported(config.levelOfLastEstimation, Spec::supportedLevelOfLastEstimation));
    assert(config.width >= Spec::minWidth && config.width <= Spec::maxWidth);
    assert(config.height >= Spec::minHeight && config.height <= Spec::maxHeight);
    assert(!(config.useMvInput && !Spec::hintSupported));
    assert(!(config.outputCost && !Spec::costSupported));

    config_ = config;
    const auto inputUsage = Image::Usage::NoStoreImageSample;
    const auto outputUsage = Image::Usage::ImageStoreSample;
    const VkExtent3D inputDims{config.width, config.height, 1};
    setupPyramidLevelDimensions(inputDims.width, inputDims.height);

    const auto granularity = static_cast<uint32_t>(std::pow(2, config.levelOfLastEstimation));
    const VkExtent3D outputDims{divideRoundUp(inputDims.width, granularity),
                                divideRoundUp(inputDims.height, granularity), 1};

    outputFlowScale_ = calcOutputFlowScale();

    assert(!inputSearchRGB_ && !inputTemplateRGB_ && !inputMV_ && !outputFlow_ && !outputCost_);

    // Create placeholder images, with no resource attached, to ensure compatability when replacing them with external
    // (application-owned) images.
    inputSearchRGB_ =
        Image::makePlaceholder(loader_, inputUsage, inputDims, config.imageFormat, config.srcSearch.layout);

    inputTemplateRGB_ =
        Image::makePlaceholder(loader_, inputUsage, inputDims, config.imageFormat, config.srcTemplate.layout);

    if (config_.useMvInput) {
        inputMV_ = Image::makePlaceholder(loader_, inputUsage, outputDims, config.flowFormat, config.srcFlow.layout);
    }

    outputFlow_ = Image::makePlaceholder(loader_, outputUsage, outputDims, config.flowFormat, config.dstFlow.layout);
    if (config_.outputCost) {
        outputCost_ =
            Image::makePlaceholder(loader_, outputUsage, outputDims, config.costFormat, config.dstCost.layout);
    }

    makeDownsamplePyramid(dsBlocksSearch_, inputSearchRGB_, rgbToYSearch_, downsampleSearchPipelines_);
    makeDownsamplePyramid(dsBlocksTemplate_, inputTemplateRGB_, rgbToYTemplate_, downsampleTemplatePipelines_);
    makeMotionEstimationBlocks();

    if (config_.useMvInput) {
        makeReplaceWithMvInput();
    }

    computeMemoryRequirements();

    for (const auto &pipeline : downsampleSearchPipelines_) {
        pipeline->makePipeline();
    }

    for (const auto &pipeline : downsampleTemplatePipelines_) {
        pipeline->makePipeline();
    }

    for (const auto &pipeline : motionEstimationPipelines_) {
        pipeline->makePipeline();
    }
}

void OpticalFlow::setInputSearch(std::shared_ptr<Image> input) {
    inputSearchRGB_ = input;
    if (rgbToYSearch_) {
        rgbToYSearch_->setInput(inputSearchRGB_);
    }
}

void OpticalFlow::setInputTemplate(std::shared_ptr<Image> input) {
    inputTemplateRGB_ = input;
    if (rgbToYTemplate_) {
        rgbToYTemplate_->setInput(inputTemplateRGB_);
    }
}

void OpticalFlow::setInputMV(std::shared_ptr<Image> input) {
    inputMV_ = input;
    if (warpByMvInput_) {
        warpByMvInput_->setInputFlow(inputMV_);
    }
    if (mvReplace_) {
        mvReplace_->setInputMv(inputMV_);
    }
}

void OpticalFlow::setOutputFlow(std::shared_ptr<Image> output) {
    outputFlow_ = output;

    if (config_.useMvInput) {
        mvReplaceFlowOut_ = outputFlow_;
        if (mvReplace_) {
            mvReplace_->setOutputFlow(outputFlow_);
        }
    } else {
        assert(config_.levelOfLastEstimation < MEBlocks_.size());
        auto &block = MEBlocks_[config_.levelOfLastEstimation];
        if (config_.performanceLevel == PerformanceLevel::FAST) {
            block.medianFilterOut = outputFlow_;
            if (block.medianFilterPipeline) {
                block.medianFilterPipeline->setOutput(outputFlow_);
            }
        } else {
            block.bilateralFilterOut = outputFlow_;
            if (block.bilateralFilterPipeline) {
                block.bilateralFilterPipeline->setOutput(outputFlow_);
            }
        }
    }
}

void OpticalFlow::setOutputCost(std::shared_ptr<Image> output) {
    outputCost_ = output;

    if (config_.useMvInput) {
        mvReplaceCostOut_ = outputCost_;
        if (mvReplace_) {
            mvReplace_->setOutputCost(outputCost_);
        }
    } else if (config_.outputCost) {
        auto lastBlockMatchPipeline = MEBlocks_[config_.levelOfLastEstimation].blockMatchPipeline;
        if (lastBlockMatchPipeline) {
            lastBlockMatchPipeline->setOutputCost(outputCost_);
        }
    }
}

void OpticalFlow::setupPyramidLevelDimensions(uint32_t initialWidth, uint32_t initialHeight) {
    pyramidDimensions_.clear();
    pyramidDimensions_.reserve(pyramidLevels_);
    pyramidDimensions_.emplace_back(VkExtent3D{initialWidth, initialHeight, 1});
    for (size_t level = 1; level < pyramidLevels_; ++level) {
        const auto &prev = pyramidDimensions_.back();
        const auto width = roundUp<uint32_t>(divideRoundUp(prev.width, 2), 2);
        const auto height = roundUp<uint32_t>(divideRoundUp(prev.height, 2), 2);
        pyramidDimensions_.emplace_back(VkExtent3D{width, height, 1});
    }
}

void OpticalFlow::makeDownsamplePyramid(std::vector<DownsampleBlock> &blocks, std::shared_ptr<Image> &srcImage,
                                        std::shared_ptr<RGBToY> &rgbToY,
                                        std::vector<std::shared_ptr<ComputePipeline>> &pipelines) {
    blocks.resize(pyramidLevels_);
    const bool cacheImages = true;
    // Handle first level separately. For 1x1 granularity variant, we need luma image in original resolution.
    const bool isRGBToYOutputFull = (config_.levelOfLastEstimation == 0);
    if (isRGBToYOutputFull) {
        auto lumaImage = makeImage(Image::Usage::BufferStoreImageSample, pyramidDimensions_[0], VK_FORMAT_R8_UNORM,
                                   VK_IMAGE_TILING_LINEAR, std::string("luma"), cacheImages);

        blocks[0].image = lumaImage;
    }

    for (size_t level = 0; level < pyramidLevels_ - 1; ++level) {
        const auto levelStr = std::to_string(level);
        if (level == 0) {
            const auto usage =
                isRGBToYOutputFull ? Image::Usage::BufferStoreImageSample : Image::Usage::ImageStoreSample;
            const auto tiling = isRGBToYOutputFull ? VK_IMAGE_TILING_LINEAR : VK_IMAGE_TILING_OPTIMAL;

            auto downscaledImage = makeImage(usage, pyramidDimensions_[level + 1], VK_FORMAT_R8_UNORM, tiling,
                                             std::string("downscaled_L") + levelStr, cacheImages);
            blocks[level + 1].image = downscaledImage;

            // RGB-to-Y + downsample
            rgbToY = makePipeline<RGBToY>(pipelines, srcImage, blocks[level + 1].image, blocks[level].image, true,
                                          isRGBToYOutputFull, 2.0f, std::string("RGBtoY_L") + levelStr);
        } else {
            auto downscaledImage =
                makeImage(Image::Usage::ImageStoreSample, pyramidDimensions_[level + 1], VK_FORMAT_R8_UNORM,
                          VK_IMAGE_TILING_OPTIMAL, std::string("downscaled_L") + levelStr, cacheImages);
            blocks[level + 1].image = downscaledImage;
            // blur + downsample
            blocks[level].pipeline = makePipeline<Downsample>(pipelines, blocks[level].image, downscaledImage,
                                                              std::string("Downsample_L") + levelStr);
        }
    }
}

void OpticalFlow::makeMotionEstimationBlocks() {
    MEBlocks_.resize(pyramidLevels_);

    setOutputFlow(outputFlow_);
    setOutputCost(outputCost_);

    // Start at the bottom of the pyramid and work up.
    const auto first = pyramidLevels_ - 1;
    const auto last = config_.levelOfLastEstimation;
    for (size_t level = first; level + 1 > last; --level) {
        auto getFlowDim = [&](size_t l) -> VkExtent3D {
            return {pyramidDimensions_[l].width, pyramidDimensions_[l].height, 1};
        };

        const auto levelStr = std::to_string(level);
        auto createFlowMem = [&](Image::Usage usage, std::string debugPrefix) -> std::shared_ptr<Image> {
            const auto tiling =
                usage == Image::Usage::ImageStoreSample ? VK_IMAGE_TILING_OPTIMAL : VK_IMAGE_TILING_LINEAR;
            return makeImage(usage, getFlowDim(level), VK_FORMAT_R16G16_SFLOAT, tiling,
                             debugPrefix.append("_flow_L").append(levelStr));
        };

        const bool isLastLevel = (level == config_.levelOfLastEstimation);

        auto &block = MEBlocks_[level];
        bool accumulatePrevFlow = true;
        if (level == first) {
            // previous flow not avaible
            accumulatePrevFlow = false;
            block.upscaledFlow = createFlowMem(Image::Usage::BufferStoreLoad, "upscaled");
            // image is not warped here
            block.warpedImage = dsBlocksSearch_[level].image;
        } else {
            // MV process and warp
            block.warpedImage =
                makeImage(Image::Usage::BufferStoreImageSample, pyramidDimensions_[level], VK_FORMAT_R8_UNORM,
                          VK_IMAGE_TILING_LINEAR, std::string("warped_L") + levelStr);
            block.upscaledFlow = createFlowMem(Image::Usage::BufferStoreLoad, "upscaled");

            block.mvWarpPipeline = makePipeline<MVProcessAndWarp>(
                motionEstimationPipelines_, dsBlocksSearch_[level].image, MEBlocks_[level + 1].filteredFlowOut,
                block.warpedImage, block.upscaledFlow, std::string("MVProcessAndWarp_L") + levelStr);
        }
        // Block match
        block.blockMatchOut = makeImage(Image::Usage::BufferStoreLoad, getFlowDim(level), VK_FORMAT_R8G8_SINT,
                                        VK_IMAGE_TILING_LINEAR, std::string("blockmatch_flow_L") + levelStr);

        std::shared_ptr<Image> costOut{};
        auto searchType = SearchType::MIN_SAD;
        if (isLastLevel) {
            if (config_.useMvInput) {
                searchType = SearchType::MIN_SAD_COST;
                minCostBlockMatch_ =
                    makeImage(Image::Usage::BufferStoreImageSample, pyramidDimensions_[level], VK_FORMAT_R16_UINT,
                              VK_IMAGE_TILING_LINEAR, std::string("blockmatch_cost_L") + levelStr);
                costOut = minCostBlockMatch_;
            } else if (config_.outputCost) {
                searchType = SearchType::MIN_SAD_COST;
                costOut = outputCost_;
            }
        }

        block.blockMatchPipeline = makePipeline<BlockMatch>(
            motionEstimationPipelines_, searchType, config_.maxSearchRange, block.warpedImage,
            dsBlocksTemplate_[level].image, block.blockMatchOut, costOut, std::string("BlockMatch_L") + levelStr);

        // Subpixel motion estimation
        block.subpixelOut = createFlowMem(Image::Usage::BufferStoreImageSample, "subpixel");
        block.subpixelMEPipeline = makePipeline<SubpixelME>(
            motionEstimationPipelines_, block.warpedImage, dsBlocksTemplate_[level].image, block.blockMatchOut,
            block.upscaledFlow, block.subpixelOut, accumulatePrevFlow, std::string("SubpixelME_L") + levelStr);

        // median filter
        if (block.medianFilterOut == nullptr) {
            block.medianFilterOut = createFlowMem(Image::Usage::ImageStoreSample, "medianF");
        }

        float medianFilterOutputScale = 1.0f;
        if (isLastLevel && config_.performanceLevel == PerformanceLevel::FAST) {
            medianFilterOutputScale = outputFlowScale_;
        }

        block.medianFilterPipeline =
            makePipeline<MedianFilter>(motionEstimationPipelines_, block.subpixelOut, block.medianFilterOut,
                                       medianFilterOutputScale, std::string("MedianFilter_L") + levelStr);
        block.filteredFlowOut = block.medianFilterOut;

        if (config_.performanceLevel != PerformanceLevel::FAST) {
            // joint bilateral filter
            if (block.bilateralFilterOut == nullptr) {
                block.bilateralFilterOut = createFlowMem(Image::Usage::BufferStoreImageSample, "bilatF");
            }

            float bilateralFilterOutputScale = 1.0f;
            if (isLastLevel) {
                bilateralFilterOutputScale = outputFlowScale_;
            }

            block.bilateralFilterPipeline = makePipeline<BilateralFilter>(
                motionEstimationPipelines_, dsBlocksTemplate_[level].image, block.medianFilterOut,
                block.bilateralFilterOut, bilateralFilterOutputScale, std::string("BilateralFilter_L") + levelStr);
            block.filteredFlowOut = block.bilateralFilterOut;
        }
    }
}

void OpticalFlow::makeReplaceWithMvInput() {
    const auto level = config_.levelOfLastEstimation;
    const auto levelStr = std::to_string(level);

    // warp search image by MV input
    warpedByMvInputImage_ = makeImage(Image::Usage::ImageStoreSample, pyramidDimensions_[level], VK_FORMAT_R8_UNORM,
                                      VK_IMAGE_TILING_OPTIMAL, std::string("warpedByMVInput_L") + levelStr);
    warpByMvInput_ =
        makePipeline<DenseWarp>(motionEstimationPipelines_, dsBlocksSearch_[level].image, inputMV_,
                                warpedByMvInputImage_, 1.f / outputFlowScale_, std::string("DenseWarp_L") + levelStr);

    // calculate cost at MV input
    costAtMvInput_ = makeImage(Image::Usage::BufferStoreImageSample, pyramidDimensions_[level], VK_FORMAT_R16_UINT,
                               VK_IMAGE_TILING_LINEAR, std::string("costAtMVInput_L") + levelStr);
    rawSad_ = makePipeline<BlockMatch>(motionEstimationPipelines_, SearchType::RAW_SAD, 0, warpedByMvInputImage_,
                                       dsBlocksTemplate_[level].image, nullptr, costAtMvInput_,
                                       std::string("BlockMatch_L") + levelStr);

    // replace MV depending on cost value
    mvReplace_ = makePipeline<MVReplace>(motionEstimationPipelines_, inputMV_, MEBlocks_[level].filteredFlowOut,
                                         costAtMvInput_, minCostBlockMatch_, mvReplaceFlowOut_, mvReplaceCostOut_,
                                         config_.outputCost, std::string("MVReplace_L") + levelStr);
}

void OpticalFlow::computeMemoryRequirements() {
    VkMemoryRequirements cacheMemReqs{0, 1, ~0u};
    VkMemoryRequirements transientMemReqs{0, 1, ~0u};

    VkPhysicalDeviceProperties physicalDeviceProperties;
    loader_->vkGetPhysicalDeviceProperties(physicalDevice_, &physicalDeviceProperties);
    const VkDeviceSize bufferImageGranularity = physicalDeviceProperties.limits.bufferImageGranularity;

    for (const auto &image : allImages_) {
        const auto mrqs = image->getMemoryRequirements();
        const auto alignment = std::max(mrqs.alignment, bufferImageGranularity);
        if (image->isCached()) {
            cacheMemReqs.size = roundUp(cacheMemReqs.size, alignment);
            image->setMemoryOffset(cacheMemReqs.size);

            cacheMemReqs.size += mrqs.size;
            cacheMemReqs.alignment = std::max(cacheMemReqs.alignment, alignment);
            cacheMemReqs.memoryTypeBits &= mrqs.memoryTypeBits;
        } else {
            transientMemReqs.size = roundUp(transientMemReqs.size, alignment);
            image->setMemoryOffset(transientMemReqs.size);

            transientMemReqs.size += mrqs.size;
            transientMemReqs.alignment = std::max(transientMemReqs.alignment, alignment);
            transientMemReqs.memoryTypeBits &= mrqs.memoryTypeBits;
        }
    }

    cacheMemoryRequirements_ = cacheMemReqs;
    transientMemoryRequirements_ = transientMemReqs;
}

VkMemoryRequirements OpticalFlow::getCacheMemoryRequirements() const { return cacheMemoryRequirements_; }
VkMemoryRequirements OpticalFlow::getTransientMemoryRequirements() const { return transientMemoryRequirements_; }

void OpticalFlow::bindSessionTransientMemory(VkDeviceMemory memory, VkDeviceSize offset) {
    for (const auto &image : allImages_) {
        if (!image->isCached()) {
            image->bindToMemory(memory, offset);
        }
    }
}

void OpticalFlow::bindSessionCacheMemory(VkDeviceMemory memory, VkDeviceSize offset) {
    for (const auto &image : allImages_) {
        if (image->isCached()) {
            image->bindToMemory(memory, offset);
        }
    }
}

void OpticalFlow::updateDescriptorSets(const OpticalFlowDescriptorMap &descriptorMap) {
    auto getDescriptorSetAndImageView =
        [&descriptorMap](const Config::InputImage &in) -> std::pair<VkDescriptorSet, VkImageView> {
        const auto it = descriptorMap.find({in.set, in.binding, 0});
        if (it != descriptorMap.end()) {
            return it->second;
        }
        throw std::runtime_error("Optical flow descriptors inconsistent with connectivity map");
    };

    VkDescriptorSet vkDescriptorSet;
    VkImageView vkImageView;

    std::tie(vkDescriptorSet, vkImageView) = getDescriptorSetAndImageView(config_.srcSearch);
    inputSearchRGB_->setExternalDescriptor(vkDescriptorSet, config_.srcSearch.binding, vkImageView);
    setInputSearch(inputSearchRGB_);

    std::tie(vkDescriptorSet, vkImageView) = getDescriptorSetAndImageView(config_.srcTemplate);
    inputTemplateRGB_->setExternalDescriptor(vkDescriptorSet, config_.srcTemplate.binding, vkImageView);
    setInputTemplate(inputTemplateRGB_);

    if (config_.useMvInput) {
        std::tie(vkDescriptorSet, vkImageView) = getDescriptorSetAndImageView(config_.srcFlow);
        inputMV_->setExternalDescriptor(vkDescriptorSet, config_.srcFlow.binding, vkImageView);
        setInputMV(inputMV_);
    }

    std::tie(vkDescriptorSet, vkImageView) = getDescriptorSetAndImageView(config_.dstFlow);
    outputFlow_->setExternalDescriptor(vkDescriptorSet, config_.dstFlow.binding, vkImageView);
    setOutputFlow(outputFlow_);

    if (config_.outputCost) {
        std::tie(vkDescriptorSet, vkImageView) = getDescriptorSetAndImageView(config_.dstCost);
        outputCost_->setExternalDescriptor(vkDescriptorSet, config_.dstCost.binding, vkImageView);
        setOutputCost(outputCost_);
    }
}

void OpticalFlow::cmdBindAndDispatch(VkCommandBuffer cmdBuf, VkDataGraphOpticalFlowExecuteFlagsARM flags,
                                     uint32_t meanFlowL1NormHint) {
    if (!validateConfiguration()) {
        throw std::runtime_error("Invalid configuration");
    }

    constexpr VkDataGraphOpticalFlowExecuteFlagsARM temporalHintFlags =
        VK_DATA_GRAPH_OPTICAL_FLOW_EXECUTE_INPUT_UNCHANGED_BIT_ARM |
        VK_DATA_GRAPH_OPTICAL_FLOW_EXECUTE_REFERENCE_UNCHANGED_BIT_ARM |
        VK_DATA_GRAPH_OPTICAL_FLOW_EXECUTE_INPUT_IS_PREVIOUS_REFERENCE_BIT_ARM |
        VK_DATA_GRAPH_OPTICAL_FLOW_EXECUTE_REFERENCE_IS_PREVIOUS_INPUT_BIT_ARM;

    const bool disableTemporalHints = (flags & VK_DATA_GRAPH_OPTICAL_FLOW_EXECUTE_DISABLE_TEMPORAL_HINTS_BIT_ARM) != 0;
    const VkDataGraphOpticalFlowExecuteFlagsARM effectiveFlags =
        disableTemporalHints ? static_cast<VkDataGraphOpticalFlowExecuteFlagsARM>(
                                   flags & static_cast<VkDataGraphOpticalFlowExecuteFlagsARM>(~temporalHintFlags))
                             : flags;

    // Indicates that the previous template image is the current search image or vice-versa
    if (effectiveFlags & (VK_DATA_GRAPH_OPTICAL_FLOW_EXECUTE_INPUT_IS_PREVIOUS_REFERENCE_BIT_ARM |
                          VK_DATA_GRAPH_OPTICAL_FLOW_EXECUTE_REFERENCE_IS_PREVIOUS_INPUT_BIT_ARM)) {
        // Swap image pyramids to avoid regenerating for the same image
        swapImagePyramids();
    }

    const int searchRangeLimit = toSearchRangeLimit(meanFlowL1NormHint);
    for (auto &me : MEBlocks_) {
        if (me.blockMatchPipeline) {
            me.blockMatchPipeline->setSearchRangeLimit(searchRangeLimit);
        }
    }

    if (!(effectiveFlags & (VK_DATA_GRAPH_OPTICAL_FLOW_EXECUTE_REFERENCE_UNCHANGED_BIT_ARM |
                            VK_DATA_GRAPH_OPTICAL_FLOW_EXECUTE_REFERENCE_IS_PREVIOUS_INPUT_BIT_ARM))) {
        for (const auto &pipeline : downsampleTemplatePipelines_) {
            pipeline->bindAndDispatch(cmdBuf);
        }
    }

    if (!(effectiveFlags & (VK_DATA_GRAPH_OPTICAL_FLOW_EXECUTE_INPUT_UNCHANGED_BIT_ARM |
                            VK_DATA_GRAPH_OPTICAL_FLOW_EXECUTE_INPUT_IS_PREVIOUS_REFERENCE_BIT_ARM))) {
        for (const auto &pipeline : downsampleSearchPipelines_) {
            pipeline->bindAndDispatch(cmdBuf);
        }
    }

    for (const auto &pipeline : motionEstimationPipelines_) {
        pipeline->bindAndDispatch(cmdBuf);
    }
}

void OpticalFlow::swapImagePyramids() {
    for (size_t i = 0; i < dsBlocksSearch_.size(); ++i) {
        if (dsBlocksSearch_[i].image) {
            dsBlocksSearch_[i].image->swapHandles(*dsBlocksTemplate_[i].image);
        }
    }
}

float OpticalFlow::calcOutputFlowScale() const { return static_cast<float>(1 << config_.levelOfLastEstimation); }

int OpticalFlow::toSearchRangeLimit(uint32_t meanFlowL1NormHint) const {
    const uint32_t maxDimension = std::max(config_.width, config_.height);
    meanFlowL1NormHint = std::min(meanFlowL1NormHint, maxDimension);
    if (meanFlowL1NormHint == 0) {
        return config_.maxSearchRange;
    }

    const int factor = (1 << (pyramidLevels_ - 1)) / 2;
    int searchRangeLimit = static_cast<int>(std::ceil(meanFlowL1NormHint / static_cast<float>(factor)));
    searchRangeLimit = std::clamp(searchRangeLimit, 1, config_.maxSearchRange);
    return searchRangeLimit;
}

bool OpticalFlow::validateConfiguration() const {
    bool anyPlaceholder = false;
    anyPlaceholder |= inputSearchRGB_->isPlaceholder();
    anyPlaceholder |= inputTemplateRGB_->isPlaceholder();
    if (config_.useMvInput) {
        anyPlaceholder |= inputMV_->isPlaceholder();
    }
    anyPlaceholder |= outputFlow_->isPlaceholder();
    if (config_.outputCost) {
        anyPlaceholder |= outputCost_->isPlaceholder();
    }

    return !anyPlaceholder;
}

std::shared_ptr<Image> OpticalFlow::makeImage(Image::Usage usage, VkExtent3D dim, VkFormat format, VkImageTiling tiling,
                                              const std::string &debugName, bool isCached) {
    auto image =
        Image::makeInternal(loader_, physicalDevice_, device_, usage, dim, format, tiling, isCached, debugName);
    allImages_.push_back(image);
    return image;
}

} // namespace mlsdk::el::compute::optical_flow
