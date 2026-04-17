/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "compute_optical_flow.hpp"
#include "image.hpp"
#include "mlel/utils.hpp"
#include "tensor.hpp"

#include <set>
#include <type_traits>
#include <vulkan/vulkan.hpp>

namespace mlsdk::el::compute::optical_flow {

// Map descriptor set index + binding + array element -> descriptor set + image view
using OpticalFlowDescriptorMap =
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, std::pair<VkDescriptorSet, VkImageView>>;

/*******************************************************************************
 * OpticalFlow
 *******************************************************************************/

class OpticalFlow {
  public:
    OpticalFlow(std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> loader,
                VkPhysicalDevice physicalDevice, VkDevice device, const std::shared_ptr<PipelineCache> &pipelineCache);
    ~OpticalFlow() = default;

    struct Spec {
        static constexpr VkBool32 hintSupported = VK_TRUE;
        static constexpr VkBool32 costSupported = VK_TRUE;

        static constexpr uint32_t minWidth = 64;
        static constexpr uint32_t minHeight = 64;
        static constexpr uint32_t maxWidth = 8192;
        static constexpr uint32_t maxHeight = 8192;

        // Correspondns to supportedOutputGridSizes and supportedHintGridSizes.
        // To convert to VkOpticalFlowGridSizeFlagsARM, do (1 << levelOfLastEstimation).
        // OutputGridSize and HintGridSize must be the same value in creating session.
        inline static const std::set<size_t> supportedLevelOfLastEstimation{0, 1, 2, 3};

        inline static const std::set<VkFormat> supportedImageFormats{
            VK_FORMAT_R8G8B8_UNORM,
            VK_FORMAT_B8G8R8_UNORM,
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_FORMAT_B8G8R8A8_UNORM,
            VK_FORMAT_B10G11R11_UFLOAT_PACK32,
            VK_FORMAT_R8_UNORM,
        };

        inline static const std::set<VkFormat> supportedFlowFormats{
            VK_FORMAT_R16G16_SFLOAT,
        };

        inline static const std::set<VkFormat> supportedCostFormats{
            VK_FORMAT_R16_UINT,
        };
    };

    enum class PerformanceLevel { MEDIUM, FAST, SLOW, UNKNOWN };

    struct Config {
        uint32_t width;
        uint32_t height;
        size_t levelOfLastEstimation = 2;
        bool useMvInput = false;
        bool outputCost = false;
        PerformanceLevel performanceLevel = PerformanceLevel::MEDIUM;
        int maxSearchRange = 3;
        VkFormat imageFormat = VK_FORMAT_R8G8B8_UNORM;
        VkFormat flowFormat = VK_FORMAT_R16G16_SFLOAT;
        VkFormat costFormat = VK_FORMAT_R16_UINT;

        struct InputImage {
            uint32_t binding;
            uint32_t set;
            VkImageLayout layout;
        };
        InputImage srcSearch;
        InputImage srcTemplate;
        InputImage srcFlow;
        InputImage dstFlow;
        InputImage dstCost;
    };

    void init(const Config &config);

    void setInputSearch() const;
    void setInputTemplate() const;
    void setInputMV() const;
    void setOutputFlow();
    void setOutputCost();

    VkMemoryRequirements getTransientMemoryRequirements() const;
    VkMemoryRequirements getCacheMemoryRequirements() const;
    void bindSessionTransientMemory(VkDeviceMemory memory, VkDeviceSize offset);
    void bindSessionCacheMemory(VkDeviceMemory memory, VkDeviceSize offset);
    void updateDescriptorSets(const OpticalFlowDescriptorMap &descriptorMap);
    void cmdBindAndDispatch(VkCommandBuffer cmdBuf, VkDataGraphOpticalFlowExecuteFlagsARM flags,
                            uint32_t meanFlowL1NormHint);

    std::shared_ptr<Image> makeImage(Image::Usage usage, VkExtent3D dim, VkFormat format, VkImageTiling tiling,
                                     const std::string &debugName = "", bool isCached = false);
    template <typename T, typename... Args>
    std::shared_ptr<T> makePipeline(std::vector<std::shared_ptr<ComputePipeline>> &pipelines, Args &&...args);

  private:
    struct MotionEstimationBlock {
        std::shared_ptr<Image> warpedImage;
        std::shared_ptr<Image> blockMatchOut;
        std::shared_ptr<Image> upscaledFlow;
        std::shared_ptr<Image> subpixelOut;
        std::shared_ptr<Image> medianFilterOut;
        std::shared_ptr<Image> bilateralFilterOut;
        // alias to the output
        std::shared_ptr<Image> filteredFlowOut;

        std::shared_ptr<MVProcessAndWarp> mvWarpPipeline;
        std::shared_ptr<BlockMatch> blockMatchPipeline;
        std::shared_ptr<SubpixelME> subpixelMEPipeline;
        std::shared_ptr<MedianFilter> medianFilterPipeline;
        std::shared_ptr<BilateralFilter> bilateralFilterPipeline;
    };

    struct DownsampleBlock {
        std::shared_ptr<Image> image;
        std::shared_ptr<Downsample> pipeline;
    };

    void makeDownsamplePyramid(std::vector<DownsampleBlock> &blocks, std::shared_ptr<Image> &srcImage,
                               std::shared_ptr<RGBToY> &rgbToY,
                               std::vector<std::shared_ptr<ComputePipeline>> &pipelines);
    void makeMotionEstimationBlocks();
    void makeReplaceWithMvInput();

    void swapImagePyramids();
    float calcOutputFlowScale() const;
    int toSearchRangeLimit(uint32_t meanFlowL1NormHint) const;
    bool validateConfiguration() const;

    void setupPyramidLevelDimensions(uint32_t initialWidth, uint32_t initialHeight);

    void computeMemoryRequirements();

    std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> loader_;
    VkPhysicalDevice physicalDevice_;
    VkDevice device_;
    std::shared_ptr<PipelineCache> pipelineCache_;

    Config config_ = {};

    std::vector<MotionEstimationBlock> MEBlocks_;

    std::vector<DownsampleBlock> dsBlocksSearch_;
    std::vector<DownsampleBlock> dsBlocksTemplate_;

    std::shared_ptr<Image> inputSearchRGB_;
    std::shared_ptr<Image> inputTemplateRGB_;
    std::shared_ptr<Image> inputMV_;
    std::shared_ptr<Image> outputFlow_;
    std::shared_ptr<Image> outputCost_;
    std::shared_ptr<RGBToY> rgbToYSearch_;
    std::shared_ptr<RGBToY> rgbToYTemplate_;

    std::shared_ptr<Image> warpedByMvInputImage_;
    std::shared_ptr<Image> costAtMvInput_;
    std::shared_ptr<Image> minCostBlockMatch_;
    std::shared_ptr<Image> mvReplaceFlowOut_;
    std::shared_ptr<Image> mvReplaceCostOut_;

    std::shared_ptr<DenseWarp> warpByMvInput_;
    std::shared_ptr<BlockMatch> rawSad_;
    std::shared_ptr<MVReplace> mvReplace_;

    size_t pyramidLevels_ = 6;
    std::vector<VkExtent3D> pyramidDimensions_;
    float outputFlowScale_ = 0.0f;

    VkMemoryRequirements cacheMemoryRequirements_;
    VkMemoryRequirements transientMemoryRequirements_;

    std::vector<std::shared_ptr<ComputePipeline>> downsampleSearchPipelines_;
    std::vector<std::shared_ptr<ComputePipeline>> downsampleTemplatePipelines_;
    std::vector<std::shared_ptr<ComputePipeline>> motionEstimationPipelines_;
    std::vector<std::shared_ptr<Image>> allImages_;
};

template <typename T, typename... Args>
inline std::shared_ptr<T> OpticalFlow::makePipeline(std::vector<std::shared_ptr<ComputePipeline>> &pipelines,
                                                    Args &&...args) {
    auto pipeline = std::make_shared<T>(loader_, device_, pipelineCache_, std::forward<Args>(args)...);
    pipelines.push_back(pipeline);
    return pipeline;
}

} // namespace mlsdk::el::compute::optical_flow
