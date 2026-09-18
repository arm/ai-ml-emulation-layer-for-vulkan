/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "compute_pipeline_common.hpp"
#include "image.hpp"
#include "mlel/utils.hpp"
#include "pipeline_cache.hpp"
#include "tensor.hpp"

#include <vulkan/vulkan.hpp>

#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace mlsdk::el::compute::optical_flow {

struct DescriptorConfig {
    uint32_t index;
    VkDescriptorType type;
};
using DescriptorConfigs = std::vector<DescriptorConfig>;

struct SpecConstants {
    const void *pointer;
    uint32_t sizeBytes;
};

class ScheduleHelper {
  public:
    ScheduleHelper(uint32_t width, uint32_t height) {
        groupCountX = utils::divideRoundUp(width, localSizeX);
        groupCountY = utils::divideRoundUp(height, localSizeY);
    }
    uint32_t localSizeX = 32;
    uint32_t localSizeY = 8;
    uint32_t groupCountX = 1;
    uint32_t groupCountY = 1;
    uint32_t groupCountZ = 1;
};

/*******************************************************************************
 * ComputePipeline
 *******************************************************************************/

class ComputePipeline {
  public:
    ComputePipeline(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &loader, VkDevice device,
                    const std::shared_ptr<PipelineCache> &pipelineCache, SpirvBinary spirv,
                    const DescriptorConfigs &descriptorConfigs, const SpecConstants &specConstants,
                    uint32_t pushConstantsSize, const ScheduleHelper &schedule, const std::string &debugName);
    ComputePipeline(const ComputePipeline &) = delete;
    ComputePipeline &operator=(const ComputePipeline &) = delete;
    virtual ~ComputePipeline();

    void makePipeline();
    void setInputStorage(VkCommandBuffer cmdBuf, uint32_t binding, const std::shared_ptr<Image> &image,
                         VkSampler sampler = VK_NULL_HANDLE);
    void setOutputStorage(VkCommandBuffer cmdBuf, uint32_t binding, const std::shared_ptr<Image> &image);

    virtual void bindAndDispatch(VkCommandBuffer cmdBuf) = 0;
    const std::string &getDebugName() const;

  protected:
    VkSampler createSampler(VkFilter filter, VkSamplerAddressMode addressMode, bool unnormalizedCoordinates);
    template <typename T> void setPushConstants(VkCommandBuffer cmdBuf, const T &constants);
    void bindPipeline(VkCommandBuffer cmdBuf);
    void dispatchPipeline(VkCommandBuffer cmdBuf);

  private:
    void setCombinedImageSampler(uint32_t binding, const std::shared_ptr<Image> &image, VkSampler sampler);
    void setOutputImage(uint32_t binding, const std::shared_ptr<Image> &image);
    void setImage(uint32_t binding, const std::shared_ptr<Image> &image, VkDescriptorType descriptorType,
                  VkSampler sampler);
    void setBuffer(uint32_t binding, const std::shared_ptr<Image> &image);

    std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> loader_;
    VkDevice device_;
    const std::shared_ptr<PipelineCache> &pipelineCache_;
    SpirvBinary spirv_;
    DescriptorConfigs descriptorConfigs_;
    VkPipeline vkPipeline_{};
    SpecConstants specConstants_;
    uint32_t pushConstantsSize_;
    ScheduleHelper scheduler_;

    VkPipelineLayout pipelineLayout_{};
    VkDescriptorSetLayout descriptorSetLayout_{};
    VkDescriptorPool descriptorPool_{};
    VkDescriptorSet descriptorSet_{};
    const VkAllocationCallbacks *pAllocator_ = nullptr;
    std::vector<VkSampler> samplers_;

    std::string debugName_;
};

using ComputePipelineDispatchDecorator = std::function<void(VkCommandBuffer, ComputePipeline &, uint32_t)>;

template <typename T> void ComputePipeline::setPushConstants(VkCommandBuffer cmdBuf, const T &constants) {
    assert(pushConstantsSize_ == sizeof(T));
    loader_->vkCmdPushConstants(cmdBuf, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(T), &constants);
}

/*******************************************************************************
 * RGBToY
 *******************************************************************************/

class RGBToY : public ComputePipeline {
  public:
    RGBToY(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &loader, VkDevice device,
           const std::shared_ptr<PipelineCache> &pipelineCache, std::shared_ptr<Image> srcRGBImage,
           const std::shared_ptr<Image> &dstDownsampledImage, std::shared_ptr<Image> dstFullImage, bool outputFullRes,
           float downsampleScale, const std::string &debugName);
    ~RGBToY() override = default;

    static SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache, bool outputFull,
                                   bool imageStore);

    struct SpecConstants {
        uint32_t threadGroupSizeX;
        uint32_t threadGroupSizeY;
        VkBool32 isLumaInput;
        float downsampleScaleX;
        float downsampleScaleY;
        uint32_t downsampledImageWidth;
        uint32_t downsampledImageHeight;
        uint32_t downsampledImageStride;
        uint32_t fullImageWidth;
        uint32_t fullImageHeight;
        uint32_t fullImageStride;
    };

    SpecConstants makeSpecConstants(float downsampleScale) const;
    void setInput(std::shared_ptr<Image> src);
    void bindAndDispatch(VkCommandBuffer cmdBuf) override;

  private:
    static constexpr std::string_view shaderBaseName = "rgb_to_y";
    static std::string makeShaderName(bool outputFull, bool imageStore);
    std::shared_ptr<Image> srcImage_;
    std::shared_ptr<Image> dstYDownsampled_;
    std::shared_ptr<Image> dstYFull_;
    bool outputFull_;
    SpecConstants specConstants_;
    VkSampler linearSampler_;

    inline static const DescriptorConfigs descriptorConfigs_{
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER}, // Src
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},          // DstDs
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},         // DstDs
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},         // DstFull
    };
};

/*******************************************************************************
 * Downsample
 *******************************************************************************/

class Downsample : public ComputePipeline {
  public:
    Downsample(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &loader, VkDevice device,
               const std::shared_ptr<PipelineCache> &pipelineCache, std::shared_ptr<Image> src,
               const std::shared_ptr<Image> &dst, const std::string &debugName);
    ~Downsample() override = default;

    static SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache);

    struct SpecConstants {
        uint32_t threadGroupSizeX;
        uint32_t threadGroupSizeY;
        VkBool32 padX;
        VkBool32 padY;
        uint32_t outputImageWidth;
        uint32_t outputImageHeight;
    };

    SpecConstants makeSpecConstants() const;

    void bindAndDispatch(VkCommandBuffer cmdBuf) override;

  private:
    static constexpr std::string_view shaderName = "downsample_img";
    std::shared_ptr<Image> srcImage_;
    std::shared_ptr<Image> dstImage_;
    SpecConstants specConstants_;
    VkSampler linearSampler_;

    inline static const DescriptorConfigs descriptorConfigs_{
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER}, // Src
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},          // Dst
    };
};

/*******************************************************************************
 * MVProcessAndWarp
 *******************************************************************************/

class MVProcessAndWarp : public ComputePipeline {
  public:
    MVProcessAndWarp(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &loader,
                     VkDevice device, const std::shared_ptr<PipelineCache> &pipelineCache,
                     std::shared_ptr<Image> srcImage, std::shared_ptr<Image> _srcFlow,
                     const std::shared_ptr<Image> &dstImage, std::shared_ptr<Image> _dstFlow,
                     const std::string &debugName);
    ~MVProcessAndWarp() override = default;

    static SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache);

    struct SpecConstants {
        uint32_t threadGroupSizeX;
        uint32_t threadGroupSizeY;
        float downsampleScaleX;
        float downsampleScaleY;
        float upsampleScaleX;
        float upsampleScaleY;
        uint32_t outputImageWidth;
        uint32_t outputImageHeight;
        uint32_t outputImageStride;
        uint32_t outputFlowStride;
    };

    SpecConstants makeSpecConstants() const;
    void bindAndDispatch(VkCommandBuffer cmdBuf) override;

  private:
    static constexpr std::string_view shaderName = "mv_process_and_warp_buf";
    std::shared_ptr<Image> srcSearch_;
    std::shared_ptr<Image> srcFlow_;
    std::shared_ptr<Image> dstWarped_;
    std::shared_ptr<Image> dstFlow_;
    SpecConstants specConstants_;
    VkSampler linearSampler_;

    inline static const DescriptorConfigs descriptorConfigs_{
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER}, // Src
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER}, // SrcFlow
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},         // Dst
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},         // DstFlow
    };
};

/*******************************************************************************
 * DenseWarp
 *******************************************************************************/

class DenseWarp : public ComputePipeline {
  public:
    DenseWarp(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &loader, VkDevice device,
              const std::shared_ptr<PipelineCache> &pipelineCache, std::shared_ptr<Image> srcImage,
              std::shared_ptr<Image> _srcFlow, const std::shared_ptr<Image> &dstImage, float inputFlowScale,
              const std::string &debugName);
    ~DenseWarp() override = default;

    static SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache);

    struct SpecConstants {
        uint32_t threadGroupSizeX;
        uint32_t threadGroupSizeY;
        float inputFlowScale;
        uint32_t outputImageWidth;
        uint32_t outputImageHeight;
    };

    SpecConstants makeSpecConstants(float inputFlowScale) const;

    void setInputFlow(std::shared_ptr<Image> _srcFlow);

    void bindAndDispatch(VkCommandBuffer cmdBuf) override;

  private:
    static constexpr std::string_view shaderName = "dense_warp_img";
    std::shared_ptr<Image> srcSearch_;
    std::shared_ptr<Image> srcFlow_;
    std::shared_ptr<Image> dstWarped_;
    SpecConstants specConstants_;
    VkSampler linearSampler_;
    VkSampler nearestSampler_;

    inline static const DescriptorConfigs descriptorConfigs_{
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER}, // Src
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER}, // SrcFlow
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},          // Dst
    };
};

/*******************************************************************************
 * MedianFilter
 *******************************************************************************/

class MedianFilter : public ComputePipeline {
  public:
    MedianFilter(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &loader, VkDevice device,
                 const std::shared_ptr<PipelineCache> &pipelineCache, std::shared_ptr<Image> srcImage,
                 const std::shared_ptr<Image> &dstImage, float outputFlowScale, const std::string &debugName);
    ~MedianFilter() override = default;

    static SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache);

    struct SpecConstants {
        uint32_t threadGroupSizeX;
        uint32_t threadGroupSizeY;
        float outputFlowScale;
        uint32_t outputImageWidth;
        uint32_t outputImageHeight;
    };

    SpecConstants makeSpecConstants(float outputFlowScale) const;

    void setOutput(std::shared_ptr<Image> dstImage);
    void bindAndDispatch(VkCommandBuffer cmdBuf) override;

  private:
    static constexpr std::string_view shaderName = "median_filter_img";
    std::shared_ptr<Image> srcFlow_;
    std::shared_ptr<Image> dstFlow_;
    SpecConstants specConstants_;
    VkSampler nearestSampler_;

    inline static const DescriptorConfigs descriptorConfigs_{
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER}, // Src
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},          // Dst
    };
};

/*******************************************************************************
 * BilateralFilter
 *******************************************************************************/

class BilateralFilter : public ComputePipeline {
  public:
    BilateralFilter(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &loader, VkDevice device,
                    const std::shared_ptr<PipelineCache> &pipelineCache, std::shared_ptr<Image> srcImage,
                    std::shared_ptr<Image> srcFlow, const std::shared_ptr<Image> &dstFlow, float outputFlowScale,
                    const std::string &debugName);
    ~BilateralFilter() override = default;

    static SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache, bool imageStore);

    struct SpecConstants {
        uint32_t threadGroupSizeX;
        uint32_t threadGroupSizeY;
        float outputFlowScale;
        uint32_t outputImageWidth;
        uint32_t outputImageHeight;
        uint32_t outputImageStride;
    };

    SpecConstants makeSpecConstants(float outputFlowScale) const;

    void setOutput(std::shared_ptr<Image> dstFlow);
    void bindAndDispatch(VkCommandBuffer cmdBuf) override;

  private:
    static constexpr std::string_view shaderBaseName = "bilateral_filter";
    static std::string makeShaderName(bool imageStore);
    std::shared_ptr<Image> srcTemplate_;
    std::shared_ptr<Image> srcFlow_;
    std::shared_ptr<Image> dstFlow_;
    SpecConstants specConstants_;
    VkSampler nearestSampler_;

    inline static const DescriptorConfigs descriptorConfigs_{
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER}, // Src
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER}, // SrcFlow
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},          // DstFlow
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},         // DstFlow
    };
};

/*******************************************************************************
 * SubpixelME
 *******************************************************************************/

class SubpixelME : public ComputePipeline {
  public:
    SubpixelME(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &loader, VkDevice device,
               const std::shared_ptr<PipelineCache> &pipelineCache, std::shared_ptr<Image> srcImageSearch,
               std::shared_ptr<Image> srcImageTemplate, std::shared_ptr<Image> srcFlow,
               std::shared_ptr<Image> prevLevelFlow, const std::shared_ptr<Image> &dstFlow, bool doAccumulate,
               const std::string &debugName);
    ~SubpixelME() override = default;

    static SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache, bool doAccumulate);

    struct SpecConstants {
        uint32_t threadGroupSizeX;
        uint32_t threadGroupSizeY;
        uint32_t outputWidth;
        uint32_t outputHeight;
        uint32_t inputFlowStride;
        uint32_t previousFlowStride;
        uint32_t outputFlowStride;
    };

    SpecConstants makeSpecConstants() const;

    void bindAndDispatch(VkCommandBuffer cmdBuf) override;

  private:
    static constexpr std::string_view shaderBaseName = "subpixel_me";
    static std::string makeShaderName(bool doAccumulate);
    std::shared_ptr<Image> srcSearch_;
    std::shared_ptr<Image> srcTemplate_;
    std::shared_ptr<Image> srcFlow_;
    std::shared_ptr<Image> srcPrevLevelFlow_;
    std::shared_ptr<Image> dstFlow_;
    bool doAccumulate_;
    SpecConstants specConstants_;
    VkSampler nearestZeroPadSampler_;
    VkSampler nearestRepeatPadSampler_;

    inline static const DescriptorConfigs descriptorConfigs_{
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER}, // SrcSearch
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER}, // SrcTemplate
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},         // SrcFlow
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},         // PrevFlow
        {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},         // DstFlow
    };
};

/*******************************************************************************
 * MVReplace
 *******************************************************************************/

class MVReplace : public ComputePipeline {
  public:
    MVReplace(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &loader, VkDevice device,
              const std::shared_ptr<PipelineCache> &pipelineCache, std::shared_ptr<Image> mvInput,
              std::shared_ptr<Image> flowBlockMatch, std::shared_ptr<Image> costAtInput,
              std::shared_ptr<Image> minCostBlockMatch, const std::shared_ptr<Image> &dstFlow,
              std::shared_ptr<Image> dstCost, bool outputCost, const std::string &debugName);
    ~MVReplace() override = default;

    static SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache, bool outputCost);

    struct SpecConstants {
        uint32_t threadGroupSizeX;
        uint32_t threadGroupSizeY;
        uint32_t outputWidth;
        uint32_t outputHeight;
    };

    SpecConstants makeSpecConstants() const;

    void setInputMv(std::shared_ptr<Image> srcMV);
    void setOutputFlow(std::shared_ptr<Image> dstFlow);
    void setOutputCost(std::shared_ptr<Image> dstCost);
    void bindAndDispatch(VkCommandBuffer cmdBuf) override;

  private:
    static constexpr std::string_view shaderBaseName = "mv_replace";
    static std::string makeShaderName(bool outputCost);
    std::shared_ptr<Image> srcInputMV_;
    std::shared_ptr<Image> srcBlockMatchFlow_;
    std::shared_ptr<Image> srcInputMVCost_;
    std::shared_ptr<Image> srcBlockMatchCost_;
    std::shared_ptr<Image> dstFlow_;
    std::shared_ptr<Image> dstCost_;
    bool outputCost_;
    SpecConstants specConstants_;
    VkSampler nearestSampler_;

    inline static const DescriptorConfigs descriptorConfigs_{
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER}, // MvInput
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER}, // FlowBlockMatch
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER}, // CostAtInputMv
        {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER}, // MinCostBlockMatchMem
        {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},          // DstFlow
        {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},          // DstCost
    };
};

/*******************************************************************************
 * BlockMatch
 *******************************************************************************/

class BlockMatch : public ComputePipeline {
  public:
    using SearchType = common::BlockMatchMode;

    BlockMatch(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &loader, VkDevice device,
               const std::shared_ptr<PipelineCache> &pipelineCache, SearchType searchType, int32_t maxSearchRange,
               const std::shared_ptr<Image> &srcSearch, std::shared_ptr<Image> srcTemplate,
               std::shared_ptr<Image> dstFlow, std::shared_ptr<Image> dstCost, const std::string &debugName);
    ~BlockMatch() override = default;

    static SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache, SearchType searchType,
                                   bool costImageStore);

    struct SpecConstants {
        uint32_t threadGroupSizeX;
        uint32_t threadGroupSizeY;
        int32_t searchType;
        uint32_t outputWidth;
        uint32_t outputHeight;
        uint32_t outputFlowStride;
        uint32_t outputCostStride;
    };

    struct PushConstants {
        int32_t searchIndexLimit;
    };

    SpecConstants makeSpecConstants() const;
    bool hasFlowOutput() const;
    bool hasCostOutput() const;

    void setOutputCost(std::shared_ptr<Image> dstImage);
    void setSearchRangeLimit(int searchRangeLimit);
    void bindAndDispatch(VkCommandBuffer cmdBuf) override;

  private:
    static constexpr std::string_view shaderBaseName = "block_match_of";
    static std::string makeShaderName(SearchType searchType, bool costImageStore);
    std::shared_ptr<Image> srcSearch_;
    std::shared_ptr<Image> srcTemplate_;
    std::shared_ptr<Image> dstFlow_;
    std::shared_ptr<Image> dstCost_;
    SearchType searchType_;
    int maxSearchRange_;
    int searchRangeLimit_;
    SpecConstants specConstants_;
    VkSampler nearestSampler_;

    inline static const DescriptorConfigs descriptorConfigs_{
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER}, // SrcTemplate
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER}, // SrcSearch
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},         // DstFlow
        {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},          // DstCost
        {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},         // DstCost
    };
};

} // namespace mlsdk::el::compute::optical_flow
