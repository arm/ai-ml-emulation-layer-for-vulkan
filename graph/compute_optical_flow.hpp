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

#include <string_view>
#include <vector>

using namespace mlsdk::el::utils;
namespace mlsdk::el::compute::optical_flow {

struct DescriptorConfig {
    uint32_t index = 0;
    VkDescriptorType type = VK_DESCRIPTOR_TYPE_MAX_ENUM;
    VkDescriptorBindingFlags flags{};
    DescriptorConfig(uint32_t index_in, VkDescriptorType type_in, VkDescriptorBindingFlags flags_in)
        : index(index_in), type(type_in), flags(flags_in) {}

  private:
    DescriptorConfig() = delete;
};
using DescriptorConfigs = std::vector<DescriptorConfig>;

struct SpecConstants {
    const void *pointer;
    uint32_t sizeBytes;
};

class ScheduleHelper {
  public:
    ScheduleHelper(uint32_t width, uint32_t height) {
        groupCountX = divideRoundUp(width, localSizeX);
        groupCountY = divideRoundUp(height, localSizeY);
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
                    const std::shared_ptr<PipelineCache> &pipelineCache, std::string_view shaderName,
                    const DescriptorConfigs &descriptorConfigs, const SpecConstants &specConstants,
                    uint32_t pushConstantsSize, const ScheduleHelper &schedule, const std::string &debugName);
    virtual ~ComputePipeline();

    void makePipeline();
    void setInputStorage(VkCommandBuffer cmdBuf, uint32_t binding, const std::shared_ptr<Image> &image,
                         VkSampler sampler = VK_NULL_HANDLE);
    void setOutputStorage(VkCommandBuffer cmdBuf, uint32_t binding, const std::shared_ptr<Image> &image);

    virtual void bindAndDispatch(VkCommandBuffer cmdBuf) = 0;

  protected:
    VkSampler createSampler(VkFilter filter, VkSamplerAddressMode addressMode, bool unnormalizedCoordinates);
    template <typename T> void setPushConstants(VkCommandBuffer cmdBuf, const T &constants);
    void bindPipeline(VkCommandBuffer cmdBuf);
    void dispatchPipeline(VkCommandBuffer cmdBuf);

  private:
    SpirvBinary createSpirv(std::string_view shaderName) const;
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
           std::shared_ptr<Image> dstDownsampledImage, std::shared_ptr<Image> dstFullImage, bool outputDownsample,
           bool outputFullRes, float downsampleScale, const std::string &debugName);
    ~RGBToY() override = default;

    struct SpecConstants {
        uint32_t threadGroupSizeX;
        uint32_t threadGroupSizeY;
        uint32_t vectorisationfactorX;
        uint32_t vectorisationfactorY;
        VkBool32 isLumaInput;
        VkBool32 outputDownsampledImage;
        VkBool32 outputFullImage;
        VkBool32 isImageStore;
        float downsampleScaleX;
        float downsampleScaleY;
        uint32_t downsampledImageWidth;
        uint32_t downsampledImageHeight;
        uint32_t downsampledImageStride;
        uint32_t fullImageWidth;
        uint32_t fullImageHeight;
        uint32_t fullImageStride;
    };

    SpecConstants makeSpecConstants(const std::shared_ptr<Image> &srcRGBImage,
                                    const std::shared_ptr<Image> &dstDownsampledImage,
                                    const std::shared_ptr<Image> &dstFullImage, bool outputDownsample,
                                    bool outputFullRes, float downsampleScale) const;
    void setInput(std::shared_ptr<Image> src);
    void bindAndDispatch(VkCommandBuffer cmdBuf) override;

  private:
    static constexpr std::string_view shaderName = "rgb_to_y";
    std::shared_ptr<Image> srcImage_;
    std::shared_ptr<Image> dstYDownsampled_;
    std::shared_ptr<Image> dstYFull_;
    bool outputDS_;
    bool outputFull_;
    float scale_ = 1.f;
    SpecConstants specConstants_;
    VkSampler linearSampler_;

    inline static const DescriptorConfigs descriptorConfigs_{
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, {}},                                // Src
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT},  // DstDs
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT}, // DstDs
        {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT},  // DstFull
        {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT}  // DstFull
    };
};

/*******************************************************************************
 * Downsample
 *******************************************************************************/

class Downsample : public ComputePipeline {
  public:
    Downsample(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &loader, VkDevice device,
               const std::shared_ptr<PipelineCache> &pipelineCache, std::shared_ptr<Image> src,
               std::shared_ptr<Image> dst, const std::string &debugName);
    ~Downsample() override = default;

    struct SpecConstants {
        uint32_t threadGroupSizeX;
        uint32_t threadGroupSizeY;
        uint32_t vectorisationfactorX;
        uint32_t vectorisationfactorY;
        VkBool32 isImageStore;
        VkBool32 padX;
        VkBool32 padY;
        uint32_t outputImageWidth;
        uint32_t outputImageHeight;
        uint32_t outputImageStride;
    };

    SpecConstants makeSpecConstants(const std::shared_ptr<Image> &srcImage,
                                    const std::shared_ptr<Image> &dstImage) const;

    void bindAndDispatch(VkCommandBuffer cmdBuf) override;

  private:
    static constexpr std::string_view shaderName = "downsample";
    std::shared_ptr<Image> srcImage_;
    std::shared_ptr<Image> dstImage_;
    SpecConstants specConstants_;
    VkSampler linearSampler_;

    inline static const DescriptorConfigs descriptorConfigs_{
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, {}},                               // Src
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT}, // Dst
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT} // Dst
    };
};

/*******************************************************************************
 * MVProcessAndWarp
 *******************************************************************************/

class MVProcessAndWarp : public ComputePipeline {
  public:
    MVProcessAndWarp(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &loader,
                     VkDevice device, const std::shared_ptr<PipelineCache> &pipelineCache,
                     std::shared_ptr<Image> srcImage, std::shared_ptr<Image> _srcFlow, std::shared_ptr<Image> dstImage,
                     std::shared_ptr<Image> _dstFlow, const std::string &debugName);
    ~MVProcessAndWarp() override = default;

    struct SpecConstants {
        uint32_t threadGroupSizeX;
        uint32_t threadGroupSizeY;
        uint32_t vectorisationfactorX;
        uint32_t vectorisationfactorY;
        VkBool32 isImageStore;
        float downsampleScaleX;
        float downsampleScaleY;
        float upsampleScaleX;
        float upsampleScaleY;
        uint32_t outputImageWidth;
        uint32_t outputImageHeight;
        uint32_t outputImageStride;
        uint32_t outputFlowStride;
    };

    SpecConstants makeSpecConstants(const std::shared_ptr<Image> &dstImage,
                                    const std::shared_ptr<Image> &dstFlow) const;
    void bindAndDispatch(VkCommandBuffer cmdBuf) override;

  private:
    static constexpr std::string_view shaderName = "mv_process_and_warp";
    std::shared_ptr<Image> srcSearch_;
    std::shared_ptr<Image> srcFlow_;
    std::shared_ptr<Image> dstWarped_;
    std::shared_ptr<Image> dstFlow_;
    SpecConstants specConstants_;
    VkSampler linearSampler_;

    inline static const DescriptorConfigs descriptorConfigs_{
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, {}},                                // Src
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, {}},                                // SrcFlow
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT},  // Dst
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT}, // Dst
        {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT},  // DstFlow
        {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT}  // DstFlow
    };
};

/*******************************************************************************
 * DenseWarp
 *******************************************************************************/

class DenseWarp : public ComputePipeline {
  public:
    DenseWarp(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &loader, VkDevice device,
              const std::shared_ptr<PipelineCache> &pipelineCache, std::shared_ptr<Image> srcImage,
              std::shared_ptr<Image> _srcFlow, std::shared_ptr<Image> dstImage, float inputFlowScale,
              const std::string &debugName);
    ~DenseWarp() override = default;

    struct SpecConstants {
        uint32_t threadGroupSizeX;
        uint32_t threadGroupSizeY;
        uint32_t vectorisationfactorX;
        uint32_t vectorisationfactorY;
        VkBool32 isImageStore;
        float inputFlowScale;
        uint32_t outputImageWidth;
        uint32_t outputImageHeight;
        uint32_t outputImageStride;
    };

    SpecConstants makeSpecConstants(const std::shared_ptr<Image> &dstImage, float inputFlowScale) const;

    void setInputFlow(std::shared_ptr<Image> _srcFlow);

    void bindAndDispatch(VkCommandBuffer cmdBuf) override;

  private:
    static constexpr std::string_view shaderName = "dense_warp";
    std::shared_ptr<Image> srcSearch_;
    std::shared_ptr<Image> srcFlow_;
    std::shared_ptr<Image> dstWarped_;
    SpecConstants specConstants_;
    VkSampler linearSampler_;
    VkSampler nearestSampler_;

    inline static const DescriptorConfigs descriptorConfigs_{
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, {}},                                // Src
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, {}},                                // SrcFlow
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT},  // Dst
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT}, // Dst
    };
};

/*******************************************************************************
 * MedianFilter
 *******************************************************************************/

class MedianFilter : public ComputePipeline {
  public:
    MedianFilter(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &loader, VkDevice device,
                 const std::shared_ptr<PipelineCache> &pipelineCache, std::shared_ptr<Image> srcImage,
                 std::shared_ptr<Image> dstImage, float outputFlowScale, const std::string &debugName);
    ~MedianFilter() override = default;

    struct SpecConstants {
        uint32_t threadGroupSizeX;
        uint32_t threadGroupSizeY;
        uint32_t vectorisationfactorX;
        uint32_t vectorisationfactorY;
        VkBool32 isImageStore;
        float outputFlowScale;
        uint32_t outputImageWidth;
        uint32_t outputImageHeight;
        uint32_t outputImageStride;
    };

    SpecConstants makeSpecConstants(const std::shared_ptr<Image> &dstImage, float outputFlowScale) const;

    void setOutput(std::shared_ptr<Image> dstImage);
    void bindAndDispatch(VkCommandBuffer cmdBuf) override;

  private:
    static constexpr std::string_view shaderName = "median_filter";
    std::shared_ptr<Image> srcFlow_;
    std::shared_ptr<Image> dstFlow_;
    SpecConstants specConstants_;
    VkSampler nearestSampler_;

    inline static const DescriptorConfigs descriptorConfigs_{
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, {}},                               // Src
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT}, // Dst
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT} // Dst
    };
};

/*******************************************************************************
 * BilateralFilter
 *******************************************************************************/

class BilateralFilter : public ComputePipeline {
  public:
    BilateralFilter(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &loader, VkDevice device,
                    const std::shared_ptr<PipelineCache> &pipelineCache, std::shared_ptr<Image> srcImage,
                    std::shared_ptr<Image> srcFlow, std::shared_ptr<Image> dstFlow, float outputFlowScale,
                    const std::string &debugName);
    ~BilateralFilter() override = default;

    struct SpecConstants {
        uint32_t threadGroupSizeX;
        uint32_t threadGroupSizeY;
        uint32_t vectorisationfactorX;
        uint32_t vectorisationfactorY;
        VkBool32 isImageStore;
        float outputFlowScale;
        uint32_t outputImageWidth;
        uint32_t outputImageHeight;
        uint32_t outputImageStride;
    };

    SpecConstants makeSpecConstants(const std::shared_ptr<Image> &dstFlow, float outputFlowScale) const;

    void setOutput(std::shared_ptr<Image> _dstFlow);
    void bindAndDispatch(VkCommandBuffer cmdBuf) override;

  private:
    static constexpr std::string_view shaderName = "bilateral_filter";
    std::shared_ptr<Image> srcTemplate_;
    std::shared_ptr<Image> srcFlow_;
    std::shared_ptr<Image> dstFlow_;
    SpecConstants specConstants_;
    VkSampler nearestSampler_;

    inline static const DescriptorConfigs descriptorConfigs_{
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, {}},                               // Src
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, {}},                               // SrcFlow
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT}, // DstFlow
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT} // DstFlow
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
               std::shared_ptr<Image> prevLevelFlow, std::shared_ptr<Image> dstFlow, bool doAccumulate,
               const std::string &debugName);
    ~SubpixelME() override = default;

    struct SpecConstants {
        uint32_t threadGroupSizeX;
        uint32_t threadGroupSizeY;
        uint32_t vectorisationfactorX;
        uint32_t vectorisationfactorY;
        VkBool32 doAccumulate;
        VkBool32 isFlowBufferLoad;
        VkBool32 isImageStore;
        uint32_t outputWidth;
        uint32_t outputHeight;
        uint32_t inputFlowStride;
        uint32_t previousFlowStride;
        uint32_t outputFlowStride;
    };

    SpecConstants makeSpecConstants(const std::shared_ptr<Image> &srcFlow, const std::shared_ptr<Image> &prevLevelFlow,
                                    const std::shared_ptr<Image> &dstFlow, bool doAccumulate) const;

    void setOutput(std::shared_ptr<Image> _dstFlow);
    void bindAndDispatch(VkCommandBuffer cmdBuf) override;

  private:
    static constexpr std::string_view shaderName = "subpixel_me";
    std::shared_ptr<Image> srcSearch_;
    std::shared_ptr<Image> srcTemplate_;
    std::shared_ptr<Image> srcFlow_;
    std::shared_ptr<Image> srcPrevLevelFlow_;
    std::shared_ptr<Image> dstFlow_;
    SpecConstants specConstants_;
    VkSampler nearestZeroPadSampler_;
    VkSampler nearestRepeatPadSampler_;

    inline static const DescriptorConfigs descriptorConfigs_{
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, {}},                                        // SrcSearch
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, {}},                                        // SrcTemplate
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, {}},                                                // SrcFlow
        {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT}, // PrevFlow
        {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT},         // PrevFlow
        {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT},          // DstFlow
        {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT}          // DstFlow
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
              std::shared_ptr<Image> minCostBlockMatch, std::shared_ptr<Image> dstFlow, std::shared_ptr<Image> dstCost,
              bool outputCost, const std::string &debugName);
    ~MVReplace() override = default;

    struct SpecConstants {
        uint32_t threadGroupSizeX;
        uint32_t threadGroupSizeY;
        uint32_t vectorisationfactorX;
        uint32_t vectorisationfactorY;
        VkBool32 outputCost;
        VkBool32 isCostBufferLoad;
        VkBool32 isImageStore;
        uint32_t outputWidth;
        uint32_t outputHeight;
        uint32_t inputCostStride;
        uint32_t outputFlowStride;
        uint32_t outputCostStride;
    };

    SpecConstants makeSpecConstants(const std::shared_ptr<Image> &costAtInput, const std::shared_ptr<Image> &dstFlow,
                                    const std::shared_ptr<Image> &dstCost, bool outputCost) const;

    void setInputMv(std::shared_ptr<Image> srcMV);
    void setOutputFlow(std::shared_ptr<Image> dstFlow);
    void setOutputCost(std::shared_ptr<Image> dstCost);
    void bindAndDispatch(VkCommandBuffer cmdBuf) override;

  private:
    static constexpr std::string_view shaderName = "mv_replace";
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
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, {}},                                        // MvInput
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, {}},                                        // FlowBlockMatch
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT}, // CostAtInputMv
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT},         // CostAtInputMv
        {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT},                                       // MinCostBlockMatchMem
        {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT}, // MinCostBlockMatchMem
        {6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT},  // DstFlow
        {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT}, // DstFlow
        {8, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT},  // DstCost
        {9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT}, // DstCost
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
               std::shared_ptr<Image> srcSearch, std::shared_ptr<Image> srcTemplate, std::shared_ptr<Image> dstFlow,
               std::shared_ptr<Image> dstCost, const std::string &debugName);
    ~BlockMatch() override = default;

    struct SpecConstants {
        uint32_t threadGroupSizeX;
        uint32_t threadGroupSizeY;
        uint32_t vectorisationfactorX;
        uint32_t vectorisationfactorY;
        int32_t searchType;
        int32_t maxSearchRange;
        uint32_t outputWidth;
        uint32_t outputHeight;
        uint32_t outputFlowStride;
        uint32_t outputCostStride;
        VkBool32 isCostImageStore;
    };

    struct PushConstants {
        int32_t searchIndexLimit;
    };

    SpecConstants makeSpecConstants(SearchType searchType, int32_t maxSearchRange,
                                    const std::shared_ptr<Image> &dstFlow, const std::shared_ptr<Image> &dstCost) const;
    bool hasFlowOutput() const;
    bool hasCostOutput() const;

    void setOutputCost(std::shared_ptr<Image> dstImage);
    void setSearchRangeLimit(int searchRangeLimit);
    void bindAndDispatch(VkCommandBuffer cmdBuf) override;

  private:
    static constexpr std::string_view shaderName = "block_match_of";
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
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, {}},                                // SrcSearch
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, {}},                                // SrcTemplate
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT}, // DstFlow
        {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT},  // DstCost
        {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT}, // DstCost
    };
};

} // namespace mlsdk::el::compute::optical_flow
