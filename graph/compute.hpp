/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "mlel/utils.hpp"
#include "pipeline_cache.hpp"
#include "tensor.hpp"

#include <spirv-tools/libspirv.hpp>
#include <vulkan/vulkan.hpp>

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace mlsdk::el::compute {

enum NanPropagationMode {
    Propagate = 1,
    Ignore = 2,
};

enum Direction { Input, Output };

// Vector of descriptor sets, where binding is always assumed to be 0
using DescriptorMap = std::vector<std::tuple<Direction, std::shared_ptr<TensorDescriptor>>>;

using TensorDescriptorMap = std::map<std::shared_ptr<TensorDescriptor>, std::shared_ptr<Tensor>>;

/*******************************************************************************
 * ComputeDescriptorSet
 *******************************************************************************/

class ComputeDescriptorSet {
  public:
    explicit ComputeDescriptorSet(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader,
                                  const VkDevice _device, VkDescriptorPool _descriptorPool,
                                  VkDescriptorSet _descriptorSet, const std::shared_ptr<Tensor> &_tensor);
    ~ComputeDescriptorSet();

    VkDescriptorSet getVkDescriptorSet() const;
    std::shared_ptr<Tensor> getTensor() const;
    VkTensorARM getVkTensorARM() const;
    VkTensorViewARM getVkTensorViewARM() const;

    void updateDescriptorSet(VkTensorARM _tensor, VkTensorViewARM tensorView);
    void updateDescriptorSet();

  private:
    std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> loader;
    VkDevice device;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;
    std::shared_ptr<Tensor> tensor;
    VkTensorARM tensorARM;
    VkTensorViewARM tensorViewARM;
};

using ComputeDescriptorSetMap = std::map<std::tuple<VkPipelineLayout, uint32_t>, std::shared_ptr<ComputeDescriptorSet>>;

/*******************************************************************************
 * ComputePipelineLayout
 *******************************************************************************/

struct PushConstant {
    const void *pointer = nullptr;
    uint32_t size = 0;
};

class ComputePipelineLayout {
  public:
    explicit ComputePipelineLayout(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader,
                                   VkDevice _device, DescriptorMap _descriptorMap,
                                   const PushConstant &_pushConstant = {});

    ~ComputePipelineLayout();

    VkPipelineLayout getVkPipelineLayout() const;
    const DescriptorMap &getDescriptorMap() const;
    std::shared_ptr<TensorDescriptor> getTensor(const uint32_t set);

    void makeDescriptorSets(ComputeDescriptorSetMap &mapping, const TensorDescriptorMap &filter) const;
    void cmdBindAndDispatch(VkCommandBuffer commandBuffer, const ComputeDescriptorSetMap &descriptorSetMap);

  private:
    std::vector<VkDescriptorSetLayoutBinding> getDescriptorSetLayoutBinding() const;
    std::vector<VkDescriptorSetLayout> createDescriptorSetLayouts() const;
    VkDescriptorPool createDescriptorPool() const;
    VkPipelineLayout createPipelineLayout() const;
    VkDescriptorSet createDescriptorSet(const VkDescriptorPool descriptorPool, const uint32_t set) const;

    void cmdBindDescriptorSets(VkCommandBuffer commandBuffer, const ComputeDescriptorSetMap &descriptorSetMap);
    void cmdPushConstants(VkCommandBuffer commandBuffer);
    void cmdPipelineBarrier(VkCommandBuffer commandBuffer, const ComputeDescriptorSetMap &descriptorSetMap) const;

    std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> loader;
    VkDevice device;
    DescriptorMap descriptorMap;
    PushConstant pushConstant;

    std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
    VkPipelineLayout pipelineLayout;
};

/*******************************************************************************
 * ComputePipelineBase
 *******************************************************************************/

class ComputePipelineBase {
  public:
    explicit ComputePipelineBase(const std::shared_ptr<ComputePipelineLayout> &_pipelineLayout);

    virtual ~ComputePipelineBase();

    virtual void cmdBindAndDispatch(VkCommandBuffer commandBuffer, const ComputeDescriptorSetMap &descriptorSetMap);

    std::shared_ptr<ComputePipelineLayout> getComputePipelineLayout() const;

    const std::vector<std::shared_ptr<VirtualTensor>> &getParents() const;
    void pushParent(const std::shared_ptr<VirtualTensor> &tensor);

    const std::vector<std::shared_ptr<VirtualTensor>> &getDescendants() const;
    void pushDescendant(const std::shared_ptr<VirtualTensor> &tensor);

  protected:
    std::shared_ptr<ComputePipelineLayout> pipelineLayout;
    std::vector<std::shared_ptr<VirtualTensor>> parents;
    std::vector<std::shared_ptr<VirtualTensor>> descendants;
};

/*******************************************************************************
 * ComputePipeline
 *******************************************************************************/

using SpecConstants = std::vector<uint32_t>;

class ComputePipeline : public ComputePipelineBase {
  public:
    explicit ComputePipeline(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader,
                             VkDevice _device, DescriptorMap descriptorMap, const PushConstant &pushConstant,
                             const std::shared_ptr<PipelineCache> &_pipelineCache, const SpirvBinary &_spirv,
                             const std::string &debugName, const SpecConstants &_constants = {});

    ~ComputePipeline() override;

    void cmdBindAndDispatch(VkCommandBuffer commandBuffer, const ComputeDescriptorSetMap &descriptorSetMap) override;

  protected:
    VkShaderModule createShaderModule(const SpirvBinary &code) const;
    VkPipeline createComputePipeline(const SpecConstants &_constants) const;
    void connectPipelines();
    virtual void cmdDispatch(VkCommandBuffer commandBuffer);

    std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> loader;
    VkDevice device;
    std::shared_ptr<PipelineCache> pipelineCache;

    VkShaderModule shaderModule;
    VkPipeline pipeline;

    static const uint32_t warp1D = 64;
    static const uint32_t MAX_CONST_LEN = 32;
};

/*******************************************************************************
 * Argmax
 *******************************************************************************/

class Argmax : public ComputePipeline {
  public:
    Argmax(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
           const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
           const std::shared_ptr<TensorDescriptor> &_output, const uint32_t _axis, const uint32_t _nanMode,
           const std::string &debugName);

  private:
    struct PushConstant {
        uint32_t axis;
        uint32_t nanMode;
    };

    PushConstant createPushConstant(const uint32_t axis, const uint32_t nanMode) const;

    DescriptorMap createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                      const std::shared_ptr<TensorDescriptor> &output) const;

    SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache,
                            const std::shared_ptr<TensorDescriptor> &input) const;

    PushConstant pushConstant;

    static constexpr std::string_view shaderName = "argmax";
};

/*******************************************************************************
 * ArithmeticRightShift
 *******************************************************************************/

class ArithmeticRightShift : public ComputePipeline {
  public:
    ArithmeticRightShift(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader,
                         VkDevice _device, const std::shared_ptr<PipelineCache> &_pipelineCache,
                         const std::shared_ptr<TensorDescriptor> &_input1,
                         const std::shared_ptr<TensorDescriptor> &_input2,
                         const std::shared_ptr<TensorDescriptor> &_output, const bool _round,
                         const std::string &debugName);

  private:
    struct PushConstant {
        uint32_t round;
    };

    PushConstant createPushConstant(const bool round) const;

    DescriptorMap createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input1,
                                      const std::shared_ptr<TensorDescriptor> &input2,
                                      const std::shared_ptr<TensorDescriptor> &output) const;

    SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache,
                            const std::shared_ptr<TensorDescriptor> &output) const;

    PushConstant pushConstant;

    static constexpr std::string_view shaderName = "arithmetic_right_shift";
};

/*******************************************************************************
 * AvgPool2D
 *******************************************************************************/

class AvgPool2D : public ComputePipeline {
  public:
    AvgPool2D(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
              const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
              const std::shared_ptr<TensorDescriptor> &_output, const std::vector<uint32_t> &_kernel,
              const std::vector<uint32_t> &_stride, const std::vector<uint32_t> &_pad, const uint32_t _accType,
              const int8_t _inputZeroPoint, const int8_t _outputZeroPoint, const std::string &debugName);

  private:
    struct PushConstant {
        uint32_t kernel[2];
        uint32_t stride[2];
        uint32_t pad[4];
        int32_t inputZeroPoint;
        int32_t outputZeroPoint;
    };

    PushConstant createPushConstant(const std::vector<uint32_t> &kernel, const std::vector<uint32_t> &stride,
                                    const std::vector<uint32_t> &pad, const int8_t inputZeroPoint,
                                    const int8_t outputZeroPoint) const;

    DescriptorMap createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                      const std::shared_ptr<TensorDescriptor> &output) const;

    SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache,
                            const std::shared_ptr<TensorDescriptor> &output, const uint32_t accType) const;

    PushConstant pushConstant;

    static constexpr std::string_view shaderName = "avgpool2d";
};

/*******************************************************************************
 * Cast
 *******************************************************************************/

class Cast : public ComputePipeline {
  public:
    Cast(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
         const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
         const std::shared_ptr<TensorDescriptor> &_output, const std::string &debugName);

  private:
    DescriptorMap createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                      const std::shared_ptr<TensorDescriptor> &output) const;

    SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache,
                            const std::shared_ptr<TensorDescriptor> &input,
                            const std::shared_ptr<TensorDescriptor> &output) const;

    static constexpr std::string_view shaderName = "cast";
};

/*******************************************************************************
 * Clamp
 *******************************************************************************/

class Clamp : public ComputePipeline {
  public:
    Clamp(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
          const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
          const std::shared_ptr<TensorDescriptor> &_output, const real_t _min, const real_t _max,
          const uint32_t _nanMode, const std::string &debugName);

  private:
    struct PushConstant {
        real_t min;
        real_t max;
        uint32_t nanMode;
    };

    PushConstant createPushConstant(const real_t min, const real_t max, const uint32_t nanMode) const;

    DescriptorMap createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                      const std::shared_ptr<TensorDescriptor> &output) const;

    SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache,
                            const std::shared_ptr<TensorDescriptor> &output) const;

    PushConstant pushConstant;

    static constexpr std::string_view shaderName = "clamp";
};

/*******************************************************************************
 * Concat
 *******************************************************************************/

class Concat : public ComputePipeline {
  public:
    Concat(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
           const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
           const std::shared_ptr<TensorDescriptor> &_output, const uint32_t _axis, const uint32_t _offset,
           const std::string &debugName);

  private:
    struct PushConstant {
        uint32_t axis;
        uint32_t offset;
    };

    PushConstant createPushConstant(const uint32_t axis, const uint32_t offset) const;

    DescriptorMap createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                      const std::shared_ptr<TensorDescriptor> &output) const;

    SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache,
                            const std::shared_ptr<TensorDescriptor> &output) const;

    void cmdDispatch(VkCommandBuffer commandBuffer) override;

    PushConstant pushConstant;

    static constexpr std::string_view shaderName = "concat";
};

/*******************************************************************************
 * Conv2D
 *******************************************************************************/

class Conv2D : public ComputePipeline {
  public:
    Conv2D(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
           const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
           const std::shared_ptr<TensorDescriptor> &_output, const std::shared_ptr<TensorDescriptor> &_weights,
           const std::shared_ptr<TensorDescriptor> &_biases, const std::vector<uint32_t> &_pad,
           const std::vector<uint32_t> &_stride, const std::vector<uint32_t> &_dilation, const int8_t _inputZeroPoint,
           const int8_t _weightZeroPoint, const uint32_t _accType, const std::string &debugName);

  private:
    struct PushConstant {
        int32_t inputZeroPoint;
        int32_t weightZeroPoint;
        uint32_t pad[4];
        uint32_t stride[2];
        uint32_t dilation[2];
    };

    PushConstant createPushConstant(const std::vector<uint32_t> &pad, const std::vector<uint32_t> &stride,
                                    const std::vector<uint32_t> &dilation, const int8_t inputZeroPoint,
                                    const int8_t weightZeroPoint) const;

    DescriptorMap createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                      const std::shared_ptr<TensorDescriptor> &output,
                                      const std::shared_ptr<TensorDescriptor> &weights,
                                      const std::shared_ptr<TensorDescriptor> &biases) const;

    SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache,
                            const std::shared_ptr<TensorDescriptor> &input,
                            const std::shared_ptr<TensorDescriptor> &output,
                            const std::shared_ptr<TensorDescriptor> &weights, const uint32_t accType) const;

    void cmdDispatch(VkCommandBuffer commandBuffer) override;

    PushConstant pushConstant;

    static constexpr std::string_view shaderName = "conv2d";

    static const uint32_t warpX = 8;
    static const uint32_t warpY = 8;
    static const uint32_t warpZ = 1;
};

/*******************************************************************************
 * Conv3D
 *******************************************************************************/

class Conv3D : public ComputePipeline {
  public:
    Conv3D(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
           const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
           const std::shared_ptr<TensorDescriptor> &_output, const std::shared_ptr<TensorDescriptor> &_weights,
           const std::shared_ptr<TensorDescriptor> &_biases, const std::vector<uint32_t> &_pad,
           const std::vector<uint32_t> &_stride, const std::vector<uint32_t> &_dilation, const int8_t _inputZeroPoint,
           const int8_t _weightZeroPoint, const uint32_t _accType, const std::string &debugName);

  private:
    struct PushConstant {
        int32_t inputZeroPoint;
        int32_t weightZeroPoint;
        uint32_t pad[6];
        uint32_t stride[3];
        uint32_t dilation[3];
    };

    PushConstant createPushConstant(const std::vector<uint32_t> &pad, const std::vector<uint32_t> &stride,
                                    const std::vector<uint32_t> &dilation, const int8_t inputZeroPoint,
                                    const int8_t weightZeroPoint) const;

    DescriptorMap createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                      const std::shared_ptr<TensorDescriptor> &output,
                                      const std::shared_ptr<TensorDescriptor> &weights,
                                      const std::shared_ptr<TensorDescriptor> &biases) const;

    SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache,
                            const std::shared_ptr<TensorDescriptor> &input,
                            const std::shared_ptr<TensorDescriptor> &output,
                            const std::shared_ptr<TensorDescriptor> &weights, const uint32_t accType) const;

    PushConstant pushConstant;

    static constexpr std::string_view shaderName = "conv3d";
};

/*******************************************************************************
 * DepthwiseConv2D
 *******************************************************************************/

class DepthwiseConv2D : public ComputePipeline {
  public:
    DepthwiseConv2D(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader,
                    VkDevice _device, const std::shared_ptr<PipelineCache> &_pipelineCache,
                    const std::shared_ptr<TensorDescriptor> &_input, const std::shared_ptr<TensorDescriptor> &_output,
                    const std::shared_ptr<TensorDescriptor> &_weights, const std::shared_ptr<TensorDescriptor> &_biases,
                    const std::vector<uint32_t> &_pad, const std::vector<uint32_t> &_stride,
                    const std::vector<uint32_t> &_dilation, const int8_t _inputZeroPoint, const int8_t _weightZeroPoint,
                    const uint32_t _accType, const std::string &debugName);

  private:
    struct PushConstant {
        int32_t inputZeroPoint;
        int32_t weightZeroPoint;
        uint32_t pad[4];
        uint32_t stride[2];
        uint32_t dilation[2];
    };

    PushConstant createPushConstant(const std::vector<uint32_t> &pad, const std::vector<uint32_t> &stride,
                                    const std::vector<uint32_t> &dilation, const int8_t inputZeroPoint,
                                    const int8_t weightZeroPoint) const;

    DescriptorMap createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                      const std::shared_ptr<TensorDescriptor> &output,
                                      const std::shared_ptr<TensorDescriptor> &weights,
                                      const std::shared_ptr<TensorDescriptor> &biases) const;

    SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache,
                            const std::shared_ptr<TensorDescriptor> &input,
                            const std::shared_ptr<TensorDescriptor> &output,
                            const std::shared_ptr<TensorDescriptor> &weights, const uint32_t accType) const;

    PushConstant pushConstant;

    static constexpr std::string_view shaderName = "depthwise_conv2d";
};

/*******************************************************************************
 * ElementwiseBinary
 *******************************************************************************/

class ElementwiseBinary : public ComputePipeline {
  public:
    ElementwiseBinary(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader,
                      VkDevice _device, const std::shared_ptr<PipelineCache> &_pipelineCache,
                      const std::shared_ptr<TensorDescriptor> &_input1,
                      const std::shared_ptr<TensorDescriptor> &_input2,
                      const std::shared_ptr<TensorDescriptor> &_output, const uint32_t _nanMode,
                      const std::string &debugName, const std::string &_operation);

  private:
    struct PushConstant {
        uint32_t nanMode;
    };

    PushConstant createPushConstant(const uint32_t nanMode) const;

    DescriptorMap createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input1,
                                      const std::shared_ptr<TensorDescriptor> &input2,
                                      const std::shared_ptr<TensorDescriptor> &output) const;

    SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache,
                            const std::shared_ptr<TensorDescriptor> &input,
                            const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName,
                            const std::string &operation) const;

    PushConstant pushConstant;
    static constexpr std::string_view shaderName = "elementwise_binary";
};

/*******************************************************************************
 * ElementwiseUnary
 *******************************************************************************/

class ElementwiseUnary : public ComputePipeline {
  public:
    ElementwiseUnary(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader,
                     VkDevice _device, const std::shared_ptr<PipelineCache> &_pipelineCache,
                     const std::shared_ptr<TensorDescriptor> &_input1, const std::shared_ptr<TensorDescriptor> &_output,
                     const std::string &debugName, const std::string &_operation);

  private:
    DescriptorMap createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input1,
                                      const std::shared_ptr<TensorDescriptor> &output) const;

    SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache,
                            const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName,
                            const std::string &operation) const;

    static constexpr std::string_view shaderName = "elementwise_unary";
};

/*******************************************************************************
 * Fft2D
 *******************************************************************************/

class Fft2D : public ComputePipeline {
  public:
    Fft2D(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
          const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_inputReal,
          const std::shared_ptr<TensorDescriptor> &_inputImag, const std::shared_ptr<TensorDescriptor> &_outputReal,
          const std::shared_ptr<TensorDescriptor> &_outputImag, const bool _inverse, const std::string &debugName);

  private:
    struct PushConstant {
        float signValue;
    };

    PushConstant createPushConstant(const bool inverse) const;

    DescriptorMap createDescriptorMap(const std::shared_ptr<TensorDescriptor> &inputReal,
                                      const std::shared_ptr<TensorDescriptor> &inputImag,
                                      const std::shared_ptr<TensorDescriptor> &outputReal,
                                      const std::shared_ptr<TensorDescriptor> &outputImag) const;

    SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache) const;
    PushConstant pushConstant;

    static constexpr std::string_view shaderName = "fft2d";
};

/*******************************************************************************
 * Gather
 *******************************************************************************/

class Gather : public ComputePipeline {
  public:
    Gather(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
           const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_values,
           const std::shared_ptr<TensorDescriptor> &_indices, const std::shared_ptr<TensorDescriptor> &_output,
           const std::string &debugName);

  private:
    DescriptorMap createDescriptorMap(const std::shared_ptr<TensorDescriptor> &values,
                                      const std::shared_ptr<TensorDescriptor> &indices,
                                      const std::shared_ptr<TensorDescriptor> &output) const;

    SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache,
                            const std::shared_ptr<TensorDescriptor> &indices,
                            const std::shared_ptr<TensorDescriptor> &output) const;

    static constexpr std::string_view shaderName = "gather";
};

/*******************************************************************************
 * Matmul
 *******************************************************************************/

class Matmul : public ComputePipeline {
  public:
    Matmul(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
           const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input1,
           const std::shared_ptr<TensorDescriptor> &_input2, const std::shared_ptr<TensorDescriptor> &_output,
           const int32_t _inputZeroPoint1, const int32_t _inputZeroPoint2, const std::string &debugName);

  private:
    struct PushConstant {
        int32_t inputZeroPoint1;
        int32_t inputZeroPoint2;
    };

    PushConstant createPushConstant(const int32_t inputZeroPoint1, const int32_t inputZeroPoint2) const;

    DescriptorMap createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input1,
                                      const std::shared_ptr<TensorDescriptor> &input2,
                                      const std::shared_ptr<TensorDescriptor> &output) const;

    SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache,
                            const std::shared_ptr<TensorDescriptor> &input1,
                            const std::shared_ptr<TensorDescriptor> &output) const;

    PushConstant pushConstant;

    static constexpr std::string_view shaderName = "matmul";
};

/*******************************************************************************
 * MaxPool2D
 *******************************************************************************/

class MaxPool2D : public ComputePipeline {
  public:
    MaxPool2D(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
              const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
              const std::shared_ptr<TensorDescriptor> &_output, const std::vector<uint32_t> &_kernel,
              const std::vector<uint32_t> &_stride, const std::vector<uint32_t> &_pad, const uint32_t _nanMode,
              const std::string &debugName);

  private:
    struct PushConstant {
        uint32_t kernel[2];
        uint32_t stride[2];
        uint32_t pad[4];
        uint32_t nanMode;
    };

    PushConstant createPushConstant(const std::vector<uint32_t> &kernel, const std::vector<uint32_t> &stride,
                                    const std::vector<uint32_t> &pad, const uint32_t nanMode) const;

    DescriptorMap createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                      const std::shared_ptr<TensorDescriptor> &output) const;

    SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache,
                            const std::shared_ptr<TensorDescriptor> &output, const uint32_t _nanMode) const;

    PushConstant pushConstant;

    static constexpr std::string_view shaderName = "maxpool2d";
};

/*******************************************************************************
 * Mul
 *******************************************************************************/

class Mul : public ComputePipeline {
  public:
    Mul(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
        const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input1,
        const std::shared_ptr<TensorDescriptor> &_input2, const std::shared_ptr<TensorDescriptor> &_output,
        const uint32_t _shift, const std::string &debugName);

  private:
    struct PushConstant {
        uint32_t shift;
    };

    PushConstant createPushConstant(const uint32_t shift) const;

    DescriptorMap createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input1,
                                      const std::shared_ptr<TensorDescriptor> &input2,
                                      const std::shared_ptr<TensorDescriptor> &output) const;

    SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache,
                            const std::shared_ptr<TensorDescriptor> &input1,
                            const std::shared_ptr<TensorDescriptor> &output) const;

    PushConstant pushConstant;

    static constexpr std::string_view shaderName = "mul";
};

/*******************************************************************************
 * Negate
 *******************************************************************************/

class Negate : public ComputePipeline {
  public:
    Negate(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
           const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
           const std::shared_ptr<TensorDescriptor> &_output, const int32_t _inputZeroPoint,
           const int32_t _outputZeroPoint, const std::string &debugName);

  private:
    struct PushConstant {
        int32_t inputZeroPoint;
        int32_t outputZeroPoint;
    };

    PushConstant createPushConstant(const int32_t inputZeroPoint, const int32_t outputZeroPoint) const;

    DescriptorMap createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                      const std::shared_ptr<TensorDescriptor> &output) const;

    SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache,
                            const std::shared_ptr<TensorDescriptor> &output) const;

    PushConstant pushConstant;

    static constexpr std::string_view shaderName = "negate";
};

/*******************************************************************************
 * Pad
 *******************************************************************************/

class Pad : public ComputePipeline {
  public:
    Pad(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
        const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
        const std::shared_ptr<TensorDescriptor> &_output, const std::shared_ptr<TensorDescriptor> &_padding,
        const real_t _padConst, const std::string &debugName);

  private:
    struct PushConstant {
        real_t padConst;
    };

    PushConstant createPushConstant(const real_t padConst) const;

    DescriptorMap createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                      const std::shared_ptr<TensorDescriptor> &output,
                                      const std::shared_ptr<TensorDescriptor> &padding) const;

    SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache,
                            const std::shared_ptr<TensorDescriptor> &output) const;

    PushConstant pushConstant;

    static constexpr std::string_view shaderName = "pad";
};

/*******************************************************************************
 * Reduce
 *******************************************************************************/

class Reduce : public ComputePipeline {
  public:
    Reduce(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
           const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
           const std::shared_ptr<TensorDescriptor> &_output, const uint32_t _axis, const uint32_t _nanMode,
           const std::string &debugName, const std::string &_init, const std::string &_operation);

  private:
    struct PushConstant {
        uint32_t axis;
        uint32_t nanMode;
    };

    PushConstant createPushConstant(const uint32_t axis, const uint32_t nanMode, const bool isInteger) const;

    DescriptorMap createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                      const std::shared_ptr<TensorDescriptor> &output) const;

    SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache,
                            const std::shared_ptr<TensorDescriptor> &output, const std::string &name,
                            const std::string &init, const std::string &operation) const;

    PushConstant pushConstant;

    static constexpr std::string_view shaderName = "reduce";
};

/*******************************************************************************
 * Rescale
 *******************************************************************************/

class Rescale : public ComputePipeline {
  public:
    Rescale(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
            const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
            const std::shared_ptr<TensorDescriptor> &_output, const int32_t _inputZeroPoint,
            const int32_t _outputZeroPoint, const std::shared_ptr<TensorDescriptor> &_multiplier,
            const std::shared_ptr<TensorDescriptor> &_shift, const bool _scale32, const bool _doubleRound,
            const bool _perChannel, const bool _inputUnsigned, const bool _outputUnsigned,
            const std::string &debugName);

  private:
    struct PushConstant {
        int32_t inputZeroPoint;
        int32_t outputZeroPoint;
    };

    PushConstant createPushConstant(const int32_t inputZeroPoint, const int32_t outputZeroPoint) const;

    DescriptorMap createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                      const std::shared_ptr<TensorDescriptor> &output,
                                      const std::shared_ptr<TensorDescriptor> &multiplier,
                                      const std::shared_ptr<TensorDescriptor> &shift) const;

    SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache,
                            const std::shared_ptr<TensorDescriptor> &input,
                            const std::shared_ptr<TensorDescriptor> &output,
                            const std::shared_ptr<TensorDescriptor> &multiplier, const bool inputUnsigned,
                            const bool outputUnsigned) const;

    PushConstant pushConstant;

    static constexpr std::string_view shaderName = "rescale";
};

/*******************************************************************************
 * Reshape
 *******************************************************************************/

class Reshape : public ComputePipeline {
  public:
    Reshape(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
            const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
            const std::shared_ptr<TensorDescriptor> &_output, const std::string &debugName);

  private:
    DescriptorMap createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                      const std::shared_ptr<TensorDescriptor> &output) const;
    SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache,
                            const std::shared_ptr<TensorDescriptor> &output) const;

    static constexpr std::string_view shaderName = "reshape";
};

/*******************************************************************************
 * Resize
 *******************************************************************************/

class Resize : public ComputePipeline {
  public:
    Resize(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
           const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
           const std::shared_ptr<TensorDescriptor> &_output, const std::vector<int32_t> &_scale,
           const std::vector<int32_t> &_offset, const std::vector<int32_t> &_border, const uint32_t _mode,
           const std::string &debugName);

  private:
    struct PushConstant {
        int32_t scale[4];
        int32_t offset[2];
        int32_t border[2];
        uint32_t mode;
    };

    PushConstant createPushConstant(const std::vector<int32_t> &scale, const std::vector<int32_t> &offset,
                                    const std::vector<int32_t> &border, const uint32_t mode) const;

    DescriptorMap createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                      const std::shared_ptr<TensorDescriptor> &output) const;

    SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache,
                            const std::shared_ptr<TensorDescriptor> &input,
                            const std::shared_ptr<TensorDescriptor> &output) const;

    PushConstant pushConstant;

    static constexpr std::string_view shaderName = "resize";
};

/*******************************************************************************
 * Reverse
 *******************************************************************************/

class Reverse : public ComputePipeline {
  public:
    Reverse(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
            const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
            const std::shared_ptr<TensorDescriptor> &_output, const uint32_t _axis, const std::string &debugName);

  private:
    struct PushConstant {
        uint32_t axis;
    };

    PushConstant createPushConstant(const uint32_t axis) const;

    DescriptorMap createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                      const std::shared_ptr<TensorDescriptor> &output) const;

    SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache,
                            const std::shared_ptr<TensorDescriptor> &output) const;

    PushConstant pushConstant;

    static constexpr std::string_view shaderName = "reverse";
};

/*******************************************************************************
 * Rfft2D
 *******************************************************************************/

class Rfft2D : public ComputePipeline {
  public:
    Rfft2D(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
           const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
           const std::shared_ptr<TensorDescriptor> &_outputReal, const std::shared_ptr<TensorDescriptor> &_outputImag,
           const std::string &debugName);

  private:
    DescriptorMap createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                      const std::shared_ptr<TensorDescriptor> &outputReal,
                                      const std::shared_ptr<TensorDescriptor> &outputImag) const;

    SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache) const;

    static constexpr std::string_view shaderName = "rfft2d";
};

/*******************************************************************************
 * Scatter
 *******************************************************************************/

class Scatter : public ComputePipeline {
  public:
    Scatter(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
            const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
            const std::shared_ptr<TensorDescriptor> &_values, const std::shared_ptr<TensorDescriptor> &_indices,
            const std::shared_ptr<TensorDescriptor> &_output, const std::string &debugName);

  private:
    DescriptorMap createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                      const std::shared_ptr<TensorDescriptor> &values,
                                      const std::shared_ptr<TensorDescriptor> &indices,
                                      const std::shared_ptr<TensorDescriptor> &output) const;

    SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache,
                            const std::shared_ptr<TensorDescriptor> &indices,
                            const std::shared_ptr<TensorDescriptor> &output) const;

    static constexpr std::string_view shaderName = "scatter";
};

/*******************************************************************************
 * Select
 *******************************************************************************/

class Select : public ComputePipeline {
  public:
    Select(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
           const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input1,
           const std::shared_ptr<TensorDescriptor> &_input2, const std::shared_ptr<TensorDescriptor> &_input3,
           const std::shared_ptr<TensorDescriptor> &_output, const std::string &debugName);

  private:
    DescriptorMap createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input1,
                                      const std::shared_ptr<TensorDescriptor> &input2,
                                      const std::shared_ptr<TensorDescriptor> &input3,
                                      const std::shared_ptr<TensorDescriptor> &output) const;

    SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache,
                            const std::shared_ptr<TensorDescriptor> &output) const;

    static constexpr std::string_view shaderName = "select";
};

/*******************************************************************************
 * Slice
 *******************************************************************************/

class Slice : public ComputePipeline {
  public:
    Slice(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
          const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
          const std::shared_ptr<TensorDescriptor> &_output, const std::vector<uint32_t> &_start,
          const std::string &debugName);

  private:
    struct PushConstant {
        uint32_t start[MAX_CONST_LEN];
    };

    PushConstant createPushConstant(const std::vector<uint32_t> &start) const;

    DescriptorMap createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                      const std::shared_ptr<TensorDescriptor> &output) const;

    SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache,
                            const std::shared_ptr<TensorDescriptor> &input) const;

    PushConstant pushConstant;
    static constexpr std::string_view shaderName = "slice";
};

/*******************************************************************************
 * Table
 *******************************************************************************/

class Table : public ComputePipeline {
  public:
    Table(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
          const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
          const std::shared_ptr<TensorDescriptor> &_output, const std::shared_ptr<TensorDescriptor> &_table,
          const std::string &debugName);

  private:
    DescriptorMap createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                      const std::shared_ptr<TensorDescriptor> &output,
                                      const std::shared_ptr<TensorDescriptor> &table) const;

    SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache,
                            const std::shared_ptr<TensorDescriptor> &input,
                            const std::shared_ptr<TensorDescriptor> &output) const;

    static constexpr std::string_view shaderName = "table";
};

/*******************************************************************************
 * Tile
 *******************************************************************************/

class Tile : public ComputePipeline {
  public:
    Tile(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
         const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
         const std::shared_ptr<TensorDescriptor> &_output, const std::string &debugName);

  private:
    DescriptorMap createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                      const std::shared_ptr<TensorDescriptor> &output) const;

    SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache,
                            const std::shared_ptr<TensorDescriptor> &output) const;

    static constexpr std::string_view shaderName = "tile";
};

/*******************************************************************************
 * Transpose
 *******************************************************************************/

class Transpose : public ComputePipeline {
  public:
    Transpose(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
              const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
              const std::shared_ptr<TensorDescriptor> &_output, const std::vector<uint32_t> &_perms,
              const std::string &debugName);

  private:
    struct PushConstant {
        uint32_t perms[MAX_CONST_LEN];
    };

    PushConstant createPushConstant(const std::vector<uint32_t> &perms) const;

    DescriptorMap createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                      const std::shared_ptr<TensorDescriptor> &output) const;

    SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache,
                            const std::shared_ptr<TensorDescriptor> &output) const;

    PushConstant pushConstant;

    static constexpr std::string_view shaderName = "transpose";
};

/*******************************************************************************
 * TransposeConv2D
 *******************************************************************************/

class TransposeConv2D : public ComputePipeline {
  public:
    TransposeConv2D(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader,
                    VkDevice _device, const std::shared_ptr<PipelineCache> &_pipelineCache,
                    const std::shared_ptr<TensorDescriptor> &_input, const std::shared_ptr<TensorDescriptor> &_output,
                    const std::shared_ptr<TensorDescriptor> &_weights, const std::shared_ptr<TensorDescriptor> &_biases,
                    const std::vector<uint32_t> &_outPad, const std::vector<uint32_t> &_stride,
                    const int8_t _inputZeroPoint, const int8_t _weightZeroPoint, const uint32_t _accType,
                    const std::string &debugName);

  private:
    struct PushConstant {
        int32_t inputZeroPoint;
        int32_t weightZeroPoint;
        uint32_t outPad[4];
        uint32_t stride[2];
    };

    PushConstant createPushConstant(const std::vector<uint32_t> &outPad, const std::vector<uint32_t> &stride,
                                    const int8_t inputZeroPoint, const int8_t weightZeroPoint) const;

    DescriptorMap createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                      const std::shared_ptr<TensorDescriptor> &output,
                                      const std::shared_ptr<TensorDescriptor> &weights,
                                      const std::shared_ptr<TensorDescriptor> &biases) const;

    SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache,
                            const std::shared_ptr<TensorDescriptor> &input,
                            const std::shared_ptr<TensorDescriptor> &output,
                            const std::shared_ptr<TensorDescriptor> &weights, const uint32_t accType) const;

    PushConstant pushConstant;

    static constexpr std::string_view shaderName = "transpose_conv2d";
};

/*******************************************************************************
 * BlockMatch
 *******************************************************************************/

class BlockMatch : public ComputePipeline {
  public:
    enum SearchType : uint32_t {
        MIN_SAD = 0,
        MIN_SAD_COST = 1,
        RAW_SAD = 2,
    };

    BlockMatch(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
               const std::shared_ptr<PipelineCache> &_pipelineCache,
               const std::shared_ptr<TensorDescriptor> &inTemplate, const std::shared_ptr<TensorDescriptor> &inSearch,
               const std::optional<std::shared_ptr<TensorDescriptor>> &outVectors,
               const std::optional<std::shared_ptr<TensorDescriptor>> &outCosts,
               const std::vector<uint32_t> &kernelSizes, const std::vector<uint32_t> &searchWindowSizes,
               const std::vector<uint32_t> &inputStrides, const std::vector<uint32_t> &windowStrides,
               const std::vector<uint32_t> &windowOffsets, const std::vector<uint32_t> &padding,
               const uint32_t searchPattern, const SearchType searchType, const std::string &debugName);

  private:
    DescriptorMap createDescriptorMap(const std::shared_ptr<TensorDescriptor> &inTemplate,
                                      const std::shared_ptr<TensorDescriptor> &inSearch,
                                      const std::optional<std::shared_ptr<TensorDescriptor>> &outVectors,
                                      const std::optional<std::shared_ptr<TensorDescriptor>> &outCosts,
                                      const SearchType searchType) const;

    SpirvBinary createSpirv(const std::shared_ptr<PipelineCache> &pipelineCache, const SearchType searchType) const;

    SpecConstants createSpecConstants(const std::vector<uint32_t> &kernelSizes,
                                      const std::vector<uint32_t> &searchWindowSizes,
                                      const std::vector<uint32_t> &inputStrides,
                                      const std::vector<uint32_t> &windowStrides,
                                      const std::vector<uint32_t> &windowOffsets, const std::vector<uint32_t> &padding,
                                      const uint32_t searchPattern) const;

    void cmdDispatch(VkCommandBuffer commandBuffer) override;

    static constexpr std::string_view shaderName = "block_match";

    static const uint32_t warpX = 8;
    static const uint32_t warpY = 8;
};

/*******************************************************************************
 * GraphPipeline
 *******************************************************************************/

class GraphPipeline {
  public:
    GraphPipeline(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader,
                  VkPhysicalDevice _physicalDevice, VkDevice _device,
                  const std::shared_ptr<PipelineCache> &_pipelineCache);

    virtual ~GraphPipeline();

    ComputePipelineBase &getInputs() { return inputs; }
    ComputePipelineBase &getOutputs() { return outputs; }

    // Constant tensors owned by the pipeline
    void makeConstTensor(const uint32_t id, const VkTensorDescriptionARM &tensorDescription, const void *data);
    std::shared_ptr<TensorDescriptor> getConstTensor(const uint32_t id) const;
    std::shared_ptr<TensorDescriptor>
    makeConstCompositeTensor(const VkFormat format, const std::vector<int64_t> &dimensions, const void *data);

    // External tensors owned by the application
    void makeDescriptorSetBinding(const uint32_t set, const uint32_t binding, const uint32_t arrayIndex,
                                  const VkTensorDescriptionARM &tensorDescription);
    std::shared_ptr<TensorDescriptor> getTensor(const uint32_t set, const uint32_t binding,
                                                const uint32_t arrayIndex = 0) const;

    // Tensors allocated in session ram
    std::shared_ptr<TensorDescriptor> makeTensor(const VkFormat format, const std::vector<int64_t> &dimensions = {},
                                                 const std::vector<int64_t> &strides = {});

    const std::vector<std::shared_ptr<TensorDescriptor>> &getTensors() const;

    // Make descriptor sets
    ComputeDescriptorSetMap makeConstantsDescriptorSets() const;
    ComputeDescriptorSetMap makeSessionRamDescriptorSets() const;
    ComputeDescriptorSetMap makeExternalDescriptorSets(uint32_t set) const;

    void cmdBindAndDispatch(VkCommandBuffer commandBuffer, const ComputeDescriptorSetMap &descriptorSetMap);

    void makeInput(const std::shared_ptr<TensorDescriptor> &tensor);

    void makeOutput(const std::shared_ptr<TensorDescriptor> &tensor);

    void makeAbs(const std::shared_ptr<TensorDescriptor> &input, const std::shared_ptr<TensorDescriptor> &output,
                 const std::string &debugName);

    void makeAdd(const std::shared_ptr<TensorDescriptor> &input1, const std::shared_ptr<TensorDescriptor> &input2,
                 const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName);

    void makeArgmax(const std::shared_ptr<TensorDescriptor> &input, const std::shared_ptr<TensorDescriptor> &output,
                    const uint32_t axis, const uint32_t nanMode, const std::string &debugName);

    void makeArithmeticRightShift(const std::shared_ptr<TensorDescriptor> &input1,
                                  const std::shared_ptr<TensorDescriptor> &input2,
                                  const std::shared_ptr<TensorDescriptor> &output, const bool round,
                                  const std::string &debugName);

    void makeAvgPool2D(const std::shared_ptr<TensorDescriptor> &input, const std::shared_ptr<TensorDescriptor> &output,
                       const std::vector<uint32_t> &kernel, const std::vector<uint32_t> &stride,
                       const std::vector<uint32_t> &pad, const uint32_t accType, const int8_t inputZeroPoint,
                       const int8_t outputZeroPoint, const std::string &debugName);

    void makeBitwiseAnd(const std::shared_ptr<TensorDescriptor> &input1,
                        const std::shared_ptr<TensorDescriptor> &input2,
                        const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName);

    void makeBitwiseNot(const std::shared_ptr<TensorDescriptor> &input, const std::shared_ptr<TensorDescriptor> &output,
                        const std::string &debugName);

    void makeBitwiseOr(const std::shared_ptr<TensorDescriptor> &input1, const std::shared_ptr<TensorDescriptor> &input2,
                       const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName);

    void makeBitwiseXor(const std::shared_ptr<TensorDescriptor> &input1,
                        const std::shared_ptr<TensorDescriptor> &input2,
                        const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName);

    void makeCast(const std::shared_ptr<TensorDescriptor> &input1, const std::shared_ptr<TensorDescriptor> &output,
                  const std::string &debugName);

    void makeCeil(const std::shared_ptr<TensorDescriptor> &input1, const std::shared_ptr<TensorDescriptor> &output,
                  const std::string &debugName);

    void makeClamp(const std::shared_ptr<TensorDescriptor> &input, const std::shared_ptr<TensorDescriptor> &output,
                   const real_t min, const real_t max, const uint32_t nanMode, const std::string &debugName);

    void makeClz(const std::shared_ptr<TensorDescriptor> &input1, const std::shared_ptr<TensorDescriptor> &output,
                 const std::string &debugName);

    void makeConcat(const std::vector<std::shared_ptr<TensorDescriptor>> &inputs,
                    const std::shared_ptr<TensorDescriptor> &output, const uint32_t axis, const std::string &debugName);

    void makeConv2D(const std::shared_ptr<TensorDescriptor> &input, const std::shared_ptr<TensorDescriptor> &output,
                    const std::shared_ptr<TensorDescriptor> &weights, const std::shared_ptr<TensorDescriptor> &biases,
                    const std::vector<uint32_t> &pad, const std::vector<uint32_t> &stride,
                    const std::vector<uint32_t> &dilation, const int8_t inputZeroPoint, const int8_t weightZeroPoint,
                    const uint32_t accType, const std::string &debugName);

    void makeConv3D(const std::shared_ptr<TensorDescriptor> &input, const std::shared_ptr<TensorDescriptor> &output,
                    const std::shared_ptr<TensorDescriptor> &weights, const std::shared_ptr<TensorDescriptor> &biases,
                    const std::vector<uint32_t> &pad, const std::vector<uint32_t> &stride,
                    const std::vector<uint32_t> &dilation, const int8_t inputZeroPoint, const int8_t weightZeroPoint,
                    const uint32_t accType, const std::string &debugName);

    void makeCos(const std::shared_ptr<TensorDescriptor> &input1, const std::shared_ptr<TensorDescriptor> &output,
                 const std::string &debugName);

    void makeDepthwiseConv2D(const std::shared_ptr<TensorDescriptor> &input,
                             const std::shared_ptr<TensorDescriptor> &output,
                             const std::shared_ptr<TensorDescriptor> &weights,
                             const std::shared_ptr<TensorDescriptor> &biases, const std::vector<uint32_t> &pad,
                             const std::vector<uint32_t> &stride, const std::vector<uint32_t> &dilation,
                             const int8_t inputZeroPoint, const int8_t weightZeroPoint, const uint32_t accType,
                             const std::string &debugName);

    void makeEqual(const std::shared_ptr<TensorDescriptor> &input1, const std::shared_ptr<TensorDescriptor> &input2,
                   const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName);

    void makeErf(const std::shared_ptr<TensorDescriptor> &input, const std::shared_ptr<TensorDescriptor> &output,
                 const std::string &debugName);

    void makeExp(const std::shared_ptr<TensorDescriptor> &input, const std::shared_ptr<TensorDescriptor> &output,
                 const std::string &debugName);

    void makeFft2D(const std::shared_ptr<TensorDescriptor> &inputReal,
                   const std::shared_ptr<TensorDescriptor> &inputImag,
                   const std::shared_ptr<TensorDescriptor> &outputReal,
                   const std::shared_ptr<TensorDescriptor> &outputImag, const bool inverse,
                   const std::string &debugName);

    void makeFloor(const std::shared_ptr<TensorDescriptor> &input1, const std::shared_ptr<TensorDescriptor> &output,
                   const std::string &debugName);

    void makeGather(const std::shared_ptr<TensorDescriptor> &values, const std::shared_ptr<TensorDescriptor> &indices,
                    const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName);

    void makeGreater(const std::shared_ptr<TensorDescriptor> &input1, const std::shared_ptr<TensorDescriptor> &input2,
                     const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName);

    void makeGreaterEqual(const std::shared_ptr<TensorDescriptor> &input1,
                          const std::shared_ptr<TensorDescriptor> &input2,
                          const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName);

    void makeIntdiv(const std::shared_ptr<TensorDescriptor> &input1, const std::shared_ptr<TensorDescriptor> &input2,
                    const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName);

    void makeLog(const std::shared_ptr<TensorDescriptor> &input1, const std::shared_ptr<TensorDescriptor> &output,
                 const std::string &debugName);

    void makeLogicalAnd(const std::shared_ptr<TensorDescriptor> &input1,
                        const std::shared_ptr<TensorDescriptor> &input2,
                        const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName);

    void makeLogicalLeftShift(const std::shared_ptr<TensorDescriptor> &input1,
                              const std::shared_ptr<TensorDescriptor> &input2,
                              const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName);

    void makeLogicalNot(const std::shared_ptr<TensorDescriptor> &input, const std::shared_ptr<TensorDescriptor> &output,
                        const std::string &debugName);

    void makeLogicalRightShift(const std::shared_ptr<TensorDescriptor> &input1,
                               const std::shared_ptr<TensorDescriptor> &input2,
                               const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName);

    void makeLogicalOr(const std::shared_ptr<TensorDescriptor> &input1, const std::shared_ptr<TensorDescriptor> &input2,
                       const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName);

    void makeLogicalXor(const std::shared_ptr<TensorDescriptor> &input1,
                        const std::shared_ptr<TensorDescriptor> &input2,
                        const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName);

    void makeMaximum(const std::shared_ptr<TensorDescriptor> &input1, const std::shared_ptr<TensorDescriptor> &input2,
                     const std::shared_ptr<TensorDescriptor> &output, const uint32_t nanMode,
                     const std::string &debugName);

    void makeMinimum(const std::shared_ptr<TensorDescriptor> &input1, const std::shared_ptr<TensorDescriptor> &input2,
                     const std::shared_ptr<TensorDescriptor> &output, const uint32_t nanMode,
                     const std::string &debugName);

    void makeMatmul(const std::shared_ptr<TensorDescriptor> &input1, const std::shared_ptr<TensorDescriptor> &input2,
                    const std::shared_ptr<TensorDescriptor> &output, const int32_t inputZeroPoint1,
                    const int32_t inputZeroPoint2, const std::string &debugName);

    void makeMaxPool2D(const std::shared_ptr<TensorDescriptor> &input, const std::shared_ptr<TensorDescriptor> &output,
                       const std::vector<uint32_t> &kernel, const std::vector<uint32_t> &stride,
                       const std::vector<uint32_t> &pad, const uint32_t nanMode, const std::string &debugName);

    void makeMul(const std::shared_ptr<TensorDescriptor> &input1, const std::shared_ptr<TensorDescriptor> &input2,
                 const std::shared_ptr<TensorDescriptor> &output, const uint32_t shift, const std::string &debugName);

    void makeNegate(const std::shared_ptr<TensorDescriptor> &input, const std::shared_ptr<TensorDescriptor> &output,
                    const int32_t inputZeroPoint, const int32_t outputZeroPoint, const std::string &debugName);

    void makePad(const std::shared_ptr<TensorDescriptor> &input, const std::shared_ptr<TensorDescriptor> &output,
                 const std::shared_ptr<TensorDescriptor> &padding, const real_t padConst, const std::string &debugName);

    void makePow(const std::shared_ptr<TensorDescriptor> &input1, const std::shared_ptr<TensorDescriptor> &input2,
                 const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName);

    void makeReciprocal(const std::shared_ptr<TensorDescriptor> &input, const std::shared_ptr<TensorDescriptor> &output,
                        const std::string &debugName);

    void makeReduceAll(const std::shared_ptr<TensorDescriptor> &input, const std::shared_ptr<TensorDescriptor> &output,
                       const uint32_t axis, const std::string &debugName);

    void makeReduceAny(const std::shared_ptr<TensorDescriptor> &input, const std::shared_ptr<TensorDescriptor> &output,
                       const uint32_t axis, const std::string &debugName);

    void makeReduceMax(const std::shared_ptr<TensorDescriptor> &input, const std::shared_ptr<TensorDescriptor> &output,
                       const uint32_t axis, const uint32_t nanMode, const std::string &debugName);

    void makeReduceMin(const std::shared_ptr<TensorDescriptor> &input, const std::shared_ptr<TensorDescriptor> &output,
                       const uint32_t axis, const uint32_t nanMode, const std::string &debugName);

    void makeReduceProduct(const std::shared_ptr<TensorDescriptor> &input,
                           const std::shared_ptr<TensorDescriptor> &output, const uint32_t axis,
                           const std::string &debugName);

    void makeReduceSum(const std::shared_ptr<TensorDescriptor> &input, const std::shared_ptr<TensorDescriptor> &output,
                       const uint32_t axis, const std::string &debugName);

    void makeRescale(const std::shared_ptr<TensorDescriptor> &input, const std::shared_ptr<TensorDescriptor> &output,
                     const int32_t inputZeroPoint, const int32_t outputZeroPoint,
                     const std::shared_ptr<TensorDescriptor> &multiplier,
                     const std::shared_ptr<TensorDescriptor> &shift, const bool scale32, const bool doubleRound,
                     const bool perChannel, const bool inputUnsigned, const bool outputUnsigned,
                     const std::string &debugName);

    void makeReshape(const std::shared_ptr<TensorDescriptor> &input, const std::shared_ptr<TensorDescriptor> &output,
                     const std::string &debugName);

    void makeResize(const std::shared_ptr<TensorDescriptor> &input, const std::shared_ptr<TensorDescriptor> &output,
                    const std::vector<int32_t> &scale, const std::vector<int32_t> &offset,
                    const std::vector<int32_t> &border, const uint32_t mode, const std::string &debugName);

    void makeReverse(const std::shared_ptr<TensorDescriptor> &input, const std::shared_ptr<TensorDescriptor> &output,
                     const uint32_t axis, const std::string &debugName);

    void makeRfft2D(const std::shared_ptr<TensorDescriptor> &input, const std::shared_ptr<TensorDescriptor> &outputReal,
                    const std::shared_ptr<TensorDescriptor> &outputImag, const std::string &debugName);

    void makeRsqrt(const std::shared_ptr<TensorDescriptor> &input1, const std::shared_ptr<TensorDescriptor> &output,
                   const std::string &debugName);

    void makeScatter(const std::shared_ptr<TensorDescriptor> &input, const std::shared_ptr<TensorDescriptor> &values,
                     const std::shared_ptr<TensorDescriptor> &indices, const std::shared_ptr<TensorDescriptor> &output,
                     const std::string &debugName);

    void makeSelect(const std::shared_ptr<TensorDescriptor> &input1, const std::shared_ptr<TensorDescriptor> &input2,
                    const std::shared_ptr<TensorDescriptor> &input3, const std::shared_ptr<TensorDescriptor> &output,
                    const std::string &debugName);

    void makeSigmoid(const std::shared_ptr<TensorDescriptor> &input, const std::shared_ptr<TensorDescriptor> &output,
                     const std::string &debugName);

    void makeSin(const std::shared_ptr<TensorDescriptor> &input1, const std::shared_ptr<TensorDescriptor> &output,
                 const std::string &debugName);

    void makeSlice(const std::shared_ptr<TensorDescriptor> &input, const std::shared_ptr<TensorDescriptor> &output,
                   const std::vector<uint32_t> &start, const std::string &debugName);

    void makeSub(const std::shared_ptr<TensorDescriptor> &input1, const std::shared_ptr<TensorDescriptor> &input2,
                 const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName);

    void makeTable(const std::shared_ptr<TensorDescriptor> &input, const std::shared_ptr<TensorDescriptor> &output,
                   const std::shared_ptr<TensorDescriptor> &table, const std::string &debugName);

    void makeTanh(const std::shared_ptr<TensorDescriptor> &input1, const std::shared_ptr<TensorDescriptor> &output,
                  const std::string &debugName);

    void makeTile(const std::shared_ptr<TensorDescriptor> &input, const std::shared_ptr<TensorDescriptor> &output,
                  const std::string &debugName);

    void makeTranspose(const std::shared_ptr<TensorDescriptor> &input, const std::shared_ptr<TensorDescriptor> &output,
                       const std::vector<uint32_t> &perms, const std::string &debugName);

    void makeTransposeConv2D(const std::shared_ptr<TensorDescriptor> &input,
                             const std::shared_ptr<TensorDescriptor> &output,
                             const std::shared_ptr<TensorDescriptor> &weights,
                             const std::shared_ptr<TensorDescriptor> &biases, const std::vector<uint32_t> &pad,
                             const std::vector<uint32_t> &stride, const int8_t inputZeroPoint,
                             const int8_t weightZeroPoint, const uint32_t accType, const std::string &debugName);

    /*******************************************************************************
     * Motion Engine Ops
     *******************************************************************************/
    void makeMinSadCost(const std::shared_ptr<TensorDescriptor> &inTemplate,
                        const std::shared_ptr<TensorDescriptor> &inSearch,
                        const std::shared_ptr<TensorDescriptor> &outVectors,
                        const std::shared_ptr<TensorDescriptor> &outCosts, const std::vector<uint32_t> &kernelSizes,
                        const std::vector<uint32_t> &searchWindowSizes, const std::vector<uint32_t> &inputStrides,
                        const std::vector<uint32_t> &windowStrides, const std::vector<uint32_t> &windowOffsets,
                        const std::vector<uint32_t> &padding, const uint32_t searchPattern,
                        const std::string &debugName);

    void makeMinSad(const std::shared_ptr<TensorDescriptor> &inTemplate,
                    const std::shared_ptr<TensorDescriptor> &inSearch,
                    const std::shared_ptr<TensorDescriptor> &outVectors, const std::vector<uint32_t> &kernelSizes,
                    const std::vector<uint32_t> &searchWindowSizes, const std::vector<uint32_t> &inputStrides,
                    const std::vector<uint32_t> &windowStrides, const std::vector<uint32_t> &windowOffsets,
                    const std::vector<uint32_t> &padding, const uint32_t searchPattern, const std::string &debugName);

    void makeRawSad(const std::shared_ptr<TensorDescriptor> &inTemplate,
                    const std::shared_ptr<TensorDescriptor> &inSearch,
                    const std::shared_ptr<TensorDescriptor> &outCosts, const std::vector<uint32_t> &kernelSizes,
                    const std::vector<uint32_t> &searchWindowSizes, const std::vector<uint32_t> &inputStrides,
                    const std::vector<uint32_t> &windowStrides, const std::vector<uint32_t> &windowOffsets,
                    const std::vector<uint32_t> &padding, const std::string &debugName);

  private:
    std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> loader;
    VkPhysicalDevice physicalDevice;
    VkDevice device;

    std::shared_ptr<PipelineCache> pipelineCache;
    std::vector<std::shared_ptr<ComputePipelineBase>> pipelines;

    // Device memory for constants
    std::vector<VkDeviceMemory> constantsDeviceMemory;

    // Mapping from SPIR-V constant id to tensor
    std::map<uint32_t, std::shared_ptr<Tensor>> constTensorMap;

    // List of composite tensors
    std::vector<std::shared_ptr<Tensor>> compositeTensors;

    // Mapping from graph descriptor set and binding to tensor array
    std::map<uint32_t, std::map<uint32_t, std::vector<std::shared_ptr<TensorDescriptor>>>> tensorMap;

    // Mapping from set to tensors
    std::map<uint32_t, TensorDescriptorMap> tensorDescriptorMap;

    // Set of all tensors allocated in session ram
    std::set<std::shared_ptr<TensorDescriptor>> tensorSet;

    std::vector<std::shared_ptr<TensorDescriptor>> tensors;

    // Virtual pipelines used to track input and output tensors
    ComputePipelineBase inputs{nullptr};
    ComputePipelineBase outputs{nullptr};
};

} // namespace mlsdk::el::compute
