/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "compute.hpp"
#include "graph_log.hpp"

#include <cmath>
#include <numeric>
#include <regex>

using namespace mlsdk::el::log;
using namespace mlsdk::el::utils;

namespace mlsdk::el::compute {

namespace {
std::shared_ptr<VirtualTensor> makeVirtualTensor(const std::shared_ptr<TensorDescriptor> &tensor,
                                                 ComputePipelineBase *descendant) {
    auto parent = tensor->getPipeline();
    auto virtualTensor = std::make_shared<VirtualTensor>(tensor, parent, descendant);

    if (parent != nullptr) {
        parent->pushDescendant(virtualTensor);
    }

    if (descendant != nullptr) {
        descendant->pushParent(virtualTensor);
    }

    return virtualTensor;
}

VkFormat accTypeVkFormat(uint32_t accType) {
    switch (accType) {
    case 1:
        return VK_FORMAT_R32_SINT;
    case 2:
        return VK_FORMAT_R16_SFLOAT;
    case 3:
        return VK_FORMAT_R32_SFLOAT;
    case 4:
        return VK_FORMAT_R64_SINT;
    default:
        throw std::runtime_error("Unsupported acc type " + std::to_string(accType));
    }
}

std::string accTypeString(uint32_t accType) {
    switch (accType) {
    case 1:
        return "int32_t";
    case 2:
        return "float16_t";
    case 3:
        return "float";
    case 4:
        return "int64_t";
    default:
        throw std::runtime_error("Unsupported AVG_POOL2D acc type " + std::to_string(accType));
    }
}

} // namespace

/*******************************************************************************
 * ComputeDescriptorSet
 *******************************************************************************/

ComputeDescriptorSet::ComputeDescriptorSet(
    const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, const VkDevice _device,
    VkDescriptorPool _descriptorPool, VkDescriptorSet _descriptorSet, const std::shared_ptr<Tensor> &_tensor)
    : loader{_loader}, device{_device}, descriptorPool{_descriptorPool}, descriptorSet{_descriptorSet}, tensor{_tensor},
      tensorARM{_tensor->getVkTensorARM()}, tensorViewARM{_tensor->getVkTensorViewARM()} {}

ComputeDescriptorSet::~ComputeDescriptorSet() {
    loader->vkFreeDescriptorSets(device, descriptorPool, 1, &descriptorSet);
    loader->vkDestroyDescriptorPool(device, descriptorPool, nullptr);
}

VkDescriptorSet ComputeDescriptorSet::getVkDescriptorSet() const { return descriptorSet; }

std::shared_ptr<Tensor> ComputeDescriptorSet::getTensor() const { return tensor; }

VkTensorARM ComputeDescriptorSet::getVkTensorARM() const { return tensorARM; }

VkTensorViewARM ComputeDescriptorSet::getVkTensorViewARM() const { return tensorViewARM; }

void ComputeDescriptorSet::updateDescriptorSet(VkTensorARM _tensor, VkTensorViewARM tensorView) {
    tensorARM = _tensor;
    tensorViewARM = tensorView;
    updateDescriptorSet();
}

void ComputeDescriptorSet::updateDescriptorSet() {
    const VkWriteDescriptorSetTensorARM descriptorInfo = {
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_TENSOR_ARM, // type
        nullptr,                                           // next
        1,                                                 // tensor view count
        &tensorViewARM,                                    // tensor views
    };

    const VkWriteDescriptorSet writeDescriptorSet = {
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, // type
        &descriptorInfo,                        // next
        descriptorSet,                          // descriptor set
        0,                                      // binding
        0,                                      // dst array element
        1,                                      // descriptor count
        VK_DESCRIPTOR_TYPE_TENSOR_ARM,          // descriptor type
        nullptr,                                // image info
        nullptr,                                // buffer info
        nullptr,                                // texel buffer view
    };

    loader->vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);
}

/*******************************************************************************
 * ComputePipelineLayout
 *******************************************************************************/

ComputePipelineLayout::ComputePipelineLayout(
    const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
    DescriptorMap _descriptorMap, const PushConstant &_pushConstant)
    : loader{_loader}, device{_device}, descriptorMap{std::move(_descriptorMap)}, pushConstant{_pushConstant},
      descriptorSetLayouts{createDescriptorSetLayouts()}, pipelineLayout{createPipelineLayout()} {}

ComputePipelineLayout::~ComputePipelineLayout() {
    for (auto descriptorSetLayout : descriptorSetLayouts) {
        loader->vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    }

    loader->vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
}

VkPipelineLayout ComputePipelineLayout::getVkPipelineLayout() const { return pipelineLayout; }

const DescriptorMap &ComputePipelineLayout::getDescriptorMap() const { return descriptorMap; }

std::shared_ptr<TensorDescriptor> ComputePipelineLayout::getTensor(const uint32_t set) {
    return std::get<1>(descriptorMap[set]);
}

void ComputePipelineLayout::cmdBindAndDispatch(VkCommandBuffer commandBuffer,
                                               const ComputeDescriptorSetMap &descriptorSetMap) {
    cmdBindDescriptorSets(commandBuffer, descriptorSetMap);
    cmdPushConstants(commandBuffer);
    cmdPipelineBarrier(commandBuffer, descriptorSetMap);
}

void ComputePipelineLayout::cmdBindDescriptorSets(VkCommandBuffer commandBuffer,
                                                  const ComputeDescriptorSetMap &descriptorSetMap) {
    for (uint32_t set = 0; set < descriptorMap.size(); set++) {
        const auto &descriptorSet = descriptorSetMap.at({pipelineLayout, set});

        const auto vkDescriptorSet = descriptorSet->getVkDescriptorSet();
        loader->vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, set, 1,
                                        &vkDescriptorSet, 0, nullptr);
    }
}

void ComputePipelineLayout::cmdPushConstants(VkCommandBuffer commandBuffer) {
    if (pushConstant.pointer != nullptr) {
        loader->vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushConstant.size,
                                   pushConstant.pointer);
    }
}

void ComputePipelineLayout::cmdPipelineBarrier(VkCommandBuffer commandBuffer,
                                               const ComputeDescriptorSetMap &descriptorSetMap) const {
    std::vector<VkTensorMemoryBarrierARM> tensorMemoryBarriers;

    for (uint32_t set = 0; set < descriptorMap.size(); set++) {
        const auto &[direction, tensor] = descriptorMap[set];

        // Only add barriers for input tensors
        if (direction != Input) {
            continue;
        }

        const auto &descriptorSet = descriptorSetMap.at({pipelineLayout, set});

        tensorMemoryBarriers.push_back({
            VK_STRUCTURE_TYPE_TENSOR_MEMORY_BARRIER_ARM, // type
            nullptr,                                     // next
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,      // src stage mask
            VK_ACCESS_2_SHADER_WRITE_BIT,                // src access mask
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,      // dst stage mask
            VK_ACCESS_2_SHADER_READ_BIT,                 // dst access mask
            VK_QUEUE_FAMILY_IGNORED,                     // src queue family index
            VK_QUEUE_FAMILY_IGNORED,                     // dst queue family index
            descriptorSet->getVkTensorARM(),             // tensor
        });
    }

    const VkTensorDependencyInfoARM tensorDependencyInfo = {
        VK_STRUCTURE_TYPE_TENSOR_DEPENDENCY_INFO_ARM,       // type
        nullptr,                                            // next
        static_cast<uint32_t>(tensorMemoryBarriers.size()), // tensorMemoryBarrierCount
        tensorMemoryBarriers.data()                         // pTensorMemoryBarriers
    };

    const VkDependencyInfo dependencyInfo = {
        VK_STRUCTURE_TYPE_DEPENDENCY_INFO, // type
        &tensorDependencyInfo,             // next
        0,                                 // dependencyFlags
        0,                                 // memoryBarrierCount
        nullptr,                           // pMemoryBarriers
        0,                                 // bufferMemoryBarrierCount
        nullptr,                           // pBufferMemoryBarriers
        0,                                 // imageMemoryBarrierCount
        nullptr                            // pImageMemoryBarriers
    };

    loader->vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
}

std::vector<VkDescriptorSetLayoutBinding> ComputePipelineLayout::getDescriptorSetLayoutBinding() const {
    std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings;

    const VkDescriptorSetLayoutBinding descriptorSetLayoutBinding = {
        0,                             // binding
        VK_DESCRIPTOR_TYPE_TENSOR_ARM, // descriptor type
        1,                             // descriptor count
        VK_SHADER_STAGE_COMPUTE_BIT,   // type
        nullptr,                       // sampler
    };

    descriptorSetLayoutBindings.emplace_back(descriptorSetLayoutBinding);

    return descriptorSetLayoutBindings;
}

std::vector<VkDescriptorSetLayout> ComputePipelineLayout::createDescriptorSetLayouts() const {
    std::vector<VkDescriptorSetLayout> layouts;
    for ([[maybe_unused]] const auto &[direction, tensor] : descriptorMap) {
        const auto &descriptorSetLayoutBindings = getDescriptorSetLayoutBinding();

        std::vector<VkDescriptorSetLayoutCreateFlags> bindingFlags(descriptorSetLayoutBindings.size(),
                                                                   VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT);

        const VkDescriptorSetLayoutBindingFlagsCreateInfo descriptorSetBindingFlagsCreateInfo{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO, // type
            nullptr,                                                           // next
            static_cast<uint32_t>(descriptorSetLayoutBindings.size()),         // binding count
            bindingFlags.data(),                                               // binding flags
        };

        const VkDescriptorSetLayoutCreateInfo descriptorSetCreateInfo = {
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,        // type
            &descriptorSetBindingFlagsCreateInfo,                       // next
            VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT, // flags
            static_cast<uint32_t>(descriptorSetLayoutBindings.size()),  // binding count
            descriptorSetLayoutBindings.data(),                         // bindings
        };

        layouts.emplace_back(nullptr);
        if (loader->vkCreateDescriptorSetLayout(device, &descriptorSetCreateInfo, nullptr, &layouts.back()) !=
            VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor set layout");
        }
    }

    return layouts;
}

VkDescriptorPool ComputePipelineLayout::createDescriptorPool() const {
    const VkDescriptorPoolSize descriptorPoolSize = {
        VK_DESCRIPTOR_TYPE_TENSOR_ARM,  // type
        uint32_t(descriptorMap.size()), // descriptor count
    };

    const VkDescriptorPoolCreateFlags descriptorPoolCreateFlags =
        VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;

    const VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, // type
        nullptr,                                       // next
        descriptorPoolCreateFlags,                     // flags,
        static_cast<uint32_t>(descriptorMap.size()),   // max sets
        1,                                             // pool size count
        &descriptorPoolSize,                           // descriptor pool size
    };

    VkDescriptorPool pool;
    if (loader->vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, nullptr, &pool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor pool");
    }

    return pool;
}

void ComputePipelineLayout::makeDescriptorSets(ComputeDescriptorSetMap &mapping,
                                               const TensorDescriptorMap &filter) const {
    for (uint32_t set = 0; set < descriptorMap.size(); set++) {
        [[maybe_unused]] const auto &[_, tensorDescriptor] = descriptorMap[set];

        // Compare if tensor descriptor from descriptor map and argument are matching
        if (auto it = filter.find(tensorDescriptor); it != filter.end()) {
            auto vkDescriptorPool = createDescriptorPool();
            mapping[{pipelineLayout, set}] = std::make_shared<ComputeDescriptorSet>(
                loader, device, vkDescriptorPool, createDescriptorSet(vkDescriptorPool, set), it->second);
        }
    }
}

VkDescriptorSet ComputePipelineLayout::createDescriptorSet(const VkDescriptorPool vkDescriptorPool,
                                                           const uint32_t set) const {
    // Allocate descriptor set
    const VkDescriptorSetAllocateInfo descriptorSetAllocInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, // type
        nullptr,                                        // next
        vkDescriptorPool,                               // descriptor pool
        1,                                              // descriptor set count
        &descriptorSetLayouts.at(set),                  // descriptor layout set
    };

    VkDescriptorSet descriptorSet;
    if (loader->vkAllocateDescriptorSets(device, &descriptorSetAllocInfo, &descriptorSet) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor set");
    }

    return descriptorSet;
}

VkPipelineLayout ComputePipelineLayout::createPipelineLayout() const {
    const VkPushConstantRange pushConstantRange = {
        VK_SHADER_STAGE_COMPUTE_BIT, // flags
        0,                           // offset
        pushConstant.size,           // size
    };

    const VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,      // type
        nullptr,                                            // next
        0,                                                  // flags
        static_cast<uint32_t>(descriptorSetLayouts.size()), // layout count
        descriptorSetLayouts.data(),                        // layout
        pushConstant.pointer != nullptr ? 1u : 0u,          // push constant count
        &pushConstantRange,                                 // push constants
    };

    VkPipelineLayout layout;
    if (loader->vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &layout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout");
    }

    return layout;
}

/*******************************************************************************
 * ComputePipelineBase
 *******************************************************************************/

ComputePipelineBase::ComputePipelineBase(const std::shared_ptr<ComputePipelineLayout> &_pipelineLayout)
    : pipelineLayout{_pipelineLayout} {}

ComputePipelineBase::~ComputePipelineBase() = default;

void ComputePipelineBase::cmdBindAndDispatch(VkCommandBuffer, const ComputeDescriptorSetMap &) {}

std::shared_ptr<ComputePipelineLayout> ComputePipelineBase::getComputePipelineLayout() const { return pipelineLayout; }

const std::vector<std::shared_ptr<VirtualTensor>> &ComputePipelineBase::getParents() const { return parents; }

void ComputePipelineBase::pushParent(const std::shared_ptr<VirtualTensor> &tensor) { parents.emplace_back(tensor); }

const std::vector<std::shared_ptr<VirtualTensor>> &ComputePipelineBase::getDescendants() const { return descendants; }

void ComputePipelineBase::pushDescendant(const std::shared_ptr<VirtualTensor> &tensor) {
    descendants.push_back(tensor);
}

/*******************************************************************************
 * ComputePipeline
 *******************************************************************************/

namespace {
std::shared_ptr<ComputePipelineLayout>
createPipelineLayout(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &loader,
                     VkDevice device, DescriptorMap descriptorMap, const PushConstant &pushConstant) {
    auto pipelineLayout =
        std::make_shared<ComputePipelineLayout>(loader, device, std::move(descriptorMap), pushConstant);

    return pipelineLayout;
}
} // namespace

ComputePipeline::ComputePipeline(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader,
                                 VkDevice _device, DescriptorMap descriptorMap, const PushConstant &pushConstant,
                                 const std::shared_ptr<PipelineCache> &_pipelineCache, const SpirvBinary &_spirv,
                                 const std::string &debugName, const SpecConstants &_constants)
    : ComputePipelineBase(createPipelineLayout(_loader, _device, std::move(descriptorMap), pushConstant)),
      loader{_loader}, device{_device}, pipelineCache{_pipelineCache},
      shaderModule{createShaderModule(_spirv)}, pipeline{createComputePipeline(_constants)} {
    connectPipelines();
    setDebugUtilsObjectName(loader, device, VK_OBJECT_TYPE_PIPELINE, (uint64_t)pipeline, debugName);
}

ComputePipeline::~ComputePipeline() {
    loader->vkDestroyPipeline(device, pipeline, nullptr);
    loader->vkDestroyShaderModule(device, shaderModule, nullptr);
}

void ComputePipeline::cmdBindAndDispatch(VkCommandBuffer commandBuffer,
                                         const ComputeDescriptorSetMap &descriptorSetMap) {
    loader->vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    pipelineLayout->cmdBindAndDispatch(commandBuffer, descriptorSetMap);
    cmdDispatch(commandBuffer);
}

void ComputePipeline::cmdDispatch(VkCommandBuffer commandBuffer) {
    // Get first output tensor
    const auto &tensor = pipelineLayout->getTensor(0);
    const auto &size = uint32_t(tensor->getShapeSize());

    const auto groupCountX = static_cast<uint32_t>(std::ceil(std::sqrt(double(divideRoundUp(size, warp1D)))));
    const auto groupCountY = groupCountX;

    loader->vkCmdDispatch(commandBuffer, groupCountX, groupCountY, 1);
}

VkPipeline ComputePipeline::createComputePipeline(const SpecConstants &_constants) const {
    std::vector<VkSpecializationMapEntry> spec_entries;
    for (uint32_t i = 0; i < _constants.size(); ++i) {
        const VkSpecializationMapEntry entry = {
            i,                                                             // constantID
            static_cast<uint32_t>(spec_entries.size() * sizeof(uint32_t)), // offset
            sizeof(uint32_t),                                              // size
        };
        spec_entries.emplace_back(entry);
    }

    const VkSpecializationInfo spec_info = {
        static_cast<uint32_t>(spec_entries.size()),                 // mapEntryCount
        spec_entries.data(),                                        // pMapEntries
        static_cast<uint32_t>(_constants.size() * sizeof(int32_t)), // dataSize
        _constants.data(),                                          // pData
    };

    const VkPipelineShaderStageCreateInfo pipelineShaderCreateInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, // type
        nullptr,                                             // next
        0,                                                   // flags
        VK_SHADER_STAGE_COMPUTE_BIT,                         // stage flag bits
        shaderModule,                                        // shader module
        "main",                                              // name
        _constants.size() > 0 ? &spec_info : nullptr,        // specialization info
    };

    const VkComputePipelineCreateInfo computePipelineCreateInfo = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, // type
        nullptr,                                        // next
        0,                                              // flags
        pipelineShaderCreateInfo,                       // create info
        pipelineLayout->getVkPipelineLayout(),          // pipeline layout
        nullptr,                                        // base pipeline handle
        0,                                              // base pipeline index
    };

    VkPipeline vkPipeline;
    if (loader->vkCreateComputePipelines(device, pipelineCache->getPipelineCache(), 1, &computePipelineCreateInfo,
                                         nullptr, &vkPipeline) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute pipelines");
    }

    return vkPipeline;
}

VkShaderModule ComputePipeline::createShaderModule(const SpirvBinary &code) const {
    const VkShaderModuleCreateInfo shaderModuleCreateInfo = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,           // type
        nullptr,                                               // next
        0,                                                     // flags
        static_cast<uint32_t>(code.size() * sizeof(uint32_t)), // code size
        code.data(),                                           // code
    };

    VkShaderModule vkShaderModule;
    if (loader->vkCreateShaderModule(device, &shaderModuleCreateInfo, nullptr, &vkShaderModule) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module");
    }

    return vkShaderModule;
}

void ComputePipeline::connectPipelines() {
    const auto &descriptorMap = pipelineLayout->getDescriptorMap();

    // Set current pipeline as producer for output tensors
    for (const auto &[direction, tensor] : descriptorMap) {
        if (direction == Output) {
            tensor->setPipeline(this);
        }
    }

    // Create connections to parent pipelines
    for (const auto &[direction, tensor] : descriptorMap) {
        if (direction == Input) {
            makeVirtualTensor(tensor, this);
        }
    }
}

/*******************************************************************************
 * Argmax
 *******************************************************************************/

Argmax::Argmax(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
               const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
               const std::shared_ptr<TensorDescriptor> &_output, const uint32_t _axis, const uint32_t _nanMode,
               const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output), {&pushConstant, sizeof(pushConstant)},
                      _pipelineCache, createSpirv(_pipelineCache, _input), debugName,
                      {_input->getRank(), _output->getRank()}),
      pushConstant{createPushConstant(_axis, _nanMode)} {}

Argmax::PushConstant Argmax::createPushConstant(const uint32_t axis, const uint32_t nanMode) const {
    PushConstant constant = {
        axis,
        nanMode,
    };

    return constant;
}

DescriptorMap Argmax::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                          const std::shared_ptr<TensorDescriptor> &output) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
    };

    return descriptorMap;
}

SpirvBinary Argmax::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                const std::shared_ptr<TensorDescriptor> &input) const {
    auto inType = makeFormat(input->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inType->glslType(),
                                  },
                                  {
                                      {"%warpX%", std::to_string(warp1D)},
                                      {"%in_t_type%", inType->toInt()},
                                      {"%in_t_lowest%", inType->lowest()},
                                      {"%in_t%", inType->glslType()},
                                  });
}

/*******************************************************************************
 * ArithmeticRightShift
 *******************************************************************************/

ArithmeticRightShift::ArithmeticRightShift(
    const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
    const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input1,
    const std::shared_ptr<TensorDescriptor> &_input2, const std::shared_ptr<TensorDescriptor> &_output,
    const bool _round, const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input1, _input2, _output),
                      {&pushConstant, sizeof(pushConstant)}, _pipelineCache, createSpirv(_pipelineCache, _output),
                      debugName, {_input1->getRank(), _output->getRank()}),
      pushConstant{createPushConstant(_round)} {}

ArithmeticRightShift::PushConstant ArithmeticRightShift::createPushConstant(const bool round) const {
    PushConstant constant = {
        round,
    };
    return constant;
}

DescriptorMap ArithmeticRightShift::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input1,
                                                        const std::shared_ptr<TensorDescriptor> &input2,
                                                        const std::shared_ptr<TensorDescriptor> &output) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input1},  // set 1
        {Input, input2},  // set 2
    };

    return descriptorMap;
}

SpirvBinary ArithmeticRightShift::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                              const std::shared_ptr<TensorDescriptor> &output) const {
    auto inOutType = makeFormat(output->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inOutType->glslType(),
                                  },
                                  {
                                      {"%warpX%", std::to_string(warp1D)},
                                      {"%in_out_t%", inOutType->glslType()},
                                  });
}

/*******************************************************************************
 * AvgPool2D
 *******************************************************************************/

AvgPool2D::AvgPool2D(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader,
                     VkDevice _device, const std::shared_ptr<PipelineCache> &_pipelineCache,
                     const std::shared_ptr<TensorDescriptor> &_input, const std::shared_ptr<TensorDescriptor> &_output,
                     const std::vector<uint32_t> &_kernel, const std::vector<uint32_t> &_stride,
                     const std::vector<uint32_t> &_pad, const uint32_t _accType, const int8_t _inputZeroPoint,
                     const int8_t _outputZeroPoint, const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output), {&pushConstant, sizeof(pushConstant)},
                      _pipelineCache, createSpirv(_pipelineCache, _output, _accType), debugName),
      pushConstant{createPushConstant(_kernel, _stride, _pad, _inputZeroPoint, _outputZeroPoint)} {}

AvgPool2D::PushConstant AvgPool2D::createPushConstant(const std::vector<uint32_t> &kernel,
                                                      const std::vector<uint32_t> &stride,
                                                      const std::vector<uint32_t> &pad, const int8_t inputZeroPoint,
                                                      const int8_t outputZeroPoint) const {
    PushConstant constant = {
        {
            kernel[0],
            kernel[1],
        },
        {
            stride[0],
            stride[1],
        },
        {
            pad[0],
            pad[1],
            pad[2],
            pad[3],
        },
        inputZeroPoint,
        outputZeroPoint,
    };

    return constant;
}

DescriptorMap AvgPool2D::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                             const std::shared_ptr<TensorDescriptor> &output) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
    };

    return descriptorMap;
}

SpirvBinary AvgPool2D::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                   const std::shared_ptr<TensorDescriptor> &output, const uint32_t accType) const {
    auto inOutType = makeFormat(output->getFormat());

    const auto &accTypeStr = accTypeString(accType);
    return _pipelineCache->lookup(shaderName,
                                  {
                                      inOutType->glslType(),
                                      accTypeStr,
                                  },
                                  {
                                      {"%warpX%", std::to_string(warp1D)},
                                      {"%acc_t%", accTypeStr},
                                      {"%in_out_t_lowest%", inOutType->lowest()},
                                      {"%in_out_t_max%", inOutType->max()},
                                      {"%in_out_t%", inOutType->glslType()},
                                      {"%in_out_t_type%", inOutType->toInt()},
                                  });
}

/*******************************************************************************
 * Cast
 *******************************************************************************/

Cast::Cast(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
           const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
           const std::shared_ptr<TensorDescriptor> &_output, const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output), {}, _pipelineCache,
                      createSpirv(_pipelineCache, _input, _output), debugName, {_input->getRank()}) {}

DescriptorMap Cast::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                        const std::shared_ptr<TensorDescriptor> &output) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
    };

    return descriptorMap;
}

SpirvBinary Cast::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                              const std::shared_ptr<TensorDescriptor> &input,
                              const std::shared_ptr<TensorDescriptor> &output) const {
    auto inType = makeFormat(input->getFormat());
    auto outType = makeFormat(output->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inType->glslType(),
                                      outType->glslType(),
                                  },
                                  {
                                      {"%warpX%", std::to_string(warp1D)},
                                      {"%in_t_type%", inType->toInt()},
                                      {"%out_t_type%", outType->toInt()},
                                      {"%in_t_lowest%", inType->lowest()},
                                      {"%in_t_max%", inType->max()},
                                      {"%out_t_lowest%", outType->lowest()},
                                      {"%out_t_max%", outType->max()},
                                      {"%in_t%", inType->glslType()},
                                      {"%out_t%", outType->glslType()},
                                  });
}

/*******************************************************************************
 * Clamp
 *******************************************************************************/

Clamp::Clamp(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
             const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
             const std::shared_ptr<TensorDescriptor> &_output, const double _min, const double _max,
             const uint32_t _nanMode, const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output), {&pushConstant, sizeof(pushConstant)},
                      _pipelineCache, createSpirv(_pipelineCache, _output), debugName, {_input->getRank()}),
      pushConstant{createPushConstant(_min, _max, _nanMode)} {}

Clamp::PushConstant Clamp::createPushConstant(const double min, const double max, const uint32_t nanMode) const {
    PushConstant constant = {
        min,
        max,
        nanMode,
    };

    return constant;
}

DescriptorMap Clamp::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                         const std::shared_ptr<TensorDescriptor> &output) const {
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
    };

    return descriptorMap;
}

SpirvBinary Clamp::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                               const std::shared_ptr<TensorDescriptor> &output) const {
    auto inOutType = makeFormat(output->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inOutType->glslType(),
                                  },
                                  {
                                      {"%warpX%", std::to_string(warp1D)},
                                      {"%in_out_t%", inOutType->glslType()},
                                      {"%in_out_t_type%", inOutType->toInt()},
                                  });
}

/*******************************************************************************
 * Concat
 *******************************************************************************/

Concat::Concat(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
               const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
               const std::shared_ptr<TensorDescriptor> &_output, const uint32_t _axis, const uint32_t _offset,
               const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output), {&pushConstant, sizeof(pushConstant)},
                      _pipelineCache, createSpirv(_pipelineCache, _output), debugName, {_input->getRank()}),
      pushConstant{createPushConstant(_axis, _offset)} {}

Concat::PushConstant Concat::createPushConstant(const uint32_t axis, const uint32_t offset) const {
    PushConstant constant = {
        axis,
        offset,
    };

    return constant;
}

DescriptorMap Concat::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                          const std::shared_ptr<TensorDescriptor> &output) const {
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
    };

    return descriptorMap;
}

void Concat::cmdDispatch(VkCommandBuffer commandBuffer) {
    const auto &tensor = pipelineLayout->getTensor(1);
    const auto &size = uint32_t(tensor->getShapeSize());

    const auto groupCountX = static_cast<uint32_t>(std::ceil(std::sqrt(double(divideRoundUp(size, warp1D)))));
    const auto groupCountY = groupCountX;

    loader->vkCmdDispatch(commandBuffer, groupCountX, groupCountY, 1);
}

SpirvBinary Concat::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                const std::shared_ptr<TensorDescriptor> &output) const {
    auto inOutType = makeFormat(output->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inOutType->glslType(),
                                  },
                                  {
                                      {"%warpX%", std::to_string(warp1D)},
                                      {"%in_out_t%", inOutType->glslType()},
                                  });
}

/*******************************************************************************
 * Conv2D
 *******************************************************************************/

Conv2D::Conv2D(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
               const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
               const std::shared_ptr<TensorDescriptor> &_output, const std::shared_ptr<TensorDescriptor> &_weights,
               const std::shared_ptr<TensorDescriptor> &_biases, const std::vector<uint32_t> &_pad,
               const std::vector<uint32_t> &_stride, const std::vector<uint32_t> &_dilation,
               const int8_t _inputZeroPoint, const int8_t _weightZeroPoint, const uint32_t _accType,
               const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output, _weights, _biases),
                      {&pushConstant, sizeof(pushConstant)}, _pipelineCache,
                      createSpirv(_pipelineCache, _input, _output, _weights, _accType), debugName),
      pushConstant{createPushConstant(_pad, _stride, _dilation, _inputZeroPoint, _weightZeroPoint)} {}

Conv2D::PushConstant Conv2D::createPushConstant(const std::vector<uint32_t> &pad, const std::vector<uint32_t> &stride,
                                                const std::vector<uint32_t> &dilation, const int8_t inputZeroPoint,
                                                const int8_t weightZeroPoint) const {
    PushConstant constant = {
        inputZeroPoint,
        weightZeroPoint,
        {
            pad[0],
            pad[1],
            pad[2],
            pad[3],
        },
        {
            stride[0],
            stride[1],
        },
        {
            dilation[0],
            dilation[1],
        },
    };

    return constant;
}

DescriptorMap Conv2D::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                          const std::shared_ptr<TensorDescriptor> &output,
                                          const std::shared_ptr<TensorDescriptor> &weights,
                                          const std::shared_ptr<TensorDescriptor> &biases) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
        {Input, weights}, // set 2
        {Input, biases},  // set 3
    };

    return descriptorMap;
}

SpirvBinary Conv2D::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                const std::shared_ptr<TensorDescriptor> &input,
                                const std::shared_ptr<TensorDescriptor> &output,
                                const std::shared_ptr<TensorDescriptor> &weights, const uint32_t accType) const {
    auto inType = makeFormat(input->getFormat());
    auto outType = makeFormat(output->getFormat());
    auto weightType = makeFormat(weights->getFormat());
    auto accTypeType = makeFormat(accTypeVkFormat(accType));

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inType->glslType(),
                                      weightType->glslType(),
                                      outType->glslType(),
                                      accTypeType->glslType(),
                                  },
                                  {
                                      {"%warpX%", std::to_string(warpX)},
                                      {"%warpY%", std::to_string(warpY)},
                                      {"%warpZ%", std::to_string(warpZ)},
                                      {"%in_t%", inType->glslType()},
                                      {"%out_t%", outType->glslType()},
                                      {"%weight_t%", weightType->glslType()},
                                      {"%acc_t_type%", accTypeType->toInt()},
                                      {"%acc_t%", accTypeType->glslType()},
                                  });
}

void Conv2D::cmdDispatch(VkCommandBuffer commandBuffer) {
    // Get first output tensor
    const auto &tensor = pipelineLayout->getTensor(0);
    const auto &dimensions = tensor->getDimensions();
    loader->vkCmdDispatch(commandBuffer, divideRoundUp(static_cast<uint32_t>(dimensions[0] * dimensions[2]), warpX),
                          divideRoundUp(static_cast<uint32_t>(dimensions[1]), warpY),
                          divideRoundUp(static_cast<uint32_t>(dimensions[3]), warpZ * 4));
}

/*******************************************************************************
 * Conv3D
 *******************************************************************************/

Conv3D::Conv3D(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
               const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
               const std::shared_ptr<TensorDescriptor> &_output, const std::shared_ptr<TensorDescriptor> &_weights,
               const std::shared_ptr<TensorDescriptor> &_biases, const std::vector<uint32_t> &_pad,
               const std::vector<uint32_t> &_stride, const std::vector<uint32_t> &_dilation,
               const int8_t _inputZeroPoint, const int8_t _weightZeroPoint, const uint32_t _accType,
               const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output, _weights, _biases),
                      {&pushConstant, sizeof(pushConstant)}, _pipelineCache,
                      createSpirv(_pipelineCache, _input, _output, _weights, _accType), debugName),
      pushConstant{createPushConstant(_pad, _stride, _dilation, _inputZeroPoint, _weightZeroPoint)} {}

Conv3D::PushConstant Conv3D::createPushConstant(const std::vector<uint32_t> &pad, const std::vector<uint32_t> &stride,
                                                const std::vector<uint32_t> &dilation, const int8_t inputZeroPoint,
                                                const int8_t weightZeroPoint) const {
    PushConstant constant = {
        inputZeroPoint,
        weightZeroPoint,
        {
            pad[0],
            pad[1],
            pad[2],
            pad[3],
            pad[4],
            pad[5],
        },
        {
            stride[0],
            stride[1],
            stride[2],
        },
        {
            dilation[0],
            dilation[1],
            dilation[2],
        },
    };

    return constant;
}

DescriptorMap Conv3D::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                          const std::shared_ptr<TensorDescriptor> &output,
                                          const std::shared_ptr<TensorDescriptor> &weights,
                                          const std::shared_ptr<TensorDescriptor> &biases) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
        {Input, weights}, // set 2
        {Input, biases},  // set 3
    };

    return descriptorMap;
}

SpirvBinary Conv3D::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                const std::shared_ptr<TensorDescriptor> &input,
                                const std::shared_ptr<TensorDescriptor> &output,
                                const std::shared_ptr<TensorDescriptor> &weights, const uint32_t accType) const {
    auto inType = makeFormat(input->getFormat());
    auto outType = makeFormat(output->getFormat());
    auto weightType = makeFormat(weights->getFormat());
    auto accTypeType = makeFormat(accTypeVkFormat(accType));

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inType->glslType(),
                                      weightType->glslType(),
                                      outType->glslType(),
                                      accTypeType->glslType(),
                                  },
                                  {
                                      {"%warpX%", std::to_string(warp1D)},
                                      {"%in_t%", inType->glslType()},
                                      {"%out_t%", outType->glslType()},
                                      {"%weight_t%", weightType->glslType()},
                                      {"%acc_t_type%", accTypeType->toInt()},
                                      {"%acc_t%", accTypeType->glslType()},
                                  });
}

/*******************************************************************************
 * DepthwiseConv2D
 *******************************************************************************/

DepthwiseConv2D::DepthwiseConv2D(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader,
                                 VkDevice _device, const std::shared_ptr<PipelineCache> &_pipelineCache,
                                 const std::shared_ptr<TensorDescriptor> &_input,
                                 const std::shared_ptr<TensorDescriptor> &_output,
                                 const std::shared_ptr<TensorDescriptor> &_weights,
                                 const std::shared_ptr<TensorDescriptor> &_biases, const std::vector<uint32_t> &_pad,
                                 const std::vector<uint32_t> &_stride, const std::vector<uint32_t> &_dilation,
                                 const int8_t _inputZeroPoint, const int8_t _weightZeroPoint, const uint32_t _accType,
                                 const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output, _weights, _biases),
                      {&pushConstant, sizeof(pushConstant)}, _pipelineCache,
                      createSpirv(_pipelineCache, _input, _output, _weights, _accType), debugName),
      pushConstant{createPushConstant(_pad, _stride, _dilation, _inputZeroPoint, _weightZeroPoint)} {}

DepthwiseConv2D::PushConstant DepthwiseConv2D::createPushConstant(const std::vector<uint32_t> &pad,
                                                                  const std::vector<uint32_t> &stride,
                                                                  const std::vector<uint32_t> &dilation,
                                                                  const int8_t inputZeroPoint,
                                                                  const int8_t weightZeroPoint) const {
    PushConstant constant = {
        inputZeroPoint,
        weightZeroPoint,
        {
            pad[0],
            pad[1],
            pad[2],
            pad[3],
        },
        {
            stride[0],
            stride[1],
        },
        {
            dilation[0],
            dilation[1],
        },
    };

    return constant;
}

DescriptorMap DepthwiseConv2D::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                                   const std::shared_ptr<TensorDescriptor> &output,
                                                   const std::shared_ptr<TensorDescriptor> &weights,
                                                   const std::shared_ptr<TensorDescriptor> &biases) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
        {Input, weights}, // set 2
        {Input, biases},  // set 3
    };

    return descriptorMap;
}

SpirvBinary DepthwiseConv2D::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                         const std::shared_ptr<TensorDescriptor> &input,
                                         const std::shared_ptr<TensorDescriptor> &output,
                                         const std::shared_ptr<TensorDescriptor> &weights,
                                         const uint32_t accType) const {
    auto inType = makeFormat(input->getFormat());
    auto outType = makeFormat(output->getFormat());
    auto weightType = makeFormat(weights->getFormat());
    auto accTypeType = makeFormat(accTypeVkFormat(accType));

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inType->glslType(),
                                      weightType->glslType(),
                                      outType->glslType(),
                                      accTypeType->glslType(),
                                  },
                                  {
                                      {"%warpX%", std::to_string(warp1D)},
                                      {"%in_t%", inType->glslType()},
                                      {"%out_t%", outType->glslType()},
                                      {"%weight_t%", weightType->glslType()},
                                      {"%acc_t_type%", accTypeType->toInt()},
                                      {"%acc_t%", accTypeType->glslType()},
                                  });
}

/*******************************************************************************
 * ElementwiseBinary
 *******************************************************************************/

ElementwiseBinary::ElementwiseBinary(
    const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
    const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input1,
    const std::shared_ptr<TensorDescriptor> &_input2, const std::shared_ptr<TensorDescriptor> &_output,
    const uint32_t _nanMode, const std::string &debugName, const std::string &_operation)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input1, _input2, _output),
                      {&pushConstant, sizeof(pushConstant)}, _pipelineCache,
                      createSpirv(_pipelineCache, _input1, _output, debugName, _operation), debugName,
                      {_input1->getRank(), _output->getRank()}),
      pushConstant{createPushConstant(_nanMode)} {}

ElementwiseBinary::PushConstant ElementwiseBinary::createPushConstant(const uint32_t nanMode) const {
    return PushConstant{nanMode};
}

DescriptorMap ElementwiseBinary::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input1,
                                                     const std::shared_ptr<TensorDescriptor> &input2,
                                                     const std::shared_ptr<TensorDescriptor> &output) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input1},  // set 1
        {Input, input2},  // set 2
    };

    return descriptorMap;
}

SpirvBinary ElementwiseBinary::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                           const std::shared_ptr<TensorDescriptor> &input,
                                           const std::shared_ptr<TensorDescriptor> &output, const std::string &name,
                                           const std::string &operation) const {
    auto inType = makeFormat(input->getFormat());
    auto outType = makeFormat(output->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      name,
                                      inType->glslType(),
                                      outType->glslType(),
                                  },
                                  {
                                      {"%warpX%", std::to_string(warp1D)},
                                      {"%operation%", operation},
                                      {"%in_t%", inType->glslType()},
                                      {"%out_t%", outType->glslType()},
                                      {"%in_t_type%", inType->toInt()},
                                  });
}

/*******************************************************************************
 * ElementwiseUnary
 *******************************************************************************/

ElementwiseUnary::ElementwiseUnary(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader,
                                   VkDevice _device, const std::shared_ptr<PipelineCache> &_pipelineCache,
                                   const std::shared_ptr<TensorDescriptor> &_input1,
                                   const std::shared_ptr<TensorDescriptor> &_output, const std::string &debugName,
                                   const std::string &_operation)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input1, _output), {}, _pipelineCache,
                      createSpirv(_pipelineCache, _output, debugName, _operation), debugName, {_output->getRank()}) {}

DescriptorMap ElementwiseUnary::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input1,
                                                    const std::shared_ptr<TensorDescriptor> &output) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input1},  // set 1
    };

    return descriptorMap;
}

SpirvBinary ElementwiseUnary::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                          const std::shared_ptr<TensorDescriptor> &output, const std::string &name,
                                          const std::string &operation) const {
    auto inOutType = makeFormat(output->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      name,
                                      inOutType->glslType(),
                                  },
                                  {
                                      {"%warpX%", std::to_string(warp1D)},
                                      {"%operation%", operation},
                                      {"%in_out_t%", inOutType->glslType()},
                                  });
}

/*******************************************************************************
 * Fft2D
 *******************************************************************************/

Fft2D::Fft2D(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
             const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_inputReal,
             const std::shared_ptr<TensorDescriptor> &_inputImag, const std::shared_ptr<TensorDescriptor> &_outputReal,
             const std::shared_ptr<TensorDescriptor> &_outputImag, const bool _inverse, const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_inputReal, _inputImag, _outputReal, _outputImag),
                      {&pushConstant, sizeof(pushConstant)}, _pipelineCache, createSpirv(_pipelineCache), debugName),
      pushConstant{createPushConstant(_inverse)} {}

Fft2D::PushConstant Fft2D::createPushConstant(const bool inverse) const {
    float signValue = inverse ? -1.0f : 1.0f;
    PushConstant constant = {
        signValue,
    };
    return constant;
}

DescriptorMap Fft2D::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &inputReal,
                                         const std::shared_ptr<TensorDescriptor> &inputImag,
                                         const std::shared_ptr<TensorDescriptor> &outputReal,
                                         const std::shared_ptr<TensorDescriptor> &outputImag) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, outputReal}, // set 0
        {Output, outputImag}, // set 1
        {Input, inputReal},   // set 2
        {Input, inputImag},   // set 3
    };

    return descriptorMap;
}

SpirvBinary Fft2D::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache) const {
    return _pipelineCache->lookup(shaderName, {},
                                  {
                                      {"%warpX%", std::to_string(warp1D)},
                                  });
}

/*******************************************************************************
 * Gather
 *******************************************************************************/

Gather::Gather(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
               const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_values,
               const std::shared_ptr<TensorDescriptor> &_indices, const std::shared_ptr<TensorDescriptor> &_output,
               const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_values, _indices, _output), {}, _pipelineCache,
                      createSpirv(_pipelineCache, _indices, _output), debugName, {}) {}

DescriptorMap Gather::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &values,
                                          const std::shared_ptr<TensorDescriptor> &indicies,
                                          const std::shared_ptr<TensorDescriptor> &output) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output},  // set 0
        {Input, values},   // set 1
        {Input, indicies}, // set 2
    };

    return descriptorMap;
}

SpirvBinary Gather::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                const std::shared_ptr<TensorDescriptor> &indices,
                                const std::shared_ptr<TensorDescriptor> &output) const {
    auto inOutType = makeFormat(output->getFormat());
    auto indicesType = makeFormat(indices->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inOutType->glslType(),
                                      indicesType->glslType(),
                                  },
                                  {
                                      {"%warpX%", std::to_string(warp1D)},
                                      {"%index_t%", indicesType->glslType()},
                                      {"%in_out_t%", inOutType->glslType()},
                                  });
}

/*******************************************************************************
 * Matmul
 *******************************************************************************/

Matmul::Matmul(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
               const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input1,
               const std::shared_ptr<TensorDescriptor> &_input2, const std::shared_ptr<TensorDescriptor> &_output,
               const int32_t _inputZeroPoint1, const int32_t _inputZeroPoint2, const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input1, _input2, _output),
                      {&pushConstant, sizeof(pushConstant)}, _pipelineCache,
                      createSpirv(_pipelineCache, _input1, _output), debugName),
      pushConstant{createPushConstant(_inputZeroPoint1, _inputZeroPoint2)} {}

Matmul::PushConstant Matmul::createPushConstant(const int32_t inputZeroPoint1, const int32_t inputZeroPoint2) const {
    PushConstant constant = {
        inputZeroPoint1,
        inputZeroPoint2,
    };

    return constant;
}

DescriptorMap Matmul::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input1,
                                          const std::shared_ptr<TensorDescriptor> &input2,
                                          const std::shared_ptr<TensorDescriptor> &output) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input1},  // set 1
        {Input, input2},  // set 2
    };

    return descriptorMap;
}

SpirvBinary Matmul::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                const std::shared_ptr<TensorDescriptor> &input1,
                                const std::shared_ptr<TensorDescriptor> &output) const {
    auto inType = makeFormat(input1->getFormat());
    auto outType = makeFormat(output->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inType->glslType(),
                                      outType->glslType(),
                                  },
                                  {
                                      {"%warpX%", std::to_string(warp1D)},
                                      {"%in_t%", inType->glslType()},
                                      {"%out_t%", outType->glslType()},
                                      {"%out_t_type%", outType->toInt()},
                                  });
}

/*******************************************************************************
 * MaxPool2D
 *******************************************************************************/

MaxPool2D::MaxPool2D(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader,
                     VkDevice _device, const std::shared_ptr<PipelineCache> &_pipelineCache,
                     const std::shared_ptr<TensorDescriptor> &_input, const std::shared_ptr<TensorDescriptor> &_output,
                     const std::vector<uint32_t> &_kernel, const std::vector<uint32_t> &_stride,
                     const std::vector<uint32_t> &_pad, const uint32_t _nanMode, const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output), {&pushConstant, sizeof(pushConstant)},
                      _pipelineCache, createSpirv(_pipelineCache, _output, _nanMode), debugName),
      pushConstant{createPushConstant(_kernel, _stride, _pad, _nanMode)} {}

MaxPool2D::PushConstant MaxPool2D::createPushConstant(const std::vector<uint32_t> &kernel,
                                                      const std::vector<uint32_t> &stride,
                                                      const std::vector<uint32_t> &pad, const uint32_t nanMode) const {
    PushConstant constant = {
        {
            kernel[0],
            kernel[1],
        },
        {
            stride[0],
            stride[1],
        },
        {
            pad[0],
            pad[1],
            pad[2],
            pad[3],
        },
        nanMode,
    };

    return constant;
}

DescriptorMap MaxPool2D::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                             const std::shared_ptr<TensorDescriptor> &output) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
    };

    return descriptorMap;
}

SpirvBinary MaxPool2D::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                   const std::shared_ptr<TensorDescriptor> &output, const uint32_t _nanMode) const {
    auto inOutType = makeFormat(output->getFormat());

    const std::string init = (_nanMode == NanPropagationMode::Ignore ? "NAN" : inOutType->lowest());
    return _pipelineCache->lookup(shaderName,
                                  {
                                      inOutType->glslType(),
                                  },
                                  {
                                      {"%warpX%", std::to_string(warp1D)},
                                      {"%in_out_t%", inOutType->glslType()},
                                      {"%in_out_t_lowest%", init},
                                      {"%in_out_t_type%", inOutType->toInt()},
                                  });
}

/*******************************************************************************
 * Mul
 *******************************************************************************/

Mul::Mul(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
         const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input1,
         const std::shared_ptr<TensorDescriptor> &_input2, const std::shared_ptr<TensorDescriptor> &_output,
         const uint32_t _shift, const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input1, _input2, _output),
                      {&pushConstant, sizeof(pushConstant)}, _pipelineCache,
                      createSpirv(_pipelineCache, _input1, _output), debugName,
                      {_input1->getRank(), _output->getRank()}),
      pushConstant{createPushConstant(_shift)} {}

Mul::PushConstant Mul::createPushConstant(const uint32_t shift) const {
    PushConstant constant = {
        shift,
    };

    return constant;
}

DescriptorMap Mul::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input1,
                                       const std::shared_ptr<TensorDescriptor> &input2,
                                       const std::shared_ptr<TensorDescriptor> &output) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input1},  // set 1
        {Input, input2},  // set 2
    };

    return descriptorMap;
}

SpirvBinary Mul::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                             const std::shared_ptr<TensorDescriptor> &input1,
                             const std::shared_ptr<TensorDescriptor> &output) const {
    auto inType = makeFormat(input1->getFormat());
    auto outType = makeFormat(output->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inType->glslType(),
                                      outType->glslType(),
                                  },
                                  {
                                      {"%warpX%", std::to_string(warp1D)},
                                      {"%in_t_type%", inType->toInt()},
                                      {"%in_t%", inType->glslType()},
                                      {"%out_t%", outType->glslType()},
                                  });
}

/*******************************************************************************
 * Negate
 *******************************************************************************/

Negate::Negate(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
               const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
               const std::shared_ptr<TensorDescriptor> &_output, const int32_t _inputZeroPoint,
               const int32_t _outputZeroPoint, const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output), {&pushConstant, sizeof(pushConstant)},
                      _pipelineCache, createSpirv(_pipelineCache, _output), debugName, {_output->getRank()}),
      pushConstant{createPushConstant(_inputZeroPoint, _outputZeroPoint)} {}

Negate::PushConstant Negate::createPushConstant(const int32_t inputZeroPoint, const int32_t outputZeroPoint) const {
    PushConstant constant = {
        inputZeroPoint,
        outputZeroPoint,
    };

    return constant;
}

DescriptorMap Negate::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                          const std::shared_ptr<TensorDescriptor> &output) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
    };

    return descriptorMap;
}

SpirvBinary Negate::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                const std::shared_ptr<TensorDescriptor> &output) const {
    auto inOutType = makeFormat(output->getFormat());

    std::string accType = inOutType->isInteger() ? "int32_t" : "float";

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inOutType->glslType(),
                                  },
                                  {
                                      {"%warpX%", std::to_string(warp1D)},
                                      {"%in_out_t%", inOutType->glslType()},
                                      {"%acc_t%", accType},
                                      {"%in_out_t_lowest%", inOutType->lowest()},
                                      {"%in_out_t_max%", inOutType->max()},
                                  });
}

/*******************************************************************************
 * Pad
 *******************************************************************************/

Pad::Pad(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
         const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
         const std::shared_ptr<TensorDescriptor> &_output, const std::shared_ptr<TensorDescriptor> &_padding,
         const double _padConst, const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output, _padding),
                      {&pushConstant, sizeof(pushConstant)}, _pipelineCache, createSpirv(_pipelineCache, _output),
                      debugName, {_input->getRank()}),
      pushConstant{createPushConstant(_padConst)} {}

Pad::PushConstant Pad::createPushConstant(const double padConst) const {
    PushConstant constant = {
        padConst,
    };

    return constant;
}

DescriptorMap Pad::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                       const std::shared_ptr<TensorDescriptor> &output,
                                       const std::shared_ptr<TensorDescriptor> &padding) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
        {Input, padding}, // set 2
    };

    return descriptorMap;
}

SpirvBinary Pad::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                             const std::shared_ptr<TensorDescriptor> &output) const {
    auto inOutType = makeFormat(output->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inOutType->glslType(),
                                  },
                                  {
                                      {"%warpX%", std::to_string(warp1D)},
                                      {"%in_out_t%", inOutType->glslType()},
                                  });
}

/*******************************************************************************
 * Reduce
 *******************************************************************************/

Reduce::Reduce(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
               const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
               const std::shared_ptr<TensorDescriptor> &_output, const uint32_t _axis, const uint32_t _nanMode,
               const std::string &debugName, const std::string &_init, const std::string &_operation)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output), {&pushConstant, sizeof(pushConstant)},
                      _pipelineCache, createSpirv(_pipelineCache, _input, debugName, _init, _operation), debugName,
                      {_output->getRank()}),
      pushConstant{createPushConstant(_axis, _nanMode, makeFormat(_input->getFormat())->isInteger())} {}

Reduce::PushConstant Reduce::createPushConstant(const uint32_t axis, const uint32_t nanMode,
                                                const bool isInteger) const {
    PushConstant constant = {
        axis,
        isInteger ? static_cast<uint32_t>(NanPropagationMode::Propagate) : nanMode, // Enforce propagate for integers
    };

    return constant;
}

DescriptorMap Reduce::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                          const std::shared_ptr<TensorDescriptor> &output) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
    };

    return descriptorMap;
}

SpirvBinary Reduce::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                const std::shared_ptr<TensorDescriptor> &output, const std::string &name,
                                const std::string &init, const std::string &operation) const {
    auto inOutType = makeFormat(output->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      name,
                                      inOutType->glslType(),
                                  },
                                  {
                                      {"%warpX%", std::to_string(warp1D)},
                                      {"%init%", init},
                                      {"%operation%", operation},
                                      {"%in_out_t%", inOutType->glslType()},
                                      {"%in_out_t_type%", inOutType->toInt()},
                                  });
}

/*******************************************************************************
 * Rescale
 *******************************************************************************/

Rescale::Rescale(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
                 const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
                 const std::shared_ptr<TensorDescriptor> &_output, const int32_t _inputZeroPoint,
                 const int32_t _outputZeroPoint, const std::shared_ptr<TensorDescriptor> &_multiplier,
                 const std::shared_ptr<TensorDescriptor> &_shift, const bool _scale32, const bool _doubleRound,
                 const bool _perChannel, const bool _inputUnsigned, const bool _outputUnsigned,
                 const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output, _multiplier, _shift),
                      {&pushConstant, sizeof(pushConstant)}, _pipelineCache,
                      createSpirv(_pipelineCache, _input, _output, _multiplier, _inputUnsigned, _outputUnsigned),
                      debugName, {_output->getRank(), _scale32, _doubleRound, _perChannel}),
      pushConstant{createPushConstant(_inputZeroPoint, _outputZeroPoint)} {}

Rescale::PushConstant Rescale::createPushConstant(const int32_t inputZeroPoint, const int32_t outputZeroPoint) const {
    PushConstant constant = {
        inputZeroPoint,
        outputZeroPoint,
    };

    return constant;
}

DescriptorMap Rescale::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                           const std::shared_ptr<TensorDescriptor> &output,
                                           const std::shared_ptr<TensorDescriptor> &multiplier,
                                           const std::shared_ptr<TensorDescriptor> &shift) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output},    // set 0
        {Input, input},      // set 1
        {Input, multiplier}, // set 2
        {Input, shift},      // set 3
    };

    return descriptorMap;
}

SpirvBinary Rescale::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                 const std::shared_ptr<TensorDescriptor> &input,
                                 const std::shared_ptr<TensorDescriptor> &output,
                                 const std::shared_ptr<TensorDescriptor> &multiplier, const bool inputUnsigned,
                                 const bool outputUnsigned) const {
    auto inType = makeFormat(input->getFormat(), inputUnsigned);
    auto outType = makeFormat(output->getFormat(), outputUnsigned);
    auto mulType = makeFormat(multiplier->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inType->glslType(),
                                      outType->glslType(),
                                      mulType->glslType(),
                                  },
                                  {
                                      {"%warpX%", std::to_string(warp1D)},
                                      {"%in_t%", inType->glslType()},
                                      {"%out_t%", outType->glslType()},
                                      {"%mul_t%", mulType->glslType()},
                                      {"%out_t_lowest%", outType->lowest()},
                                      {"%out_t_max%", outType->max()},
                                  });
}

/*******************************************************************************
 * Reshape
 *******************************************************************************/

Reshape::Reshape(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
                 const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
                 const std::shared_ptr<TensorDescriptor> &_output, const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output), {}, _pipelineCache,
                      createSpirv(_pipelineCache, _output), debugName, {_input->getRank(), _output->getRank()}) {}

DescriptorMap Reshape::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                           const std::shared_ptr<TensorDescriptor> &output) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
    };

    return descriptorMap;
}

SpirvBinary Reshape::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                 const std::shared_ptr<TensorDescriptor> &output) const {
    auto inOutType = makeFormat(output->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inOutType->glslType(),
                                  },
                                  {
                                      {"%warpX%", std::to_string(warp1D)},
                                      {"%in_out_t%", inOutType->glslType()},
                                  });
}

/*******************************************************************************
 * Resize
 *******************************************************************************/

Resize::Resize(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
               const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
               const std::shared_ptr<TensorDescriptor> &_output, const std::vector<int32_t> &_scale,
               const std::vector<int32_t> &_offset, const std::vector<int32_t> &_border, const uint32_t _mode,
               const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output), {&pushConstant, sizeof(pushConstant)},
                      _pipelineCache, createSpirv(_pipelineCache, _input, _output), debugName),
      pushConstant{createPushConstant(_scale, _offset, _border, _mode)} {}

Resize::PushConstant Resize::createPushConstant(const std::vector<int32_t> &scale, const std::vector<int32_t> &offset,
                                                const std::vector<int32_t> &border, const uint32_t mode) const {
    PushConstant constant = {
        {
            scale[0],
            scale[1],
            scale[2],
            scale[3],
        },
        {
            offset[0],
            offset[1],
        },
        {
            border[0],
            border[1],
        },
        mode,
    };

    return constant;
}

DescriptorMap Resize::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                          const std::shared_ptr<TensorDescriptor> &output) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
    };

    return descriptorMap;
}

SpirvBinary Resize::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                const std::shared_ptr<TensorDescriptor> &input,
                                const std::shared_ptr<TensorDescriptor> &output) const {
    auto inType = makeFormat(input->getFormat());
    auto outType = makeFormat(output->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inType->glslType(),
                                      outType->glslType(),
                                  },
                                  {
                                      {"%warpX%", std::to_string(warp1D)},
                                      {"%in_t%", inType->glslType()},
                                      {"%out_t%", outType->glslType()},
                                      {"%out_t_type%", outType->toInt()},
                                  });
}

/*******************************************************************************
 * Reverse
 *******************************************************************************/

Reverse::Reverse(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
                 const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
                 const std::shared_ptr<TensorDescriptor> &_output, const uint32_t _axis, const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output), {&pushConstant, sizeof(pushConstant)},
                      _pipelineCache, createSpirv(_pipelineCache, _output), debugName, {_output->getRank()}),
      pushConstant{createPushConstant(_axis)} {}

Reverse::PushConstant Reverse::createPushConstant(const uint32_t axis) const {
    PushConstant constant = {
        axis,
    };

    return constant;
}

DescriptorMap Reverse::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                           const std::shared_ptr<TensorDescriptor> &output) const {
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
    };

    return descriptorMap;
}

SpirvBinary Reverse::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                 const std::shared_ptr<TensorDescriptor> &output) const {
    auto inOutType = makeFormat(output->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inOutType->glslType(),
                                  },
                                  {
                                      {"%warpX%", std::to_string(warp1D)},
                                      {"%in_out_t%", inOutType->glslType()},
                                  });
}

/*******************************************************************************
 * Rfft2D
 *******************************************************************************/

Rfft2D::Rfft2D(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
               const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
               const std::shared_ptr<TensorDescriptor> &_outputReal,
               const std::shared_ptr<TensorDescriptor> &_outputImag, const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _outputReal, _outputImag), {}, _pipelineCache,
                      createSpirv(_pipelineCache), debugName) {}

DescriptorMap Rfft2D::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                          const std::shared_ptr<TensorDescriptor> &outputReal,
                                          const std::shared_ptr<TensorDescriptor> &outputImag) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, outputReal}, // set 0
        {Output, outputImag}, // set 1
        {Input, input},       // set 2
    };

    return descriptorMap;
}

SpirvBinary Rfft2D::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache) const {
    return _pipelineCache->lookup(shaderName, {},
                                  {
                                      {"%warpX%", std::to_string(warp1D)},
                                  });
}

/*******************************************************************************
 * Scatter
 *******************************************************************************/

Scatter::Scatter(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
                 const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
                 const std::shared_ptr<TensorDescriptor> &_values, const std::shared_ptr<TensorDescriptor> &_indices,
                 const std::shared_ptr<TensorDescriptor> &_output, const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _values, _indices, _output), {}, _pipelineCache,
                      createSpirv(_pipelineCache, _indices, _output), debugName, {}) {}

DescriptorMap Scatter::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                           const std::shared_ptr<TensorDescriptor> &values,
                                           const std::shared_ptr<TensorDescriptor> &indicies,
                                           const std::shared_ptr<TensorDescriptor> &output) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output},  // set 0
        {Input, input},    // set 1
        {Input, values},   // set 2
        {Input, indicies}, // set 3
    };

    return descriptorMap;
}

SpirvBinary Scatter::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                 const std::shared_ptr<TensorDescriptor> &indices,
                                 const std::shared_ptr<TensorDescriptor> &output) const {
    auto inOutType = makeFormat(output->getFormat());
    auto indicesType = makeFormat(indices->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inOutType->glslType(),
                                      indicesType->glslType(),
                                  },
                                  {
                                      {"%warpX%", std::to_string(warp1D)},
                                      {"%index_t%", indicesType->glslType()},
                                      {"%in_out_t%", inOutType->glslType()},
                                  });
}

/*******************************************************************************
 * Select
 *******************************************************************************/

Select::Select(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
               const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input1,
               const std::shared_ptr<TensorDescriptor> &_input2, const std::shared_ptr<TensorDescriptor> &_input3,
               const std::shared_ptr<TensorDescriptor> &_output, const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input1, _input2, _input3, _output), {}, _pipelineCache,
                      createSpirv(_pipelineCache, _output), debugName, {_output->getRank()}) {}

DescriptorMap Select::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input1,
                                          const std::shared_ptr<TensorDescriptor> &input2,
                                          const std::shared_ptr<TensorDescriptor> &input3,
                                          const std::shared_ptr<TensorDescriptor> &output) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input1},  // set 1
        {Input, input2},  // set 2
        {Input, input3},  // set 3
    };

    return descriptorMap;
}

SpirvBinary Select::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                const std::shared_ptr<TensorDescriptor> &output) const {
    auto inOutType = makeFormat(output->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inOutType->glslType(),
                                  },
                                  {
                                      {"%warpX%", std::to_string(warp1D)},
                                      {"%in_out_t%", inOutType->glslType()},
                                  });
}

/*******************************************************************************
 * Slice
 *******************************************************************************/

Slice::Slice(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
             const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
             const std::shared_ptr<TensorDescriptor> &_output, const std::vector<uint32_t> &_start,
             const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output),
                      {&pushConstant, static_cast<uint32_t>(_input->getRank() * sizeof(uint32_t))}, _pipelineCache,
                      createSpirv(_pipelineCache, _input), debugName, {_input->getRank()}),
      pushConstant{createPushConstant(_start)} {}

Slice::PushConstant Slice::createPushConstant(const std::vector<uint32_t> &start) const {
    PushConstant constant{};
    std::copy(start.begin(), start.end(), constant.start);
    return constant;
}

DescriptorMap Slice::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                         const std::shared_ptr<TensorDescriptor> &output) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
    };

    return descriptorMap;
}

SpirvBinary Slice::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                               const std::shared_ptr<TensorDescriptor> &input) const {
    auto inOutType = makeFormat(input->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inOutType->glslType(),
                                  },
                                  {
                                      {"%warpX%", std::to_string(warp1D)},
                                      {"%in_out_t%", inOutType->glslType()},
                                  });
}

/*******************************************************************************
 * Table
 *******************************************************************************/

Table::Table(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
             const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
             const std::shared_ptr<TensorDescriptor> &_output, const std::shared_ptr<TensorDescriptor> &_table,
             const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output, _table), {}, _pipelineCache,
                      createSpirv(_pipelineCache, _input, _output), debugName, {_output->getRank()}) {}

DescriptorMap Table::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                         const std::shared_ptr<TensorDescriptor> &output,
                                         const std::shared_ptr<TensorDescriptor> &table) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
        {Input, table},   // set 2
    };

    return descriptorMap;
}

SpirvBinary Table::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                               const std::shared_ptr<TensorDescriptor> &input,
                               const std::shared_ptr<TensorDescriptor> &output) const {
    auto inType = makeFormat(input->getFormat());
    auto outType = makeFormat(output->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inType->glslType(),
                                      outType->glslType(),
                                  },
                                  {
                                      {"%warpX%", std::to_string(warp1D)},
                                      {"%in_t%", inType->glslType()},
                                      {"%out_t%", outType->glslType()},
                                  });
}

/*******************************************************************************
 * Tile
 *******************************************************************************/

Tile::Tile(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
           const std::shared_ptr<PipelineCache> &_pipelineCache, const std::shared_ptr<TensorDescriptor> &_input,
           const std::shared_ptr<TensorDescriptor> &_output, const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output), {}, _pipelineCache,
                      createSpirv(_pipelineCache, _output), debugName, {_input->getRank()}) {}

DescriptorMap Tile::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                        const std::shared_ptr<TensorDescriptor> &output) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
    };

    return descriptorMap;
}

SpirvBinary Tile::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                              const std::shared_ptr<TensorDescriptor> &output) const {
    auto inOutType = makeFormat(output->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inOutType->glslType(),
                                  },
                                  {
                                      {"%warpX%", std::to_string(warp1D)},
                                      {"%in_out_t%", inOutType->glslType()},
                                  });
}

/*******************************************************************************
 * Transpose
 *******************************************************************************/

Transpose::Transpose(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader,
                     VkDevice _device, const std::shared_ptr<PipelineCache> &_pipelineCache,
                     const std::shared_ptr<TensorDescriptor> &_input, const std::shared_ptr<TensorDescriptor> &_output,
                     const std::vector<uint32_t> &_perms, const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output),
                      {&pushConstant, static_cast<uint32_t>(_input->getRank() * sizeof(uint32_t))}, _pipelineCache,
                      createSpirv(_pipelineCache, _output), debugName, {_output->getRank()}),
      pushConstant{createPushConstant(_perms)} {}

Transpose::PushConstant Transpose::createPushConstant(const std::vector<uint32_t> &perms) const {
    PushConstant constant{};
    std::copy(perms.begin(), perms.end(), constant.perms);
    return constant;
}

DescriptorMap Transpose::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                             const std::shared_ptr<TensorDescriptor> &output) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
    };

    return descriptorMap;
}

SpirvBinary Transpose::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                   const std::shared_ptr<TensorDescriptor> &output) const {
    auto inOutType = makeFormat(output->getFormat());

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inOutType->glslType(),
                                  },
                                  {
                                      {"%warpX%", std::to_string(warp1D)},
                                      {"%in_out_t%", inOutType->glslType()},
                                  });
}

/*******************************************************************************
 * TransposeConv2D
 *******************************************************************************/

TransposeConv2D::TransposeConv2D(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader,
                                 VkDevice _device, const std::shared_ptr<PipelineCache> &_pipelineCache,
                                 const std::shared_ptr<TensorDescriptor> &_input,
                                 const std::shared_ptr<TensorDescriptor> &_output,
                                 const std::shared_ptr<TensorDescriptor> &_weights,
                                 const std::shared_ptr<TensorDescriptor> &_biases, const std::vector<uint32_t> &_outPad,
                                 const std::vector<uint32_t> &_stride, const int8_t _inputZeroPoint,
                                 const int8_t _weightZeroPoint, const uint32_t _accType, const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(_input, _output, _weights, _biases),
                      {&pushConstant, sizeof(pushConstant)}, _pipelineCache,
                      createSpirv(_pipelineCache, _input, _output, _weights, _accType), debugName),
      pushConstant{createPushConstant(_outPad, _stride, _inputZeroPoint, _weightZeroPoint)} {}

TransposeConv2D::PushConstant TransposeConv2D::createPushConstant(const std::vector<uint32_t> &outPad,
                                                                  const std::vector<uint32_t> &stride,
                                                                  const int8_t inputZeroPoint,
                                                                  const int8_t weightZeroPoint) const {
    PushConstant constant = {
        inputZeroPoint,
        weightZeroPoint,
        {
            outPad[0],
            outPad[1],
            outPad[2],
            outPad[3],
        },
        {
            stride[0],
            stride[1],
        },
    };

    return constant;
}

DescriptorMap TransposeConv2D::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &input,
                                                   const std::shared_ptr<TensorDescriptor> &output,
                                                   const std::shared_ptr<TensorDescriptor> &weights,
                                                   const std::shared_ptr<TensorDescriptor> &biases) const {
    // Configure descriptor map
    DescriptorMap descriptorMap = {
        {Output, output}, // set 0
        {Input, input},   // set 1
        {Input, weights}, // set 2
        {Input, biases},  // set 3
    };

    return descriptorMap;
}

SpirvBinary TransposeConv2D::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                         const std::shared_ptr<TensorDescriptor> &input,
                                         const std::shared_ptr<TensorDescriptor> &output,
                                         const std::shared_ptr<TensorDescriptor> &weights,
                                         const uint32_t accType) const {
    auto inType = makeFormat(input->getFormat());
    auto outType = makeFormat(output->getFormat());
    auto weightType = makeFormat(weights->getFormat());
    auto accTypeType = makeFormat(accTypeVkFormat(accType));

    return _pipelineCache->lookup(shaderName,
                                  {
                                      inType->glslType(),
                                      weightType->glslType(),
                                      outType->glslType(),
                                      accTypeType->glslType(),
                                  },
                                  {
                                      {"%warpX%", std::to_string(warp1D)},
                                      {"%in_t%", inType->glslType()},
                                      {"%out_t%", outType->glslType()},
                                      {"%weight_t%", weightType->glslType()},
                                      {"%acc_t_type%", accTypeType->toInt()},
                                      {"%acc_t%", accTypeType->glslType()},
                                  });
}

/*******************************************************************************
 * BlockMatch
 *******************************************************************************/

BlockMatch::BlockMatch(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader,
                       VkDevice _device, const std::shared_ptr<PipelineCache> &_pipelineCache,
                       const std::shared_ptr<TensorDescriptor> &inTemplate,
                       const std::shared_ptr<TensorDescriptor> &inSearch,
                       const std::optional<std::shared_ptr<TensorDescriptor>> &outVectors,
                       const std::optional<std::shared_ptr<TensorDescriptor>> &outCosts,
                       const std::vector<uint32_t> &kernelSizes, const std::vector<uint32_t> &searchWindowSizes,
                       const std::vector<uint32_t> &inputStrides, const std::vector<uint32_t> &windowStrides,
                       const std::vector<uint32_t> &windowOffsets, const std::vector<uint32_t> &padding,
                       const uint32_t searchPattern, const SearchType searchType, const std::string &debugName)
    : ComputePipeline(_loader, _device, createDescriptorMap(inTemplate, inSearch, outVectors, outCosts, searchType), {},
                      _pipelineCache, createSpirv(_pipelineCache, searchType), debugName,
                      createSpecConstants(kernelSizes, searchWindowSizes, inputStrides, windowStrides, windowOffsets,
                                          padding, searchPattern)) {}

DescriptorMap BlockMatch::createDescriptorMap(const std::shared_ptr<TensorDescriptor> &inTemplate,
                                              const std::shared_ptr<TensorDescriptor> &inSearch,
                                              const std::optional<std::shared_ptr<TensorDescriptor>> &outVectors,
                                              const std::optional<std::shared_ptr<TensorDescriptor>> &outCosts,
                                              const SearchType searchType) const {

    DescriptorMap descriptorMap = {
        {Input, inTemplate}, // set 0
        {Input, inSearch},   // set 1
    };

    switch (searchType) {
    case SearchType::MIN_SAD: {
        assert(outVectors);
        descriptorMap.emplace_back(Output, outVectors.value()); // set 2
        break;
    }
    case SearchType::RAW_SAD: {
        assert(outCosts);
        descriptorMap.emplace_back(Output, outCosts.value()); // set 2
        break;
    }
    case SearchType::MIN_SAD_COST: {
        assert(outVectors && outCosts);
        descriptorMap.emplace_back(Output, outVectors.value()); // set 2
        descriptorMap.emplace_back(Output, outCosts.value());   // set 3
        break;
    }
    }

    return descriptorMap;
}

SpirvBinary BlockMatch::createSpirv(const std::shared_ptr<PipelineCache> &_pipelineCache,
                                    const SearchType searchType) const {
    return _pipelineCache->lookup(shaderName, {std::to_string(searchType)},
                                  {
                                      {"%search_type%", std::to_string(searchType)},
                                      {"%warpX%", std::to_string(warpX)},
                                      {"%warpY%", std::to_string(warpY)},
                                  });
}

SpecConstants BlockMatch::createSpecConstants(const std::vector<uint32_t> &kernelSizes,
                                              const std::vector<uint32_t> &searchWindowSizes,
                                              const std::vector<uint32_t> &inputStrides,
                                              const std::vector<uint32_t> &windowStrides,
                                              const std::vector<uint32_t> &windowOffsets,
                                              const std::vector<uint32_t> &padding,
                                              const uint32_t searchPattern) const {
    SpecConstants specConstants = {
        kernelSizes[0],  kernelSizes[1],   searchWindowSizes[0], searchWindowSizes[1], inputStrides[0],
        inputStrides[1], windowStrides[0], windowStrides[1],     windowOffsets[0],     windowOffsets[1],
        padding[0],      padding[1],       searchPattern,
    };

    return specConstants;
}

void BlockMatch::cmdDispatch(VkCommandBuffer commandBuffer) {
    const auto &tensor = pipelineLayout->getTensor(0);
    const auto &dimensions = tensor->getDimensions();
    loader->vkCmdDispatch(commandBuffer, divideRoundUp(static_cast<uint32_t>(dimensions[3]), warpX),
                          divideRoundUp(static_cast<uint32_t>(dimensions[2]), warpY), 1);
}

/*******************************************************************************
 * GraphPipeline
 *******************************************************************************/

GraphPipeline::GraphPipeline(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader,
                             VkPhysicalDevice _physicalDevice, VkDevice _device,
                             const std::shared_ptr<PipelineCache> &_pipelineCache)
    : loader{_loader}, physicalDevice{_physicalDevice}, device{_device}, pipelineCache{_pipelineCache} {}

GraphPipeline::~GraphPipeline() {
    for (auto &deviceMemory : constantsDeviceMemory) {
        loader->vkFreeMemory(device, deviceMemory, nullptr);
    }
}

void GraphPipeline::makeConstTensor(const uint32_t id, const VkTensorDescriptionARM &tensorDescription,
                                    const void *data) {
    const auto tensorDescriptor = std::make_shared<TensorDescriptor>(loader, physicalDevice, device, tensorDescription);
    auto tensor = TensorDescriptor::makeTensor(tensorDescriptor);
    const auto deviceMemory = tensorDescriptor->createInitializeDeviceMemory(data);

    (void)tensor->bindTensorMemory(deviceMemory, 0);
    constantsDeviceMemory.push_back(deviceMemory);
    constTensorMap[id] = std::move(tensor);
}

std::shared_ptr<TensorDescriptor> GraphPipeline::getConstTensor(const uint32_t id) const {
    return constTensorMap.at(id)->getTensorDescriptor();
}

std::shared_ptr<TensorDescriptor> GraphPipeline::makeConstCompositeTensor(const VkFormat format,
                                                                          const std::vector<int64_t> &dimensions,
                                                                          const void *data) {
    auto tensorDescriptor = std::make_shared<TensorDescriptor>(loader, physicalDevice, device, format, dimensions);
    auto tensor = TensorDescriptor::makeTensor(tensorDescriptor);
    const auto deviceMemory = tensorDescriptor->createInitializeDeviceMemory(data);

    (void)tensor->bindTensorMemory(deviceMemory, 0);
    constantsDeviceMemory.push_back(deviceMemory);
    compositeTensors.emplace_back(std::move(tensor));

    return tensorDescriptor;
}

ComputeDescriptorSetMap GraphPipeline::makeConstantsDescriptorSets() const {
    TensorDescriptorMap filter;

    for (const auto &[id, tensor] : constTensorMap) {
        filter[tensor->getTensorDescriptor()] = tensor;
    }

    for (const auto &tensor : compositeTensors) {
        filter[tensor->getTensorDescriptor()] = tensor;
    }

    ComputeDescriptorSetMap mapping;
    for (const auto &pipeline : pipelines) {
        const auto &pipelineLayout = pipeline->getComputePipelineLayout();
        pipelineLayout->makeDescriptorSets(mapping, filter);
    }

    return mapping;
}

void GraphPipeline::makeDescriptorSetBinding(const uint32_t set, const uint32_t binding, const uint32_t arrayIndex,
                                             const VkTensorDescriptionARM &tensorDescription) {
    auto tensorDescriptor = std::make_shared<TensorDescriptor>(loader, physicalDevice, device, tensorDescription);

    auto &vec = tensorMap[set][binding];
    vec.resize(std::max(vec.size(), size_t(arrayIndex + 1)));
    vec[arrayIndex] = tensorDescriptor;

    tensorDescriptorMap[set][tensorDescriptor] = TensorDescriptor::makeTensor(tensorDescriptor);
}

std::shared_ptr<TensorDescriptor> GraphPipeline::getTensor(const uint32_t set, const uint32_t binding,
                                                           const uint32_t arrayIndex) const {
    return tensorMap.at(set).at(binding).at(arrayIndex);
}

std::shared_ptr<TensorDescriptor> GraphPipeline::makeTensor(const VkFormat format,
                                                            const std::vector<int64_t> &dimensions,
                                                            const std::vector<int64_t> &strides) {
    auto tensor = std::make_shared<TensorDescriptor>(loader, physicalDevice, device, format, dimensions, strides);

    auto [iterator, inserted] = tensorSet.insert(tensor);

    if (inserted) {
        tensors.push_back(tensor);
    }

    return tensor;
}

const std::vector<std::shared_ptr<TensorDescriptor>> &GraphPipeline::getTensors() const { return tensors; }

ComputeDescriptorSetMap GraphPipeline::makeSessionRamDescriptorSets() const {
    TensorDescriptorMap filter;
    for (const auto &tensorDescriptor : tensorSet) {
        filter[tensorDescriptor] = TensorDescriptor::makeTensor(tensorDescriptor);
    }

    ComputeDescriptorSetMap mapping;
    for (const auto &pipeline : pipelines) {
        const auto &pipelineLayout = pipeline->getComputePipelineLayout();
        pipelineLayout->makeDescriptorSets(mapping, filter);
    }

    return mapping;
}

ComputeDescriptorSetMap GraphPipeline::makeExternalDescriptorSets(const uint32_t set) const {
    const auto &filter = tensorDescriptorMap.at(set);
    ComputeDescriptorSetMap mapping;
    for (const auto &pipeline : pipelines) {
        const auto &pipelineLayout = pipeline->getComputePipelineLayout();
        pipelineLayout->makeDescriptorSets(mapping, filter);
    }

    return mapping;
}

void GraphPipeline::cmdBindAndDispatch(VkCommandBuffer commandBuffer, const ComputeDescriptorSetMap &descriptorSetMap) {
    for (auto &pipeline : pipelines) {
        pipeline->cmdBindAndDispatch(commandBuffer, descriptorSetMap);
    }
}

void GraphPipeline::makeInput(const std::shared_ptr<TensorDescriptor> &tensor) {
    // Register inputs pipeline as producer of tensors
    tensor->setPipeline(&inputs);
}

void GraphPipeline::makeOutput(const std::shared_ptr<TensorDescriptor> &tensor) {
    // Connect outputs pipeline with parent pipelines
    makeVirtualTensor(tensor, &outputs);
}

void GraphPipeline::makeAbs(const std::shared_ptr<TensorDescriptor> &input,
                            const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    auto pipeline =
        std::make_shared<ElementwiseUnary>(loader, device, pipelineCache, input, output, debugName, "abs(value1)");
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeAdd(const std::shared_ptr<TensorDescriptor> &input1,
                            const std::shared_ptr<TensorDescriptor> &input2,
                            const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    auto pipeline = std::make_shared<ElementwiseBinary>(loader, device, pipelineCache, input1, input2, output,
                                                        NanPropagationMode::Propagate, debugName, "value1 + value2");
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeArgmax(const std::shared_ptr<TensorDescriptor> &input,
                               const std::shared_ptr<TensorDescriptor> &output, const uint32_t axis,
                               const uint32_t nanMode, const std::string &debugName) {
    auto pipeline = std::make_shared<Argmax>(loader, device, pipelineCache, input, output, axis, nanMode, debugName);
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeArithmeticRightShift(const std::shared_ptr<TensorDescriptor> &input1,
                                             const std::shared_ptr<TensorDescriptor> &input2,
                                             const std::shared_ptr<TensorDescriptor> &output, const bool round,
                                             const std::string &debugName) {
    auto pipeline =
        std::make_shared<ArithmeticRightShift>(loader, device, pipelineCache, input1, input2, output, round, debugName);
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeAvgPool2D(const std::shared_ptr<TensorDescriptor> &input,
                                  const std::shared_ptr<TensorDescriptor> &output, const std::vector<uint32_t> &kernel,
                                  const std::vector<uint32_t> &stride, const std::vector<uint32_t> &pad,
                                  const uint32_t accType, const int8_t inputZeroPoint, const int8_t outputZeroPoint,
                                  const std::string &debugName) {
    auto pipeline = std::make_shared<AvgPool2D>(loader, device, pipelineCache, input, output, kernel, stride, pad,
                                                accType, inputZeroPoint, outputZeroPoint, debugName);
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeBitwiseAnd(const std::shared_ptr<TensorDescriptor> &input1,
                                   const std::shared_ptr<TensorDescriptor> &input2,
                                   const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    auto pipeline = std::make_shared<ElementwiseBinary>(loader, device, pipelineCache, input1, input2, output,
                                                        NanPropagationMode::Propagate, debugName, "value1 & value2");
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeBitwiseNot(const std::shared_ptr<TensorDescriptor> &input,
                                   const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    auto pipeline =
        std::make_shared<ElementwiseUnary>(loader, device, pipelineCache, input, output, debugName, "~value1");
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeBitwiseOr(const std::shared_ptr<TensorDescriptor> &input1,
                                  const std::shared_ptr<TensorDescriptor> &input2,
                                  const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    auto pipeline = std::make_shared<ElementwiseBinary>(loader, device, pipelineCache, input1, input2, output,
                                                        NanPropagationMode::Propagate, debugName, "value1 | value2");
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeBitwiseXor(const std::shared_ptr<TensorDescriptor> &input1,
                                   const std::shared_ptr<TensorDescriptor> &input2,
                                   const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    auto pipeline = std::make_shared<ElementwiseBinary>(loader, device, pipelineCache, input1, input2, output,
                                                        NanPropagationMode::Propagate, debugName, "value1 ^ value2");
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeCast(const std::shared_ptr<TensorDescriptor> &input,
                             const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    auto pipeline = std::make_shared<Cast>(loader, device, pipelineCache, input, output, debugName);
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeCeil(const std::shared_ptr<TensorDescriptor> &input1,
                             const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    auto pipeline =
        std::make_shared<ElementwiseUnary>(loader, device, pipelineCache, input1, output, debugName, "ceil(value1)");
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeClamp(const std::shared_ptr<TensorDescriptor> &input,
                              const std::shared_ptr<TensorDescriptor> &output, const double min, const double max,
                              const uint32_t nanMode, const std::string &debugName) {
    auto pipeline = std::make_shared<Clamp>(loader, device, pipelineCache, input, output, min, max, nanMode, debugName);
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeClz(const std::shared_ptr<TensorDescriptor> &input1,
                            const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    auto pipeline =
        std::make_shared<ElementwiseUnary>(loader, device, pipelineCache, input1, output, debugName, "clz(value1)");
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeConcat(const std::vector<std::shared_ptr<TensorDescriptor>> &_inputs,
                               const std::shared_ptr<TensorDescriptor> &output, const uint32_t axis,
                               const std::string &debugName) {
    uint32_t offset = 0;
    for (const auto &input : _inputs) {
        auto pipeline = std::make_shared<Concat>(loader, device, pipelineCache, input, output, axis, offset, debugName);
        pipelines.emplace_back(pipeline);
        offset += static_cast<uint32_t>(input->getDimensions()[axis]);
    }
}

void GraphPipeline::makeConv2D(const std::shared_ptr<TensorDescriptor> &input,
                               const std::shared_ptr<TensorDescriptor> &output,
                               const std::shared_ptr<TensorDescriptor> &weights,
                               const std::shared_ptr<TensorDescriptor> &biases, const std::vector<uint32_t> &pad,
                               const std::vector<uint32_t> &stride, const std::vector<uint32_t> &dilation,
                               const int8_t inputZeroPoint, const int8_t weightZeroPoint, const uint32_t accType,
                               const std::string &debugName) {
    auto pipeline = std::make_shared<Conv2D>(loader, device, pipelineCache, input, output, weights, biases, pad, stride,
                                             dilation, inputZeroPoint, weightZeroPoint, accType, debugName);
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeConv3D(const std::shared_ptr<TensorDescriptor> &input,
                               const std::shared_ptr<TensorDescriptor> &output,
                               const std::shared_ptr<TensorDescriptor> &weights,
                               const std::shared_ptr<TensorDescriptor> &biases, const std::vector<uint32_t> &pad,
                               const std::vector<uint32_t> &stride, const std::vector<uint32_t> &dilation,
                               const int8_t inputZeroPoint, const int8_t weightZeroPoint, const uint32_t accType,
                               const std::string &debugName) {
    auto pipeline = std::make_shared<Conv3D>(loader, device, pipelineCache, input, output, weights, biases, pad, stride,
                                             dilation, inputZeroPoint, weightZeroPoint, accType, debugName);
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeCos(const std::shared_ptr<TensorDescriptor> &input1,
                            const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    auto pipeline =
        std::make_shared<ElementwiseUnary>(loader, device, pipelineCache, input1, output, debugName, "cos(value1)");
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeDepthwiseConv2D(
    const std::shared_ptr<TensorDescriptor> &input, const std::shared_ptr<TensorDescriptor> &output,
    const std::shared_ptr<TensorDescriptor> &weights, const std::shared_ptr<TensorDescriptor> &biases,
    const std::vector<uint32_t> &pad, const std::vector<uint32_t> &stride, const std::vector<uint32_t> &dilation,
    const int8_t inputZeroPoint, const int8_t weightZeroPoint, const uint32_t accType, const std::string &debugName) {
    auto pipeline =
        std::make_shared<DepthwiseConv2D>(loader, device, pipelineCache, input, output, weights, biases, pad, stride,
                                          dilation, inputZeroPoint, weightZeroPoint, accType, debugName);
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeEqual(const std::shared_ptr<TensorDescriptor> &input1,
                              const std::shared_ptr<TensorDescriptor> &input2,
                              const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    auto pipeline = std::make_shared<ElementwiseBinary>(loader, device, pipelineCache, input1, input2, output,
                                                        NanPropagationMode::Propagate, debugName, "value1 == value2");
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeErf(const std::shared_ptr<TensorDescriptor> &input,
                            const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    auto pipeline =
        std::make_shared<ElementwiseUnary>(loader, device, pipelineCache, input, output, debugName, "erf(value1)");
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeExp(const std::shared_ptr<TensorDescriptor> &input,
                            const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    auto pipeline =
        std::make_shared<ElementwiseUnary>(loader, device, pipelineCache, input, output, debugName, "exp(value1)");
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeFft2D(const std::shared_ptr<TensorDescriptor> &inputReal,
                              const std::shared_ptr<TensorDescriptor> &inputImag,
                              const std::shared_ptr<TensorDescriptor> &outputReal,
                              const std::shared_ptr<TensorDescriptor> &outputImag, const bool inverse,
                              const std::string &debugName) {
    auto pipeline = std::make_shared<Fft2D>(loader, device, pipelineCache, inputReal, inputImag, outputReal, outputImag,
                                            inverse, debugName);
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeFloor(const std::shared_ptr<TensorDescriptor> &input1,
                              const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    auto pipeline =
        std::make_shared<ElementwiseUnary>(loader, device, pipelineCache, input1, output, debugName, "floor(value1)");
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeGather(const std::shared_ptr<TensorDescriptor> &values,
                               const std::shared_ptr<TensorDescriptor> &indices,
                               const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    auto pipeline = std::make_shared<Gather>(loader, device, pipelineCache, values, indices, output, debugName);
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeGreater(const std::shared_ptr<TensorDescriptor> &input1,
                                const std::shared_ptr<TensorDescriptor> &input2,
                                const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    auto pipeline = std::make_shared<ElementwiseBinary>(loader, device, pipelineCache, input1, input2, output,
                                                        NanPropagationMode::Propagate, debugName, "value1 > value2");
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeGreaterEqual(const std::shared_ptr<TensorDescriptor> &input1,
                                     const std::shared_ptr<TensorDescriptor> &input2,
                                     const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    auto pipeline = std::make_shared<ElementwiseBinary>(loader, device, pipelineCache, input1, input2, output,
                                                        NanPropagationMode::Propagate, debugName, "value1 >= value2");
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeIntdiv(const std::shared_ptr<TensorDescriptor> &input1,
                               const std::shared_ptr<TensorDescriptor> &input2,
                               const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    auto pipeline = std::make_shared<ElementwiseBinary>(loader, device, pipelineCache, input1, input2, output,
                                                        NanPropagationMode::Propagate, debugName, "value1 / value2");
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeLog(const std::shared_ptr<TensorDescriptor> &input1,
                            const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    auto pipeline = std::make_shared<ElementwiseUnary>(loader, device, pipelineCache, input1, output, debugName,
                                                       "log_guarded(value1)");
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeLogicalAnd(const std::shared_ptr<TensorDescriptor> &input1,
                                   const std::shared_ptr<TensorDescriptor> &input2,
                                   const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    auto pipeline = std::make_shared<ElementwiseBinary>(loader, device, pipelineCache, input1, input2, output,
                                                        NanPropagationMode::Propagate, debugName, "value1 && value2");
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeLogicalLeftShift(const std::shared_ptr<TensorDescriptor> &input1,
                                         const std::shared_ptr<TensorDescriptor> &input2,
                                         const std::shared_ptr<TensorDescriptor> &output,
                                         const std::string &debugName) {
    auto pipeline =
        std::make_shared<ElementwiseBinary>(loader, device, pipelineCache, input1, input2, output,
                                            NanPropagationMode::Propagate, debugName, "uint(value1) << uint(value2)");
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeLogicalNot(const std::shared_ptr<TensorDescriptor> &input,
                                   const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    auto pipeline =
        std::make_shared<ElementwiseUnary>(loader, device, pipelineCache, input, output, debugName, "!value1");
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeLogicalRightShift(const std::shared_ptr<TensorDescriptor> &input1,
                                          const std::shared_ptr<TensorDescriptor> &input2,
                                          const std::shared_ptr<TensorDescriptor> &output,
                                          const std::string &debugName) {
    auto pipeline = std::make_shared<ElementwiseBinary>(loader, device, pipelineCache, input1, input2, output,
                                                        NanPropagationMode::Propagate, debugName,
                                                        "zeroExtend(value1) >> uint(value2)");
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeLogicalOr(const std::shared_ptr<TensorDescriptor> &input1,
                                  const std::shared_ptr<TensorDescriptor> &input2,
                                  const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    auto pipeline = std::make_shared<ElementwiseBinary>(loader, device, pipelineCache, input1, input2, output,
                                                        NanPropagationMode::Propagate, debugName, "value1 || value2");
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeLogicalXor(const std::shared_ptr<TensorDescriptor> &input1,
                                   const std::shared_ptr<TensorDescriptor> &input2,
                                   const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    auto pipeline = std::make_shared<ElementwiseBinary>(loader, device, pipelineCache, input1, input2, output,
                                                        NanPropagationMode::Propagate, debugName, "value1 ^^ value2");
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeMatmul(const std::shared_ptr<TensorDescriptor> &input1,
                               const std::shared_ptr<TensorDescriptor> &input2,
                               const std::shared_ptr<TensorDescriptor> &output, const int32_t inputZeroPoint1,
                               const int32_t inputZeroPoint2, const std::string &debugName) {
    auto pipeline = std::make_shared<Matmul>(loader, device, pipelineCache, input1, input2, output, inputZeroPoint1,
                                             inputZeroPoint2, debugName);
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeMaxPool2D(const std::shared_ptr<TensorDescriptor> &input,
                                  const std::shared_ptr<TensorDescriptor> &output, const std::vector<uint32_t> &kernel,
                                  const std::vector<uint32_t> &stride, const std::vector<uint32_t> &pad,
                                  const uint32_t nanMode, const std::string &debugName) {
    auto pipeline = std::make_shared<MaxPool2D>(loader, device, pipelineCache, input, output, kernel, stride, pad,
                                                nanMode, debugName);
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeMaximum(const std::shared_ptr<TensorDescriptor> &input1,
                                const std::shared_ptr<TensorDescriptor> &input2,
                                const std::shared_ptr<TensorDescriptor> &output, const uint32_t nanMode,
                                const std::string &debugName) {
    auto pipeline = std::make_shared<ElementwiseBinary>(loader, device, pipelineCache, input1, input2, output, nanMode,
                                                        debugName, "applyMax(value1, value2, pushConstants.nanMode)");
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeMinimum(const std::shared_ptr<TensorDescriptor> &input1,
                                const std::shared_ptr<TensorDescriptor> &input2,
                                const std::shared_ptr<TensorDescriptor> &output, const uint32_t nanMode,
                                const std::string &debugName) {
    auto pipeline = std::make_shared<ElementwiseBinary>(loader, device, pipelineCache, input1, input2, output, nanMode,
                                                        debugName, "applyMin(value1, value2, pushConstants.nanMode)");
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeMul(const std::shared_ptr<TensorDescriptor> &input1,
                            const std::shared_ptr<TensorDescriptor> &input2,
                            const std::shared_ptr<TensorDescriptor> &output, const uint32_t shift,
                            const std::string &debugName) {
    auto pipeline = std::make_shared<Mul>(loader, device, pipelineCache, input1, input2, output, shift, debugName);
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeNegate(const std::shared_ptr<TensorDescriptor> &input,
                               const std::shared_ptr<TensorDescriptor> &output, const int32_t inputZeroPoint,
                               const int32_t outputZeroPoint, const std::string &debugName) {
    auto pipeline = std::make_shared<Negate>(loader, device, pipelineCache, input, output, inputZeroPoint,
                                             outputZeroPoint, debugName);
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makePad(const std::shared_ptr<TensorDescriptor> &input,
                            const std::shared_ptr<TensorDescriptor> &output,
                            const std::shared_ptr<TensorDescriptor> &padding, const double padConst,
                            const std::string &debugName) {
    auto pipeline = std::make_shared<Pad>(loader, device, pipelineCache, input, output, padding, padConst, debugName);
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makePow(const std::shared_ptr<TensorDescriptor> &input1,
                            const std::shared_ptr<TensorDescriptor> &input2,
                            const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    auto pipeline =
        std::make_shared<ElementwiseBinary>(loader, device, pipelineCache, input1, input2, output,
                                            NanPropagationMode::Propagate, debugName, "power(value1, value2)");
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeReciprocal(const std::shared_ptr<TensorDescriptor> &input,
                                   const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    auto pipeline =
        std::make_shared<ElementwiseUnary>(loader, device, pipelineCache, input, output, debugName, "1.0 / value1");
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeReduceAll(const std::shared_ptr<TensorDescriptor> &input,
                                  const std::shared_ptr<TensorDescriptor> &output, const uint32_t axis,
                                  const std::string &debugName) {
    auto pipeline = std::make_shared<Reduce>(loader, device, pipelineCache, input, output, axis,
                                             NanPropagationMode::Propagate, debugName, "true", "result && value");
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeReduceAny(const std::shared_ptr<TensorDescriptor> &input,
                                  const std::shared_ptr<TensorDescriptor> &output, const uint32_t axis,
                                  const std::string &debugName) {
    auto pipeline = std::make_shared<Reduce>(loader, device, pipelineCache, input, output, axis,
                                             NanPropagationMode::Propagate, debugName, "false", "result || value");
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeReduceMax(const std::shared_ptr<TensorDescriptor> &input,
                                  const std::shared_ptr<TensorDescriptor> &output, const uint32_t axis,
                                  const uint32_t nanMode, const std::string &debugName) {
    auto inOutType = makeFormat(output->getFormat());
    const std::string init =
        "(pushConstants.nanMode == NAN_MODE_IGNORE) ? IN_OUT_T(NAN) : IN_OUT_T(" + inOutType->lowest() + ')';
    auto pipeline = std::make_shared<Reduce>(loader, device, pipelineCache, input, output, axis, nanMode, debugName,
                                             init, "max(result, value)");
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeReduceMin(const std::shared_ptr<TensorDescriptor> &input,
                                  const std::shared_ptr<TensorDescriptor> &output, const uint32_t axis,
                                  const uint32_t nanMode, const std::string &debugName) {
    auto inOutType = makeFormat(output->getFormat());
    const std::string init =
        "(pushConstants.nanMode == NAN_MODE_IGNORE) ? IN_OUT_T(NAN) : IN_OUT_T(" + inOutType->max() + ')';
    auto pipeline = std::make_shared<Reduce>(loader, device, pipelineCache, input, output, axis, nanMode, debugName,
                                             init, "min(result, value)");
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeReduceProduct(const std::shared_ptr<TensorDescriptor> &input,
                                      const std::shared_ptr<TensorDescriptor> &output, const uint32_t axis,
                                      const std::string &debugName) {
    auto pipeline = std::make_shared<Reduce>(loader, device, pipelineCache, input, output, axis,
                                             NanPropagationMode::Propagate, debugName, "1", "result * value");
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeReduceSum(const std::shared_ptr<TensorDescriptor> &input,
                                  const std::shared_ptr<TensorDescriptor> &output, const uint32_t axis,
                                  const std::string &debugName) {
    auto pipeline = std::make_shared<Reduce>(loader, device, pipelineCache, input, output, axis,
                                             NanPropagationMode::Propagate, debugName, "0", "result + value");
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeRescale(const std::shared_ptr<TensorDescriptor> &input,
                                const std::shared_ptr<TensorDescriptor> &output, const int32_t inputZeroPoint,
                                const int32_t outputZeroPoint, const std::shared_ptr<TensorDescriptor> &multiplier,
                                const std::shared_ptr<TensorDescriptor> &shift, const bool scale32,
                                const bool doubleRound, const bool perChannel, const bool inputUnsigned,
                                const bool outputUnsigned, const std::string &debugName) {
    auto pipeline = std::make_shared<Rescale>(loader, device, pipelineCache, input, output, inputZeroPoint,
                                              outputZeroPoint, multiplier, shift, scale32, doubleRound, perChannel,
                                              inputUnsigned, outputUnsigned, debugName);
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeReshape(const std::shared_ptr<TensorDescriptor> &input,
                                const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    auto pipeline = std::make_shared<Reshape>(loader, device, pipelineCache, input, output, debugName);
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeResize(const std::shared_ptr<TensorDescriptor> &input,
                               const std::shared_ptr<TensorDescriptor> &output, const std::vector<int32_t> &scale,
                               const std::vector<int32_t> &offset, const std::vector<int32_t> &border,
                               const uint32_t mode, const std::string &debugName) {
    auto pipeline =
        std::make_shared<Resize>(loader, device, pipelineCache, input, output, scale, offset, border, mode, debugName);
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeReverse(const std::shared_ptr<TensorDescriptor> &input,
                                const std::shared_ptr<TensorDescriptor> &output, const uint32_t axis,
                                const std::string &debugName) {
    auto pipeline = std::make_shared<Reverse>(loader, device, pipelineCache, input, output, axis, debugName);
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeRfft2D(const std::shared_ptr<TensorDescriptor> &input,
                               const std::shared_ptr<TensorDescriptor> &outputReal,
                               const std::shared_ptr<TensorDescriptor> &outputImag, const std::string &debugName) {
    auto pipeline = std::make_shared<Rfft2D>(loader, device, pipelineCache, input, outputReal, outputImag, debugName);
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeRsqrt(const std::shared_ptr<TensorDescriptor> &input1,
                              const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    auto pipeline = std::make_shared<ElementwiseUnary>(loader, device, pipelineCache, input1, output, debugName,
                                                       "inversesqrt(value1)");
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeScatter(const std::shared_ptr<TensorDescriptor> &input,
                                const std::shared_ptr<TensorDescriptor> &values,
                                const std::shared_ptr<TensorDescriptor> &indices,
                                const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    auto pipeline = std::make_shared<Scatter>(loader, device, pipelineCache, input, values, indices, output, debugName);
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeSelect(const std::shared_ptr<TensorDescriptor> &input1,
                               const std::shared_ptr<TensorDescriptor> &input2,
                               const std::shared_ptr<TensorDescriptor> &input3,
                               const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    auto pipeline = std::make_shared<Select>(loader, device, pipelineCache, input1, input2, input3, output, debugName);
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeSigmoid(const std::shared_ptr<TensorDescriptor> &input,
                                const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    auto pipeline = std::make_shared<ElementwiseUnary>(loader, device, pipelineCache, input, output, debugName,
                                                       "1.0 / (1.0 + exp(-value1))");
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeSin(const std::shared_ptr<TensorDescriptor> &input1,
                            const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    auto pipeline =
        std::make_shared<ElementwiseUnary>(loader, device, pipelineCache, input1, output, debugName, "sin(value1)");
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeSlice(const std::shared_ptr<TensorDescriptor> &input,
                              const std::shared_ptr<TensorDescriptor> &output, const std::vector<uint32_t> &start,
                              const std::string &debugName) {
    auto pipeline = std::make_shared<Slice>(loader, device, pipelineCache, input, output, start, debugName);
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeSub(const std::shared_ptr<TensorDescriptor> &input1,
                            const std::shared_ptr<TensorDescriptor> &input2,
                            const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    auto pipeline = std::make_shared<ElementwiseBinary>(loader, device, pipelineCache, input1, input2, output,
                                                        NanPropagationMode::Propagate, debugName, "value1 - value2");
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeTable(const std::shared_ptr<TensorDescriptor> &input,
                              const std::shared_ptr<TensorDescriptor> &output,
                              const std::shared_ptr<TensorDescriptor> &table, const std::string &debugName) {
    auto pipeline = std::make_shared<Table>(loader, device, pipelineCache, input, output, table, debugName);
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeTanh(const std::shared_ptr<TensorDescriptor> &input1,
                             const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    auto pipeline = std::make_shared<ElementwiseUnary>(loader, device, pipelineCache, input1, output, debugName,
                                                       "tanh_clamped(value1)");
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeTile(const std::shared_ptr<TensorDescriptor> &input,
                             const std::shared_ptr<TensorDescriptor> &output, const std::string &debugName) {
    auto pipeline = std::make_shared<Tile>(loader, device, pipelineCache, input, output, debugName);
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeTranspose(const std::shared_ptr<TensorDescriptor> &input,
                                  const std::shared_ptr<TensorDescriptor> &output, const std::vector<uint32_t> &perms,
                                  const std::string &debugName) {
    auto pipeline = std::make_shared<Transpose>(loader, device, pipelineCache, input, output, perms, debugName);
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeTransposeConv2D(const std::shared_ptr<TensorDescriptor> &input,
                                        const std::shared_ptr<TensorDescriptor> &output,
                                        const std::shared_ptr<TensorDescriptor> &weights,
                                        const std::shared_ptr<TensorDescriptor> &biases,
                                        const std::vector<uint32_t> &pad, const std::vector<uint32_t> &stride,
                                        const int8_t inputZeroPoint, const int8_t weightZeroPoint,
                                        const uint32_t accType, const std::string &debugName) {
    auto pipeline = std::make_shared<TransposeConv2D>(loader, device, pipelineCache, input, output, weights, biases,
                                                      pad, stride, inputZeroPoint, weightZeroPoint, accType, debugName);
    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeMinSadCost(
    const std::shared_ptr<TensorDescriptor> &inTemplate, const std::shared_ptr<TensorDescriptor> &inSearch,
    const std::shared_ptr<TensorDescriptor> &outVectors, const std::shared_ptr<TensorDescriptor> &outCosts,
    const std::vector<uint32_t> &kernelSizes, const std::vector<uint32_t> &searchWindowSizes,
    const std::vector<uint32_t> &inputStrides, const std::vector<uint32_t> &windowStrides,
    const std::vector<uint32_t> &windowOffsets, const std::vector<uint32_t> &padding, const uint32_t searchPattern,
    const std::string &debugName) {
    auto pipeline =
        std::make_shared<BlockMatch>(loader, device, pipelineCache, inTemplate, inSearch, outVectors, outCosts,
                                     kernelSizes, searchWindowSizes, inputStrides, windowStrides, windowOffsets,
                                     padding, searchPattern, BlockMatch::SearchType::MIN_SAD_COST, debugName);

    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeMinSad(const std::shared_ptr<TensorDescriptor> &inTemplate,
                               const std::shared_ptr<TensorDescriptor> &inSearch,
                               const std::shared_ptr<TensorDescriptor> &outVectors,
                               const std::vector<uint32_t> &kernelSizes, const std::vector<uint32_t> &searchWindowSizes,
                               const std::vector<uint32_t> &inputStrides, const std::vector<uint32_t> &windowStrides,
                               const std::vector<uint32_t> &windowOffsets, const std::vector<uint32_t> &padding,
                               const uint32_t searchPattern, const std::string &debugName) {
    auto pipeline = std::make_shared<BlockMatch>(
        loader, device, pipelineCache, inTemplate, inSearch, outVectors, std::nullopt, kernelSizes, searchWindowSizes,
        inputStrides, windowStrides, windowOffsets, padding, searchPattern, BlockMatch::SearchType::MIN_SAD, debugName);

    pipelines.emplace_back(pipeline);
}

void GraphPipeline::makeRawSad(const std::shared_ptr<TensorDescriptor> &inTemplate,
                               const std::shared_ptr<TensorDescriptor> &inSearch,
                               const std::shared_ptr<TensorDescriptor> &outCosts,
                               const std::vector<uint32_t> &kernelSizes, const std::vector<uint32_t> &searchWindowSizes,
                               const std::vector<uint32_t> &inputStrides, const std::vector<uint32_t> &windowStrides,
                               const std::vector<uint32_t> &windowOffsets, const std::vector<uint32_t> &padding,
                               const std::string &debugName) {
    auto pipeline = std::make_shared<BlockMatch>(loader, device, pipelineCache, inTemplate, inSearch, std::nullopt,
                                                 outCosts, kernelSizes, searchWindowSizes, inputStrides, windowStrides,
                                                 windowOffsets, padding, 0, BlockMatch::SearchType::RAW_SAD, debugName);

    pipelines.emplace_back(pipeline);
}

} // namespace mlsdk::el::compute
