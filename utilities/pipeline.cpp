/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "mlel/pipeline.hpp"

#include "mlel/exception.hpp"
#include "mlel/utils.hpp"

#include <numeric>

namespace mlsdk::el::utilities {

/*******************************************************************************
 * GraphPipelineConstantTensor
 *******************************************************************************/

GraphPipelineConstantTensor::GraphPipelineConstantTensor(const Shape &_shape, const uint8_t *pointer, const size_t size)
    : shape{_shape}, _data{pointer, pointer + size}, constantTensorDescription{createConstantTensorDescription()} {
    if (_data.size() != _shape.getSize()) {
        throw std::runtime_error("Size of constant tensors does not match data size");
    }
}

const vk::TensorDescriptionARM &GraphPipelineConstantTensor::getConstantTensorDescription() const {
    return constantTensorDescription;
}

size_t GraphPipelineConstantTensor::size() const { return _data.size(); }

const uint8_t *GraphPipelineConstantTensor::data() const { return _data.data(); }

void GraphPipelineConstantTensor::print() const {
    std::ios_base::fmtflags coutFlags(std::cout.flags());
    std::cout << std::hex << std::setfill('0');

    for (uint32_t i = 0; i < size(); i++) {
        if ((i % 16) == 0) {
            std::cout << std::endl << std::setw(8) << i << ": ";
        }

        std::cout << std::setw(2) << static_cast<unsigned>(_data[i]) << " ";
    }

    std::cout << std::endl;
    std::cout.flags(coutFlags);
}

vk::TensorDescriptionARM GraphPipelineConstantTensor::createConstantTensorDescription() const {
    return {
        vk::TensorTilingARM::eLinear,                        // tiling
        shape.getFormat(),                                   // format
        static_cast<uint32_t>(shape.getDimensions().size()), // dimension count
        shape.getDimensions().data(),                        // dimensions
        nullptr,                                             // strides
        vk::TensorUsageFlagBitsARM::eDataGraph,              // usage
    };
}

/*******************************************************************************
 * GraphConstants
 *******************************************************************************/

void GraphConstants::makeGraphPipelineConstantTensor(uint32_t id, const Shape &shape, const uint8_t *pointer,
                                                     const size_t size) {
    constants[id] = std::make_shared<GraphPipelineConstantTensor>(shape, pointer, size);
}

std::vector<vk::DataGraphPipelineConstantARM> GraphConstants::getGraphPipelineConstants() const {
    std::vector<vk::DataGraphPipelineConstantARM> graphPipelineConstants;

    for (auto [id, constant] : constants) {
        graphPipelineConstants.push_back({
            id,                                       // id
            constant->data(),                         // data
            &constant->getConstantTensorDescription() // next
        });
    }

    return graphPipelineConstants;
}

const std::map<uint32_t, std::shared_ptr<GraphPipelineConstantTensor>> &GraphConstants::operator&() const {
    return constants;
}

GraphPipelineConstantTensor &GraphConstants::operator[](size_t size) { return *constants[size]; }

/*******************************************************************************
 * PipelineBase
 *******************************************************************************/

PipelineBase::PipelineBase(std::shared_ptr<Device> &_device, const DescriptorMap &_descriptorMap,
                           const std::vector<uint32_t> &_spirv)
    : device(_device), descriptorMap{_descriptorMap}, descriptorPool{createDescriptorPool()},
      descriptorSetLayouts{createDescriptorSetLayouts()}, pipelineLayout{createPipelineLayout()},
      shaderModule{createShaderModule(_spirv)}, commandPool{createCommandPool()} {}

PipelineBase::PipelineBase(std::shared_ptr<Device> &_device, const DescriptorMap &_descriptorMap,
                           const std::shared_ptr<vk::raii::PipelineLayout> &_pipelineLayout,
                           const std::vector<uint32_t> &_spirv)
    : device(_device), descriptorMap{_descriptorMap}, descriptorPool{createDescriptorPool()},
      descriptorSetLayouts{createDescriptorSetLayouts()}, pipelineLayout{_pipelineLayout},
      shaderModule{createShaderModule(_spirv)}, commandPool{createCommandPool()} {}

void PipelineBase::updateDescriptorSet(const vk::raii::DescriptorSets &descriptorSets,
                                       const DescriptorMap &descriptorMap) const {
    uint32_t set = 0;
    for (const auto &bindingMap : descriptorMap) {
        for (const auto &[binding, tensors] : bindingMap) {
            for (uint32_t arrayIndex = 0; arrayIndex < tensors.size(); arrayIndex++) {
                tensors[arrayIndex]->updateDescriptorSet(descriptorSets[set], binding, arrayIndex);
            }
        }

        set++;
    }
}

const std::shared_ptr<vk::raii::PipelineLayout> &PipelineBase::getPipelineLayout() const { return pipelineLayout; }

void PipelineBase::submitWork(vk::raii::CommandBuffer &commandBuffer) const {
    vk::raii::Queue queue(&(*device), device->getPhysicalDevice()->getComputeFamilyIndex(), 0);

    vk::raii::Fence fence(&(*device), vk::FenceCreateInfo());

    const vk::SubmitInfo submitInfo{
        0,                 // wait semaphore count
        nullptr,           // wait semaphore
        nullptr,           // pipeline stage flags
        1,                 // command buffer count
        &(*commandBuffer), // command buffers
        0,                 // signal semaphore count
    };

    queue.submit({1, &submitInfo}, *fence);
    while (vk::Result::eTimeout == (&(*device)).waitForFences({*fence}, vk::True, uint64_t(-1)))
        ;
}

uint32_t PipelineBase::getDescriptorCount() const {
    uint32_t descriptorCount = 0;

    for (const auto &bindingMap : descriptorMap) {
        descriptorCount =
            std::accumulate(bindingMap.begin(), bindingMap.end(), descriptorCount,
                            [](uint32_t acc, const auto &kv) { return acc + static_cast<uint32_t>(kv.second.size()); });
    }

    return descriptorCount;
}

vk::raii::DescriptorPool PipelineBase::createDescriptorPool() const {
    const vk::DescriptorPoolSize descriptorPoolSize{
        vk::DescriptorType::eTensorARM, // type
        getDescriptorCount()            // descriptor count
    };

    auto descriptorPoolCreateFlags =
        vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet | vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind;

    const vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo{
        descriptorPoolCreateFlags,                   // flags
        static_cast<uint32_t>(descriptorMap.size()), // max sets
        1,                                           // pool size count
        &descriptorPoolSize                          // descriptor pool size
    };

    return {&(*device), descriptorPoolCreateInfo};
}

std::vector<vk::raii::DescriptorSetLayout> PipelineBase::createDescriptorSetLayouts() const {
    std::vector<vk::raii::DescriptorSetLayout> descriptorSetLayouts;

    for (auto &bindingMap : descriptorMap) {
        std::vector<vk::DescriptorSetLayoutBinding> descriptorSetLayoutBindings;

        for (const auto &[binding, tensors] : bindingMap) {
            descriptorSetLayoutBindings.push_back(vk::DescriptorSetLayoutBinding{
                binding,                        // binding
                vk::DescriptorType::eTensorARM, // descriptor type
                uint32_t(tensors.size()),       // descriptor count
                vk::ShaderStageFlagBits::eAll,  // flags
            });
        }

        std::vector<vk::DescriptorBindingFlags> descriptorBindingFlags(descriptorSetLayoutBindings.size(),
                                                                       vk::DescriptorBindingFlagBits::eUpdateAfterBind);

        const vk::DescriptorSetLayoutBindingFlagsCreateInfo descriptorSetBindingFlagsCreateInfo{
            static_cast<uint32_t>(descriptorBindingFlags.size()), // binding count
            descriptorBindingFlags.data(),                        // binding flags
        };

        const vk::DescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo{
            vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool, // flags
            static_cast<uint32_t>(descriptorSetLayoutBindings.size()),   // binding count
            descriptorSetLayoutBindings.data(),                          // bindings
            &descriptorSetBindingFlagsCreateInfo,                        // next
        };

        descriptorSetLayouts.push_back(vk::raii::DescriptorSetLayout(&(*device), descriptorSetLayoutCreateInfo));
    }

    return descriptorSetLayouts;
}

PipelineBase::PoolAndSet PipelineBase::createDescriptorSets(const DescriptorMap &descriptorMap) const {
    auto descriptorPool = createDescriptorPool();

    std::vector<vk::DescriptorSetLayout> vkDescriptorSetLayouts;
    std::transform(descriptorSetLayouts.begin(), descriptorSetLayouts.end(), std::back_inserter(vkDescriptorSetLayouts),
                   [](const auto &layout) { return *layout; });

    const vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo{
        descriptorPool,                          // descriptor pool
        uint32_t(vkDescriptorSetLayouts.size()), // descriptor set layout count
        vkDescriptorSetLayouts.data()            // descriptor set layouts
    };

    vk::raii::DescriptorSets descriptorSets{&(*device), descriptorSetAllocateInfo};

    updateDescriptorSet(descriptorSets, descriptorMap);

    return {std::move(descriptorPool), std::move(descriptorSets)};
}

std::shared_ptr<vk::raii::PipelineLayout> PipelineBase::createPipelineLayout() const {
    std::vector<vk::DescriptorSetLayout> vkDescriptorSetLayouts;
    std::transform(descriptorSetLayouts.begin(), descriptorSetLayouts.end(), std::back_inserter(vkDescriptorSetLayouts),
                   [](const auto &layout) { return *layout; });

    const vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo{
        {},                                                   // flags
        static_cast<uint32_t>(vkDescriptorSetLayouts.size()), // descriptor set layout count
        vkDescriptorSetLayouts.data()                         // descriptor set layouts
    };

    return std::make_shared<vk::raii::PipelineLayout>(&(*device), pipelineLayoutCreateInfo);
}

vk::raii::ShaderModule PipelineBase::createShaderModule(const std::vector<uint32_t> &code) const {
    const vk::ShaderModuleCreateInfo info{
        {},                                                    // flags
        static_cast<uint32_t>(code.size() * sizeof(uint32_t)), // code size
        code.data()                                            // code
    };

    return vk::raii::ShaderModule(&(*device), info);
}

vk::raii::CommandPool PipelineBase::createCommandPool() const {
    const vk::CommandPoolCreateInfo commandPoolCreateInfo{
        {},                                                  // flags
        device->getPhysicalDevice()->getComputeFamilyIndex() // queue family index
    };

    return vk::raii::CommandPool(&(*device), commandPoolCreateInfo);
}

vk::raii::CommandBuffer PipelineBase::createCommandBuffer() const {
    const vk::CommandBufferAllocateInfo commandBufferAllocInfo{
        *commandPool,                     // command pool
        vk::CommandBufferLevel::ePrimary, // command buffer level
        1                                 // command buffer count
    };

    vk::raii::CommandBuffers commandBuffers(&(*device), commandBufferAllocInfo);

    return std::move(commandBuffers.front());
}

/*******************************************************************************
 * GraphPipeline
 *******************************************************************************/

GraphPipeline::GraphPipeline(std::shared_ptr<Device> &_device, const DescriptorMap &_descriptorMap,
                             const GraphConstants &_graphConstants, const std::vector<uint32_t> &_spirv,
                             bool _hostMemory)
    : PipelineBase(_device, _descriptorMap, _spirv), graphConstants{_graphConstants},
      hostMemory{_hostMemory}, pipeline{createPipeline()} {}

GraphPipeline::GraphPipeline(std::shared_ptr<Device> &_device, const DescriptorMap &_descriptorMap,
                             const std::shared_ptr<vk::raii::PipelineLayout> &_pipelineLayout,
                             const GraphConstants &_graphConstants, const std::vector<uint32_t> &_spirv,
                             bool _hostMemory)
    : PipelineBase(_device, _descriptorMap, _pipelineLayout, _spirv), graphConstants{_graphConstants},
      hostMemory{_hostMemory}, pipeline{createPipeline()} {}

void GraphPipeline::dispatch(const vk::raii::CommandBuffer &commandBuffer,
                             const vk::raii::DescriptorSets &descriptorSets) {
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eDataGraphARM, *pipeline);

    std::vector<vk::DescriptorSet> descriptorSetVec;
    std::transform(descriptorSets.begin(), descriptorSets.end(), std::back_inserter(descriptorSetVec),
                   [](const auto &set) { return *set; });

    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eDataGraphARM, *pipelineLayout, 0,
                                     {uint32_t(descriptorSetVec.size()), descriptorSetVec.data()}, nullptr);
    Session session;
    session.graphPipelineSession = createGraphPipelineSession();
    session.memoryRequirements = createMemoryRequirements(session.graphPipelineSession);
    session.deviceMemory = bindGraphPipelineSessionMemory(session.graphPipelineSession, session.memoryRequirements);
    session.pointer = mapGraphPipelineSessionMemory(session.deviceMemory);
    sessions.push_back(std::move(session));
    commandBuffer.dispatchDataGraphARM(sessions.back().graphPipelineSession);
}

void GraphPipeline::dispatchSubmit() {
    auto [descriptorPool, descriptorSets] = createDescriptorSets(descriptorMap);

    auto commandBuffer = createCommandBuffer();

    const vk::CommandBufferBeginInfo commandBufferBeginInfo{
        vk::CommandBufferUsageFlagBits::eOneTimeSubmit, // flags
    };

    commandBuffer.begin(commandBufferBeginInfo);
    dispatch(commandBuffer, descriptorSets);
    commandBuffer.end();

    submitWork(commandBuffer);
}

void GraphPipeline::dispatchUpdateSubmit() {
    auto commandBuffer = createCommandBuffer();

    const vk::CommandBufferBeginInfo commandBufferBeginInfo{
        vk::CommandBufferUsageFlagBits::eOneTimeSubmit, // flags
    };

    commandBuffer.begin(commandBufferBeginInfo);

    auto [descriptorPool, descriptorSets] = createDescriptorSets(descriptorMap);
    dispatch(commandBuffer, descriptorSets);

    // Update just one of them to prove the point
    {
        const uint32_t set = 0;
        const auto &bindingMap = descriptorMap.at(0);
        const uint32_t binding = 0;
        const auto &tensors = bindingMap.at(binding);
        const uint32_t arrayIndex = 0;
        tensors[arrayIndex]->updateDescriptorSet(descriptorSets[set], binding, arrayIndex);
    }

    commandBuffer.end();

    submitWork(commandBuffer);
}

void GraphPipeline::printGraphPipelineSessionMemory() const {
    int i = 0;
    for (const auto &session : sessions) {
        if (session.pointer == nullptr) {
            continue;
        }
        std::cout << "Session: " << i++ << " at address: " << session.pointer << '\n';
        const auto p = static_cast<const uint8_t *>(session.pointer);
        std::ios_base::fmtflags coutFlags(std::cout.flags());
        std::cout << std::hex << std::setfill('0');

        const auto &memReq = session.memoryRequirements.at(vk::DataGraphPipelineSessionBindPointARM::eTransient);
        for (uint32_t i = 0; i < memReq.size; i++) {
            if ((i % 16) == 0) {
                std::cout << std::endl << std::setw(8) << i << ": ";
            }

            std::cout << std::setw(2) << static_cast<unsigned>(p[i]) << " ";
        }

        std::cout << std::endl;
        std::cout.flags(coutFlags);
    }
}

void GraphPipeline::clearSessions() { sessions.clear(); }

vk::raii::Pipeline GraphPipeline::createPipeline() const {
    std::vector<vk::DataGraphPipelineResourceInfoARM> graphPipelineResourceInfos;
    uint32_t set = 0;

    for (auto &bindingMap : descriptorMap) {
        for (const auto &[binding, tensors] : bindingMap) {
            for (uint32_t i = 0; i < tensors.size(); i++) {
                const auto &tensor = tensors[i];

                graphPipelineResourceInfos.push_back(vk::DataGraphPipelineResourceInfoARM{
                    set,                            // descriptor set
                    binding,                        // binding
                    i,                              // array element
                    &tensor->getTensorDescription() // next
                });
            }
        }
        set++;
    }

    const auto graphPipelineConstants = graphConstants.getGraphPipelineConstants();

    const vk::DataGraphPipelineShaderModuleCreateInfoARM shaderModuleCreateInfo{
        *shaderModule,                           // shader module
        "Graph Pipeline",                        // name
        nullptr,                                 // specialization info
        uint32_t(graphPipelineConstants.size()), // constant count
        graphPipelineConstants.data(),           // constants
    };

    const vk::DataGraphPipelineCreateInfoARM graphPipelineCreateInfo{
        {},                                          // flags
        *pipelineLayout,                             // pipeline layout
        uint32_t(graphPipelineResourceInfos.size()), // resource info count
        graphPipelineResourceInfos.data(),           // resource infos
        &shaderModuleCreateInfo,                     // next
    };

    return {&(*device), nullptr, nullptr, graphPipelineCreateInfo};
}

vk::raii::DataGraphPipelineSessionARM GraphPipeline::createGraphPipelineSession() const {
    const vk::DataGraphPipelineSessionCreateInfoARM graphPipelineSessionCreateInfo{
        {},        // flags
        *pipeline, // pipeline
    };

    return vk::raii::DataGraphPipelineSessionARM{&(*device), graphPipelineSessionCreateInfo};
}

std::map<vk::DataGraphPipelineSessionBindPointARM, vk::MemoryRequirements>
GraphPipeline::createMemoryRequirements(const vk::raii::DataGraphPipelineSessionARM &graphPipelineSession) const {
    // Get bind points
    const auto bindPoints = (&(*device))
                                .getDataGraphPipelineSessionBindPointRequirementsARM(
                                    vk::DataGraphPipelineSessionBindPointRequirementsInfoARM{graphPipelineSession});

    std::map<vk::DataGraphPipelineSessionBindPointARM, vk::MemoryRequirements> memoryRequirements;
    for (const auto &bindPoint : bindPoints) {
        const auto requirement =
            (&(*device))
                .getDataGraphPipelineSessionMemoryRequirementsARM(
                    vk::DataGraphPipelineSessionMemoryRequirementsInfoARM{graphPipelineSession, bindPoint.bindPoint});

        memoryRequirements[bindPoint.bindPoint] = requirement.memoryRequirements;
    }

    return memoryRequirements;
}

std::shared_ptr<vk::raii::DeviceMemory> GraphPipeline::bindGraphPipelineSessionMemory(
    const vk::raii::DataGraphPipelineSessionARM &graphPipelineSession,
    const std::map<vk::DataGraphPipelineSessionBindPointARM, vk::MemoryRequirements> &memoryRequirements) const {
    std::vector<vk::BindDataGraphPipelineSessionMemoryInfoARM> bindInfos;
    std::shared_ptr<vk::raii::DeviceMemory> deviceMemory;

    const auto it = memoryRequirements.find(vk::DataGraphPipelineSessionBindPointARM::eTransient);
    if (it == memoryRequirements.end()) {
        return deviceMemory;
    }
    const auto &memReq = it->second;

    if (memReq.size > 0) {
        deviceMemory = device->allocateDeviceMemory(
            memReq.size, hostMemory ? vk::MemoryPropertyFlagBits::eHostVisible : vk::MemoryPropertyFlags{},
            memReq.memoryTypeBits);

        bindInfos.push_back(vk::BindDataGraphPipelineSessionMemoryInfoARM{
            *graphPipelineSession,                                // session
            vk::DataGraphPipelineSessionBindPointARM::eTransient, // bind point
            0,                                                    // resourceIndex
            **deviceMemory,                                       // device memory
            {}                                                    // memory offset
        });
    }

    if (!bindInfos.empty()) {
        (&(*device)).bindDataGraphPipelineSessionMemoryARM({uint32_t(bindInfos.size()), bindInfos.data()});
    }

    return deviceMemory;
}

void *GraphPipeline::mapGraphPipelineSessionMemory(std::shared_ptr<vk::raii::DeviceMemory> deviceMemory) const {
    if (deviceMemory == nullptr || &(*deviceMemory.get()) == nullptr || !hostMemory) {
        return nullptr;
    }
    void *pointer = deviceMemory->mapMemory({}, VK_WHOLE_SIZE);

    return pointer;
}

/*******************************************************************************
 * TensorComputePipeline
 *******************************************************************************/

TensorComputePipeline::TensorComputePipeline(std::shared_ptr<Device> &_device, const DescriptorMap &_descriptorMap,
                                             const std::vector<uint32_t> &_spirv)
    : PipelineBase(_device, _descriptorMap, _spirv), pipeline{createPipeline()} {}

void TensorComputePipeline::dispatchSubmit(uint32_t _x, uint32_t _y, uint32_t _z) {
    auto [descriptorPool, descriptorSets] = createDescriptorSets(descriptorMap);

    auto commandBuffer = createCommandBuffer();

    const vk::CommandBufferBeginInfo commandBufferBeginInfo{
        vk::CommandBufferUsageFlagBits::eOneTimeSubmit, // flags
    };

    commandBuffer.begin(commandBufferBeginInfo);
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);

    std::vector<vk::DescriptorSet> descriptorSetVec;
    std::transform(descriptorSets.begin(), descriptorSets.end(), std::back_inserter(descriptorSetVec),
                   [](const auto &set) { return *set; });

    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipelineLayout, 0,
                                     {uint32_t(descriptorSetVec.size()), descriptorSetVec.data()}, nullptr);
    commandBuffer.dispatch(_x, _y, _z);
    commandBuffer.end();

    submitWork(commandBuffer);
}

vk::raii::Pipeline TensorComputePipeline::createPipeline() const {
    const vk::PipelineShaderStageCreateInfo stageInfo{
        {},                                // flags
        vk::ShaderStageFlagBits::eCompute, // stage
        *shaderModule,                     // module
        "main"                             // name
    };

    const vk::ComputePipelineCreateInfo pipelineInfo{
        {},             // flags
        stageInfo,      // stage
        *pipelineLayout // layout
    };

    return vk::raii::Pipeline{&(*device), nullptr, pipelineInfo};
}

} // namespace mlsdk::el::utilities
