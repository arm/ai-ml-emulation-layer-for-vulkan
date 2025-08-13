/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "device.hpp"
#include "tensor.hpp"

#include <map>
#include <vector>

namespace mlsdk::el::utilities {

/*******************************************************************************
 * GraphPipelineConstantTensor
 *******************************************************************************/

class GraphPipelineConstantTensor {
  public:
    explicit GraphPipelineConstantTensor(const Shape &_shape, const uint8_t *pointer = nullptr, const size_t size = 0);

    size_t size() const;
    const uint8_t *data() const;
    void print() const;

    const vk::TensorDescriptionARM &getConstantTensorDescription() const;

  private:
    vk::TensorDescriptionARM createConstantTensorDescription() const;

    Shape shape;
    std::vector<uint8_t> _data;
    vk::TensorDescriptionARM constantTensorDescription;
};

/*******************************************************************************
 * GraphConstants
 *******************************************************************************/

class GraphConstants {
  public:
    void makeGraphPipelineConstantTensor(uint32_t id, const Shape &shape, const uint8_t *pointer, const size_t size);

    template <typename T>
    void makeGraphPipelineConstantTensor(uint32_t id, const Shape &shape, const std::vector<T> &data) {
        makeGraphPipelineConstantTensor(id, shape, reinterpret_cast<const uint8_t *>(data.data()),
                                        data.size() * sizeof(T));
    }

    std::vector<vk::DataGraphPipelineConstantARM> getGraphPipelineConstants() const;
    const std::map<uint32_t, std::shared_ptr<GraphPipelineConstantTensor>> &operator&() const;
    GraphPipelineConstantTensor &operator[](size_t size);

  private:
    std::map<uint32_t, std::shared_ptr<GraphPipelineConstantTensor>> constants;
};

/*******************************************************************************
 * PipelineBase
 *******************************************************************************/

class PipelineBase {
  public:
    using BindingMap = std::map<uint32_t, std::vector<std::shared_ptr<Tensor>>>;
    using DescriptorMap = std::vector<BindingMap>;
    struct PoolAndSet {
        vk::raii::DescriptorPool pool;
        vk::raii::DescriptorSets sets;
    };

    PipelineBase(std::shared_ptr<Device> &_device, const DescriptorMap &_descriptorMap,
                 const std::vector<uint32_t> &_spirv);
    PipelineBase(std::shared_ptr<Device> &_device, const DescriptorMap &_descriptorMap,
                 const std::shared_ptr<vk::raii::PipelineLayout> &_pipelineLayout, const std::vector<uint32_t> &_spirv);

    vk::raii::CommandBuffer createCommandBuffer() const;
    PoolAndSet createDescriptorSets(const DescriptorMap &descriptorMap) const;
    const std::shared_ptr<vk::raii::PipelineLayout> &getPipelineLayout() const;
    void submitWork(vk::raii::CommandBuffer &commandBuffer) const;

  protected:
    ~PipelineBase() = default;

    uint32_t getDescriptorCount() const;
    vk::raii::DescriptorPool createDescriptorPool() const;
    std::vector<vk::raii::DescriptorSetLayout> createDescriptorSetLayouts() const;
    std::shared_ptr<vk::raii::PipelineLayout> createPipelineLayout() const;
    vk::raii::ShaderModule createShaderModule(const std::vector<uint32_t> &code) const;
    vk::raii::CommandPool createCommandPool() const;
    void updateDescriptorSet(const vk::raii::DescriptorSets &descriptorSets, const DescriptorMap &descriptorMap) const;

    std::shared_ptr<Device> device;
    DescriptorMap descriptorMap;

    vk::raii::DescriptorPool descriptorPool;
    std::vector<vk::raii::DescriptorSetLayout> descriptorSetLayouts;
    std::shared_ptr<vk::raii::PipelineLayout> pipelineLayout;
    vk::raii::ShaderModule shaderModule;
    vk::raii::CommandPool commandPool;
};

/*******************************************************************************
 * GraphPipeline
 *******************************************************************************/

class GraphPipeline : public PipelineBase {
  public:
    GraphPipeline(std::shared_ptr<Device> &_device, const DescriptorMap &_descriptorMap,
                  const GraphConstants &_graphConstants, const std::vector<uint32_t> &_spirv, bool _hostMemory = true);

    GraphPipeline(std::shared_ptr<Device> &_device, const DescriptorMap &_descriptorMap,
                  const std::shared_ptr<vk::raii::PipelineLayout> &_pipelineLayout,
                  const GraphConstants &_graphConstants, const std::vector<uint32_t> &_spirv, bool _hostMemory = true);

    void dispatch(const vk::raii::CommandBuffer &commandBuffer, const vk::raii::DescriptorSets &descriptorSets);
    void dispatchSubmit();
    void dispatchUpdateSubmit();
    void printGraphPipelineSessionMemory() const;

    void clearSessions();

  private:
    struct Session {
        vk::raii::DataGraphPipelineSessionARM graphPipelineSession{nullptr};
        std::map<vk::DataGraphPipelineSessionBindPointARM, vk::MemoryRequirements> memoryRequirements;
        std::shared_ptr<vk::raii::DeviceMemory> deviceMemory;
        void *pointer;
    };

    vk::raii::Pipeline createPipeline() const;
    vk::raii::DataGraphPipelineSessionARM createGraphPipelineSession() const;
    std::map<vk::DataGraphPipelineSessionBindPointARM, vk::MemoryRequirements>
    createMemoryRequirements(const vk::raii::DataGraphPipelineSessionARM &graphPipelineSession) const;
    std::shared_ptr<vk::raii::DeviceMemory> bindGraphPipelineSessionMemory(
        const vk::raii::DataGraphPipelineSessionARM &graphPipelineSession,
        const std::map<vk::DataGraphPipelineSessionBindPointARM, vk::MemoryRequirements> &memoryRequirements) const;
    void *mapGraphPipelineSessionMemory(std::shared_ptr<vk::raii::DeviceMemory> deviceMemory) const;

    const GraphConstants &graphConstants;
    bool hostMemory = false;
    vk::raii::Pipeline pipeline;
    std::vector<Session> sessions;
};

/*******************************************************************************
 * TensorComputePipeline
 *******************************************************************************/

class TensorComputePipeline : public PipelineBase {
  public:
    TensorComputePipeline(std::shared_ptr<Device> &_device, const DescriptorMap &_descriptorMap,
                          const std::vector<uint32_t> &_spirv);

    void dispatchSubmit(uint32_t _x, uint32_t _y, uint32_t _z);

  private:
    vk::raii::Pipeline createPipeline() const;

    vk::raii::Pipeline pipeline;
};

} // namespace mlsdk::el::utilities
