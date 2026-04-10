/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include <gtest/gtest.h>

#include "mlel/device.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <vector>

using namespace mlsdk::el::utilities;

namespace {

std::shared_ptr<Device> createDevice() {
    std::vector<const char *> layers = {"VK_LAYER_ML_Graph_Emulation", "VK_LAYER_ML_Tensor_Emulation"};
    std::vector<const char *> extensions = {
        VK_ARM_DATA_GRAPH_EXTENSION_NAME,
        VK_ARM_DATA_GRAPH_OPTICAL_FLOW_EXTENSION_NAME,
        VK_ARM_TENSORS_EXTENSION_NAME,
    };

    auto *const envValidation = std::getenv("VMEL_VALIDATION");
    if (envValidation && !std::string(envValidation).empty() && std::string(envValidation) != "0") {
        layers.emplace_back("VK_LAYER_KHRONOS_validation");
    }

    vk::PhysicalDeviceFeatures baseFeatures = {};
    baseFeatures.shaderInt64 = VK_TRUE;
    baseFeatures.shaderFloat64 = VK_TRUE;

    vk::PhysicalDeviceFeatures2 features2 = {};
    features2.setFeatures(baseFeatures);

    auto context = std::make_shared<vk::raii::Context>();
    auto instance = std::make_shared<Instance>(context, layers);
    auto physicalDevice = std::make_shared<PhysicalDevice>(instance, extensions);

    return std::make_shared<Device>(physicalDevice, extensions, &features2);
}

struct ImageResource {
    vk::raii::Image image{nullptr};
    vk::raii::DeviceMemory memory{nullptr};
    vk::raii::ImageView view{nullptr};
};

ImageResource createImageResource(const std::shared_ptr<Device> &device, vk::Format format, uint32_t width,
                                  uint32_t height, vk::ImageUsageFlags usage) {
    const vk::ImageCreateInfo imageCreateInfo{
        {},
        vk::ImageType::e2D,
        format,
        vk::Extent3D{width, height, 1},
        1,
        1,
        vk::SampleCountFlagBits::e1,
        vk::ImageTiling::eOptimal,
        usage,
        vk::SharingMode::eExclusive,
        0,
        nullptr,
        vk::ImageLayout::eUndefined,
    };

    ImageResource out;
    out.image = vk::raii::Image(&(*device), imageCreateInfo);

    const auto memReq = out.image.getMemoryRequirements();
    const auto memoryTypeIndices = device->getPhysicalDevice()->getMemoryTypeIndices(
        vk::MemoryPropertyFlagBits::eDeviceLocal, memReq.memoryTypeBits);
    if (memoryTypeIndices.empty()) {
        throw std::runtime_error("Failed to find device local memory type for image");
    }

    out.memory = vk::raii::DeviceMemory(&(*device), vk::MemoryAllocateInfo{memReq.size, memoryTypeIndices[0]});
    out.image.bindMemory(*out.memory, 0);

    const vk::ImageViewCreateInfo imageViewCreateInfo{
        {},
        *out.image,
        vk::ImageViewType::e2D,
        format,
        vk::ComponentMapping{},
        vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    };
    out.view = vk::raii::ImageView(&(*device), imageViewCreateInfo);

    return out;
}

class OpticalFlow {
  public:
    struct Config {
        VkDataGraphOpticalFlowGridSizeFlagsARM outputGridSize = VK_DATA_GRAPH_OPTICAL_FLOW_GRID_SIZE_4X4_BIT_ARM;
        VkDataGraphOpticalFlowPerformanceLevelARM performanceLevel =
            VK_DATA_GRAPH_OPTICAL_FLOW_PERFORMANCE_LEVEL_MEDIUM_ARM;
        bool enableHint = false;
        bool enableCost = false;
    };

    OpticalFlow(const std::shared_ptr<Device> &device, uint32_t width, uint32_t height, const Config &config)
        : device_(device), vkDevice_(&(*device)), config_(config) {
        std::vector<vk::DescriptorSetLayoutBinding> bindings;
        for (uint32_t binding = 0; binding < descriptorCount(); binding++) {
            bindings.push_back(vk::DescriptorSetLayoutBinding{binding, vk::DescriptorType::eStorageImage, 1,
                                                              vk::ShaderStageFlagBits::eAll});
        }

        const vk::DescriptorSetLayoutCreateInfo dsLayoutCI{{}, static_cast<uint32_t>(bindings.size()), bindings.data()};
        descriptorSetLayout_ = vk::raii::DescriptorSetLayout{&(*device), dsLayoutCI};

        const vk::DescriptorPoolSize poolSize{vk::DescriptorType::eStorageImage, descriptorCount()};
        const vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo{{}, 1, 1, &poolSize};
        descriptorPool_ = vk::raii::DescriptorPool{&(*device), descriptorPoolCreateInfo};

        const vk::DescriptorSetLayout layouts[] = {*descriptorSetLayout_};
        const vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo{*descriptorPool_, 1, layouts};
        descriptorSets_ = vk::raii::DescriptorSets{&(*device), descriptorSetAllocateInfo};

        const vk::PipelineLayoutCreateInfo pipelineLayoutCI{{}, 1, layouts};
        pipelineLayout_ = vk::raii::PipelineLayout{&(*device), pipelineLayoutCI};

        std::vector<VkDataGraphPipelineResourceInfoImageLayoutARM> resourceLayouts(descriptorCount());
        std::vector<VkDataGraphPipelineResourceInfoARM> resourceInfos(descriptorCount());
        for (uint32_t i = 0; i < descriptorCount(); i++) {
            resourceLayouts[i] = {
                VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_RESOURCE_INFO_IMAGE_LAYOUT_ARM,
                nullptr,
                VK_IMAGE_LAYOUT_GENERAL,
            };
            resourceInfos[i] = {
                VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_RESOURCE_INFO_ARM, &resourceLayouts[i], 0, i, 0,
            };
        }

        std::vector<VkDataGraphPipelineSingleNodeConnectionARM> connections;
        connections.push_back({VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SINGLE_NODE_CONNECTION_ARM, nullptr, 0, 0,
                               VK_DATA_GRAPH_PIPELINE_NODE_CONNECTION_TYPE_OPTICAL_FLOW_INPUT_ARM});
        connections.push_back({VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SINGLE_NODE_CONNECTION_ARM, nullptr, 0, 1,
                               VK_DATA_GRAPH_PIPELINE_NODE_CONNECTION_TYPE_OPTICAL_FLOW_REFERENCE_ARM});
        connections.push_back({VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SINGLE_NODE_CONNECTION_ARM, nullptr, 0, 2,
                               VK_DATA_GRAPH_PIPELINE_NODE_CONNECTION_TYPE_OPTICAL_FLOW_FLOW_VECTOR_ARM});
        if (config_.enableHint) {
            connections.push_back({VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SINGLE_NODE_CONNECTION_ARM, nullptr, 0,
                                   hintBinding(), VK_DATA_GRAPH_PIPELINE_NODE_CONNECTION_TYPE_OPTICAL_FLOW_HINT_ARM});
        }
        if (config_.enableCost) {
            connections.push_back({VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SINGLE_NODE_CONNECTION_ARM, nullptr, 0,
                                   costBinding(), VK_DATA_GRAPH_PIPELINE_NODE_CONNECTION_TYPE_OPTICAL_FLOW_COST_ARM});
        }

        VkDataGraphPipelineOpticalFlowCreateInfoARM opticalFlowCreateInfo{};
        opticalFlowCreateInfo.sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_OPTICAL_FLOW_CREATE_INFO_ARM;
        opticalFlowCreateInfo.flags = 0;
        if (config_.enableHint) {
            opticalFlowCreateInfo.flags |= VK_DATA_GRAPH_OPTICAL_FLOW_CREATE_ENABLE_HINT_BIT_ARM;
        }
        if (config_.enableCost) {
            opticalFlowCreateInfo.flags |= VK_DATA_GRAPH_OPTICAL_FLOW_CREATE_ENABLE_COST_BIT_ARM;
        }
        opticalFlowCreateInfo.outputGridSize = config_.outputGridSize;
        opticalFlowCreateInfo.hintGridSize = config_.enableHint ? config_.outputGridSize : 0;
        opticalFlowCreateInfo.performanceLevel = config_.performanceLevel;
        opticalFlowCreateInfo.imageFormat = VK_FORMAT_R8_UNORM;
        opticalFlowCreateInfo.flowVectorFormat = VK_FORMAT_R16G16_SFLOAT;
        opticalFlowCreateInfo.costFormat = VK_FORMAT_R16_UINT;
        opticalFlowCreateInfo.width = width;
        opticalFlowCreateInfo.height = height;

        VkDataGraphPipelineSingleNodeCreateInfoARM singleNodeCreateInfo{};
        singleNodeCreateInfo.sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SINGLE_NODE_CREATE_INFO_ARM;
        singleNodeCreateInfo.pNext = &opticalFlowCreateInfo;
        singleNodeCreateInfo.nodeType = VK_DATA_GRAPH_PIPELINE_NODE_TYPE_OPTICAL_FLOW_ARM;
        singleNodeCreateInfo.connectionCount = static_cast<uint32_t>(connections.size());
        singleNodeCreateInfo.pConnections = connections.data();

        VkDataGraphPipelineCreateInfoARM createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_CREATE_INFO_ARM;
        createInfo.pNext = &singleNodeCreateInfo;
        createInfo.layout = *pipelineLayout_;
        createInfo.resourceInfoCount = static_cast<uint32_t>(resourceInfos.size());
        createInfo.pResourceInfos = resourceInfos.data();

        VkResult result = vkDevice_.getDispatcher()->vkCreateDataGraphPipelinesARM(
            *vkDevice_, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &createInfo, nullptr, &pipeline_);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to create data graph optical flow pipeline");
        }

        VkDataGraphPipelineSessionCreateInfoARM sessionCreateInfo{};
        sessionCreateInfo.sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_CREATE_INFO_ARM;
        sessionCreateInfo.flags = VK_DATA_GRAPH_PIPELINE_SESSION_CREATE_OPTICAL_FLOW_CACHE_BIT_ARM;
        sessionCreateInfo.dataGraphPipeline = pipeline_;

        result = vkDevice_.getDispatcher()->vkCreateDataGraphPipelineSessionARM(*vkDevice_, &sessionCreateInfo, nullptr,
                                                                                &session_);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to create data graph optical flow session_");
        }

        allocateAndBindSessionMemory();
    }

    ~OpticalFlow() {
        if (session_ != VK_NULL_HANDLE) {
            vkDevice_.getDispatcher()->vkDestroyDataGraphPipelineSessionARM(*vkDevice_, session_, nullptr);
        }
        if (pipeline_ != VK_NULL_HANDLE) {
            vkDevice_.getDispatcher()->vkDestroyPipeline(*vkDevice_, pipeline_, nullptr);
        }
    }

    void bindImages(vk::ImageView inputView, vk::ImageView referenceView, vk::ImageView flowVectorView,
                    std::optional<vk::ImageView> hintView = std::nullopt,
                    std::optional<vk::ImageView> costView = std::nullopt) {
        std::vector<vk::DescriptorImageInfo> imageInfos;
        imageInfos.reserve(descriptorCount());
        imageInfos.push_back(vk::DescriptorImageInfo{VK_NULL_HANDLE, inputView, vk::ImageLayout::eGeneral});
        imageInfos.push_back(vk::DescriptorImageInfo{VK_NULL_HANDLE, referenceView, vk::ImageLayout::eGeneral});
        imageInfos.push_back(vk::DescriptorImageInfo{VK_NULL_HANDLE, flowVectorView, vk::ImageLayout::eGeneral});

        if (config_.enableHint) {
            if (!hintView.has_value()) {
                throw std::runtime_error("Hint image view is required when hint is enabled");
            }
            imageInfos.push_back(vk::DescriptorImageInfo{VK_NULL_HANDLE, *hintView, vk::ImageLayout::eGeneral});
        }
        if (config_.enableCost) {
            if (!costView.has_value()) {
                throw std::runtime_error("Cost image view is required when cost output is enabled");
            }
            imageInfos.push_back(vk::DescriptorImageInfo{VK_NULL_HANDLE, *costView, vk::ImageLayout::eGeneral});
        }

        std::vector<vk::WriteDescriptorSet> descriptorWrites;
        descriptorWrites.reserve(descriptorCount());
        for (uint32_t i = 0; i < descriptorCount(); i++) {
            descriptorWrites.push_back(vk::WriteDescriptorSet{*descriptorSets_[0], i, 0, 1,
                                                              vk::DescriptorType::eStorageImage, &imageInfos[i]});
        }

        vkDevice_.updateDescriptorSets(descriptorWrites, {});
    }

    void dispatchSubmit(const std::vector<vk::Image> &images) {
        const vk::CommandPoolCreateInfo commandPoolCreateInfo{{},
                                                              device_->getPhysicalDevice()->getComputeFamilyIndex()};
        vk::raii::CommandPool commandPool{&(*device_), commandPoolCreateInfo};
        const vk::CommandBufferAllocateInfo commandBufferAllocInfo{*commandPool, vk::CommandBufferLevel::ePrimary, 1};
        vk::raii::CommandBuffers commandBuffers{&(*device_), commandBufferAllocInfo};
        auto commandBuffer = std::move(commandBuffers.front());

        commandBuffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

        for (const auto &image : images) {
            VkImageMemoryBarrier2 imageBarrier{};
            imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            imageBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            imageBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            imageBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            imageBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
            imageBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            imageBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            imageBarrier.image = image;
            imageBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

            VkDependencyInfo depInfo{};
            depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            depInfo.imageMemoryBarrierCount = 1;
            depInfo.pImageMemoryBarriers = &imageBarrier;
            vkDevice_.getDispatcher()->vkCmdPipelineBarrier2(*commandBuffer, &depInfo);
        }

        vkDevice_.getDispatcher()->vkCmdBindPipeline(*commandBuffer, VK_PIPELINE_BIND_POINT_DATA_GRAPH_ARM, pipeline_);
        const VkDescriptorSet vkDescriptorSet = *descriptorSets_[0];
        vkDevice_.getDispatcher()->vkCmdBindDescriptorSets(*commandBuffer, VK_PIPELINE_BIND_POINT_DATA_GRAPH_ARM,
                                                           *pipelineLayout_, 0, 1, &vkDescriptorSet, 0, nullptr);
        vkDevice_.getDispatcher()->vkCmdDispatchDataGraphARM(*commandBuffer, session_, nullptr);
        commandBuffer.end();

        vk::raii::Queue queue(&(*device_), device_->getPhysicalDevice()->getComputeFamilyIndex(), 0);
        vk::raii::Fence fence(&(*device_), vk::FenceCreateInfo{});
        const vk::SubmitInfo submitInfo{0, nullptr, nullptr, 1, &(*commandBuffer), 0, nullptr};
        queue.submit({1, &submitInfo}, *fence);
        const auto waitResult = (&(*device_)).waitForFences({*fence}, vk::True, uint64_t(-1));
        if (waitResult != vk::Result::eSuccess) {
            throw std::runtime_error("Failed waiting for optical flow dispatch completion");
        }
    }

  private:
    void allocateAndBindSessionMemory() {
        const VkDataGraphPipelineSessionBindPointRequirementsInfoARM bindPointReqInfo = {
            VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_REQUIREMENTS_INFO_ARM,
            nullptr,
            session_,
        };

        uint32_t bindPointCount = 0;
        VkResult result = vkDevice_.getDispatcher()->vkGetDataGraphPipelineSessionBindPointRequirementsARM(
            *vkDevice_, &bindPointReqInfo, &bindPointCount, nullptr);
        if (result != VK_SUCCESS || bindPointCount == 0) {
            throw std::runtime_error("Failed querying optical flow bind point requirements");
        }

        std::vector<VkDataGraphPipelineSessionBindPointRequirementARM> bindPointRequirements(bindPointCount);
        for (auto &r : bindPointRequirements) {
            r = {};
            r.sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_REQUIREMENT_ARM;
        }
        uint32_t retrievedBindPointCount = bindPointCount;
        result = vkDevice_.getDispatcher()->vkGetDataGraphPipelineSessionBindPointRequirementsARM(
            *vkDevice_, &bindPointReqInfo, &retrievedBindPointCount, bindPointRequirements.data());
        if (result != VK_SUCCESS || retrievedBindPointCount != bindPointCount) {
            throw std::runtime_error("Failed retrieving optical flow bind point requirements");
        }

        std::vector<VkBindDataGraphPipelineSessionMemoryInfoARM> bindInfos;
        for (const auto &req : bindPointRequirements) {
            for (uint32_t objectIndex = 0; objectIndex < req.numObjects; objectIndex++) {
                VkMemoryRequirements2 memReq{};
                memReq.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;

                VkDataGraphPipelineSessionMemoryRequirementsInfoARM memReqInfo{};
                memReqInfo.sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_MEMORY_REQUIREMENTS_INFO_ARM;
                memReqInfo.session = session_;
                memReqInfo.bindPoint = req.bindPoint;
                memReqInfo.objectIndex = objectIndex;
                vkDevice_.getDispatcher()->vkGetDataGraphPipelineSessionMemoryRequirementsARM(*vkDevice_, &memReqInfo,
                                                                                              &memReq);

                const auto memoryTypeIndices = device_->getPhysicalDevice()->getMemoryTypeIndices(
                    vk::MemoryPropertyFlagBits::eDeviceLocal, memReq.memoryRequirements.memoryTypeBits);
                if (memoryTypeIndices.empty()) {
                    throw std::runtime_error("Failed finding optical flow session_ memory type index");
                }

                sessionMemories_.emplace_back(
                    &(*device_), vk::MemoryAllocateInfo{memReq.memoryRequirements.size, memoryTypeIndices[0]});

                VkBindDataGraphPipelineSessionMemoryInfoARM bindInfo{};
                bindInfo.sType = VK_STRUCTURE_TYPE_BIND_DATA_GRAPH_PIPELINE_SESSION_MEMORY_INFO_ARM;
                bindInfo.session = session_;
                bindInfo.bindPoint = req.bindPoint;
                bindInfo.objectIndex = objectIndex;
                bindInfo.memory = *sessionMemories_.back();
                bindInfo.memoryOffset = 0;
                bindInfos.push_back(bindInfo);
            }
        }

        result = vkDevice_.getDispatcher()->vkBindDataGraphPipelineSessionMemoryARM(
            *vkDevice_, static_cast<uint32_t>(bindInfos.size()), bindInfos.data());
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed binding optical flow session_ memory");
        }
    }

    uint32_t descriptorCount() const { return 3u + (config_.enableHint ? 1u : 0u) + (config_.enableCost ? 1u : 0u); }
    uint32_t hintBinding() const { return 3u; }
    uint32_t costBinding() const { return config_.enableHint ? 4u : 3u; }

    std::shared_ptr<Device> device_;
    const vk::raii::Device &vkDevice_;
    Config config_;
    vk::raii::DescriptorSetLayout descriptorSetLayout_{nullptr};
    vk::raii::DescriptorPool descriptorPool_{nullptr};
    vk::raii::DescriptorSets descriptorSets_{nullptr};
    vk::raii::PipelineLayout pipelineLayout_{nullptr};
    VkPipeline pipeline_{VK_NULL_HANDLE};
    VkDataGraphPipelineSessionARM session_{VK_NULL_HANDLE};
    std::vector<vk::raii::DeviceMemory> sessionMemories_;
};

void initializeImages(const std::shared_ptr<Device> &device, vk::Image input, vk::Image reference, vk::Image flow,
                      uint8_t inputValue, uint8_t referenceValue, uint8_t flowSeedValue) {
    const vk::CommandPoolCreateInfo commandPoolCreateInfo{{}, device->getPhysicalDevice()->getComputeFamilyIndex()};
    vk::raii::CommandPool commandPool{&(*device), commandPoolCreateInfo};
    const vk::CommandBufferAllocateInfo commandBufferAllocInfo{*commandPool, vk::CommandBufferLevel::ePrimary, 1};
    vk::raii::CommandBuffers commandBuffers{&(*device), commandBufferAllocInfo};
    auto commandBuffer = std::move(commandBuffers.front());

    commandBuffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    for (const auto image : {input, reference, flow}) {
        VkImageMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_NONE;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.imageMemoryBarrierCount = 1;
        depInfo.pImageMemoryBarriers = &barrier;
        (&(*device)).getDispatcher()->vkCmdPipelineBarrier2(*commandBuffer, &depInfo);
    }

    const VkClearColorValue inputColor = {{float(inputValue) / 255.0f, 0.0f, 0.0f, 0.0f}};
    const VkClearColorValue referenceColor = {{float(referenceValue) / 255.0f, 0.0f, 0.0f, 0.0f}};
    const VkClearColorValue flowColor = {{float(flowSeedValue) / 255.0f, 0.0f, 0.0f, 0.0f}};
    const VkImageSubresourceRange subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    (&(*device))
        .getDispatcher()
        ->vkCmdClearColorImage(*commandBuffer, input, VK_IMAGE_LAYOUT_GENERAL, &inputColor, 1, &subresourceRange);
    (&(*device))
        .getDispatcher()
        ->vkCmdClearColorImage(*commandBuffer, reference, VK_IMAGE_LAYOUT_GENERAL, &referenceColor, 1,
                               &subresourceRange);
    (&(*device))
        .getDispatcher()
        ->vkCmdClearColorImage(*commandBuffer, flow, VK_IMAGE_LAYOUT_GENERAL, &flowColor, 1, &subresourceRange);

    commandBuffer.end();

    vk::raii::Queue queue(&(*device), device->getPhysicalDevice()->getComputeFamilyIndex(), 0);
    vk::raii::Fence fence(&(*device), vk::FenceCreateInfo{});
    const vk::SubmitInfo submitInfo{0, nullptr, nullptr, 1, &(*commandBuffer), 0, nullptr};
    queue.submit({1, &submitInfo}, *fence);
    const auto waitResult = (&(*device)).waitForFences({*fence}, vk::True, uint64_t(-1));
    if (waitResult != vk::Result::eSuccess) {
        throw std::runtime_error("Failed waiting for image initialization");
    }
}

std::vector<uint8_t> readFlowImage(const std::shared_ptr<Device> &device, vk::Image image, uint32_t width,
                                   uint32_t height) {
    const auto byteSize = vk::DeviceSize(width) * vk::DeviceSize(height) * 4;
    const vk::BufferCreateInfo bufferCreateInfo{
        {}, byteSize, vk::BufferUsageFlagBits::eTransferDst, vk::SharingMode::eExclusive, 0, nullptr};
    vk::raii::Buffer stagingBuffer{&(*device), bufferCreateInfo};

    const auto memReq = stagingBuffer.getMemoryRequirements();
    const auto memoryTypeIndices = device->getPhysicalDevice()->getMemoryTypeIndices(
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, memReq.memoryTypeBits);
    if (memoryTypeIndices.empty()) {
        throw std::runtime_error("Failed to find host visible memory type for readback");
    }

    vk::raii::DeviceMemory stagingMemory{&(*device), vk::MemoryAllocateInfo{memReq.size, memoryTypeIndices[0]}};
    stagingBuffer.bindMemory(*stagingMemory, 0);

    const vk::CommandPoolCreateInfo commandPoolCreateInfo{{}, device->getPhysicalDevice()->getComputeFamilyIndex()};
    vk::raii::CommandPool commandPool{&(*device), commandPoolCreateInfo};
    const vk::CommandBufferAllocateInfo commandBufferAllocInfo{*commandPool, vk::CommandBufferLevel::ePrimary, 1};
    vk::raii::CommandBuffers commandBuffers{&(*device), commandBufferAllocInfo};
    auto commandBuffer = std::move(commandBuffers.front());

    commandBuffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    VkImageMemoryBarrier2 imageBarrier{};
    imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    imageBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    imageBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    imageBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    imageBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    imageBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.image = image;
    imageBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkDependencyInfo depInfo{};
    depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    depInfo.imageMemoryBarrierCount = 1;
    depInfo.pImageMemoryBarriers = &imageBarrier;
    (&(*device)).getDispatcher()->vkCmdPipelineBarrier2(*commandBuffer, &depInfo);

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {width, height, 1};

    (&(*device))
        .getDispatcher()
        ->vkCmdCopyImageToBuffer(*commandBuffer, image, VK_IMAGE_LAYOUT_GENERAL, *stagingBuffer, 1, &region);

    commandBuffer.end();

    vk::raii::Queue queue(&(*device), device->getPhysicalDevice()->getComputeFamilyIndex(), 0);
    vk::raii::Fence fence(&(*device), vk::FenceCreateInfo{});
    const vk::SubmitInfo submitInfo{0, nullptr, nullptr, 1, &(*commandBuffer), 0, nullptr};
    queue.submit({1, &submitInfo}, *fence);
    const auto waitResult = (&(*device)).waitForFences({*fence}, vk::True, uint64_t(-1));
    if (waitResult != vk::Result::eSuccess) {
        throw std::runtime_error("Failed waiting for flow readback");
    }

    std::vector<uint8_t> out(static_cast<size_t>(byteSize));
    void *mapped = stagingMemory.mapMemory(0, byteSize);
    std::memcpy(out.data(), mapped, out.size());
    stagingMemory.unmapMemory();
    return out;
}

uint32_t granularityFromGrid(VkDataGraphOpticalFlowGridSizeFlagsARM gridSize) {
    switch (gridSize) {
    case VK_DATA_GRAPH_OPTICAL_FLOW_GRID_SIZE_1X1_BIT_ARM:
        return 1;
    case VK_DATA_GRAPH_OPTICAL_FLOW_GRID_SIZE_2X2_BIT_ARM:
        return 2;
    case VK_DATA_GRAPH_OPTICAL_FLOW_GRID_SIZE_4X4_BIT_ARM:
        return 4;
    case VK_DATA_GRAPH_OPTICAL_FLOW_GRID_SIZE_8X8_BIT_ARM:
        return 8;
    default:
        throw std::runtime_error("Unsupported optical flow grid size");
    }
}

void runOpticalFlowAndExpectOutputChange(const std::shared_ptr<Device> &device, const OpticalFlow::Config &cfg,
                                         std::string_view contextName) {
    constexpr uint32_t width = 64;
    constexpr uint32_t height = 64;
    const uint32_t granularity = granularityFromGrid(cfg.outputGridSize);
    const uint32_t flowWidth = width / granularity;
    const uint32_t flowHeight = height / granularity;

    auto srcInput = createImageResource(device, vk::Format::eR8Unorm, width, height,
                                        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage |
                                            vk::ImageUsageFlagBits::eTransferDst);
    auto srcReference = createImageResource(device, vk::Format::eR8Unorm, width, height,
                                            vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage |
                                                vk::ImageUsageFlagBits::eTransferDst);
    auto dstFlow = createImageResource(device, vk::Format::eR16G16Sfloat, flowWidth, flowHeight,
                                       vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
                                           vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst);

    std::optional<ImageResource> srcHint;
    std::optional<ImageResource> dstCost;

    if (cfg.enableHint) {
        srcHint = createImageResource(device, vk::Format::eR16G16Sfloat, flowWidth, flowHeight,
                                      vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage |
                                          vk::ImageUsageFlagBits::eTransferDst);
    }
    if (cfg.enableCost) {
        dstCost = createImageResource(device, vk::Format::eR16Uint, flowWidth, flowHeight,
                                      vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
                                          vk::ImageUsageFlagBits::eTransferDst);
    }

    initializeImages(device, *srcInput.image, *srcReference.image, *dstFlow.image, 48, 176, 255);
    if (srcHint.has_value()) {
        initializeImages(device, *srcHint->image, *srcHint->image, *srcHint->image, 5, 5, 5);
    }
    if (dstCost.has_value()) {
        initializeImages(device, *dstCost->image, *dstCost->image, *dstCost->image, 255, 255, 255);
    }

    OpticalFlow opticalFlow{device, width, height, cfg};
    opticalFlow.bindImages(*srcInput.view, *srcReference.view, *dstFlow.view,
                           srcHint.has_value() ? std::optional<vk::ImageView>{*srcHint->view} : std::nullopt,
                           dstCost.has_value() ? std::optional<vk::ImageView>{*dstCost->view} : std::nullopt);

    std::vector<vk::Image> images = {*srcInput.image, *srcReference.image, *dstFlow.image};
    if (srcHint.has_value()) {
        images.push_back(*srcHint->image);
    }
    if (dstCost.has_value()) {
        images.push_back(*dstCost->image);
    }

    opticalFlow.dispatchSubmit(images);

    const auto flowOutput = readFlowImage(device, *dstFlow.image, flowWidth, flowHeight);
    ASSERT_FALSE(flowOutput.empty()) << contextName;
    ASSERT_TRUE(std::any_of(flowOutput.begin(), flowOutput.end(), [](uint8_t value) { return value != 0xFF; }))
        << contextName;
}

TEST(MLEmulationLayerOpticalFlowForVulkan, RGBToY_Smoke_Grid4x4) { // cppcheck-suppress syntaxError
    const auto device = createDevice();
    runOpticalFlowAndExpectOutputChange(device, OpticalFlow::Config{}, "RGBToY smoke");
}

TEST(MLEmulationLayerOpticalFlowForVulkan, RGBToY_Grid1x1) {
    const auto device = createDevice();
    OpticalFlow::Config cfg;
    cfg.outputGridSize = VK_DATA_GRAPH_OPTICAL_FLOW_GRID_SIZE_1X1_BIT_ARM;
    runOpticalFlowAndExpectOutputChange(device, cfg, "RGBToY class path");
}

TEST(MLEmulationLayerOpticalFlowForVulkan, Downsample_Grid8x8) {
    const auto device = createDevice();
    OpticalFlow::Config cfg;
    cfg.outputGridSize = VK_DATA_GRAPH_OPTICAL_FLOW_GRID_SIZE_8X8_BIT_ARM;
    runOpticalFlowAndExpectOutputChange(device, cfg, "Downsample class path");
}

TEST(MLEmulationLayerOpticalFlowForVulkan, DenseWarp_WithHint) {
    const auto device = createDevice();
    OpticalFlow::Config cfg;
    cfg.enableHint = true;
    runOpticalFlowAndExpectOutputChange(device, cfg, "DenseWarp class path");
}

TEST(MLEmulationLayerOpticalFlowForVulkan, MedianFilter_Fast) {
    const auto device = createDevice();
    OpticalFlow::Config cfg;
    cfg.performanceLevel = VK_DATA_GRAPH_OPTICAL_FLOW_PERFORMANCE_LEVEL_FAST_ARM;
    runOpticalFlowAndExpectOutputChange(device, cfg, "MedianFilter class path");
}

TEST(MLEmulationLayerOpticalFlowForVulkan, BilateralFilter_Medium) {
    const auto device = createDevice();
    OpticalFlow::Config cfg;
    cfg.performanceLevel = VK_DATA_GRAPH_OPTICAL_FLOW_PERFORMANCE_LEVEL_MEDIUM_ARM;
    runOpticalFlowAndExpectOutputChange(device, cfg, "BilateralFilter class path");
}

TEST(MLEmulationLayerOpticalFlowForVulkan, MVReplace_WithHintAndCost) {
    const auto device = createDevice();
    OpticalFlow::Config cfg;
    cfg.enableHint = true;
    cfg.enableCost = true;
    runOpticalFlowAndExpectOutputChange(device, cfg, "MVReplace class path");
}

TEST(MLEmulationLayerOpticalFlowForVulkan, BlockMatch_CostEnabled) {
    const auto device = createDevice();
    OpticalFlow::Config cfg;
    cfg.enableCost = true;
    runOpticalFlowAndExpectOutputChange(device, cfg, "BlockMatch class path");
}

} // namespace
