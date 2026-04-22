/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/*****************************************************************************
 * Includes
 *****************************************************************************/

#include "mlel/vulkan_layer.hpp"

#include "compute_graph_op.hpp"
#include "graph_log.hpp"
#include "memory_planner.hpp"
#include "optical_flow.hpp"
#include "pipeline_cache.hpp"
#include "source/opt/build_module.h"
#include "source/opt/ir_context.h"
#include "source/opt/module.h"
#include "spirv_pass.hpp"
#include "spirv_pass_tosaspv_v100.hpp"

#include <chrono>
#include <cstdlib>
#include <optional>
#include <regex>

using namespace mlsdk::el::compute;
using namespace mlsdk::el::compute::graph_op;
using namespace mlsdk::el::compute::optical_flow;
using namespace mlsdk::el::log;

/*****************************************************************************
 * Graph layer
 *****************************************************************************/

namespace mlsdk::el::layer {
namespace {
constexpr char graphPipelineCreatedLog[] = "Graph pipeline created";
}

/**************************************************************************
 * DataGraphDescriptorSet
 **************************************************************************/

class DataGraphDescriptorSet : public DescriptorSet {
  public:
    explicit DataGraphDescriptorSet(const std::shared_ptr<DescriptorSetLayout> &_descriptorSetLayout)
        : DescriptorSet(_descriptorSetLayout) {
        for (const auto &[binding, descriptorSetLayoutBinding] : descriptorSetLayout->bindings) {
            tensorViews[binding].resize(descriptorSetLayoutBinding.descriptorCount);
            imageViews[binding].resize(descriptorSetLayoutBinding.descriptorCount);
        }
    }

    void update(const VkWriteDescriptorSet &set) {
        [[maybe_unused]] const auto &bindingInfo = descriptorSetLayout->bindings.at(set.dstBinding);

        assert(bindingInfo.descriptorType == set.descriptorType);
        assert(bindingInfo.descriptorCount >= set.dstArrayElement + set.descriptorCount);

        switch (set.descriptorType) {
        case VK_DESCRIPTOR_TYPE_TENSOR_ARM: {
            const auto *tensorInfo =
                findType<VkWriteDescriptorSetTensorARM>(set.pNext, VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_TENSOR_ARM);
            assert(tensorInfo);
            assert(tensorInfo->tensorViewCount == set.descriptorCount);

            for (uint32_t i = 0; i < set.descriptorCount; i++) {
                tensorViews[set.dstBinding][set.dstArrayElement + i] = tensorInfo->pTensorViews[i];
            }
            break;
        }
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: {
            for (uint32_t i = 0; i < set.descriptorCount; i++) {
                // Only grab image view since we get image layout from connectivity map and we don't care about the
                // sampler
                imageViews[set.dstBinding][set.dstArrayElement + i] = set.pImageInfo[i].imageView;
            }
            break;
        }
        default:
            break;
        }
    }

    // Mapping from [binding, arrayIndex] to tensor/image view
    std::map<uint32_t, std::vector<VkTensorViewARM>> tensorViews;
    std::map<uint32_t, std::vector<VkImageView>> imageViews;

    // Mapping from [pipeline, set] to external descriptor sets bound by the application
    std::map<std::tuple<VkPipeline, uint32_t>, ComputeDescriptorSetMap> externalDescriptorSets;
};

/*****************************************************************************
 * DataGraphPipelineARM
 *****************************************************************************/

class DataGraphPipelineARM : public Loader {
  public:
    enum class Type {
        GRAPH,
        OPTICAL_FLOW,
    };

    explicit DataGraphPipelineARM(const std::shared_ptr<Device> &device,
                                  const std::shared_ptr<PipelineCache> &_pipelineCache, Type pipelineType)
        : Loader(*device) {
        if (pipelineType == Type::GRAPH) {
            graphPipeline = std::make_shared<GraphPipeline>(device->loader, device->physicalDevice->physicalDevice,
                                                            device->device, _pipelineCache);
        } else {
            opticalFlow = std::make_shared<OpticalFlow>(device->loader, device->physicalDevice->physicalDevice,
                                                        device->device, _pipelineCache);
        }
    }

    std::shared_ptr<GraphPipeline> graphPipeline;
    std::shared_ptr<OpticalFlow> opticalFlow;
    ComputeDescriptorSetMap constantsDescriptorSets;

    void makeConstantsDescriptorSets() {
        constantsDescriptorSets = graphPipeline->makeConstantsDescriptorSets();
        for ([[maybe_unused]] const auto &[_, descriptorSet] : constantsDescriptorSets) {
            descriptorSet->updateDescriptorSet();
        }
    }

    bool isGraph() const { return graphPipeline != nullptr; }
    bool isOpticalFlow() const { return opticalFlow != nullptr; }
};

/*****************************************************************************
 * DataGraphPipelineSessionARM
 *****************************************************************************/

class DataGraphPipelineSessionARM : public Loader {
  public:
    explicit DataGraphPipelineSessionARM(const std::shared_ptr<Device> &device,
                                         const std::shared_ptr<DataGraphPipelineARM> &_pipeline,
                                         VkDataGraphPipelineSessionCreateFlagsARM _createFlags)
        : Loader(*device), pipeline{_pipeline}, createFlags{_createFlags} {
        if (pipeline->isGraph()) {
            sessionRamDescriptorSets = pipeline->graphPipeline->makeSessionRamDescriptorSets();
            memoryPlanner = createMemoryPlanner();
        }
    }

    std::shared_ptr<DataGraphPipelineARM> pipeline;

    // Session ram descriptor sets
    ComputeDescriptorSetMap sessionRamDescriptorSets;

    bool transientMemoryBound = false;
    bool opticalFlowCacheMemoryBound = false;

    bool hasOpticalFlowCache() const {
        return (createFlags & VK_DATA_GRAPH_PIPELINE_SESSION_CREATE_OPTICAL_FLOW_CACHE_BIT_ARM) != 0;
    }

    bool needsTransientRequirements() const {
        return pipeline->isOpticalFlow() ? true : (memoryPlanner->getGraphPipelineSessionMemoryRequirements().size > 0);
    }
    bool needsOpticalFlowCacheRequirements() const { return pipeline->isOpticalFlow() && hasOpticalFlowCache(); }

    VkMemoryRequirements getGraphPipelineMemoryRequirements(VkDataGraphPipelineSessionBindPointARM bindPoint) const {
        if (pipeline->isGraph()) {
            return memoryPlanner->getGraphPipelineSessionMemoryRequirements();
        }
        if (pipeline->isOpticalFlow()) {
            if (bindPoint == VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TRANSIENT_ARM) {
                return pipeline->opticalFlow->getTransientMemoryRequirements();
            }
            if (bindPoint == VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_OPTICAL_FLOW_CACHE_ARM &&
                hasOpticalFlowCache()) {
                return pipeline->opticalFlow->getCacheMemoryRequirements();
            }
        }
        return {0, 1, 0};
    }

    void bindTransientMemory(VkDeviceMemory memory, VkDeviceSize offset) {
        if (pipeline->isGraph()) {
            memoryPlanner->bindGraphPipelineSessionMemory(memory, offset, sessionRamDescriptorSets);

            for ([[maybe_unused]] const auto &[_, descriptorSet] : sessionRamDescriptorSets) {
                descriptorSet->updateDescriptorSet();
            }
        } else if (pipeline->isOpticalFlow()) {
            pipeline->opticalFlow->bindSessionTransientMemory(memory, offset);
        }
        transientMemoryBound = true;
    }

    void bindOpticalFlowCacheMemory(VkDeviceMemory memory, VkDeviceSize offset) {
        pipeline->opticalFlow->bindSessionCacheMemory(memory, offset);
        opticalFlowCacheMemoryBound = true;
    }

  private:
    std::shared_ptr<MemoryPlanner> memoryPlanner;
    VkDataGraphPipelineSessionCreateFlagsARM createFlags;

    std::shared_ptr<MemoryPlanner> createMemoryPlanner() const {
        auto *const envMemoryPlanner = std::getenv("VMEL_MEMORY_PLANNER");

        if (envMemoryPlanner && std::string(envMemoryPlanner) == "Linear") {
            graphLog(Severity::Info) << "Using linear memory planner" << std::endl;
            return std::make_shared<LinearMemoryPlanner>(pipeline->graphPipeline);
        }

        graphLog(Severity::Info) << "Using best-fit memory planner" << std::endl;
        return std::make_shared<BestFitMemoryPlanner>(pipeline->graphPipeline);
    }
};

/**************************************************************************
 * Tensor
 **************************************************************************/
class TensorView {
  public:
    explicit TensorView(const VkTensorViewCreateInfoARM *_info) : info{*_info} {}

    const VkTensorViewCreateInfoARM info;
};

/*****************************************************************************
 * Device
 *****************************************************************************/

class GraphDevice : public Device {
  public:
    explicit GraphDevice(const std::shared_ptr<PhysicalDevice> &_physicalDevice, VkDevice _device,
                         PFN_vkGetInstanceProcAddr _gipr, PFN_vkGetDeviceProcAddr _gdpr,
                         const VkAllocationCallbacks *_callbacks)
        : Device(_physicalDevice, _device, _gipr, _gdpr, _callbacks) {}

    std::map<VkDescriptorSet, std::shared_ptr<DataGraphDescriptorSet>> descriptorSetMap;
    std::map<VkPipeline, std::shared_ptr<DataGraphPipelineARM>> dataGraphPipelineMap;
    std::map<VkTensorViewARM, std::shared_ptr<TensorView>> tensorViewMap;
    std::map<VkShaderModule, std::shared_ptr<ShaderModule>> shaderModuleMap;
};

/*****************************************************************************
 * Layer
 *****************************************************************************/
namespace {

void sprivMessageConsumer(spv_message_level_t level, const char *, const spv_position_t &position,
                          const char *message) {
    Severity severity = Severity::Info;
    switch (level) {
    case SPV_MSG_FATAL:
        severity = Severity::Error;
        break;
    case SPV_MSG_INTERNAL_ERROR:
        severity = Severity::Error;
        break;
    case SPV_MSG_ERROR:
        severity = Severity::Error;
        break;
    case SPV_MSG_WARNING:
        severity = Severity::Warning;
        break;
    case SPV_MSG_INFO:
        severity = Severity::Info;
        break;
    case SPV_MSG_DEBUG:
        severity = Severity::Debug;
        break;
    }

    graphLog(severity) << "SPIRV-Tools message: " << message << " at position " << position.index << std::endl;
}

inline std::optional<bool> isGraphSpirv(const std::vector<uint32_t> &spirv) {
    auto ir = spvtools::BuildModule(SPV_ENV_UNIVERSAL_1_6, sprivMessageConsumer, spirv.data(), spirv.size());
    if (ir == nullptr || ir->module() == nullptr) {
        return std::nullopt;
    }
    return !ir->module()->graphs().empty();
}

std::optional<std::string> tryGetExtInstVersion(const uint32_t *spirvCode, const size_t spirvSize,
                                                const std::regex &pattern) {
    auto ir = spvtools::BuildModule(SPV_ENV_UNIVERSAL_1_6, sprivMessageConsumer, spirvCode, spirvSize);
    for (const auto &inst : ir->module()->ext_inst_imports()) {
        const auto name = inst.GetInOperand(0).AsString();
        if (std::regex_search(name, pattern)) {
            return name;
        }
    }
    return std::nullopt;
}

} // namespace

constexpr std::array<const VkExtensionProperties, 3> extensions{
    VkExtensionProperties{VK_ARM_DATA_GRAPH_EXTENSION_NAME, VK_ARM_DATA_GRAPH_SPEC_VERSION},
    VkExtensionProperties{VK_ARM_DATA_GRAPH_INSTRUCTION_SET_TOSA_EXTENSION_NAME,
                          VK_ARM_DATA_GRAPH_INSTRUCTION_SET_TOSA_SPEC_VERSION},
    VkExtensionProperties{VK_ARM_DATA_GRAPH_OPTICAL_FLOW_EXTENSION_NAME, VK_ARM_DATA_GRAPH_OPTICAL_FLOW_SPEC_VERSION},
};

constexpr std::array<const VkExtensionProperties, 2> requiredExtensions = {
    VkExtensionProperties{VK_ARM_TENSORS_EXTENSION_NAME, VK_ARM_TENSORS_SPEC_VERSION},
    VkExtensionProperties{VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME, VK_KHR_SYNCHRONIZATION_2_SPEC_VERSION},
};

constexpr VkLayerProperties layerProperties = {
    "VK_LAYER_ML_Graph_Emulation",
    VK_MAKE_VERSION(1, 3, 0),
    VK_ARM_DATA_GRAPH_SPEC_VERSION,
    "ML Graph Emulation Layer",
};

using VulkanLayerImpl = VulkanLayer<layerProperties, extensions, requiredExtensions, GraphDevice>;

class GraphLayer : public VulkanLayerImpl {
  public:
    static PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char *name) {
        static const vTable vtable = {
            // Instance functions
            {"vkGetInstanceProcAddr", PFN_vkVoidFunction(vkGetInstanceProcAddr)},
            {"vk_layerGetPhysicalDeviceProcAddr", PFN_vkVoidFunction(vk_layerGetPhysicalDeviceProcAddr)},

            // PhysicalDevice functions
            {"vkGetPhysicalDeviceQueueFamilyDataGraphEngineOperationPropertiesARM",
             PFN_vkVoidFunction(vkGetPhysicalDeviceQueueFamilyDataGraphEngineOperationPropertiesARM)},
            {"vkGetPhysicalDeviceQueueFamilyDataGraphOpticalFlowImageFormatsARM",
             PFN_vkVoidFunction(vkGetPhysicalDeviceQueueFamilyDataGraphOpticalFlowImageFormatsARM)},
            {"vkGetPhysicalDeviceQueueFamilyDataGraphProcessingEnginePropertiesARM",
             PFN_vkVoidFunction(vkGetPhysicalDeviceQueueFamilyDataGraphProcessingEnginePropertiesARM)},
            {"vkGetPhysicalDeviceQueueFamilyDataGraphPropertiesARM",
             PFN_vkVoidFunction(vkGetPhysicalDeviceQueueFamilyDataGraphPropertiesARM)},
            {"vkGetPhysicalDeviceQueueFamilyProperties", PFN_vkVoidFunction(vkGetPhysicalDeviceQueueFamilyProperties)},
            {"vkGetPhysicalDeviceQueueFamilyProperties2",
             PFN_vkVoidFunction(vkGetPhysicalDeviceQueueFamilyProperties2)},
            {"vkGetPhysicalDeviceFeatures2", PFN_vkVoidFunction(vkGetPhysicalDeviceFeatures2)},
            {"vkGetPhysicalDeviceFeatures2KHR", PFN_vkVoidFunction(vkGetPhysicalDeviceFeatures2KHR)},
            {"vkGetPhysicalDeviceToolPropertiesEXT", PFN_vkVoidFunction(vkGetPhysicalDeviceToolPropertiesEXT)},
            {"vkCreateDevice", PFN_vkVoidFunction(vkCreateDevice)},

            // Device functions
            {"vkSetDebugUtilsObjectNameEXT", PFN_vkVoidFunction(vkSetDebugUtilsObjectNameEXT)}};

        if (auto it = vtable.find(name); it != vtable.end()) {
            return it->second;
        }

        return VulkanLayerImpl::vkGetInstanceProcAddr(instance, name);
    }

    static PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char *name) {
        static const vTable vtable = {
            // Device functions
            {"vkGetDeviceProcAddr", PFN_vkVoidFunction(vkGetDeviceProcAddr)},

            // Graph extension
            {"vkBindDataGraphPipelineSessionMemoryARM", PFN_vkVoidFunction(vkBindDataGraphPipelineSessionMemoryARM)},
            {"vkCreateDataGraphPipelinesARM", PFN_vkVoidFunction(vkCreateDataGraphPipelinesARM)},
            {"vkCreateDataGraphPipelineSessionARM", PFN_vkVoidFunction(vkCreateDataGraphPipelineSessionARM)},
            {"vkDestroyDataGraphPipelineSessionARM", PFN_vkVoidFunction(vkDestroyDataGraphPipelineSessionARM)},
            {"vkGetDataGraphPipelineAvailablePropertiesARM",
             PFN_vkVoidFunction(vkGetDataGraphPipelineAvailablePropertiesARM)},
            {"vkGetDataGraphPipelinePropertiesARM", PFN_vkVoidFunction(vkGetDataGraphPipelinePropertiesARM)},
            {"vkGetDataGraphPipelineSessionBindPointRequirementsARM",
             PFN_vkVoidFunction(vkGetDataGraphPipelineSessionBindPointRequirementsARM)},
            {"vkGetDataGraphPipelineSessionMemoryRequirementsARM",
             PFN_vkVoidFunction(vkGetDataGraphPipelineSessionMemoryRequirementsARM)},

            // Pipeline
            {"vkDestroyPipeline", PFN_vkVoidFunction(vkDestroyPipeline)},

            // DescriptorSet
            {"vkAllocateDescriptorSets", PFN_vkVoidFunction(vkAllocateDescriptorSets)},
            {"vkFreeDescriptorSets", PFN_vkVoidFunction(vkFreeDescriptorSets)},
            {"vkUpdateDescriptorSets", PFN_vkVoidFunction(vkUpdateDescriptorSets)},

            // Command buffer
            {"vkCmdBindPipeline", PFN_vkVoidFunction(vkCmdBindPipeline)},
            {"vkCmdBindDescriptorSets", PFN_vkVoidFunction(vkCmdBindDescriptorSets)},
            {"vkCmdDispatchDataGraphARM", PFN_vkVoidFunction(vkCmdDispatchDataGraphARM)},

            // Tensor extension
            {"vkCreateTensorViewARM", PFN_vkVoidFunction(vkCreateTensorViewARM)},
            {"vkDestroyTensorViewARM", PFN_vkVoidFunction(vkDestroyTensorViewARM)},

            // ShaderModule
            {"vkCreateShaderModule", PFN_vkVoidFunction(vkCreateShaderModule)},
            {"vkDestroyShaderModule", PFN_vkVoidFunction(vkDestroyShaderModule)},

            // Barrier
            {"vkCmdPipelineBarrier2", PFN_vkVoidFunction(vkCmdPipelineBarrier2)}};

        if (auto it = vtable.find(name); it != vtable.end()) {
            return it->second;
        }

        return VulkanLayerImpl::vkGetDeviceProcAddr(device, name);
    }

    static PFN_vkVoidFunction VKAPI_CALL vk_layerGetPhysicalDeviceProcAddr(VkInstance instance, const char *name) {
        static const vTable vtable = {
            {"vk_layerGetPhysicalDeviceProcAddr", PFN_vkVoidFunction(vk_layerGetPhysicalDeviceProcAddr)},
            // PhysicalDevice functions
            {"vkGetPhysicalDeviceQueueFamilyDataGraphEngineOperationPropertiesARM",
             PFN_vkVoidFunction(vkGetPhysicalDeviceQueueFamilyDataGraphEngineOperationPropertiesARM)},
            {"vkGetPhysicalDeviceQueueFamilyDataGraphOpticalFlowImageFormatsARM",
             PFN_vkVoidFunction(vkGetPhysicalDeviceQueueFamilyDataGraphOpticalFlowImageFormatsARM)},
            {"vkGetPhysicalDeviceQueueFamilyDataGraphProcessingEnginePropertiesARM",
             PFN_vkVoidFunction(vkGetPhysicalDeviceQueueFamilyDataGraphProcessingEnginePropertiesARM)},
            {"vkGetPhysicalDeviceQueueFamilyDataGraphPropertiesARM",
             PFN_vkVoidFunction(vkGetPhysicalDeviceQueueFamilyDataGraphPropertiesARM)},
            {"vkGetPhysicalDeviceQueueFamilyProperties", PFN_vkVoidFunction(vkGetPhysicalDeviceQueueFamilyProperties)},
            {"vkGetPhysicalDeviceQueueFamilyProperties2",
             PFN_vkVoidFunction(vkGetPhysicalDeviceQueueFamilyProperties2)},
            {"vkGetPhysicalDeviceFeatures2", PFN_vkVoidFunction(vkGetPhysicalDeviceFeatures2)},
            {"vkGetPhysicalDeviceFeatures2KHR", PFN_vkVoidFunction(vkGetPhysicalDeviceFeatures2KHR)},
            {"vkGetPhysicalDeviceToolPropertiesEXT", PFN_vkVoidFunction(vkGetPhysicalDeviceToolPropertiesEXT)},
            {"vkCreateDevice", PFN_vkVoidFunction(vkCreateDevice)}};

        if (auto it = vtable.find(name); it != vtable.end()) {
            return it->second;
        }

        if (instance == VK_NULL_HANDLE) {
            return nullptr;
        }

        return VulkanLayerImpl::vk_layerGetPhysicalDeviceProcAddr(instance, name);
    }

    /*******************************************************************************
     * PhysicalDevice
     *******************************************************************************/

    static void VKAPI_CALL vkGetPhysicalDeviceQueueFamilyProperties(VkPhysicalDevice physicalDevice,
                                                                    uint32_t *pQueueFamilyPropertyCount,
                                                                    VkQueueFamilyProperties *pQueueFamilyProperties) {
        auto handle = VulkanLayerImpl::getHandle(physicalDevice);
        handle->loader->vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, pQueueFamilyPropertyCount,
                                                                 pQueueFamilyProperties);

        if (pQueueFamilyProperties) {
            for (uint32_t i = 0; i < *pQueueFamilyPropertyCount; i++) {
                auto &property = pQueueFamilyProperties;
                if (property->queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    property->queueFlags |= VK_QUEUE_DATA_GRAPH_BIT_ARM;
                }
                pQueueFamilyProperties++;
            }
        }
    }

    static void VKAPI_CALL vkGetPhysicalDeviceQueueFamilyProperties2(VkPhysicalDevice physicalDevice,
                                                                     uint32_t *pQueueFamilyPropertyCount,
                                                                     VkQueueFamilyProperties2 *pQueueFamilyProperties) {
        auto handle = VulkanLayerImpl::getHandle(physicalDevice);
        handle->loader->vkGetPhysicalDeviceQueueFamilyProperties2(physicalDevice, pQueueFamilyPropertyCount,
                                                                  pQueueFamilyProperties);

        if (pQueueFamilyProperties) {
            for (uint32_t i = 0; i < *pQueueFamilyPropertyCount; i++) {
                auto &property = pQueueFamilyProperties->queueFamilyProperties;
                if (property.queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    property.queueFlags |= VK_QUEUE_DATA_GRAPH_BIT_ARM;
                }
                pQueueFamilyProperties++;
            }
        }
    }

    static VkResult VKAPI_CALL vkGetPhysicalDeviceQueueFamilyDataGraphOpticalFlowImageFormatsARM(
        VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex,
        const VkQueueFamilyDataGraphPropertiesARM *pQueueFamilyDataGraphProperties,
        const VkDataGraphOpticalFlowImageFormatInfoARM *pOpticalFlowImageFormatInfo, uint32_t *pFormatCount,
        VkDataGraphOpticalFlowImageFormatPropertiesARM *pImageFormatProperties) {
        if (!pFormatCount || !pQueueFamilyDataGraphProperties || !pOpticalFlowImageFormatInfo) {
            return VK_ERROR_UNKNOWN;
        }

        auto handle = VulkanLayerImpl::getHandle(physicalDevice);
        uint32_t familyCount = 0;
        handle->loader->vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &familyCount, nullptr);
        if (queueFamilyIndex >= familyCount) {
            return VK_ERROR_UNKNOWN;
        }
        if (pQueueFamilyDataGraphProperties->operation.operationType !=
            VK_PHYSICAL_DEVICE_DATA_GRAPH_OPERATION_TYPE_OPTICAL_FLOW_ARM) {
            *pFormatCount = 0;
            return VK_ERROR_UNKNOWN;
        }
        const std::set<VkFormat> *pSupportedFormats = nullptr;

        switch (pOpticalFlowImageFormatInfo->usage) {
        case VK_DATA_GRAPH_OPTICAL_FLOW_IMAGE_USAGE_INPUT_BIT_ARM:
            pSupportedFormats = &OpticalFlow::Spec::supportedImageFormats;
            break;
        case VK_DATA_GRAPH_OPTICAL_FLOW_IMAGE_USAGE_OUTPUT_BIT_ARM:
        case VK_DATA_GRAPH_OPTICAL_FLOW_IMAGE_USAGE_HINT_BIT_ARM:
            pSupportedFormats = &OpticalFlow::Spec::supportedFlowFormats;
            break;
        case VK_DATA_GRAPH_OPTICAL_FLOW_IMAGE_USAGE_COST_BIT_ARM:
            pSupportedFormats = &OpticalFlow::Spec::supportedCostFormats;
            break;
        default:
            *pFormatCount = 0;
            return VK_SUCCESS;
        }

        const auto availableFormatCount = static_cast<uint32_t>(pSupportedFormats->size());

        // First call: return count
        if (pImageFormatProperties == nullptr) {
            *pFormatCount = availableFormatCount;
            return VK_SUCCESS;
        }

        // Second call: return formats
        const uint32_t capacity = *pFormatCount;
        const uint32_t numToWrite = std::min(capacity, availableFormatCount);

        auto it = pSupportedFormats->cbegin();
        for (uint32_t i = 0; i < numToWrite; ++i, ++it) {
            pImageFormatProperties[i].sType = VK_STRUCTURE_TYPE_DATA_GRAPH_OPTICAL_FLOW_IMAGE_FORMAT_PROPERTIES_ARM;
            pImageFormatProperties[i].pNext = nullptr;
            pImageFormatProperties[i].format = *it;
        }

        *pFormatCount = numToWrite;
        return (numToWrite < availableFormatCount) ? VK_INCOMPLETE : VK_SUCCESS;
    }

    /**************************************************************************
     * Graph layer
     **************************************************************************/

    static VkResult VKAPI_CALL vkCreateDataGraphPipelinesARM(VkDevice device, VkDeferredOperationKHR,
                                                             VkPipelineCache pipelineCache, uint32_t createInfoCount,
                                                             const VkDataGraphPipelineCreateInfoARM *createInfos,
                                                             const VkAllocationCallbacks *callbacks,
                                                             VkPipeline *pipelines) {
        auto deviceHandle = VulkanLayerImpl::getHandle(device);
        auto pipelineCacheHandle = getHandle(pipelineCache);

        for (uint32_t i = 0; i < createInfoCount; i++) {
            const auto &createInfo = createInfos[i];

            const auto *creationFeedbackInfo = findType<VkPipelineCreationFeedbackCreateInfo>(
                createInfo.pNext, VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO);
            std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
            if (creationFeedbackInfo != nullptr) {
                startTime = std::chrono::high_resolution_clock::now();
            }

            const auto *dataGraphPipelineShaderModuleCreateInfo =
                findType<VkDataGraphPipelineShaderModuleCreateInfoARM>(
                    createInfo.pNext, VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SHADER_MODULE_CREATE_INFO_ARM);

            const auto *singleNodeCreateInfo = findType<VkDataGraphPipelineSingleNodeCreateInfoARM>(
                createInfo.pNext, VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SINGLE_NODE_CREATE_INFO_ARM);

            const VkDataGraphPipelineOpticalFlowCreateInfoARM *opticalFlowCreateInfo = nullptr;
            if (singleNodeCreateInfo != nullptr &&
                singleNodeCreateInfo->nodeType == VK_DATA_GRAPH_PIPELINE_NODE_TYPE_OPTICAL_FLOW_ARM) {
                opticalFlowCreateInfo = findType<VkDataGraphPipelineOpticalFlowCreateInfoARM>(
                    createInfo.pNext, VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_OPTICAL_FLOW_CREATE_INFO_ARM);
                if (opticalFlowCreateInfo == nullptr) {
                    graphLog(Severity::Error) << "Missing OF create info in single node create info" << std::endl;
                    return VK_ERROR_UNKNOWN;
                }
            }

            if (!dataGraphPipelineShaderModuleCreateInfo && !opticalFlowCreateInfo) {
                graphLog(Severity::Error) << "DataGraphPipelineCreateInfo Missing pNext struct" << std::endl;
                return VK_ERROR_UNKNOWN;
            }
            if (dataGraphPipelineShaderModuleCreateInfo && opticalFlowCreateInfo) {
                graphLog(Severity::Error) << "Multiple DataGraphPipelineCreateInfo pNext structs" << std::endl;
                return VK_ERROR_UNKNOWN;
            }

            const auto type = dataGraphPipelineShaderModuleCreateInfo != nullptr
                                  ? DataGraphPipelineARM::Type::GRAPH
                                  : DataGraphPipelineARM::Type::OPTICAL_FLOW;
            // Create pipeline handle
            auto pipeline = std::allocate_shared<DataGraphPipelineARM>(Allocator<GraphPipeline>{callbacks},
                                                                       deviceHandle, pipelineCacheHandle, type);
            pipelines[i] = reinterpret_cast<VkPipeline>(pipeline.get());
            graphLog(Severity::Info) << graphPipelineCreatedLog << std::endl;

            if (pipeline->isGraph()) {
                // Given by type check above, this should never be nullptr
                assert(dataGraphPipelineShaderModuleCreateInfo);
                auto graphPipeline = pipeline->graphPipeline;
                // Copy tensor resources to pipeline
                for (uint32_t j = 0; j < createInfo.resourceInfoCount; j++) {
                    const auto &resourceInfo = createInfo.pResourceInfos[j];
                    const auto *tensorDescription =
                        findType<VkTensorDescriptionARM>(resourceInfo.pNext, VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM);

                    if (tensorDescription == nullptr) {
                        graphLog(Severity::Error) << "Missing tensor description" << std::endl;
                        return VK_ERROR_UNKNOWN;
                    }

                    graphPipeline->makeDescriptorSetBinding(resourceInfo.descriptorSet, resourceInfo.binding,
                                                            resourceInfo.arrayElement, *tensorDescription);
                }

                // Constants
                for (uint32_t j = 0; j < dataGraphPipelineShaderModuleCreateInfo->constantCount; j++) {
                    const auto &constant = dataGraphPipelineShaderModuleCreateInfo->pConstants[j];

                    const auto *graphPipelineConstantTensor =
                        findType<VkTensorDescriptionARM>(constant.pNext, VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM);

                    if (graphPipelineConstantTensor == nullptr) {
                        graphLog(Severity::Error) << "Missing const tensor description" << std::endl;
                        return VK_ERROR_UNKNOWN;
                    }

                    graphPipeline->makeConstTensor(constant.id, *graphPipelineConstantTensor, constant.pConstantData);
                }
                std::shared_ptr<ShaderModule> shaderModule;
                if (dataGraphPipelineShaderModuleCreateInfo->module == VK_NULL_HANDLE) {
                    const auto *shaderModuleCreateInfo = findType<VkShaderModuleCreateInfo>(
                        createInfo.pNext, VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
                    if (shaderModuleCreateInfo == nullptr) {
                        graphLog(Severity::Error) << "Missing both shader handle and shader create info" << std::endl;
                        return VK_ERROR_UNKNOWN;
                    }

                    std::vector<uint32_t> spirvSource = {shaderModuleCreateInfo->pCode,
                                                         shaderModuleCreateInfo->pCode +
                                                             shaderModuleCreateInfo->codeSize / sizeof(uint32_t)};
                    auto isGraph = isGraphSpirv(spirvSource);
                    if (!isGraph.has_value()) {
                        graphLog(Severity::Error) << "Failed to compile spirv code." << std::endl;
                        return VK_ERROR_UNKNOWN;
                    }
                    if (isGraph.value()) {
                        shaderModule = std::make_shared<ShaderModule>(shaderModuleCreateInfo);
                    } else {
                        graphLog(Severity::Error) << "spirv code does not contain graph." << std::endl;
                        return VK_ERROR_UNKNOWN;
                    }
                } else {
                    shaderModule = getHandle(deviceHandle, dataGraphPipelineShaderModuleCreateInfo->module);
                }

                if (!shaderModule) {
                    graphLog(Severity::Error) << "Shader module not recognized by Graph layer" << std::endl;
                    return VK_ERROR_FEATURE_NOT_PRESENT;
                }

                // Create optimizer
                spvtools::Optimizer optimizer{SPV_ENV_UNIVERSAL_1_6};

                // Register passes
                const auto tosaVersion = tryGetExtInstVersion(shaderModule->code.data(), shaderModule->code.size(),
                                                              std::regex("^TOSA\\.\\d{6}\\.\\d"));
                const auto motionEngineVersion = tryGetExtInstVersion(
                    shaderModule->code.data(), shaderModule->code.size(), std::regex("^Arm\\.MotionEngine\\.\\d{3}"));

                const bool isTosaVersionUnsupported = tosaVersion.has_value() && tosaVersion != tosaSpv100;
                if (isTosaVersionUnsupported) {
                    graphLog(Severity::Error) << "Unsupported Tosa version provided." << std::endl;
                    return VK_ERROR_UNKNOWN;
                }

                const bool isMotionEngineVersionUnsupported =
                    motionEngineVersion.has_value() && motionEngineVersion != motionEngine100;
                if (isMotionEngineVersionUnsupported) {
                    graphLog(Severity::Error) << "Unsupported MotionEngine version provided." << std::endl;
                    return VK_ERROR_UNKNOWN;
                }

                optimizer.RegisterPass(spvtools::CreateGraphPass<spvtools::opt::GraphPassTosaSpv100>(*graphPipeline));

                // Run passes
                std::vector<uint32_t> optimizedModule;
                if (!optimizer.Run(shaderModule->code.data(), shaderModule->code.size(), &optimizedModule,
                                   spvtools::ValidatorOptions(), true)) {
                    graphLog(Severity::Error) << "Failed to run optimizer passes" << std::endl;
                    return VK_ERROR_UNKNOWN;
                }

                // Create constants descriptor sets
                pipeline->makeConstantsDescriptorSets();
            } else if (pipeline->isOpticalFlow()) {
                graphLog(Severity::Debug) << "Creating Optical Flow pipeline" << std::endl;
                // Initialise OpticalFlow
                const auto &opticalFlowPipeline = pipeline->opticalFlow;
                OpticalFlow::Config config;
                config.useMvInput =
                    (opticalFlowCreateInfo->flags & VK_DATA_GRAPH_OPTICAL_FLOW_CREATE_ENABLE_HINT_BIT_ARM) != 0;
                config.outputCost =
                    (opticalFlowCreateInfo->flags & VK_DATA_GRAPH_OPTICAL_FLOW_CREATE_ENABLE_COST_BIT_ARM) != 0;
                config.maxSearchRange = 3;

                constexpr uint32_t supportedFlags = VK_DATA_GRAPH_OPTICAL_FLOW_CREATE_ENABLE_HINT_BIT_ARM |
                                                    VK_DATA_GRAPH_OPTICAL_FLOW_CREATE_ENABLE_COST_BIT_ARM;
                if (opticalFlowCreateInfo->flags & ~supportedFlags) {
                    graphLog(Severity::Error) << "Invalid OF flags" << std::endl;
                    return VK_ERROR_UNKNOWN;
                }

                if (config.useMvInput && !OpticalFlow::Spec::hintSupported) {
                    graphLog(Severity::Error) << "OF hint is not supported by this implementation" << std::endl;
                    return VK_ERROR_UNKNOWN;
                }
                if (config.outputCost && !OpticalFlow::Spec::costSupported) {
                    graphLog(Severity::Error) << "OF cost output is not supported by this implementation" << std::endl;
                    return VK_ERROR_UNKNOWN;
                }

                switch (opticalFlowCreateInfo->outputGridSize) {
                case VK_DATA_GRAPH_OPTICAL_FLOW_GRID_SIZE_1X1_BIT_ARM:
                    config.levelOfLastEstimation = 0;
                    break;
                case VK_DATA_GRAPH_OPTICAL_FLOW_GRID_SIZE_2X2_BIT_ARM:
                    config.levelOfLastEstimation = 1;
                    break;
                case VK_DATA_GRAPH_OPTICAL_FLOW_GRID_SIZE_4X4_BIT_ARM:
                    config.levelOfLastEstimation = 2;
                    break;
                case VK_DATA_GRAPH_OPTICAL_FLOW_GRID_SIZE_8X8_BIT_ARM:
                    config.levelOfLastEstimation = 3;
                    break;
                default:
                    graphLog(Severity::Error) << "Invalid OF output grid size" << std::endl;
                    return VK_ERROR_UNKNOWN;
                }

                if (config.useMvInput && opticalFlowCreateInfo->hintGridSize == 0) {
                    graphLog(Severity::Error) << "OF hint grid size cannot be zero when hint is enabled" << std::endl;
                    return VK_ERROR_UNKNOWN;
                }
                if (opticalFlowCreateInfo->hintGridSize != 0 &&
                    opticalFlowCreateInfo->hintGridSize != opticalFlowCreateInfo->outputGridSize) {
                    graphLog(Severity::Error)
                        << "Output and hint grid sizes must match when hint grid size is set" << std::endl;
                    return VK_ERROR_UNKNOWN;
                }
                if (opticalFlowCreateInfo->hintGridSize != 0) {
                    switch (opticalFlowCreateInfo->hintGridSize) {
                    case VK_DATA_GRAPH_OPTICAL_FLOW_GRID_SIZE_1X1_BIT_ARM:
                    case VK_DATA_GRAPH_OPTICAL_FLOW_GRID_SIZE_2X2_BIT_ARM:
                    case VK_DATA_GRAPH_OPTICAL_FLOW_GRID_SIZE_4X4_BIT_ARM:
                    case VK_DATA_GRAPH_OPTICAL_FLOW_GRID_SIZE_8X8_BIT_ARM:
                        break;
                    default:
                        graphLog(Severity::Error) << "Invalid OF hint grid size" << std::endl;
                        return VK_ERROR_UNKNOWN;
                    }
                }

                switch (opticalFlowCreateInfo->performanceLevel) {
                case VK_DATA_GRAPH_OPTICAL_FLOW_PERFORMANCE_LEVEL_SLOW_ARM:
                    config.performanceLevel = OpticalFlow::PerformanceLevel::SLOW;
                    break;
                case VK_DATA_GRAPH_OPTICAL_FLOW_PERFORMANCE_LEVEL_MEDIUM_ARM:
                    config.performanceLevel = OpticalFlow::PerformanceLevel::MEDIUM;
                    break;
                case VK_DATA_GRAPH_OPTICAL_FLOW_PERFORMANCE_LEVEL_FAST_ARM:
                    config.performanceLevel = OpticalFlow::PerformanceLevel::FAST;
                    break;
                case VK_DATA_GRAPH_OPTICAL_FLOW_PERFORMANCE_LEVEL_UNKNOWN_ARM:
                    config.performanceLevel = OpticalFlow::PerformanceLevel::UNKNOWN;
                    break;
                default:
                    graphLog(Severity::Error) << "Invalid OF performance level" << std::endl;
                    return VK_ERROR_UNKNOWN;
                }

                config.imageFormat = opticalFlowCreateInfo->imageFormat;
                config.flowFormat = opticalFlowCreateInfo->flowVectorFormat;
                config.costFormat = opticalFlowCreateInfo->costFormat;

                config.width = opticalFlowCreateInfo->width;
                config.height = opticalFlowCreateInfo->height;

                const auto *opticalFlowNodeCreateInfo = singleNodeCreateInfo;
                if (opticalFlowNodeCreateInfo == nullptr ||
                    opticalFlowNodeCreateInfo->nodeType != VK_DATA_GRAPH_PIPELINE_NODE_TYPE_OPTICAL_FLOW_ARM) {
                    graphLog(Severity::Error) << "Missing OF single node create info" << std::endl;
                    return VK_ERROR_UNKNOWN;
                }
                if (opticalFlowNodeCreateInfo->connectionCount == 0 ||
                    opticalFlowNodeCreateInfo->pConnections == nullptr) {
                    graphLog(Severity::Error) << "Missing OF connectivity map" << std::endl;
                    return VK_ERROR_UNKNOWN;
                }

                const auto isFormatSupported = [](VkFormat format, const auto &supported) {
                    return supported.find(format) != supported.end();
                };
                if (!isFormatSupported(config.imageFormat, OpticalFlow::Spec::supportedImageFormats)) {
                    graphLog(Severity::Error) << "Invalid OF input/reference image format" << std::endl;
                    return VK_ERROR_UNKNOWN;
                }
                if (!isFormatSupported(config.flowFormat, OpticalFlow::Spec::supportedFlowFormats)) {
                    graphLog(Severity::Error) << "Invalid OF flow vector format" << std::endl;
                    return VK_ERROR_UNKNOWN;
                }
                if (config.outputCost &&
                    !isFormatSupported(config.costFormat, OpticalFlow::Spec::supportedCostFormats)) {
                    graphLog(Severity::Error) << "Invalid OF cost format" << std::endl;
                    return VK_ERROR_UNKNOWN;
                }
                if (config.width < OpticalFlow::Spec::minWidth || config.width > OpticalFlow::Spec::maxWidth) {
                    graphLog(Severity::Error) << "Invalid OF width" << std::endl;
                    return VK_ERROR_UNKNOWN;
                }
                if (config.height < OpticalFlow::Spec::minHeight || config.height > OpticalFlow::Spec::maxHeight) {
                    graphLog(Severity::Error) << "Invalid OF height" << std::endl;
                    return VK_ERROR_UNKNOWN;
                }

                auto getLayout = [&createInfo](uint32_t binding, uint32_t set) -> std::optional<VkImageLayout> {
                    for (uint32_t i = 0; i < createInfo.resourceInfoCount; ++i) {
                        if (createInfo.pResourceInfos[i].descriptorSet == set &&
                            createInfo.pResourceInfos[i].binding == binding) {
                            const auto *const resourceInfoImageLayout =
                                findType<VkDataGraphPipelineResourceInfoImageLayoutARM>(
                                    createInfo.pResourceInfos[i].pNext,
                                    VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_RESOURCE_INFO_IMAGE_LAYOUT_ARM);
                            if (resourceInfoImageLayout == nullptr) {
                                graphLog(Severity::Error)
                                    << "Missing pipeline resource image info layout struct" << std::endl;
                                return std::nullopt;
                            }
                            return resourceInfoImageLayout->layout;
                        }
                    }
                    graphLog(Severity::Error) << "Missing OF resource info for connection set/binding" << std::endl;
                    return std::nullopt;
                };

                bool hasInput = false;
                bool hasReference = false;
                bool hasHint = false;
                bool hasFlowVector = false;
                bool hasCost = false;
                std::set<VkDataGraphPipelineNodeConnectionTypeARM> seenConnectionTypes;
                std::set<std::pair<uint32_t, uint32_t>> seenSetBindingPairs;

                for (uint32_t connection = 0; connection < opticalFlowNodeCreateInfo->connectionCount; connection++) {
                    if (opticalFlowNodeCreateInfo->pConnections[connection].pNext != nullptr) {
                        graphLog(Severity::Error) << "OF connection pNext must be null" << std::endl;
                        return VK_ERROR_UNKNOWN;
                    }

                    const uint32_t set = opticalFlowNodeCreateInfo->pConnections[connection].set;
                    const uint32_t binding = opticalFlowNodeCreateInfo->pConnections[connection].binding;
                    const auto connectionType = opticalFlowNodeCreateInfo->pConnections[connection].connection;

                    if (!seenSetBindingPairs.insert({set, binding}).second) {
                        graphLog(Severity::Error) << "Duplicate OF set/binding in connectivity map" << std::endl;
                        return VK_ERROR_UNKNOWN;
                    }
                    if (!seenConnectionTypes.insert(connectionType).second) {
                        graphLog(Severity::Error) << "Duplicate OF connection type in connectivity map" << std::endl;
                        return VK_ERROR_UNKNOWN;
                    }

                    const auto layout = getLayout(binding, set);
                    if (!layout.has_value()) {
                        return VK_ERROR_UNKNOWN;
                    }

                    /* Create configuration */
                    switch (connectionType) {
                    case VK_DATA_GRAPH_PIPELINE_NODE_CONNECTION_TYPE_OPTICAL_FLOW_REFERENCE_ARM:
                        /* Input Image storage */
                        hasReference = true;
                        config.srcSearch.binding = binding;
                        config.srcSearch.set = set;
                        config.srcSearch.layout = *layout;
                        break;
                    case VK_DATA_GRAPH_PIPELINE_NODE_CONNECTION_TYPE_OPTICAL_FLOW_INPUT_ARM:
                        /* Input Template Image storage */
                        hasInput = true;
                        config.srcTemplate.binding = binding;
                        config.srcTemplate.set = set;
                        config.srcTemplate.layout = *layout;
                        break;
                    case VK_DATA_GRAPH_PIPELINE_NODE_CONNECTION_TYPE_OPTICAL_FLOW_HINT_ARM:
                        /* Input Flow Input hint storage */
                        hasHint = true;
                        config.srcFlow.binding = binding;
                        config.srcFlow.set = set;
                        config.srcFlow.layout = *layout;
                        break;
                    case VK_DATA_GRAPH_PIPELINE_NODE_CONNECTION_TYPE_OPTICAL_FLOW_FLOW_VECTOR_ARM:
                        /* Output Flow Image storage */
                        hasFlowVector = true;
                        config.dstFlow.binding = binding;
                        config.dstFlow.set = set;
                        config.dstFlow.layout = *layout;
                        break;
                    case VK_DATA_GRAPH_PIPELINE_NODE_CONNECTION_TYPE_OPTICAL_FLOW_COST_ARM:
                        /* Output Cost Image storage */
                        hasCost = true;
                        config.dstCost.binding = binding;
                        config.dstCost.set = set;
                        config.dstCost.layout = *layout;
                        break;
                    default:
                        graphLog(Severity::Error) << "Invalid OF connection" << std::endl;
                        return VK_ERROR_UNKNOWN;
                    }
                }

                if (!hasInput || !hasReference || !hasFlowVector) {
                    graphLog(Severity::Error)
                        << "Missing required OF connections (input/reference/flow output)" << std::endl;
                    return VK_ERROR_UNKNOWN;
                }
                if (config.useMvInput != hasHint) {
                    graphLog(Severity::Error) << "OF hint connection does not match hint create flag" << std::endl;
                    return VK_ERROR_UNKNOWN;
                }
                if (config.outputCost != hasCost) {
                    graphLog(Severity::Error) << "OF cost connection does not match cost create flag" << std::endl;
                    return VK_ERROR_UNKNOWN;
                }

                opticalFlowPipeline->init(config);
            }

            {
                scopedMutex l(globalMutex);
                deviceHandle->dataGraphPipelineMap[pipelines[i]] = pipeline;
            }

            if (creationFeedbackInfo != nullptr) {
                auto endTime = std::chrono::high_resolution_clock::now();
                creationFeedbackInfo->pPipelineCreationFeedback->flags |= VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT;
                creationFeedbackInfo->pPipelineCreationFeedback->duration = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count());
            }
        }

        return VK_SUCCESS;
    }

    static void VKAPI_CALL vkGetPhysicalDeviceFeatures2KHR(VkPhysicalDevice physicalDevice,
                                                           VkPhysicalDeviceFeatures2 *pFeatures) {
        vkGetPhysicalDeviceFeatures2(physicalDevice, pFeatures);
    }

    static void VKAPI_CALL vkGetPhysicalDeviceFeatures2(VkPhysicalDevice physicalDevice,
                                                        VkPhysicalDeviceFeatures2 *pFeatures) {
        auto handle = VulkanLayerImpl::getHandle(physicalDevice);
        handle->loader->vkGetPhysicalDeviceFeatures2(physicalDevice, pFeatures);

        auto *pDataGraphFeatures = findTypeMutable<VkPhysicalDeviceDataGraphFeaturesARM>(
            pFeatures->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_FEATURES_ARM);
        if (pDataGraphFeatures) {
            pDataGraphFeatures->dataGraph = VK_TRUE;
            pDataGraphFeatures->dataGraphUpdateAfterBind =
                supportsDataGraphUpdateAfterBind(physicalDevice, handle) ? VK_TRUE : VK_FALSE;
            pDataGraphFeatures->dataGraphShaderModule = VK_TRUE;
        }
        auto *pPipelineCreationCacheControlFeatures =
            findTypeMutable<VkPhysicalDevicePipelineCreationCacheControlFeatures>(
                pFeatures->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_CREATION_CACHE_CONTROL_FEATURES);
        // Pipeline caching is currently not supported
        if (pPipelineCreationCacheControlFeatures) {
            pPipelineCreationCacheControlFeatures->pipelineCreationCacheControl = VK_FALSE;
        }

        auto *pDataGraphOpticalFlowFeatures = findTypeMutable<VkPhysicalDeviceDataGraphOpticalFlowFeaturesARM>(
            pFeatures->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_OPTICAL_FLOW_FEATURES_ARM);
        if (pDataGraphOpticalFlowFeatures) {
            pDataGraphOpticalFlowFeatures->dataGraphOpticalFlow = VK_TRUE;
        }
    }

    static bool supportsDataGraphUpdateAfterBind(VkPhysicalDevice physicalDevice,
                                                 const std::shared_ptr<PhysicalDevice> &handle) {
        VkPhysicalDeviceVulkan12Features vulkan12Features{};
        vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        VkPhysicalDeviceFeatures2 features2{};
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = &vulkan12Features;

        handle->loader->vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);

        return vulkan12Features.descriptorBindingUniformBufferUpdateAfterBind == VK_TRUE;
    }

    static VkResult VKAPI_CALL vkCreateDevice(VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo *createInfo,
                                              const VkAllocationCallbacks *allocator, VkDevice *device) {
        auto originCreateInfoChain = dumpVkStructureList(createInfo);

        VkDeviceCreateInfo newCreateInfo{*createInfo};
        findAndRemoveType<VkPhysicalDeviceDataGraphFeaturesARM>(
            &newCreateInfo, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_FEATURES_ARM);
        findAndRemoveType<VkPhysicalDeviceDataGraphOpticalFlowFeaturesARM>(
            &newCreateInfo, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_OPTICAL_FLOW_FEATURES_ARM);

        auto result = VulkanLayerImpl::vkCreateDevice(physicalDevice, &newCreateInfo, allocator, device);

        loadVkStructureList(const_cast<VkDeviceCreateInfo *>(createInfo), originCreateInfoChain);
        return result;
    }

    static void VKAPI_CALL vkDestroyPipeline(VkDevice device, VkPipeline pipeline,
                                             const VkAllocationCallbacks *allocator) {
        auto handle = VulkanLayerImpl::getHandle(device);
        auto deviceHandle = VulkanLayerImpl::getHandle(device);
        auto pipelineImpl = getHandle(deviceHandle, pipeline);

        if (!pipelineImpl) {
            handle->loader->vkDestroyPipeline(device, pipeline, allocator);
            return;
        }

        {
            scopedMutex l(globalMutex);
            deviceHandle->dataGraphPipelineMap.erase(pipeline);
        }
    }

    static VkResult VKAPI_CALL vkCreateDataGraphPipelineSessionARM(
        VkDevice device, const VkDataGraphPipelineSessionCreateInfoARM *createInfo,
        const VkAllocationCallbacks *callbacks, VkDataGraphPipelineSessionARM *session) {
        if (!createInfo || !session) {
            return VK_ERROR_UNKNOWN;
        }

        auto deviceHandle = VulkanLayerImpl::getHandle(device);
        auto pipelineImpl = getHandle(deviceHandle, createInfo->dataGraphPipeline);
        if (!pipelineImpl) {
            return VK_ERROR_UNKNOWN;
        }

        constexpr VkDataGraphPipelineSessionCreateFlagsARM supportedSessionCreateFlags =
            VK_DATA_GRAPH_PIPELINE_SESSION_CREATE_OPTICAL_FLOW_CACHE_BIT_ARM;
        if ((createInfo->flags & ~supportedSessionCreateFlags) != 0) {
            graphLog(Severity::Error) << "Unsupported data graph session create flags" << std::endl;
            return VK_ERROR_UNKNOWN;
        }
        if (pipelineImpl->isGraph() &&
            (createInfo->flags & VK_DATA_GRAPH_PIPELINE_SESSION_CREATE_OPTICAL_FLOW_CACHE_BIT_ARM) != 0) {
            graphLog(Severity::Error) << "OF cache create flag is invalid for non-OF pipelines" << std::endl;
            return VK_ERROR_UNKNOWN;
        }
        if (pipelineImpl->isOpticalFlow() &&
            (createInfo->flags & VK_DATA_GRAPH_PIPELINE_SESSION_CREATE_OPTICAL_FLOW_CACHE_BIT_ARM) == 0) {
            graphLog(Severity::Error) << "OF sessions currently require OF cache create flag" << std::endl;
            return VK_ERROR_UNKNOWN;
        }

        *session = reinterpret_cast<VkDataGraphPipelineSessionARM>(
            allocateObject<DataGraphPipelineSessionARM>(callbacks, deviceHandle, pipelineImpl, createInfo->flags));

        return VK_SUCCESS;
    }

    static VkResult VKAPI_CALL vkGetDataGraphPipelineSessionBindPointRequirementsARM(
        VkDevice, const VkDataGraphPipelineSessionBindPointRequirementsInfoARM *info,
        uint32_t *bindPointRequirementCount, VkDataGraphPipelineSessionBindPointRequirementARM *bindPointRequirements) {
        const auto *const session = reinterpret_cast<DataGraphPipelineSessionARM *>(info->session);

        const auto needsTransient = session->needsTransientRequirements();
        const auto needsOpticalFlowCache = session->needsOpticalFlowCacheRequirements();

        uint32_t requiredCount = 0;
        if (needsTransient) {
            ++requiredCount;
        }
        if (needsOpticalFlowCache) {
            ++requiredCount;
        }

        if (bindPointRequirements == nullptr) {
            *bindPointRequirementCount = requiredCount;
            return VK_SUCCESS;
        }

        const auto capacity = *bindPointRequirementCount;
        uint32_t written = 0;

        auto writeRequirement = [&](VkDataGraphPipelineSessionBindPointARM bindPoint) {
            if (written < capacity) {
                bindPointRequirements[written] = VkDataGraphPipelineSessionBindPointRequirementARM{
                    VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_REQUIREMENT_ARM,
                    nullptr,
                    bindPoint,
                    VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TYPE_MEMORY_ARM,
                    1,
                };
            }
            ++written;
        };

        if (needsTransient) {
            writeRequirement(VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TRANSIENT_ARM);
        }

        if (needsOpticalFlowCache) {
            writeRequirement(VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_OPTICAL_FLOW_CACHE_ARM);
        }

        *bindPointRequirementCount = written;

        return (capacity < requiredCount) ? VK_INCOMPLETE : VK_SUCCESS;
    }

    static void VKAPI_CALL vkGetDataGraphPipelineSessionMemoryRequirementsARM(
        VkDevice, const VkDataGraphPipelineSessionMemoryRequirementsInfoARM *info,
        VkMemoryRequirements2 *requirements) {
        const auto *const session = reinterpret_cast<DataGraphPipelineSessionARM *>(info->session);

        // Calculate how much memory pipelines hidden layers require
        requirements->memoryRequirements = session->getGraphPipelineMemoryRequirements(info->bindPoint);
    }

    static VkResult VKAPI_CALL vkBindDataGraphPipelineSessionMemoryARM(
        VkDevice, uint32_t bindInfoCount, const VkBindDataGraphPipelineSessionMemoryInfoARM *bindInfos) {
        auto *const session = reinterpret_cast<DataGraphPipelineSessionARM *>(bindInfos->session);

        // Bind session memory to hidden layers
        for (uint32_t i = 0; i < bindInfoCount; i++) {
            switch (bindInfos[i].bindPoint) {
            case VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TRANSIENT_ARM: {
                session->bindTransientMemory(bindInfos[i].memory, bindInfos[i].memoryOffset);
                break;
            }
            case VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_OPTICAL_FLOW_CACHE_ARM: {
                if (!session->pipeline->isOpticalFlow()) {
                    graphLog(Severity::Error) << "Invalid bind point for pipeline type" << std::endl;
                    return VK_ERROR_UNKNOWN;
                }
                if (!session->hasOpticalFlowCache()) {
                    graphLog(Severity::Error) << "OF cache bind point requires session create cache flag" << std::endl;
                    return VK_ERROR_UNKNOWN;
                }
                session->bindOpticalFlowCacheMemory(bindInfos[i].memory, bindInfos[i].memoryOffset);
                break;
            }
            default:
                return VK_ERROR_UNKNOWN;
            }
        }

        return VK_SUCCESS;
    }

    static void VKAPI_CALL vkDestroyDataGraphPipelineSessionARM(VkDevice, VkDataGraphPipelineSessionARM session,
                                                                const VkAllocationCallbacks *callbacks) {
        destroyObject(callbacks, reinterpret_cast<DataGraphPipelineSessionARM *>(session));
    }

    static VkResult VKAPI_CALL vkGetDataGraphPipelineAvailablePropertiesARM(
        VkDevice, const VkDataGraphPipelineInfoARM *, uint32_t *pPropertiesCount,
        VkDataGraphPipelinePropertyARM *pProperties) {
        if (!pProperties) {
            // This property is always available
            *pPropertiesCount = 1;
            return VK_SUCCESS;
        }

        if (*pPropertiesCount == 0) {
            return VK_INCOMPLETE;
        }

        *pProperties = VK_DATA_GRAPH_PIPELINE_PROPERTY_CREATION_LOG_ARM;
        *pPropertiesCount = 1;

        return VK_SUCCESS;
    }

    static VkResult VKAPI_CALL
    vkGetDataGraphPipelinePropertiesARM(VkDevice, const VkDataGraphPipelineInfoARM *, uint32_t propertiesCount,
                                        VkDataGraphPipelinePropertyQueryResultARM *pProperties) {
        if (propertiesCount == 0) {
            return VK_SUCCESS;
        }
        if (!pProperties->pData) {
            pProperties->dataSize = sizeof(graphPipelineCreatedLog);
            return VK_SUCCESS;
        }
        pProperties->property = VK_DATA_GRAPH_PIPELINE_PROPERTY_CREATION_LOG_ARM;
        pProperties->isText = VK_TRUE;
        const auto dataSize = std::min(pProperties->dataSize, sizeof(graphPipelineCreatedLog));
        pProperties->dataSize = dataSize;
        std::memcpy(pProperties->pData, &graphPipelineCreatedLog[0], dataSize);
        return (dataSize < sizeof(graphPipelineCreatedLog)) ? VK_INCOMPLETE : VK_SUCCESS;
    }

    static void VKAPI_CALL vkGetPhysicalDeviceQueueFamilyDataGraphProcessingEnginePropertiesARM(
        VkPhysicalDevice /*physicalDevice*/,
        const VkPhysicalDeviceQueueFamilyDataGraphProcessingEngineInfoARM
            * /*pQueueFamilyDataGraphProcessingEngineInfo*/,
        VkQueueFamilyDataGraphProcessingEnginePropertiesARM * /*pQueueFamilyDataGraphProcessingEngineProperties*/) {
        // No properties available
    }

    static VkResult VKAPI_CALL vkGetPhysicalDeviceQueueFamilyDataGraphEngineOperationPropertiesARM(
        VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex,
        const VkQueueFamilyDataGraphPropertiesARM *pQueueFamilyDataGraphProperties, VkBaseOutStructure *pProperties) {
        auto handle = VulkanLayerImpl::getHandle(physicalDevice);
        uint32_t familyCount = 0;
        handle->loader->vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &familyCount, nullptr);
        if (queueFamilyIndex >= familyCount || pQueueFamilyDataGraphProperties == nullptr || pProperties == nullptr) {
            return VK_ERROR_UNKNOWN;
        }

        if ((pQueueFamilyDataGraphProperties->operation.operationType ==
             VK_PHYSICAL_DEVICE_DATA_GRAPH_OPERATION_TYPE_OPTICAL_FLOW_ARM) &&
            (pProperties->sType == VK_STRUCTURE_TYPE_QUEUE_FAMILY_DATA_GRAPH_OPTICAL_FLOW_PROPERTIES_ARM)) {
            auto *opticalFlowProps = reinterpret_cast<VkQueueFamilyDataGraphOpticalFlowPropertiesARM *>(pProperties);

            VkDataGraphOpticalFlowGridSizeFlagsARM gridSizes = 0;
            for (size_t lvl : OpticalFlow::Spec::supportedLevelOfLastEstimation) {
                gridSizes = static_cast<VkDataGraphOpticalFlowGridSizeFlagsARM>(gridSizes | (1u << lvl));
            }
            opticalFlowProps->supportedOutputGridSizes = gridSizes;
            opticalFlowProps->supportedHintGridSizes = gridSizes;
            opticalFlowProps->hintSupported = OpticalFlow::Spec::hintSupported;
            opticalFlowProps->costSupported = OpticalFlow::Spec::costSupported;
            opticalFlowProps->minWidth = OpticalFlow::Spec::minWidth;
            opticalFlowProps->minHeight = OpticalFlow::Spec::minHeight;
            opticalFlowProps->maxWidth = OpticalFlow::Spec::maxWidth;
            opticalFlowProps->maxHeight = OpticalFlow::Spec::maxHeight;

            return VK_SUCCESS;
        }

        if (pQueueFamilyDataGraphProperties->engine.type !=
                VK_PHYSICAL_DEVICE_DATA_GRAPH_PROCESSING_ENGINE_TYPE_DEFAULT_ARM ||
            pQueueFamilyDataGraphProperties->operation.operationType !=
                VK_PHYSICAL_DEVICE_DATA_GRAPH_OPERATION_TYPE_SPIRV_EXTENDED_INSTRUCTION_SET_ARM) {
            return VK_ERROR_UNKNOWN;
        }

        auto *tosaProperties = findTypeMutable<VkQueueFamilyDataGraphTOSAPropertiesARM>(
            pProperties, VK_STRUCTURE_TYPE_QUEUE_FAMILY_DATA_GRAPH_TOSA_PROPERTIES_ARM);
        if (tosaProperties == nullptr) {
            return VK_SUCCESS;
        }

        const static VkDataGraphTOSANameQualityARM profile = {"Emulation Layer",
                                                              VK_DATA_GRAPH_TOSA_QUALITY_CONFORMANT_ARM};

        tosaProperties->profileCount = 1;
        tosaProperties->pProfiles = &profile;
        tosaProperties->extensionCount = 0;
        tosaProperties->pExtensions = nullptr;
        tosaProperties->level = VK_DATA_GRAPH_TOSA_LEVEL_8K_ARM;

        return VK_SUCCESS;
    }

    static VkResult VKAPI_CALL vkGetPhysicalDeviceQueueFamilyDataGraphPropertiesARM(
        VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex, uint32_t *pQueueFamilyDataGraphPropertyCount,
        VkQueueFamilyDataGraphPropertiesARM *pQueueFamilyDataGraphProperties) {
        auto handle = VulkanLayerImpl::getHandle(physicalDevice);
        uint32_t familyCount = 0;
        handle->loader->vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &familyCount, nullptr);
        if (queueFamilyIndex >= familyCount) {
            return VK_ERROR_UNKNOWN;
        }

        constexpr uint32_t propertyCount = 2;

        if (pQueueFamilyDataGraphProperties == nullptr) {
            *pQueueFamilyDataGraphPropertyCount = propertyCount;
            return VK_SUCCESS;
        }

        const auto capacity = *pQueueFamilyDataGraphPropertyCount;
        const uint32_t toWrite = std::min(capacity, propertyCount);

        const VkPhysicalDeviceDataGraphProcessingEngineARM processingEngine = {
            VK_PHYSICAL_DEVICE_DATA_GRAPH_PROCESSING_ENGINE_TYPE_DEFAULT_ARM,
            VK_FALSE,
        };

        const VkPhysicalDeviceDataGraphOperationSupportARM operationSupportTOSA = {
            VK_PHYSICAL_DEVICE_DATA_GRAPH_OPERATION_TYPE_SPIRV_EXTENDED_INSTRUCTION_SET_ARM,
            "TOSA.001000.1",
            {},
        };

        const VkPhysicalDeviceDataGraphOperationSupportARM operationSupportOF = {
            VK_PHYSICAL_DEVICE_DATA_GRAPH_OPERATION_TYPE_OPTICAL_FLOW_ARM,
            "OpticalFlow",
            {},
        };

        const VkQueueFamilyDataGraphPropertiesARM availableProperties[propertyCount] = {
            {
                VK_STRUCTURE_TYPE_QUEUE_FAMILY_DATA_GRAPH_PROPERTIES_ARM,
                nullptr,
                processingEngine,
                operationSupportTOSA,
            },
            {
                VK_STRUCTURE_TYPE_QUEUE_FAMILY_DATA_GRAPH_PROPERTIES_ARM,
                nullptr,
                processingEngine,
                operationSupportOF,
            },
        };

        for (uint32_t i = 0; i < toWrite; ++i) {
            pQueueFamilyDataGraphProperties[i] = availableProperties[i];
        }
        *pQueueFamilyDataGraphPropertyCount = toWrite;

        return (toWrite < propertyCount) ? VK_INCOMPLETE : VK_SUCCESS;
    }

    /**************************************************************************
     * DescriptorSet
     **************************************************************************/

    static VkResult VKAPI_CALL vkAllocateDescriptorSets(VkDevice device,
                                                        const VkDescriptorSetAllocateInfo *allocateInfo,
                                                        VkDescriptorSet *descriptorSets) {
        auto deviceHandle = VulkanLayerImpl::getHandle(device);
        auto res = deviceHandle->loader->vkAllocateDescriptorSets(device, allocateInfo, descriptorSets);

        if (res == VK_SUCCESS) {
            scopedMutex l(globalMutex);

            for (uint32_t i = 0; i < allocateInfo->descriptorSetCount; i++) {
                const auto descriptorSetLayout = VulkanLayerImpl::getHandle(allocateInfo->pSetLayouts[i]);
                deviceHandle->descriptorSetMap[descriptorSets[i]] =
                    std::make_shared<DataGraphDescriptorSet>(descriptorSetLayout);
            }
        }

        return res;
    }

    static VkResult VKAPI_CALL vkFreeDescriptorSets(VkDevice device, VkDescriptorPool descriptorPool,
                                                    uint32_t descriptorSetCount,
                                                    const VkDescriptorSet *descriptorSets) {
        auto deviceHandle = VulkanLayerImpl::getHandle(device);
        auto res =
            deviceHandle->loader->vkFreeDescriptorSets(device, descriptorPool, descriptorSetCount, descriptorSets);

        while (descriptorSetCount-- > 0) {
            scopedMutex l(globalMutex);
            deviceHandle->descriptorSetMap.erase(descriptorSets[descriptorSetCount]);
        }

        return res;
    }

    static void updateDescriptorSet(const std::shared_ptr<GraphDevice> &deviceHandle,
                                    const std::vector<VkTensorViewARM> &tensorViews, const uint32_t arrayIndex,
                                    const std::shared_ptr<GraphPipeline> &graphPipeline, const uint32_t set,
                                    const uint32_t binding, const ComputeDescriptorSetMap &computeDescriptorSetMap) {
        const auto tensorView = getHandle(deviceHandle, tensorViews[arrayIndex]);

        // Get tensor descriptor associated with this set, binding and array index
        const auto tensorDescriptor = graphPipeline->getTensor(set, binding, arrayIndex);

        // Find and update all descriptor sets with matching tensor descriptor
        for ([[maybe_unused]] const auto &[_, descSet] : computeDescriptorSetMap) {
            if (descSet->getTensor()->getTensorDescriptor() == tensorDescriptor) {
                // Store tensor and tensor view and update descriptor set
                descSet->updateDescriptorSet(tensorView->info.tensor, tensorViews[arrayIndex]);
            }
        }
    }

    static void VKAPI_CALL vkUpdateDescriptorSets(VkDevice device, uint32_t descriptorWriteCount,
                                                  const VkWriteDescriptorSet *descriptorWrites,
                                                  uint32_t descriptorCopyCount,
                                                  const VkCopyDescriptorSet *descriptorCopies) {
        auto deviceHandle = VulkanLayerImpl::getHandle(device);
        deviceHandle->loader->vkUpdateDescriptorSets(device, descriptorWriteCount, descriptorWrites,
                                                     descriptorCopyCount, descriptorCopies);

        for (uint32_t i = 0; i < descriptorWriteCount; i++) {
            const auto &vkWriteDescriptorSet = descriptorWrites[i];
            const auto descriptorSet = getHandle(deviceHandle, vkWriteDescriptorSet.dstSet);
            descriptorSet->update(vkWriteDescriptorSet);

            for (const auto &[pipelineSet, computeDescriptorSetMap] : descriptorSet->externalDescriptorSets) {
                const auto &[vkPipeline, set] = pipelineSet;

                std::shared_ptr<DataGraphPipelineARM> dataGraphPipelineArm;
                {
                    scopedMutex l(globalMutex);
                    const auto it = deviceHandle->dataGraphPipelineMap.find(vkPipeline);
                    if (it == deviceHandle->dataGraphPipelineMap.end()) {
                        continue; // To avoid adding nullptr
                    }
                    dataGraphPipelineArm = it->second;
                }

                const auto binding = vkWriteDescriptorSet.dstBinding;
                const auto arrayIndex = vkWriteDescriptorSet.dstArrayElement;

                updateDescriptorSet(deviceHandle, descriptorSet->tensorViews[binding], arrayIndex,
                                    dataGraphPipelineArm->graphPipeline, set, binding, computeDescriptorSetMap);
            }
        }
    }

    /**************************************************************************
     * Command buffer
     **************************************************************************/

    static void VKAPI_CALL vkCmdBindPipeline(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint,
                                             VkPipeline pipeline) {
        auto handle = VulkanLayerImpl::getHandle(commandBuffer);

        if (pipelineBindPoint != VK_PIPELINE_BIND_POINT_DATA_GRAPH_ARM) {
            handle->loader->vkCmdBindPipeline(commandBuffer, pipelineBindPoint, pipeline);
            return;
        }
    }

    static void VKAPI_CALL vkCmdBindDescriptorSets(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint,
                                                   VkPipelineLayout layout, uint32_t firstSet,
                                                   uint32_t descriptorSetCount, const VkDescriptorSet *descriptorSets,
                                                   uint32_t dynamicOffsetCount, const uint32_t *dynamicOffsets) {
        auto handle = VulkanLayerImpl::getHandle(commandBuffer);

        if (pipelineBindPoint != VK_PIPELINE_BIND_POINT_DATA_GRAPH_ARM) {
            handle->loader->vkCmdBindDescriptorSets(commandBuffer, pipelineBindPoint, layout, firstSet,
                                                    descriptorSetCount, descriptorSets, dynamicOffsetCount,
                                                    dynamicOffsets);
            return;
        }

        // Clear descriptor set map if pipeline layout changes
        if (handle->pipelineLayout != layout) {
            handle->descriptorSets.clear();
        }

        // Remember current pipeline layout
        handle->pipelineLayout = layout;

        // Graph pipeline
        for (uint32_t i = 0; i < descriptorSetCount; i++) {
            auto set = firstSet + i;

            // Store reference to descriptor set
            handle->descriptorSets[set] = descriptorSets[i];
        }
    }

    static void VKAPI_CALL vkCmdDispatchDataGraphARM(VkCommandBuffer commandBuffer,
                                                     VkDataGraphPipelineSessionARM _session,
                                                     const VkDataGraphPipelineDispatchInfoARM *pInfo) {
        auto handle = VulkanLayerImpl::getHandle(commandBuffer);
        const auto *session = reinterpret_cast<DataGraphPipelineSessionARM *>(_session);
        const auto &pipeline = session->pipeline;
        auto *vkPipeline = reinterpret_cast<VkPipeline>(pipeline.get());
        auto deviceHandle = VulkanLayerImpl::getHandle(handle->device->device);

        if (pipeline->isGraph()) {
            const auto &graphPipeline = pipeline->graphPipeline;
            /*
             * Merge descriptor sets, they can have three different origins:
             * - Constants owned by the pipeline
             * - Session ram owned by the session
             * - External owned by the application
             */
            ComputeDescriptorSetMap allDescriptorSetMap;

            for (const auto &[set, vkDescriptorSet] : handle->descriptorSets) {
                auto descriptorSet = getHandle(deviceHandle, vkDescriptorSet);

                auto &externalDescriptorSets = descriptorSet->externalDescriptorSets;
                if (externalDescriptorSets.find({vkPipeline, set}) == externalDescriptorSets.end()) {
                    /*
                     * A resource bound to the graph with {set, binding} can be used by multiple compute jobs,
                     * with different {set, binding}.
                     *
                     * The list of compute jobs is first known when the pipeline is dispatched. A DescriptorSet is bound
                     * to a PipelineLayout, which is why the compute DescriptorSets must be created here.
                     *
                     *               <- Defined by the PipelineLayout ->
                     * +----------+    +----------+     +------------+
                     * | GRAPH    |    | COMPUTE1 |     | COMPUTE<n> |
                     * +----------+    +----------+     +------------+
                     * | set      | => | set1     | ... | set<n>     |
                     * | binding  |    | binding1 |     | binding<n> |
                     * | resource |    | resource |     | resource   |
                     * +----------+    +----------+     +------------+
                     */

                    // Create compute descriptor sets
                    auto descriptorSetMapTemp = graphPipeline->makeExternalDescriptorSets(set);
                    auto &computeDescriptorSetMap = externalDescriptorSets[{vkPipeline, set}];
                    computeDescriptorSetMap.merge(descriptorSetMapTemp);

                    for (const auto &[binding, tensorViews] : descriptorSet->tensorViews) {
                        for (uint32_t arrayIndex = 0; arrayIndex < tensorViews.size(); arrayIndex++) {
                            if (tensorViews[arrayIndex] == nullptr) {
                                continue;
                            }
                            updateDescriptorSet(deviceHandle, tensorViews, arrayIndex, graphPipeline, set, binding,
                                                computeDescriptorSetMap);
                        }
                    }
                } // end if no entry

                auto &externals = descriptorSet->externalDescriptorSets.at({vkPipeline, set});
                allDescriptorSetMap.insert(externals.begin(), externals.end());
            }

            allDescriptorSetMap.insert(pipeline->constantsDescriptorSets.begin(),
                                       pipeline->constantsDescriptorSets.end());
            allDescriptorSetMap.insert(session->sessionRamDescriptorSets.begin(),
                                       session->sessionRamDescriptorSets.end());

            graphPipeline->cmdBindAndDispatch(commandBuffer, allDescriptorSetMap);
        } else if (pipeline->isOpticalFlow()) {
            const auto &opticalFlowPipeline = pipeline->opticalFlow;

            VkDataGraphOpticalFlowExecuteFlagsARM opticalFlowFlags = 0;
            uint32_t meanFlowL1NormHint = 0;
            if (pInfo != nullptr) {
                const auto *opticalFlowDispatchInfo = findType<VkDataGraphPipelineOpticalFlowDispatchInfoARM>(
                    pInfo, VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_OPTICAL_FLOW_DISPATCH_INFO_ARM);
                if (opticalFlowDispatchInfo) {
                    opticalFlowFlags = opticalFlowDispatchInfo->flags;
                    meanFlowL1NormHint = opticalFlowDispatchInfo->meanFlowL1NormHint;
                }
            }

            constexpr VkDataGraphOpticalFlowExecuteFlagsARM cacheDependentExecuteFlags =
                VK_DATA_GRAPH_OPTICAL_FLOW_EXECUTE_INPUT_UNCHANGED_BIT_ARM |
                VK_DATA_GRAPH_OPTICAL_FLOW_EXECUTE_REFERENCE_UNCHANGED_BIT_ARM |
                VK_DATA_GRAPH_OPTICAL_FLOW_EXECUTE_INPUT_IS_PREVIOUS_REFERENCE_BIT_ARM |
                VK_DATA_GRAPH_OPTICAL_FLOW_EXECUTE_REFERENCE_IS_PREVIOUS_INPUT_BIT_ARM;
            constexpr VkDataGraphOpticalFlowExecuteFlagsARM supportedExecuteFlags =
                VK_DATA_GRAPH_OPTICAL_FLOW_EXECUTE_DISABLE_TEMPORAL_HINTS_BIT_ARM | cacheDependentExecuteFlags;

            if (opticalFlowFlags & ~supportedExecuteFlags) {
                graphLog(Severity::Error) << "Unsupported OF execute flags" << std::endl;
                return;
            }
            if (!session->transientMemoryBound) {
                graphLog(Severity::Error) << "OF session transient memory is not bound" << std::endl;
                return;
            }
            if ((opticalFlowFlags & cacheDependentExecuteFlags) && !session->hasOpticalFlowCache()) {
                graphLog(Severity::Error) << "OF execute flags require session create OF cache flag" << std::endl;
                return;
            }
            if ((opticalFlowFlags & cacheDependentExecuteFlags) && !session->opticalFlowCacheMemoryBound) {
                graphLog(Severity::Error) << "OF execute flags require OF cache memory to be bound" << std::endl;
                return;
            }

            OpticalFlowDescriptorMap descriptorMap;
            for (const auto &[set, vkDescriptorSet] : handle->descriptorSets) {
                auto descriptorSet = getHandle(deviceHandle, vkDescriptorSet);
                for (const auto &[binding, imageViews] : descriptorSet->imageViews) {
                    for (uint32_t arrayIndex = 0; arrayIndex < imageViews.size(); arrayIndex++) {
                        if (imageViews[arrayIndex] == VK_NULL_HANDLE) {
                            continue;
                        }
                        descriptorMap[{set, binding, arrayIndex}] = {vkDescriptorSet, imageViews[arrayIndex]};
                    }
                }
                opticalFlowPipeline->updateDescriptorSets(descriptorMap);
            }
            opticalFlowPipeline->cmdBindAndDispatch(commandBuffer, opticalFlowFlags, meanFlowL1NormHint);
        }
    }

    /*******************************************************************************
     * TensorView
     *******************************************************************************/

    static VkResult VKAPI_CALL vkCreateTensorViewARM(VkDevice device, const VkTensorViewCreateInfoARM *createInfo,
                                                     const VkAllocationCallbacks *allocator,
                                                     VkTensorViewARM *tensorView) {
        auto deviceHandle = VulkanLayerImpl::getHandle(device);
        auto res = deviceHandle->loader->vkCreateTensorViewARM(device, createInfo, allocator, tensorView);

        if (res == VK_SUCCESS) {
            scopedMutex l(globalMutex);
            deviceHandle->tensorViewMap[*tensorView] = std::make_shared<TensorView>(createInfo);
        }

        return res;
    }

    static void VKAPI_CALL vkDestroyTensorViewARM(VkDevice device, VkTensorViewARM tensorView,
                                                  const VkAllocationCallbacks *allocator) {
        auto deviceHandle = VulkanLayerImpl::getHandle(device);
        deviceHandle->loader->vkDestroyTensorViewARM(device, tensorView, allocator);

        {
            scopedMutex l(globalMutex);
            deviceHandle->tensorViewMap.erase(tensorView);
        }
    }

    /*******************************************************************************
     * ShaderModule
     *******************************************************************************/

    static VkResult VKAPI_CALL vkCreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo *pCreateInfo,
                                                    const VkAllocationCallbacks *pAllocator,
                                                    VkShaderModule *pShaderModule) {
        auto deviceHandle = VulkanLayerImpl::getHandle(device);
        std::vector<uint32_t> spirvSource = {pCreateInfo->pCode,
                                             pCreateInfo->pCode + pCreateInfo->codeSize / sizeof(uint32_t)};
        auto isGraph = isGraphSpirv(spirvSource);
        if (!isGraph.has_value()) {
            graphLog(Severity::Error) << "Failed to compile spirv code." << std::endl;
            return VK_ERROR_UNKNOWN;
        }
        if (isGraph.value()) {
            auto shaderModule = std::make_shared<ShaderModule>(pCreateInfo);
            *pShaderModule = reinterpret_cast<VkShaderModule>(shaderModule.get());
            {
                scopedMutex l(globalMutex);
                deviceHandle->shaderModuleMap[*pShaderModule] = std::move(shaderModule);
            }
            return VK_SUCCESS;
        }
        return deviceHandle->loader->vkCreateShaderModule(device, pCreateInfo, pAllocator, pShaderModule);
    }

    static void VKAPI_CALL vkDestroyShaderModule(VkDevice device, VkShaderModule shaderModule,
                                                 const VkAllocationCallbacks *allocator) {
        auto deviceHandle = VulkanLayerImpl::getHandle(device);
        scopedMutex l(globalMutex);
        if (deviceHandle->shaderModuleMap.count(shaderModule)) {
            deviceHandle->shaderModuleMap.erase(shaderModule);
        } else {
            deviceHandle->loader->vkDestroyShaderModule(device, shaderModule, allocator);
        }
    }

    /*******************************************************************************
     * Barrier
     *******************************************************************************/

    static void VKAPI_CALL vkCmdPipelineBarrier2(VkCommandBuffer commandBuffer,
                                                 const VkDependencyInfo *pDependencyInfo) {
        auto handle = VulkanLayerImpl::getHandle(commandBuffer);

        const auto *tensorDependencyInfo =
            findType<VkTensorDependencyInfoARM>(pDependencyInfo->pNext, VK_STRUCTURE_TYPE_TENSOR_DEPENDENCY_INFO_ARM);
        if (tensorDependencyInfo == nullptr && pDependencyInfo->pMemoryBarriers == nullptr &&
            pDependencyInfo->pImageMemoryBarriers == nullptr && pDependencyInfo->pBufferMemoryBarriers == nullptr) {
            return handle->loader->vkCmdPipelineBarrier2(commandBuffer, pDependencyInfo);
        }

        auto replaceAccessFlag = [](const auto flag) {
            auto newFlag = flag;
            if (newFlag & VK_ACCESS_2_DATA_GRAPH_READ_BIT_ARM) {
                newFlag = (newFlag ^ VK_ACCESS_2_DATA_GRAPH_READ_BIT_ARM) | VK_ACCESS_2_SHADER_READ_BIT;
            }
            if (newFlag & VK_ACCESS_2_DATA_GRAPH_WRITE_BIT_ARM) {
                newFlag = (newFlag ^ VK_ACCESS_2_DATA_GRAPH_WRITE_BIT_ARM) | VK_ACCESS_2_SHADER_WRITE_BIT;
            }
            return newFlag;
        };

        auto replaceStageFlag = [](const auto flag) {
            auto newFlag = flag;
            if (newFlag & VK_PIPELINE_STAGE_2_DATA_GRAPH_BIT_ARM) {
                newFlag = (newFlag ^ VK_PIPELINE_STAGE_2_DATA_GRAPH_BIT_ARM) | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            }
            return newFlag;
        };

        auto replaceBarriersGraphFlag = [&](auto &barriers) {
            for (auto &barrier : barriers) {
                barrier.srcAccessMask = replaceAccessFlag(barrier.srcAccessMask);
                barrier.srcStageMask = replaceStageFlag(barrier.srcStageMask);

                barrier.dstAccessMask = replaceAccessFlag(barrier.dstAccessMask);
                barrier.dstStageMask = replaceStageFlag(barrier.dstStageMask);
            }
        };

        // replace pipeline memory barrier graph flag
        std::vector<VkMemoryBarrier2> memoryBarriers{
            pDependencyInfo->pMemoryBarriers, pDependencyInfo->pMemoryBarriers + pDependencyInfo->memoryBarrierCount};
        replaceBarriersGraphFlag(memoryBarriers);

        // replace image memory barrier graph flag
        std::vector<VkImageMemoryBarrier2> imageBarriers{pDependencyInfo->pImageMemoryBarriers,
                                                         pDependencyInfo->pImageMemoryBarriers +
                                                             pDependencyInfo->imageMemoryBarrierCount};
        replaceBarriersGraphFlag(imageBarriers);

        std::vector<VkBufferMemoryBarrier2> bufferBarriers{pDependencyInfo->pBufferMemoryBarriers,
                                                           pDependencyInfo->pBufferMemoryBarriers +
                                                               pDependencyInfo->bufferMemoryBarrierCount};
        replaceBarriersGraphFlag(bufferBarriers);

        // replace tensor memory barrier graph flag
        if (tensorDependencyInfo != nullptr) {
            std::vector<VkTensorMemoryBarrierARM> tensorMemoryBarriers{
                tensorDependencyInfo->pTensorMemoryBarriers,
                tensorDependencyInfo->pTensorMemoryBarriers + tensorDependencyInfo->tensorMemoryBarrierCount};

            replaceBarriersGraphFlag(tensorMemoryBarriers);

            const VkTensorDependencyInfoARM newTensorDependencyInfo{
                VK_STRUCTURE_TYPE_TENSOR_DEPENDENCY_INFO_ARM,       // sType
                nullptr,                                            // pNext
                static_cast<uint32_t>(tensorMemoryBarriers.size()), // tensorMemoryBarrierCount
                tensorMemoryBarriers.data()                         // pTensorMemoryBarriers
            };

            const VkDependencyInfo newDependencyInfo{
                VK_STRUCTURE_TYPE_DEPENDENCY_INFO,            // sType
                &newTensorDependencyInfo,                     // pNext
                pDependencyInfo->dependencyFlags,             // dependencyFlags
                static_cast<uint32_t>(memoryBarriers.size()), // memoryBarrierCount
                memoryBarriers.data(),                        // pMemoryBarriers
                static_cast<uint32_t>(bufferBarriers.size()), // bufferMemoryBarrierCount
                bufferBarriers.data(),                        // pBufferMemoryBarriers
                static_cast<uint32_t>(imageBarriers.size()),  // imageMemoryBarrierCount
                imageBarriers.data()                          // pImageMemoryBarriers
            };
            handle->loader->vkCmdPipelineBarrier2(commandBuffer, &newDependencyInfo);
        } else {
            const VkDependencyInfo newDependencyInfo{
                VK_STRUCTURE_TYPE_DEPENDENCY_INFO,            // sType
                pDependencyInfo->pNext,                       // pNext
                pDependencyInfo->dependencyFlags,             // dependencyFlags
                static_cast<uint32_t>(memoryBarriers.size()), // memoryBarrierCount
                memoryBarriers.data(),                        // pMemoryBarriers
                static_cast<uint32_t>(bufferBarriers.size()), // bufferMemoryBarrierCount
                bufferBarriers.data(),                        // pBufferMemoryBarriers
                static_cast<uint32_t>(imageBarriers.size()),  // imageMemoryBarrierCount
                imageBarriers.data()                          // pImageMemoryBarriers
            };
            handle->loader->vkCmdPipelineBarrier2(commandBuffer, &newDependencyInfo);
        }
    }

    /*******************************************************************************
     * Debugging
     *******************************************************************************/

    static VkResult VKAPI_CALL vkSetDebugUtilsObjectNameEXT(VkDevice device,
                                                            const VkDebugUtilsObjectNameInfoEXT *pNameInfo) {
        auto deviceHandle = VulkanLayerImpl::getHandle(device);

        switch (pNameInfo->objectType) {
        case VK_OBJECT_TYPE_PIPELINE: {
            auto *pipeline = reinterpret_cast<VkPipeline>(pNameInfo->objectHandle);
            scopedMutex l(globalMutex);
            if (deviceHandle->dataGraphPipelineMap.find(pipeline) != deviceHandle->dataGraphPipelineMap.end()) {
                return VK_SUCCESS;
            }
        } break;
        case VK_OBJECT_TYPE_SHADER_MODULE: {
            auto *shaderModule = reinterpret_cast<VkShaderModule>(pNameInfo->objectHandle);
            scopedMutex l(globalMutex);
            if (deviceHandle->shaderModuleMap.find(shaderModule) != deviceHandle->shaderModuleMap.end()) {
                return VK_SUCCESS;
            }
        } break;
        default:
            break;
        }
        return deviceHandle->loader->vkSetDebugUtilsObjectNameEXT(device, pNameInfo);
    }

    static VkResult VKAPI_CALL vkGetPhysicalDeviceToolPropertiesEXT(VkPhysicalDevice device, uint32_t *pToolCount,
                                                                    VkPhysicalDeviceToolProperties *pToolProperties) {
        auto handle = VulkanLayerImpl::getHandle(device);

        VkPhysicalDeviceToolProperties tool = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TOOL_PROPERTIES_EXT,
                                               nullptr,
                                               "Graph Layer",
                                               "1.0",
                                               VK_TOOL_PURPOSE_ADDITIONAL_FEATURES_BIT,
                                               "Graph Layer",
                                               "VK_LAYER_ML_Graph_Emulation"};

        // Query mode
        if (pToolProperties == nullptr) {
            VkResult result = handle->loader->vkGetPhysicalDeviceToolPropertiesEXT(device, pToolCount, nullptr);

            if (result == VK_SUCCESS) {
                *pToolCount += 1;
            }
            return result;
        }

        const uint32_t capacity = *pToolCount;
        if (capacity == 0) {
            *pToolCount = 0;
            return VK_INCOMPLETE;
        }

        // Reserve one slot
        uint32_t downstreamCapacity = capacity - 1;

        VkResult result =
            handle->loader->vkGetPhysicalDeviceToolPropertiesEXT(device, &downstreamCapacity, pToolProperties);

        const uint32_t written = downstreamCapacity;

        if (result == VK_SUCCESS) {
            pToolProperties[written] = tool;
            *pToolCount = written + 1;
            return VK_SUCCESS;
        }

        *pToolCount = written;
        return result;
    }

    /**************************************************************************
     * Handles
     **************************************************************************/

    static std::shared_ptr<DataGraphDescriptorSet> getHandle(const std::shared_ptr<GraphDevice> &graphDevice,
                                                             const VkDescriptorSet handle) {
        scopedMutex l(globalMutex);
        return graphDevice->descriptorSetMap[handle];
    }

    static std::shared_ptr<DataGraphPipelineARM> getHandle(const std::shared_ptr<GraphDevice> &graphDevice,
                                                           const VkPipeline handle) {
        scopedMutex l(globalMutex);
        return graphDevice->dataGraphPipelineMap[handle];
    }

    static std::shared_ptr<TensorView> getHandle(const std::shared_ptr<GraphDevice> &graphDevice,
                                                 const VkTensorViewARM handle) {
        scopedMutex l(globalMutex);
        return graphDevice->tensorViewMap[handle];
    }

    static std::shared_ptr<ShaderModule> getHandle(const std::shared_ptr<GraphDevice> &graphDevice,
                                                   const VkShaderModule handle) {
        scopedMutex l(globalMutex);
        return graphDevice->shaderModuleMap[handle];
    }
    static std::shared_ptr<PipelineCache> getHandle(const VkPipelineCache handle) {
        scopedMutex l(globalMutex);
        if (handle != VK_NULL_HANDLE) {
            graphLog(Severity::Warning) << "Using an externally provided pipeline cache is not supported" << std::endl;
        }
        // Null handle means no (persistent) pipeline caching
        return std::make_shared<PipelineCache>(nullptr, 0, handle);
    }
};

} // namespace mlsdk::el::layer

/*******************************************************************************
 * External functions
 *******************************************************************************/
extern "C" {
using namespace mlsdk::el::layer;

MLEL_EXPORT PFN_vkVoidFunction VKAPI_CALL vk_layerGetPhysicalDeviceProcAddr(VkInstance instance, const char *name) {
    return GraphLayer::vk_layerGetPhysicalDeviceProcAddr(instance, name);
}

MLEL_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface *pNegotiateLayerInterface) {

    if (!pNegotiateLayerInterface || pNegotiateLayerInterface->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    if (pNegotiateLayerInterface->loaderLayerInterfaceVersion >= 2) {
        pNegotiateLayerInterface->pfnGetInstanceProcAddr = GraphLayer::vkGetInstanceProcAddr;
        pNegotiateLayerInterface->pfnGetDeviceProcAddr = GraphLayer::vkGetDeviceProcAddr;
        pNegotiateLayerInterface->pfnGetPhysicalDeviceProcAddr = GraphLayer::vk_layerGetPhysicalDeviceProcAddr;
    }

    return VK_SUCCESS;
}

MLEL_EXPORT PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char *name) {
    return GraphLayer::vkGetInstanceProcAddr(instance, name);
}

MLEL_EXPORT PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char *name) {
    return GraphLayer::vkGetDeviceProcAddr(device, name);
}

MLEL_EXPORT VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(uint32_t *pPropertyCount,
                                                                   VkLayerProperties *pProperties) {
    return GraphLayer::vkEnumerateInstanceLayerProperties(pPropertyCount, pProperties);
}

#ifdef __ANDROID__
MLEL_EXPORT VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(const char *pLayerName, uint32_t *pPropertyCount,
                                                                       VkExtensionProperties *pProperties) {
    return GraphLayer::vkEnumerateInstanceExtensionProperties(pLayerName, pPropertyCount, pProperties);
}
#endif

MLEL_EXPORT VkResult VKAPI_CALL vkEnumerateDeviceLayerProperties(VkPhysicalDevice physicalDevice,
                                                                 uint32_t *pPropertyCount,
                                                                 VkLayerProperties *pProperties) {
    return GraphLayer::vkEnumerateDeviceLayerProperties(physicalDevice, pPropertyCount, pProperties);
}

MLEL_EXPORT VkResult VKAPI_CALL vkEnumerateDeviceExtensionProperties(VkPhysicalDevice physicalDevice,
                                                                     const char *pLayerName, uint32_t *pPropertyCount,
                                                                     VkExtensionProperties *pProperties) {
    return GraphLayer::vkEnumerateDeviceExtensionProperties(physicalDevice, pLayerName, pPropertyCount, pProperties);
}
}
