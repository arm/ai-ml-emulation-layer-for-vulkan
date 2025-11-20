/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/*****************************************************************************
 * Includes
 *****************************************************************************/

#include "mlel/vulkan_layer.hpp"

#include "compute.hpp"
#include "graph_log.hpp"
#include "memory_planner.hpp"
#include "pipeline_cache.hpp"
#include "source/opt/build_module.h"
#include "source/opt/ir_context.h"
#include "source/opt/module.h"
#include "spirv_pass.hpp"
#include "spirv_pass_tosaspv_v100.hpp"

#include <chrono>
#include <optional>
#include <regex>

using namespace mlsdk::el::compute;
using namespace mlsdk::el::log;

/*****************************************************************************
 * Graph layer
 *****************************************************************************/

namespace mlsdk::el::layer {
namespace {
constexpr char graphPipelineCreatedLog[] = "Graph pipeline created";
}

/*****************************************************************************
 * Instance
 *****************************************************************************/

class GraphInstance : public Instance {
  public:
    explicit GraphInstance(VkInstance _instance, PFN_vkGetInstanceProcAddr _gipr,
                           const VkAllocationCallbacks *_callbacks, PFN_vkGetInstanceProcAddr nextGetInstanceProcAddr,
                           PFN_GetPhysicalDeviceProcAddr nextGetPhysicalDeviceProcAddr)
        : Instance(_instance, _gipr, _callbacks, nextGetInstanceProcAddr, nextGetPhysicalDeviceProcAddr) {}
};

/*****************************************************************************
 * PhysicalDevice
 *****************************************************************************/

class GraphPhysicalDevice : public PhysicalDevice {
  public:
    explicit GraphPhysicalDevice(const std::shared_ptr<Instance> &_instance, VkPhysicalDevice _physicalDevice)
        : PhysicalDevice(_instance, _physicalDevice) {}
};

/**************************************************************************
 * DataGraphDescriptorSet
 **************************************************************************/

class DataGraphDescriptorSet : public DescriptorSet {
  public:
    explicit DataGraphDescriptorSet(const std::shared_ptr<DescriptorSetLayout> &_descriptorSetLayout)
        : DescriptorSet(_descriptorSetLayout) {
        for (const auto &[binding, descriptorSetLayoutBinding] : descriptorSetLayout->bindings) {
            tensorViews[binding].resize(descriptorSetLayoutBinding.descriptorCount);
        }
    }

    void update(const VkWriteDescriptorSet &set) {
        [[maybe_unused]] const auto &bindingInfo = descriptorSetLayout->bindings.at(set.dstBinding);

        assert(bindingInfo.descriptorType == set.descriptorType);
        assert(bindingInfo.descriptorCount >= set.dstArrayElement + set.descriptorCount);

        switch (set.descriptorType) {
        case VK_DESCRIPTOR_TYPE_TENSOR_ARM: {
            auto tensorInfo =
                findType<VkWriteDescriptorSetTensorARM>(set.pNext, VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_TENSOR_ARM);
            assert(tensorInfo);
            assert(tensorInfo->tensorViewCount == set.descriptorCount);

            for (uint32_t i = 0; i < set.descriptorCount; i++) {
                tensorViews[set.dstBinding][set.dstArrayElement + i] = tensorInfo->pTensorViews[i];
            }
            break;
        }
        default:
            break;
        }
    }

    // Mapping from [binding, arrayIndex] to tensor view
    std::map<uint32_t, std::vector<VkTensorViewARM>> tensorViews;

    // Mapping from [pipeline, set] to external descriptor sets bound by the application
    std::map<std::tuple<VkPipeline, uint32_t>, ComputeDescriptorSetMap> externalDescriptorSets;
};

/*****************************************************************************
 * DataGraphPipelineARM
 *****************************************************************************/

class DataGraphPipelineARM : public Loader {
  public:
    explicit DataGraphPipelineARM(const std::shared_ptr<Device> &device,
                                  const std::shared_ptr<PipelineCache> &_pipelineCache)
        : Loader(*device), graphPipeline{std::make_shared<GraphPipeline>(device->loader,
                                                                         device->physicalDevice->physicalDevice,
                                                                         device->device, _pipelineCache)} {}

    std::shared_ptr<GraphPipeline> graphPipeline;
    ComputeDescriptorSetMap constantsDescriptorSets;

    void makeConstantsDescriptorSets() {
        constantsDescriptorSets = graphPipeline->makeConstantsDescriptorSets();
        for ([[maybe_unused]] const auto &[_, descriptorSet] : constantsDescriptorSets) {
            descriptorSet->updateDescriptorSet();
        }
    }
};

/*****************************************************************************
 * DataGraphPipelineSessionARM
 *****************************************************************************/

class DataGraphPipelineSessionARM : public Loader {
  public:
    explicit DataGraphPipelineSessionARM(const std::shared_ptr<Device> &device,
                                         const std::shared_ptr<DataGraphPipelineARM> &_pipeline)
        : Loader(*device), pipeline{_pipeline}, memoryPlanner{createMemoryPlanner(pipeline->graphPipeline)},
          sessionRamDescriptorSets{pipeline->graphPipeline->makeSessionRamDescriptorSets()} {}

    std::shared_ptr<DataGraphPipelineARM> pipeline;
    std::shared_ptr<MemoryPlanner> memoryPlanner;

    // Session ram descriptor sets
    ComputeDescriptorSetMap sessionRamDescriptorSets;

  private:
    std::shared_ptr<MemoryPlanner> createMemoryPlanner(const std::shared_ptr<GraphPipeline> &graphPipeline) const {
        const auto envMemoryPlanner = std::getenv("VMEL_MEMORY_PLANNER");

        if (envMemoryPlanner && std::string(envMemoryPlanner) == "Linear") {
            graphLog(Severity::Info) << "Using linear memory planner" << std::endl;
            return std::make_shared<LinearMemoryPlanner>(graphPipeline);
        }

        graphLog(Severity::Info) << "Using best-fit memory planner" << std::endl;
        return std::make_shared<BestFitMemoryPlanner>(graphPipeline);
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

inline std::optional<bool> isGraphSpirv(const std::vector<uint32_t> &spirv) {
    auto ir = spvtools::BuildModule(SPV_ENV_UNIVERSAL_1_6, nullptr, spirv.data(), spirv.size());
    if (ir == nullptr || ir->module() == nullptr) {
        return std::nullopt;
    }
    return ir->module()->graphs().size() > 0;
}

std::optional<std::string> tryGetExtInstVersion(const uint32_t *spirvCode, const size_t spirvSize,
                                                const std::regex &pattern) {
    auto ir = spvtools::BuildModule(SPV_ENV_UNIVERSAL_1_6, nullptr, spirvCode, spirvSize);
    for (const auto &inst : ir->module()->ext_inst_imports()) {
        const auto name = inst.GetInOperand(0).AsString();
        if (std::regex_search(name, pattern))
            return name;
    }
    return std::nullopt;
}

} // namespace

constexpr std::array<const VkExtensionProperties, 1> extensions{
    VkExtensionProperties{VK_ARM_DATA_GRAPH_EXTENSION_NAME, VK_ARM_DATA_GRAPH_SPEC_VERSION},
};

constexpr std::array<const VkExtensionProperties, 1> requiredExtensions = {
    VkExtensionProperties{VK_ARM_TENSORS_EXTENSION_NAME, VK_ARM_TENSORS_SPEC_VERSION},
};

constexpr VkLayerProperties layerProperties = {
    "VK_LAYER_ML_Graph_Emulation",
    VK_MAKE_VERSION(1, 3, 0),
    VK_ARM_DATA_GRAPH_SPEC_VERSION,
    "ML Graph Emulation Layer",
};

using VulkanLayerImpl =
    VulkanLayer<layerProperties, extensions, requiredExtensions, Instance, PhysicalDevice, GraphDevice>;

class GraphLayer : public VulkanLayerImpl {
  public:
    static PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char *name) {
        static const vTable vtable = {
            // Instance functions
            {"vkGetInstanceProcAddr", PFN_vkVoidFunction(vkGetInstanceProcAddr)},
            {"vk_layerGetPhysicalDeviceProcAddr", PFN_vkVoidFunction(vk_layerGetPhysicalDeviceProcAddr)},

            // PhysicalDevice functions
            {"vkGetPhysicalDeviceQueueFamilyDataGraphProcessingEnginePropertiesARM",
             PFN_vkVoidFunction(vkGetPhysicalDeviceQueueFamilyDataGraphProcessingEnginePropertiesARM)},
            {"vkGetPhysicalDeviceQueueFamilyDataGraphPropertiesARM",
             PFN_vkVoidFunction(vkGetPhysicalDeviceQueueFamilyDataGraphPropertiesARM)},
            {"vkGetPhysicalDeviceQueueFamilyProperties", PFN_vkVoidFunction(vkGetPhysicalDeviceQueueFamilyProperties)},
            {"vkGetPhysicalDeviceQueueFamilyProperties2",
             PFN_vkVoidFunction(vkGetPhysicalDeviceQueueFamilyProperties2)},
            {"vkGetPhysicalDeviceFeatures2", PFN_vkVoidFunction(vkGetPhysicalDeviceFeatures2)},
            {"vkGetPhysicalDeviceFeatures2KHR", PFN_vkVoidFunction(vkGetPhysicalDeviceFeatures2KHR)},
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
            {"vkGetPhysicalDeviceQueueFamilyDataGraphProcessingEnginePropertiesARM",
             PFN_vkVoidFunction(vkGetPhysicalDeviceQueueFamilyDataGraphProcessingEnginePropertiesARM)},
            {"vkGetPhysicalDeviceQueueFamilyDataGraphPropertiesARM",
             PFN_vkVoidFunction(vkGetPhysicalDeviceQueueFamilyDataGraphPropertiesARM)},
            {"vkGetPhysicalDeviceQueueFamilyProperties", PFN_vkVoidFunction(vkGetPhysicalDeviceQueueFamilyProperties)},
            {"vkGetPhysicalDeviceQueueFamilyProperties2",
             PFN_vkVoidFunction(vkGetPhysicalDeviceQueueFamilyProperties2)},
            {"vkGetPhysicalDeviceFeatures2", PFN_vkVoidFunction(vkGetPhysicalDeviceFeatures2)},
            {"vkGetPhysicalDeviceFeatures2KHR", PFN_vkVoidFunction(vkGetPhysicalDeviceFeatures2KHR)},
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
            if (dataGraphPipelineShaderModuleCreateInfo == nullptr) {
                graphLog(Severity::Error) << "Missing shader module create info" << std::endl;
                return VK_ERROR_UNKNOWN;
            }

            // Create pipeline handle
            auto pipeline = std::allocate_shared<DataGraphPipelineARM>(Allocator<GraphPipeline>{callbacks},
                                                                       deviceHandle, pipelineCacheHandle);
            pipelines[i] = reinterpret_cast<VkPipeline>(pipeline.get());
            auto graphPipeline = pipeline->graphPipeline;
            graphLog(Severity::Info) << graphPipelineCreatedLog << std::endl;

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
                auto shaderModuleCreateInfo =
                    findType<VkShaderModuleCreateInfo>(createInfo.pNext, VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
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
                } else if (isGraph.value()) {
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
            const auto motionEngineVersion = tryGetExtInstVersion(shaderModule->code.data(), shaderModule->code.size(),
                                                                  std::regex("^Arm\\.MotionEngine\\.\\d{3}"));

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

        auto pDataGraphFeatures =
            const_cast<VkPhysicalDeviceDataGraphFeaturesARM *>(findType<VkPhysicalDeviceDataGraphFeaturesARM>(
                pFeatures->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_FEATURES_ARM));
        if (pDataGraphFeatures) {
            pDataGraphFeatures->dataGraph = VK_TRUE;
            pDataGraphFeatures->dataGraphUpdateAfterBind = VK_TRUE;
            pDataGraphFeatures->dataGraphShaderModule = VK_TRUE;
        }
        auto pPipelineCreationCacheControlFeatures = const_cast<VkPhysicalDevicePipelineCreationCacheControlFeatures *>(
            findType<VkPhysicalDevicePipelineCreationCacheControlFeatures>(
                pFeatures->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_CREATION_CACHE_CONTROL_FEATURES));
        // Pipeline caching is currently not supported
        if (pPipelineCreationCacheControlFeatures) {
            pPipelineCreationCacheControlFeatures->pipelineCreationCacheControl = VK_FALSE;
        }
    }

    static VkResult VKAPI_CALL vkCreateDevice(VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo *createInfo,
                                              const VkAllocationCallbacks *allocator, VkDevice *device) {
        auto originCreateInfoChain = dumpVkStructureList(createInfo);

        VkDeviceCreateInfo newCreateInfo{*createInfo};
        findAndRemoveType<VkPhysicalDeviceDataGraphFeaturesARM>(
            &newCreateInfo, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_FEATURES_ARM);
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
        auto deviceHandle = VulkanLayerImpl::getHandle(device);
        auto pipelineImpl = getHandle(deviceHandle, createInfo->dataGraphPipeline);

        *session = reinterpret_cast<VkDataGraphPipelineSessionARM>(
            allocateObject<DataGraphPipelineSessionARM>(callbacks, deviceHandle, pipelineImpl));

        return VK_SUCCESS;
    }

    static VkResult VKAPI_CALL vkGetDataGraphPipelineSessionBindPointRequirementsARM(
        VkDevice, const VkDataGraphPipelineSessionBindPointRequirementsInfoARM *info,
        uint32_t *bindPointRequirementCount, VkDataGraphPipelineSessionBindPointRequirementARM *bindPointRequirements) {
        const auto session = reinterpret_cast<DataGraphPipelineSessionARM *>(info->session);

        *bindPointRequirementCount = 0;

        // Calculate how much memory pipelines hidden layers require
        const auto memoryRequirements = session->memoryPlanner->getGraphPipelineSessionMemoryRequirements();
        if (memoryRequirements.size > 0) {
            (*bindPointRequirementCount)++;
        }

        if (bindPointRequirements != nullptr) {
            bindPointRequirements[0] = VkDataGraphPipelineSessionBindPointRequirementARM{
                VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_REQUIREMENTS_INFO_ARM, // type
                nullptr,                                                                        // next
                VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TRANSIENT_ARM,                        // bind point
                VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TYPE_MEMORY_ARM,                      // bind point type
                1,                                                                              // number of resources
            };
        }

        return VK_SUCCESS;
    }

    static void VKAPI_CALL vkGetDataGraphPipelineSessionMemoryRequirementsARM(
        VkDevice, const VkDataGraphPipelineSessionMemoryRequirementsInfoARM *info,
        VkMemoryRequirements2 *requirements) {
        const auto session = reinterpret_cast<DataGraphPipelineSessionARM *>(info->session);

        // Calculate how much memory pipelines hidden layers require
        requirements->memoryRequirements = session->memoryPlanner->getGraphPipelineSessionMemoryRequirements();
    }

    static VkResult VKAPI_CALL vkBindDataGraphPipelineSessionMemoryARM(
        VkDevice, uint32_t bindInfoCount, const VkBindDataGraphPipelineSessionMemoryInfoARM *bindInfos) {
        const auto session = reinterpret_cast<DataGraphPipelineSessionARM *>(bindInfos->session);

        // Bind session memory to hidden layers
        for (uint32_t i = 0; i < bindInfoCount; i++) {
            switch (bindInfos[i].bindPoint) {
            case VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TRANSIENT_ARM: {
                session->memoryPlanner->bindGraphPipelineSessionMemory(bindInfos[i].memory, bindInfos[i].memoryOffset,
                                                                       session->sessionRamDescriptorSets);

                for ([[maybe_unused]] const auto &[_, descriptorSet] : session->sessionRamDescriptorSets) {
                    descriptorSet->updateDescriptorSet();
                }

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

    static VkResult VKAPI_CALL vkGetPhysicalDeviceQueueFamilyDataGraphPropertiesARM(
        VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex, uint32_t *pQueueFamilyDataGraphPropertyCount,
        VkQueueFamilyDataGraphPropertiesARM *pQueueFamilyDataGraphProperties) {
        auto handle = VulkanLayerImpl::getHandle(physicalDevice);
        uint32_t familyCount = 0;
        handle->loader->vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &familyCount, nullptr);
        if (queueFamilyIndex >= familyCount) {
            return VK_ERROR_UNKNOWN;
        }

        if (!pQueueFamilyDataGraphProperties) {
            *pQueueFamilyDataGraphPropertyCount = 1;
            return VK_SUCCESS;
        }

        if (*pQueueFamilyDataGraphPropertyCount == 0) {
            return VK_INCOMPLETE;
        }

        *pQueueFamilyDataGraphProperties = VkQueueFamilyDataGraphPropertiesARM{};
        *pQueueFamilyDataGraphPropertyCount = 1;

        return VK_SUCCESS;
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
                                                     VkDataGraphPipelineSessionARM _session) {
        auto handle = VulkanLayerImpl::getHandle(commandBuffer);
        auto session = reinterpret_cast<DataGraphPipelineSessionARM *>(_session);
        auto pipeline = session->pipeline;
        auto vkPipeline = reinterpret_cast<VkPipeline>(pipeline.get());
        auto graphPipeline = pipeline->graphPipeline;
        auto deviceHandle = VulkanLayerImpl::getHandle(handle->device->device);

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
                 * The list of compute jobs is first known when the pipeline is dispatched. A DescriptorSet is bound to
                 * a PipelineLayout, which is why the compute DescriptorSets must be created here.
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
                computeDescriptorSetMap.insert(descriptorSetMapTemp.begin(), descriptorSetMapTemp.end());

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

        allDescriptorSetMap.insert(pipeline->constantsDescriptorSets.begin(), pipeline->constantsDescriptorSets.end());
        allDescriptorSetMap.insert(session->sessionRamDescriptorSets.begin(), session->sessionRamDescriptorSets.end());

        graphPipeline->cmdBindAndDispatch(commandBuffer, allDescriptorSetMap);
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
        } else if (isGraph.value()) {
            std::shared_ptr<ShaderModule> shaderModule = std::make_shared<ShaderModule>(pCreateInfo);
            *pShaderModule = reinterpret_cast<VkShaderModule>(shaderModule.get());
            {
                scopedMutex l(globalMutex);
                deviceHandle->shaderModuleMap[*pShaderModule] = std::move(shaderModule);
            }
            return VK_SUCCESS;
        } else {
            return deviceHandle->loader->vkCreateShaderModule(device, pCreateInfo, pAllocator, pShaderModule);
        }
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

        auto tensorDependencyInfo =
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
            auto pipeline = reinterpret_cast<VkPipeline>(pNameInfo->objectHandle);
            scopedMutex l(globalMutex);
            if (deviceHandle->dataGraphPipelineMap.find(pipeline) != deviceHandle->dataGraphPipelineMap.end()) {
                return VK_SUCCESS;
            }
        } break;
        case VK_OBJECT_TYPE_SHADER_MODULE: {
            auto shaderModule = reinterpret_cast<VkShaderModule>(pNameInfo->objectHandle);
            scopedMutex l(globalMutex);
            if (deviceHandle->shaderModuleMap.find(shaderModule) != deviceHandle->shaderModuleMap.end()) {
                return VK_SUCCESS;
            }
        } break;
        default:
            break;
        }
        return deviceHandle->physicalDevice->instance->loader->vkSetDebugUtilsObjectNameEXT(device, pNameInfo);
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

PFN_vkVoidFunction VKAPI_CALL graphGetInstanceProcAddr(VkInstance instance, const char *name) {
    return GraphLayer::vkGetInstanceProcAddr(instance, name);
}

PFN_vkVoidFunction VKAPI_CALL graphGetDeviceProcAddr(VkDevice device, const char *name) {
    return GraphLayer::vkGetDeviceProcAddr(device, name);
}

PFN_vkVoidFunction VKAPI_CALL vk_layerGetPhysicalDeviceProcAddr(VkInstance instance, const char *name) {
    return GraphLayer::vk_layerGetPhysicalDeviceProcAddr(instance, name);
}

VKAPI_ATTR VkResult VKAPI_CALL
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

PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char *name) {
    return GraphLayer::vkGetInstanceProcAddr(instance, name);
}

PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char *name) {
    return GraphLayer::vkGetDeviceProcAddr(device, name);
}

VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(uint32_t *pPropertyCount, VkLayerProperties *pProperties) {
    return GraphLayer::vkEnumerateInstanceLayerProperties(pPropertyCount, pProperties);
}

#ifdef __ANDROID__
VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(const char *pLayerName, uint32_t *pPropertyCount,
                                                           VkExtensionProperties *pProperties) {
    return GraphLayer::vkEnumerateInstanceExtensionProperties(pLayerName, pPropertyCount, pProperties);
}
#endif

VkResult VKAPI_CALL vkEnumerateDeviceLayerProperties(VkPhysicalDevice physicalDevice, uint32_t *pPropertyCount,
                                                     VkLayerProperties *pProperties) {
    return GraphLayer::vkEnumerateDeviceLayerProperties(physicalDevice, pPropertyCount, pProperties);
}

VkResult VKAPI_CALL vkEnumerateDeviceExtensionProperties(VkPhysicalDevice physicalDevice, const char *pLayerName,
                                                         uint32_t *pPropertyCount, VkExtensionProperties *pProperties) {
    return GraphLayer::vkEnumerateDeviceExtensionProperties(physicalDevice, pLayerName, pPropertyCount, pProperties);
}
}
