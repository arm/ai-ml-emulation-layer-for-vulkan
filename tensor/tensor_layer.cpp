/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "mlel/vulkan_layer.hpp"

#include "descriptor_binding.hpp"
#include "memory_aliasing.hpp"
#include "temporary_shader_modules.hpp"
#include "tensor_arm.hpp"
#include "tensor_log.hpp"
#include "tensor_processor.hpp"
#include "tensor_view.hpp"
#include "version.hpp"

#include <limits>
#include <map>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <utility>
#include <variant>

using namespace mlsdk::el::log;

/*******************************************************************************
 * Tensor layer
 *******************************************************************************/

namespace mlsdk::el::layer {
namespace {

/*******************************************************************************
 * Device
 *******************************************************************************/

class TensorDevice : public Device {
  public:
    TensorDevice(const std::shared_ptr<PhysicalDevice> &_physicalDevice, VkDevice _device,
                 PFN_vkGetInstanceProcAddr _gipr, PFN_vkGetDeviceProcAddr _gdpr,
                 const VkAllocationCallbacks *_callbacks)
        : Device(_physicalDevice, _device, _gipr, _gdpr, _callbacks), loweringMode{getTensorLoweringMode()} {}

    const TensorLoweringMode loweringMode;
    bool uniformBufferUpdateAfterBindEnabled = false;
    std::map<VkDeviceMemory, VkDeviceSize> allocationSizes;

    VkDeviceSize availableSize(VkDeviceMemory memory, VkDeviceSize offset) const {
        const auto it = allocationSizes.find(memory);
        return it != allocationSizes.end() && offset < it->second ? it->second - offset : 0;
    }
};

/*******************************************************************************
 * Hash calculation
 *******************************************************************************/
std::size_t spirvHash(const uint32_t *spirvCode, std::size_t spirvSize) {
    std::size_t hash = spirvSize;
    for (std::size_t i = 0; i < spirvSize; ++i) {
        hash ^= std::hash<uint32_t>()(spirvCode[i]) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    return hash;
}

/*******************************************************************************
 * Layer
 *******************************************************************************/

#ifdef VK_ARM_tensors2
constexpr std::array<const VkExtensionProperties, 2> extensions = {
    VkExtensionProperties{VK_ARM_TENSORS_EXTENSION_NAME, VK_ARM_TENSORS_SPEC_VERSION},
    VkExtensionProperties{VK_ARM_TENSORS_2_EXTENSION_NAME, VK_ARM_TENSORS_2_SPEC_VERSION},
};
#else
constexpr std::array<const VkExtensionProperties, 1> extensions = {
    VkExtensionProperties{VK_ARM_TENSORS_EXTENSION_NAME, VK_ARM_TENSORS_SPEC_VERSION},
};
#endif
constexpr std::array<const VkExtensionProperties, 0> requiredExtensions = {};
constexpr VkLayerProperties layerProperties = {
    "VK_LAYER_ML_Tensor_Emulation",
    VK_MAKE_VERSION(1, 3, 0),
    VK_ARM_TENSORS_SPEC_VERSION,
    "ML Tensor Emulation Layer",
};

constexpr bool isSupportedShaderTensorFormat(VkFormat format) {
    return format != VK_FORMAT_R16_SFLOAT_FPENCODING_BFLOAT16_ARM &&
           format != VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E4M3_ARM &&
           format != VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E5M2_ARM;
}

using VulkanLayerImpl = VulkanLayer<layerProperties, extensions, requiredExtensions, TensorDevice, TensorCommandBuffer>;

class TensorLayer : public VulkanLayerImpl {
  public:
    static PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char *name) {
        static const vTable vtable = {
            // Device functions
            {"vkGetDeviceProcAddr", PFN_vkVoidFunction(vkGetDeviceProcAddr)},
            {"vkSetDebugUtilsObjectNameEXT", PFN_vkVoidFunction(vkSetDebugUtilsObjectNameEXT)},

            // Tensor extension
            {"vkCreateTensorARM", PFN_vkVoidFunction(vkCreateTensorARM)},
            {"vkDestroyTensorARM", PFN_vkVoidFunction(vkDestroyTensorARM)},
            {"vkCreateTensorViewARM", PFN_vkVoidFunction(vkCreateTensorViewARM)},
            {"vkDestroyTensorViewARM", PFN_vkVoidFunction(vkDestroyTensorViewARM)},
            {"vkGetTensorMemoryRequirementsARM", PFN_vkVoidFunction(vkGetTensorMemoryRequirementsARM)},
            {"vkBindTensorMemoryARM", PFN_vkVoidFunction(vkBindTensorMemoryARM)},
            {"vkGetDeviceTensorMemoryRequirementsARM", PFN_vkVoidFunction(vkGetDeviceTensorMemoryRequirementsARM)},
            {"vkCmdCopyTensorARM", PFN_vkVoidFunction(vkCmdCopyTensorARM)},
            {"vkGetTensorOpaqueCaptureDescriptorDataARM",
             PFN_vkVoidFunction(vkGetTensorOpaqueCaptureDescriptorDataARM)},
            {"vkGetTensorViewOpaqueCaptureDescriptorDataARM",
             PFN_vkVoidFunction(vkGetTensorViewOpaqueCaptureDescriptorDataARM)},

            // Shader
            {"vkCreateShaderModule", PFN_vkVoidFunction(vkCreateShaderModule)},

            // Compute pipeline
            {"vkCreateComputePipelines", PFN_vkVoidFunction(vkCreateComputePipelines)},

            // Descriptor set
            {"vkCreateDescriptorPool", PFN_vkVoidFunction(vkCreateDescriptorPool)},
            {"vkCreateDescriptorSetLayout", PFN_vkVoidFunction(vkCreateDescriptorSetLayout)},
            {"vkUpdateDescriptorSets", PFN_vkVoidFunction(vkUpdateDescriptorSets)},
            {"vkCmdPushDescriptorSetKHR", PFN_vkVoidFunction(vkCmdPushDescriptorSetKHR)},

            // Barrier
            {"vkCmdPipelineBarrier", PFN_vkVoidFunction(vkCmdPipelineBarrier)},
            {"vkCmdPipelineBarrier2", PFN_vkVoidFunction(vkCmdPipelineBarrier2)},

            // Image
            {"vkCreateImage", PFN_vkVoidFunction(vkCreateImage)},
            {"vkBindImageMemory", PFN_vkVoidFunction(vkBindImageMemory)},
            {"vkBindImageMemory2", PFN_vkVoidFunction(vkBindImageMemory2)},
            {"vkDestroyImage", PFN_vkVoidFunction(vkDestroyImage)},

            // Memory
            {"vkAllocateMemory", PFN_vkVoidFunction(vkAllocateMemory)},
            {"vkFreeMemory", PFN_vkVoidFunction(vkFreeMemory)}};

        if (auto it = vtable.find(name); it != vtable.end()) {
            if (it->first == "vkSetDebugUtilsObjectNameEXT") {
                // Preserve downstream availability of this optional command.
                return VulkanLayerImpl::vkGetDeviceProcAddr(device, name) ? it->second : nullptr;
            }
            return it->second;
        }

        return VulkanLayerImpl::vkGetDeviceProcAddr(device, name);
    }

    static PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char *name) {
        static const vTable vtable = {
            {"vkGetInstanceProcAddr", PFN_vkVoidFunction(vkGetInstanceProcAddr)},
            {"vk_layerGetPhysicalDeviceProcAddr", PFN_vkVoidFunction(vk_layerGetPhysicalDeviceProcAddr)},
            // PhysicalDevice functions
            {"vkGetPhysicalDeviceProperties2", PFN_vkVoidFunction(vkGetPhysicalDeviceProperties2)},
            {"vkGetPhysicalDeviceProperties2KHR", PFN_vkVoidFunction(vkGetPhysicalDeviceProperties2KHR)},
            {"vkGetPhysicalDeviceFormatProperties2", PFN_vkVoidFunction(vkGetPhysicalDeviceFormatProperties2)},
            {"vkGetPhysicalDeviceFeatures2", PFN_vkVoidFunction(vkGetPhysicalDeviceFeatures2)},
            {"vkGetPhysicalDeviceFeatures2KHR", PFN_vkVoidFunction(vkGetPhysicalDeviceFeatures2KHR)},
            {"vkGetPhysicalDeviceExternalTensorPropertiesARM",
             PFN_vkVoidFunction(vkGetPhysicalDeviceExternalTensorPropertiesARM)},
            {"vkGetPhysicalDeviceToolPropertiesEXT", PFN_vkVoidFunction(vkGetPhysicalDeviceToolPropertiesEXT)},
            {"vkCreateDevice", PFN_vkVoidFunction(vkCreateDevice)},
            // Device functions
            {"vkSetDebugUtilsObjectNameEXT", PFN_vkVoidFunction(vkSetDebugUtilsObjectNameEXT)}};

        if (auto it = vtable.find(name); it != vtable.end()) {
            return it->second;
        }

        return VulkanLayerImpl::vkGetInstanceProcAddr(instance, name);
    }

    static PFN_vkVoidFunction VKAPI_CALL vk_layerGetPhysicalDeviceProcAddr(VkInstance instance, const char *name) {
        static const vTable vtable = {
            {"vk_layerGetPhysicalDeviceProcAddr", PFN_vkVoidFunction(vk_layerGetPhysicalDeviceProcAddr)},
            // PhysicalDevice functions
            {"vkGetPhysicalDeviceProperties2", PFN_vkVoidFunction(vkGetPhysicalDeviceProperties2)},
            {"vkGetPhysicalDeviceProperties2KHR", PFN_vkVoidFunction(vkGetPhysicalDeviceProperties2KHR)},
            {"vkGetPhysicalDeviceFormatProperties2", PFN_vkVoidFunction(vkGetPhysicalDeviceFormatProperties2)},
            {"vkGetPhysicalDeviceFeatures2", PFN_vkVoidFunction(vkGetPhysicalDeviceFeatures2)},
            {"vkGetPhysicalDeviceFeatures2KHR", PFN_vkVoidFunction(vkGetPhysicalDeviceFeatures2KHR)},
            {"vkGetPhysicalDeviceExternalTensorPropertiesARM",
             PFN_vkVoidFunction(vkGetPhysicalDeviceExternalTensorPropertiesARM)},
            {"vkGetPhysicalDeviceToolPropertiesEXT", PFN_vkVoidFunction(vkGetPhysicalDeviceToolPropertiesEXT)},
            {"vkCreateDevice", PFN_vkVoidFunction(vkCreateDevice)}};

        if (auto it = vtable.find(name); it != vtable.end()) {
            return it->second;
        }

        return VulkanLayerImpl::vk_layerGetPhysicalDeviceProcAddr(instance, name);
    }

    static VkResult VKAPI_CALL vkCreateTensorARM(VkDevice device, const VkTensorCreateInfoARM *createInfo,
                                                 const VkAllocationCallbacks *allocator, VkTensorARM *tensor) {
#ifdef VK_ARM_tensors2
        const auto *scaled = findType<VkBlockScaledTensorDescriptionARM>(
            createInfo->pNext, VK_STRUCTURE_TYPE_BLOCK_SCALED_TENSOR_DESCRIPTION_ARM);
        if ((scaled && (createInfo->pDescription ||
                        scaled->blockScalingScheme != VK_TENSOR_BLOCK_SCALING_SCHEME_PLANAR_OCP_MICROSCALING_ARM ||
                        scaled->planeCount != 2 || !scaled->pPlaneTypes || !scaled->pPlaneDescriptions ||
                        scaled->pPlaneTypes[0] != VK_TENSOR_PLANE_VALUES_ARM ||
                        scaled->pPlaneTypes[1] != VK_TENSOR_PLANE_SCALES_ARM)) ||
            (!scaled && !createInfo->pDescription)) {
            tensorLog(Severity::Error) << "Unsupported tensor creation description" << std::endl;
            *tensor = VK_NULL_HANDLE;
            return VK_ERROR_VALIDATION_FAILED;
        }
#endif
        auto *tensorARM = allocateObject<TensorARM>(allocator);
        VkResult result;
        try {
            result = tensorARM->create(*VulkanLayerImpl::getHandle(device), *createInfo, allocator);
        } catch (const std::invalid_argument &error) {
            tensorLog(Severity::Error) << error.what() << std::endl;
            result = VK_ERROR_VALIDATION_FAILED;
        } catch (const std::runtime_error &error) {
            tensorLog(Severity::Error) << error.what() << std::endl;
            result = VK_ERROR_UNKNOWN;
        }
        if (result != VK_SUCCESS) {
            *tensor = VK_NULL_HANDLE;
            destroyObject(allocator, tensorARM);
            return result;
        }
        *tensor = reinterpret_cast<VkTensorARM>(tensorARM);
        return result;
    }

    static void VKAPI_CALL vkDestroyTensorARM(VkDevice device, VkTensorARM tensor,
                                              const VkAllocationCallbacks *allocator) {
        {
            scopedMutex l(globalMutex);
            memoryAliasing.removeAliasingResource(tensor);
        }
        if (tensor) {
            auto *tensorARM = reinterpret_cast<TensorARM *>(tensor);
            tensorARM->destroy(*VulkanLayerImpl::getHandle(device), allocator);
            destroyObject(allocator, tensorARM);
        }
    }

    static VkResult VKAPI_CALL vkCreateTensorViewARM(VkDevice device, const VkTensorViewCreateInfoARM *createInfo,
                                                     const VkAllocationCallbacks *allocator,
                                                     VkTensorViewARM *tensorView) {
        auto *tensorViewARM = allocateObject<TensorViewARM>(allocator);
        VkResult result = tensorViewARM->create(*VulkanLayerImpl::getHandle(device), createInfo, allocator);
        if (result != VK_SUCCESS) {
            destroyObject(allocator, tensorViewARM);
            return result;
        }
        *tensorView = reinterpret_cast<VkTensorViewARM>(tensorViewARM);
        return result;
    }

    static void VKAPI_CALL vkDestroyTensorViewARM(VkDevice device, VkTensorViewARM tensorView,
                                                  const VkAllocationCallbacks *allocator) {
        if (tensorView) {
            auto *tensorViewARM = reinterpret_cast<TensorViewARM *>(tensorView);
            tensorViewARM->destroy(*VulkanLayerImpl::getHandle(device), allocator);
            destroyObject(allocator, tensorViewARM);
        }
    }

    static void VKAPI_CALL vkGetTensorMemoryRequirementsARM(VkDevice device,
                                                            const VkTensorMemoryRequirementsInfoARM *info,
                                                            VkMemoryRequirements2 *requirements) {
        const auto *tensor = reinterpret_cast<const TensorARM *>(info->tensor);
        std::optional<uint32_t> planeIndex;
#ifdef VK_ARM_tensors2
        if (const auto *plane = findType<VkTensorPlaneInfoARM>(info->pNext, VK_STRUCTURE_TYPE_TENSOR_PLANE_INFO_ARM)) {
            planeIndex = plane->planeIndex;
        }
#endif
        if (!tensor || !tensor->getMemoryRequirements(*VulkanLayerImpl::getHandle(device),
                                                      &requirements->memoryRequirements, planeIndex)) {
            requirements->memoryRequirements = {};
            if (auto *dedicated = findTypeMutable<VkMemoryDedicatedRequirements>(
                    requirements->pNext, VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS)) {
                dedicated->prefersDedicatedAllocation = VK_FALSE;
                dedicated->requiresDedicatedAllocation = VK_FALSE;
            }
            tensorLog(Severity::Error) << "Tensor memory query requires a valid plane selector" << std::endl;
        }
    }

    static VkResult VKAPI_CALL vkBindTensorMemoryARM(VkDevice device, uint32_t bindInfoCount,
                                                     const VkBindTensorMemoryInfoARM *bindInfos) {
        auto handle = VulkanLayerImpl::getHandle(device);
        try {
            for (uint32_t i = 0; i < bindInfoCount; ++i) {
                const auto &bind = bindInfos[i];
                auto *tensor = reinterpret_cast<TensorARM *>(bind.tensor);
                if (!tensor) {
                    return VK_ERROR_VALIDATION_FAILED;
                }
                std::optional<uint32_t> planeIndex;
#ifdef VK_ARM_tensors2
                if (const auto *plane =
                        findType<VkTensorPlaneInfoARM>(bind.pNext, VK_STRUCTURE_TYPE_TENSOR_PLANE_INFO_ARM)) {
                    planeIndex = plane->planeIndex;
                }
#endif
                const auto result = tensor->bindTensorMemory(*handle, bind.memory, bind.memoryOffset, planeIndex);
                if (result != VK_SUCCESS) {
                    return result;
                }
                if (!planeIndex) {
                    // Image aliasing tracking handles only ordinary tensors.
                    scopedMutex l(globalMutex);
                    memoryAliasing.addAliasingResource(bind.memory, bind.memoryOffset, bind.tensor);
                    const auto images = memoryAliasing.getAliasingResources<VkImage>(bind.memory, bind.memoryOffset);
                    if (!images.empty()) {
                        const auto aliasResult =
                            tensor->updateAliasedTensorInfo(*handle, images[0], bind.memory, bind.memoryOffset,
                                                            handle->availableSize(bind.memory, bind.memoryOffset));
                        if (aliasResult != VK_SUCCESS) {
                            return aliasResult;
                        }
                    }
                }
            }
            return VK_SUCCESS;
        } catch (const std::bad_alloc &) {
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
    }

    static void VKAPI_CALL vkGetDeviceTensorMemoryRequirementsARM(VkDevice device,
                                                                  const VkDeviceTensorMemoryRequirementsARM *info,
                                                                  VkMemoryRequirements2 *requirements) {
#ifdef VK_ARM_tensors2
        const auto *createInfo = info->pCreateInfo;
        if (findType<VkBaseInStructure>(createInfo->pNext, VK_STRUCTURE_TYPE_BLOCK_SCALED_TENSOR_DESCRIPTION_ARM) ||
            !createInfo->pDescription) {
            // Implementation limit: return no requirements for block-scaled tensors here.
            // The draft provides neither a plane selector nor a rule for combining plane requirements.
            tensorLog(Severity::Error) << "Unsupported device tensor memory query" << std::endl;
            requirements->memoryRequirements = {};
            if (auto *dedicated = findTypeMutable<VkMemoryDedicatedRequirements>(
                    requirements->pNext, VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS)) {
                dedicated->prefersDedicatedAllocation = VK_FALSE;
                dedicated->requiresDedicatedAllocation = VK_FALSE;
            }
            return;
        }
#endif
        TensorARM::getDeviceTensorMemoryRequirements(*VulkanLayerImpl::getHandle(device), *info->pCreateInfo,
                                                     requirements);
    }

    static void VKAPI_CALL vkCmdCopyTensorARM(VkCommandBuffer commandBuffer,
                                              const VkCopyTensorInfoARM *copyTensorInfo) {
        assert(copyTensorInfo->regionCount == 1 && "Only support single region to copy tensor.");
        auto *srcTensor = reinterpret_cast<TensorARM *>(copyTensorInfo->srcTensor);
        const auto *dstTensor = reinterpret_cast<const TensorARM *>(copyTensorInfo->dstTensor);
        std::optional<uint32_t> planeIndex;
#ifdef VK_ARM_tensors2
        // Assumption: pNext selects the same plane in both tensors; each copy covers that complete plane.
        // The draft does not yet allow VkTensorPlaneInfoARM on VkCopyTensorInfoARM.
        if (const auto *plane =
                findType<VkTensorPlaneInfoARM>(copyTensorInfo->pNext, VK_STRUCTURE_TYPE_TENSOR_PLANE_INFO_ARM)) {
            planeIndex = plane->planeIndex;
        }
#endif
        srcTensor->copyToTensor(*VulkanLayerImpl::getHandle(commandBuffer), *dstTensor, planeIndex);
    }

    static VkResult vkGetTensorViewOpaqueCaptureDescriptorDataARM(VkDevice device,
                                                                  const VkTensorViewCaptureDescriptorDataInfoARM *pInfo,
                                                                  void *pData) {
        const auto *tensorView = reinterpret_cast<const TensorViewARM *>(pInfo->tensorView);
        return tensorView->getOpaqueCaptureDescriptorDataEXT(*VulkanLayerImpl::getHandle(device), pData);
    }

    static VkResult vkGetTensorOpaqueCaptureDescriptorDataARM(VkDevice device,
                                                              const VkTensorCaptureDescriptorDataInfoARM *pInfo,
                                                              void *pData) {
        const auto *tensor = reinterpret_cast<const TensorARM *>(pInfo->tensor);
        return tensor->getOpaqueCaptureDescriptorDataEXT(*VulkanLayerImpl::getHandle(device), pData);
    }

    static void VKAPI_CALL vkGetPhysicalDeviceExternalTensorPropertiesARM(
        VkPhysicalDevice physicalDevice, const VkPhysicalDeviceExternalTensorInfoARM *pExternalTensorInfo,
        VkExternalTensorPropertiesARM *pExternalTensorProperties) {
        auto handle = VulkanLayerImpl::getHandle(physicalDevice);
        VkExternalMemoryHandleTypeFlagBits handleType = pExternalTensorInfo->handleType;

        VkPhysicalDeviceExternalBufferInfo externalBufferInfo{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO, nullptr,
            TensorLayout::convertToBufferCreateFlags(pExternalTensorInfo->flags),
            TensorLayout::convertToBufferUsageFlags(pExternalTensorInfo->pDescription->usage), handleType};

        VkExternalBufferProperties externalBufferProperties{VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES, nullptr, {}};

        handle->loader->vkGetPhysicalDeviceExternalBufferProperties(physicalDevice, &externalBufferInfo,
                                                                    &externalBufferProperties);

        pExternalTensorProperties->externalMemoryProperties = externalBufferProperties.externalMemoryProperties;
    }

    static VkResult VKAPI_CALL vkCreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo *pCreateInfo,
                                                    const VkAllocationCallbacks *pAllocator,
                                                    VkShaderModule *pShaderModule) {
        auto handle = VulkanLayerImpl::getHandle(device);
        const auto mode = handle->loweringMode;
        if (mode == TensorLoweringMode::DirectSpirv &&
            (pCreateInfo == nullptr || pCreateInfo->pCode == nullptr || pCreateInfo->codeSize == 0 ||
             pCreateInfo->codeSize % sizeof(uint32_t) != 0)) {
            tensorLog(Severity::Error) << "Malformed SPIR-V shader module size or code pointer" << std::endl;
            return VK_ERROR_UNKNOWN;
        }
        if (pCreateInfo != nullptr && pCreateInfo->pCode != nullptr && pCreateInfo->codeSize > 0) {
            const uint32_t *spirvCode = pCreateInfo->pCode;
            const std::size_t spirvSize = pCreateInfo->codeSize / sizeof(uint32_t);
            const auto cacheKey = std::make_pair(mode, spirvHash(spirvCode, spirvSize));
            bool hasCacheEntry;
            std::size_t shaderModuleCodeSize = 0;
            const uint32_t *shaderModulepCode = nullptr;

            {
                scopedMutex l(globalMutex);
                const auto it = spirvCache.find(cacheKey);
                hasCacheEntry = (it != spirvCache.end());
                if (hasCacheEntry) {
                    shaderModuleCodeSize = it->second.size() * sizeof(uint32_t);
                    shaderModulepCode = it->second.data();
                }
            }

            if (!hasCacheEntry) {
                std::vector<uint32_t> spirvSource = {spirvCode, spirvCode + spirvSize};
                const TensorProcessor tensorProcessor(std::move(spirvSource), mode);
                if (!tensorProcessor.isValidShader()) {
                    return VK_ERROR_UNKNOWN;
                }
                if (tensorProcessor.isTensorComputeShader()) {
                    scopedMutex l(globalMutex);
                    auto &spirvSourceNew = spirvCache[cacheKey];
                    spirvSourceNew = tensorProcessor.getNewSpirv();
                    shaderModuleCodeSize = spirvSourceNew.size() * sizeof(uint32_t);
                    shaderModulepCode = spirvSourceNew.data();
                }
            }

            if ((shaderModuleCodeSize > 0) && (shaderModulepCode != nullptr)) {
                const VkShaderModuleCreateInfo shaderModuleInfo = {
                    VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, // type
                    pCreateInfo->pNext,                          // next
                    pCreateInfo->flags,                          // flags
                    shaderModuleCodeSize,                        // size
                    shaderModulepCode                            // code
                };

                return handle->loader->vkCreateShaderModule(device, &shaderModuleInfo, pAllocator, pShaderModule);
            }
        }
        return handle->loader->vkCreateShaderModule(device, pCreateInfo, pAllocator, pShaderModule);
    }

    static VkResult VKAPI_CALL vkCreateComputePipelines(VkDevice device, VkPipelineCache pipelineCache,
                                                        uint32_t createInfoCount,
                                                        const VkComputePipelineCreateInfo *pCreateInfos,
                                                        const VkAllocationCallbacks *pAllocator,
                                                        VkPipeline *pPipelines) {
        auto handle = VulkanLayerImpl::getHandle(device);

        for (uint32_t i = 0; i < createInfoCount; ++i) {
            pPipelines[i] = VK_NULL_HANDLE;
        }

        std::vector<VkComputePipelineCreateInfo> createInfosNew;
        const auto mode = handle->loweringMode;
        TemporaryShaderModules shaderCache(device, pAllocator, handle->loader->vkCreateShaderModule,
                                           handle->loader->vkDestroyShaderModule);

        // Inspect all VkComputePipelineCreateInfo for VkShaderModuleCreateInfo to find uses tensorARM in shaders
        for (uint32_t i = 0; i < createInfoCount; i++) {
            const auto &pipelineCreateInfo = pCreateInfos[i];
            const auto &shaderStageCreateInfo = pipelineCreateInfo.stage;
            if (shaderStageCreateInfo.module != VK_NULL_HANDLE) {
                // shaderStageCreateInfo uses "module" instead of "pNext" to specify shader
                continue;
            }
            // Find VkShaderModuleCreateInfo in pNext chain
            const auto *pShaderCreateInfo = findType<VkShaderModuleCreateInfo>(
                shaderStageCreateInfo.pNext, VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
            if (mode == TensorLoweringMode::DirectSpirv && pShaderCreateInfo != nullptr &&
                (pShaderCreateInfo->pCode == nullptr || pShaderCreateInfo->codeSize == 0 ||
                 pShaderCreateInfo->codeSize % sizeof(uint32_t) != 0)) {
                tensorLog(Severity::Error) << "Malformed inline SPIR-V shader module size or code pointer" << std::endl;
                return VK_ERROR_UNKNOWN;
            }
            if (pShaderCreateInfo == nullptr || pShaderCreateInfo->pCode == nullptr ||
                pShaderCreateInfo->codeSize == 0) {
                continue;
            }
            // If a shaderModule has already been created for this shaderModuleCreateInfo, it can be reused for this
            // pipeline
            if (const VkShaderModule module = shaderCache.find(pShaderCreateInfo); module != VK_NULL_HANDLE) {
                createInfosNew[i].stage.module = module;
                continue;
            }
            // Check if the shader uses tensors
            const uint32_t *spirvCode = pShaderCreateInfo->pCode;
            const std::size_t spirvSize = pShaderCreateInfo->codeSize / sizeof(uint32_t);
            std::vector<uint32_t> spirvSource = {spirvCode, spirvCode + spirvSize};
            const TensorProcessor tensorProcessor(std::move(spirvSource), mode);
            if (!tensorProcessor.isValidShader()) {
                return VK_ERROR_UNKNOWN;
            }
            if (!tensorProcessor.isTensorComputeShader()) {
                continue;
            }
            // Can't modify pCreateInfos, so we have to make a copy
            if (createInfosNew.empty()) {
                createInfosNew = std::vector<VkComputePipelineCreateInfo>(pCreateInfos, pCreateInfos + createInfoCount);
            }
            std::size_t shaderModuleCodeSize;
            const uint32_t *shaderModulepCode;
            // Replace tensors with buffers in shader
            {
                scopedMutex l(globalMutex);
                const auto cacheKey = std::make_pair(mode, spirvHash(spirvCode, spirvSize));
                auto &spirvSourceNew = spirvCache[cacheKey];
                if (spirvSourceNew.empty()) {
                    spirvSourceNew = tensorProcessor.getNewSpirv();
                }
                shaderModuleCodeSize = spirvSourceNew.size() * sizeof(uint32_t);
                shaderModulepCode = spirvSourceNew.data();
            }
            // Replace incoming VkShaderModuleCreateInfo with modified shader
            VkShaderModuleCreateInfo shaderModuleCreateInfo{
                VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, // type
                pShaderCreateInfo->pNext,                    // next
                pShaderCreateInfo->flags,                    // flags
                shaderModuleCodeSize,                        // size
                shaderModulepCode                            // code
            };
            auto &shaderStageCreateInfoNew = createInfosNew[i].stage;
            // The incoming shader is provided via a "const *" pNext chain. To replace the old shader with the new
            // one, we would have to copy all structs in the chain to work around pNext being immutable. Instead, we
            // explicitly call vkCreateShaderModule and set ".module" in the copied VkPipelineShaderStageCreateInfo,
            // as it will take preference over the VkShaderModuleCreateInfo in the pNext chain.
            VkShaderModule shaderModule = VK_NULL_HANDLE;
            const auto result = shaderCache.create(pShaderCreateInfo, shaderModuleCreateInfo, shaderModule);
            if (result != VK_SUCCESS) {
                return result;
            }
            shaderStageCreateInfoNew.module = shaderModule;
        }
        const auto *pCreateInfosNew = createInfosNew.empty() ? pCreateInfos : createInfosNew.data();
        VkResult res = handle->loader->vkCreateComputePipelines(device, pipelineCache, createInfoCount, pCreateInfosNew,
                                                                pAllocator, pPipelines);
        return res;
    }

    static VkResult VKAPI_CALL vkCreateDescriptorPool(VkDevice device, const VkDescriptorPoolCreateInfo *pCreateInfo,
                                                      const VkAllocationCallbacks *pAllocator,
                                                      VkDescriptorPool *pDescriptorPool) {
        auto handle = VulkanLayerImpl::getHandle(device);

        auto poolSizes = descriptor_binding::substituteTensorDescriptorPoolSizes(std::vector<VkDescriptorPoolSize>{
            pCreateInfo->pPoolSizes, pCreateInfo->pPoolSizes + pCreateInfo->poolSizeCount});

        VkDescriptorPoolCreateInfo newPoolInfo(*pCreateInfo);
        newPoolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        newPoolInfo.pPoolSizes = poolSizes.data();

        return handle->loader->vkCreateDescriptorPool(device, &newPoolInfo, pAllocator, pDescriptorPool);
    }

    static VkResult VKAPI_CALL vkCreateDescriptorSetLayout(VkDevice device,
                                                           const VkDescriptorSetLayoutCreateInfo *pCreateInfo,
                                                           const VkAllocationCallbacks *pAllocator,
                                                           VkDescriptorSetLayout *pSetLayout) {
        auto handle = VulkanLayerImpl::getHandle(device);
        const bool supportsBufferUpdateAfterBind = handle->uniformBufferUpdateAfterBindEnabled;

        const auto *bindingInfo = findType<VkDescriptorSetLayoutBindingFlagsCreateInfo>(
            pCreateInfo, VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO);

        const auto bindings =
            descriptor_binding::substituteTensorBinding(pCreateInfo->bindingCount, pCreateInfo->pBindings);

        const bool substituteBindingFlags =
            bindingInfo && std::any_of(pCreateInfo->pBindings, pCreateInfo->pBindings + pCreateInfo->bindingCount,
                                       descriptor_binding::hasTensor<VkDescriptorSetLayoutBinding>);
        std::vector<VkDescriptorBindingFlags> bindingFlags;
        if (substituteBindingFlags) {
            bindingFlags = descriptor_binding::substituteTensorBindingFlags(
                pCreateInfo->bindingCount, pCreateInfo->pBindings, *bindingInfo, supportsBufferUpdateAfterBind);
        }

        const void *pNext = substituteBindingFlags ? bindingInfo->pNext : pCreateInfo->pNext;
        const VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsCreateInfo{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
            pNext,
            uint32_t(bindingFlags.size()),
            bindingFlags.data(),
        };

        const VkDescriptorSetLayoutCreateInfo newCreateInfo{
            pCreateInfo->sType,                                       // type
            substituteBindingFlags ? &bindingFlagsCreateInfo : pNext, // next
            pCreateInfo->flags,                                       // flags
            uint32_t(bindings.size()),                                // binding count
            bindings.data(),                                          // bindings
        };

        return handle->loader->vkCreateDescriptorSetLayout(device, &newCreateInfo, pAllocator, pSetLayout);
    }

    static void VKAPI_CALL vkUpdateDescriptorSets(VkDevice device, uint32_t descriptorWriteCount,
                                                  const VkWriteDescriptorSet *pDescriptorWrites,
                                                  uint32_t descriptorCopyCount,
                                                  const VkCopyDescriptorSet *pDescriptorCopies) {
        auto handle = VulkanLayerImpl::getHandle(device);

        auto [writes, _bufferInfos, _imageInfos] =
            descriptor_binding::substituteTensorWriteDescriptorSet(*handle, descriptorWriteCount, pDescriptorWrites);

        handle->loader->vkUpdateDescriptorSets(device, uint32_t(writes.size()), writes.data(), descriptorCopyCount,
                                               pDescriptorCopies);
    }

    static void VKAPI_CALL vkCmdPushDescriptorSetKHR(VkCommandBuffer commandBuffer,
                                                     VkPipelineBindPoint pipelineBindPoint, VkPipelineLayout layout,
                                                     uint32_t set, uint32_t descriptorWriteCount,
                                                     const VkWriteDescriptorSet *pDescriptorWrites) {
        auto handle = VulkanLayerImpl::getHandle(commandBuffer);

        auto [writes, _bufferInfos, _imageInfos] = descriptor_binding::substituteTensorWriteDescriptorSet(
            *handle->device, descriptorWriteCount, pDescriptorWrites);

        handle->loader->vkCmdPushDescriptorSetKHR(commandBuffer, pipelineBindPoint, layout, set,
                                                  static_cast<uint32_t>(writes.size()), writes.data());
    }

    static void VKAPI_CALL vkCmdPipelineBarrier2(VkCommandBuffer commandBuffer,
                                                 const VkDependencyInfo *pDependencyInfo) {
        auto handle = VulkanLayerImpl::getHandle(commandBuffer);

        const auto *tensorDependencyInfo =
            findType<VkTensorDependencyInfoARM>(pDependencyInfo->pNext, VK_STRUCTURE_TYPE_TENSOR_DEPENDENCY_INFO_ARM);
        const auto *tensorBarrier =
            findType<VkTensorMemoryBarrierARM>(pDependencyInfo->pNext, VK_STRUCTURE_TYPE_TENSOR_MEMORY_BARRIER_ARM);

        if (tensorDependencyInfo == nullptr && tensorBarrier == nullptr &&
            pDependencyInfo->pImageMemoryBarriers == nullptr) {
            handle->loader->vkCmdPipelineBarrier2(commandBuffer, pDependencyInfo);
            return;
        }

        // replace tensor/image aliasing flag
        std::vector<VkImageMemoryBarrier2> imageMemoryBarriers{pDependencyInfo->pImageMemoryBarriers,
                                                               pDependencyInfo->pImageMemoryBarriers +
                                                                   pDependencyInfo->imageMemoryBarrierCount};
        for (auto &barrier : imageMemoryBarriers) {
            if (barrier.oldLayout == VK_IMAGE_LAYOUT_TENSOR_ALIASING_ARM) {
                barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            }
            if (barrier.newLayout == VK_IMAGE_LAYOUT_TENSOR_ALIASING_ARM) {
                barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            }
        }

        // replace tensor memory barrier with buffer memory barrier
        std::vector<VkBufferMemoryBarrier2> bufferMemoryBarriers{pDependencyInfo->pBufferMemoryBarriers,
                                                                 pDependencyInfo->pBufferMemoryBarriers +
                                                                     pDependencyInfo->bufferMemoryBarrierCount};
        const auto appendTensorBarriers = [&bufferMemoryBarriers](const VkTensorMemoryBarrierARM &barrier) {
            const auto *tensorARM = reinterpret_cast<const TensorARM *>(barrier.tensor);
            // Assumption: a block-scaled tensor barrier covers all planes, including queue ownership.
            // The draft does not explicitly define barriers for block-scaled tensors.
            for (const auto &plane : tensorARM->getPlanes()) {
                bufferMemoryBarriers.emplace_back(VkBufferMemoryBarrier2{
                    VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2, nullptr, barrier.srcStageMask, barrier.srcAccessMask,
                    barrier.dstStageMask, barrier.dstAccessMask, barrier.srcQueueFamilyIndex,
                    barrier.dstQueueFamilyIndex, plane.buffer, 0, VK_WHOLE_SIZE});
            }
        };
        if (tensorDependencyInfo != nullptr) {
            std::vector<VkTensorMemoryBarrierARM> tensorMemoryBarriers{
                tensorDependencyInfo->pTensorMemoryBarriers,
                tensorDependencyInfo->pTensorMemoryBarriers + tensorDependencyInfo->tensorMemoryBarrierCount};

            for (const auto &barrier : tensorMemoryBarriers) {
                appendTensorBarriers(barrier);
            }
        } else if (tensorBarrier != nullptr) {
            appendTensorBarriers(*tensorBarrier);
        }
        const VkDependencyInfo newDependencyInfo{
            VK_STRUCTURE_TYPE_DEPENDENCY_INFO,                  // sType
            nullptr,                                            // pNext
            pDependencyInfo->dependencyFlags,                   // dependencyFlags
            pDependencyInfo->memoryBarrierCount,                // memoryBarrierCount
            pDependencyInfo->pMemoryBarriers,                   // pMemoryBarriers
            static_cast<uint32_t>(bufferMemoryBarriers.size()), // bufferMemoryBarrierCount
            bufferMemoryBarriers.data(),                        // pBufferMemoryBarriers
            static_cast<uint32_t>(imageMemoryBarriers.size()),  // imageMemoryBarrierCount
            imageMemoryBarriers.data()                          // pImageMemoryBarriers
        };
        handle->loader->vkCmdPipelineBarrier2(commandBuffer, &newDependencyInfo);
    }

    static void VKAPI_CALL vkCmdPipelineBarrier(VkCommandBuffer commandBuffer, VkPipelineStageFlags srcStageMask,
                                                VkPipelineStageFlags dstStageMask, VkDependencyFlags dependencyFlags,
                                                uint32_t memoryBarrierCount, const VkMemoryBarrier *pMemoryBarriers,
                                                uint32_t bufferMemoryBarrierCount,
                                                const VkBufferMemoryBarrier *pBufferMemoryBarriers,
                                                uint32_t imageMemoryBarrierCount,
                                                const VkImageMemoryBarrier *pImageMemoryBarriers) {
        auto handle = VulkanLayerImpl::getHandle(commandBuffer);

        if (pImageMemoryBarriers == nullptr) {
            handle->loader->vkCmdPipelineBarrier(commandBuffer, srcStageMask, dstStageMask, dependencyFlags,
                                                 memoryBarrierCount, pMemoryBarriers, bufferMemoryBarrierCount,
                                                 pBufferMemoryBarriers, imageMemoryBarrierCount, pImageMemoryBarriers);
            return;
        }

        // Replace any `VK_IMAGE_LAYOUT_TENSOR_ALIASING_ARM` flags
        std::vector<VkImageMemoryBarrier> imageMemoryBarriers(pImageMemoryBarriers,
                                                              pImageMemoryBarriers + imageMemoryBarrierCount);

        for (auto &barrier : imageMemoryBarriers) {
            if (barrier.oldLayout == VK_IMAGE_LAYOUT_TENSOR_ALIASING_ARM) {
                barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            }
            if (barrier.newLayout == VK_IMAGE_LAYOUT_TENSOR_ALIASING_ARM) {
                barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            }
        }

        handle->loader->vkCmdPipelineBarrier(commandBuffer, srcStageMask, dstStageMask, dependencyFlags,
                                             memoryBarrierCount, pMemoryBarriers, bufferMemoryBarrierCount,
                                             pBufferMemoryBarriers, uint32_t(imageMemoryBarriers.size()),
                                             imageMemoryBarriers.data());
    }

    static VkResult VKAPI_CALL vkCreateImage(VkDevice device, const VkImageCreateInfo *pCreateInfo,
                                             const VkAllocationCallbacks *pAllocator, VkImage *pImage) {
        auto handle = VulkanLayerImpl::getHandle(device);
        if (pCreateInfo && pCreateInfo->usage & VK_IMAGE_USAGE_TENSOR_ALIASING_BIT_ARM) {
            auto imageCreateInfo = *pCreateInfo;
            imageCreateInfo.usage ^= VK_IMAGE_USAGE_TENSOR_ALIASING_BIT_ARM;
            imageCreateInfo.tiling = VK_IMAGE_TILING_LINEAR;
            return handle->loader->vkCreateImage(device, &imageCreateInfo, pAllocator, pImage);
        }
        return handle->loader->vkCreateImage(device, pCreateInfo, pAllocator, pImage);
    }

    static VkResult VKAPI_CALL vkBindImageMemory(VkDevice device, VkImage image, VkDeviceMemory memory,
                                                 VkDeviceSize memoryOffset) {
        const VkBindImageMemoryInfo bind{VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO, nullptr, image, memory,
                                         memoryOffset};
        return vkBindImageMemory2(device, 1, &bind);
    }

    static VkResult VKAPI_CALL vkBindImageMemory2(VkDevice device, uint32_t bindInfoCount,
                                                  const VkBindImageMemoryInfo *pBindInfos) {
        auto handle = VulkanLayerImpl::getHandle(device);
        const auto result = handle->loader->vkBindImageMemory2(device, bindInfoCount, pBindInfos);
        if (result != VK_SUCCESS) {
            return result;
        }
        scopedMutex l(globalMutex);
        try {
            for (uint32_t i = 0; i < bindInfoCount; ++i) {
                const auto &bind = pBindInfos[i];
                memoryAliasing.addAliasingResource(bind.memory, bind.memoryOffset, bind.image);
                const auto tensors = memoryAliasing.getAliasingResources<VkTensorARM>(bind.memory, bind.memoryOffset);
                for (const VkTensorARM tensor : tensors) {
                    auto *underlyingTensor = reinterpret_cast<TensorARM *>(tensor);
                    const auto aliasResult = underlyingTensor->updateAliasedTensorInfo(
                        *handle, bind.image, bind.memory, bind.memoryOffset,
                        handle->availableSize(bind.memory, bind.memoryOffset));
                    if (aliasResult != VK_SUCCESS) {
                        return aliasResult;
                    }
                }
            }
            return VK_SUCCESS;
        } catch (const std::bad_alloc &) {
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
    }

    static void VKAPI_CALL vkDestroyImage(VkDevice device, VkImage image, const VkAllocationCallbacks *pAllocator) {
        auto handle = VulkanLayerImpl::getHandle(device);
        handle->loader->vkDestroyImage(device, image, pAllocator);
        {
            scopedMutex l(globalMutex);
            memoryAliasing.removeAliasingResource(image);
        }
    }

    static VkResult VKAPI_CALL vkAllocateMemory(VkDevice device, const VkMemoryAllocateInfo *pAllocateInfo,
                                                const VkAllocationCallbacks *pAllocator, VkDeviceMemory *pMemory) {
        const auto originalAllocateChain = dumpVkStructureList(pAllocateInfo);
        VkMemoryAllocateInfo newAllocateInfo{*pAllocateInfo};
        const auto originalAllocateFlags = findAndRemoveType<VkMemoryAllocateFlagsInfo>(
            &newAllocateInfo, VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO);

        auto newAllocateFlagInfo = originalAllocateFlags.current ? *originalAllocateFlags.current
                                                                 : VkMemoryAllocateFlagsInfo{
                                                                       VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
                                                                       nullptr,
                                                                       VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
                                                                       0,
                                                                   };

        newAllocateFlagInfo.flags |= VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        appendType(&newAllocateInfo, &newAllocateFlagInfo);

        auto handle = VulkanLayerImpl::getHandle(device);
        const auto result = handle->loader->vkAllocateMemory(device, &newAllocateInfo, pAllocator, pMemory);

        loadVkStructureList(const_cast<VkMemoryAllocateInfo *>(pAllocateInfo), originalAllocateChain);
        if (result == VK_SUCCESS) {
            scopedMutex l(globalMutex);
            try {
                handle->allocationSizes[*pMemory] = pAllocateInfo->allocationSize;
            } catch (const std::bad_alloc &) {
                handle->loader->vkFreeMemory(device, *pMemory, pAllocator);
                *pMemory = VK_NULL_HANDLE;
                return VK_ERROR_OUT_OF_HOST_MEMORY;
            }
        }
        return result;
    }

    static void VKAPI_CALL vkFreeMemory(VkDevice device, VkDeviceMemory memory,
                                        const VkAllocationCallbacks *pAllocator) {
        auto handle = VulkanLayerImpl::getHandle(device);
        {
            scopedMutex l(globalMutex);
            handle->allocationSizes.erase(memory);
            memoryAliasing.removeAliasingMemory(memory);
        }
        handle->loader->vkFreeMemory(device, memory, pAllocator);
    }

    static void VKAPI_CALL vkGetPhysicalDeviceFormatProperties2(VkPhysicalDevice physicalDevice, VkFormat format,
                                                                VkFormatProperties2 *pFormatProperties) {
        auto handle = VulkanLayerImpl::getHandle(physicalDevice);
        auto *pTensorFormatProp = findTypeMutable<VkTensorFormatPropertiesARM>(
            pFormatProperties->pNext, VK_STRUCTURE_TYPE_TENSOR_FORMAT_PROPERTIES_ARM);
        handle->loader->vkGetPhysicalDeviceFormatProperties2(physicalDevice, format, pFormatProperties);
        if (pTensorFormatProp) {
#ifdef VK_ARM_tensors2
            switch (format) {
            case VK_FORMAT_R8_UFLOAT_FPENCODING_FLOAT8UNSIGNEDE8M0_ARM:
                // Storage and copies are supported; shader/graph lowering is not yet implemented.
                pTensorFormatProp->optimalTilingTensorFeatures =
                    VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT;
                pTensorFormatProp->linearTilingTensorFeatures = pTensorFormatProp->optimalTilingTensorFeatures;
                return;
            case VK_FORMAT_R8_SFLOAT_FPENCODING_MXINT8_ARM:
            case VK_FORMAT_OCPMXFP_32_SFLOAT_BLOCK_FPENCODING_FLOAT4E2M1_ARM:
            case VK_FORMAT_OCPMXFP_32_SFLOAT_BLOCK_FPENCODING_FLOAT6E2M3_ARM:
            case VK_FORMAT_OCPMXFP_32_SFLOAT_BLOCK_FPENCODING_FLOAT6E3M2_ARM:
                pTensorFormatProp->optimalTilingTensorFeatures = 0;
                pTensorFormatProp->linearTilingTensorFeatures = 0;
                return;
            default:
                break;
            }
#endif
            pTensorFormatProp->optimalTilingTensorFeatures = VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT |
                                                             VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT |
                                                             VK_FORMAT_FEATURE_2_TENSOR_DATA_GRAPH_BIT_ARM;
            if (isSupportedShaderTensorFormat(format)) {
                pTensorFormatProp->optimalTilingTensorFeatures |= VK_FORMAT_FEATURE_2_TENSOR_SHADER_BIT_ARM;
            }
            pTensorFormatProp->linearTilingTensorFeatures = pTensorFormatProp->optimalTilingTensorFeatures;
        }
    }

    static void VKAPI_CALL vkGetPhysicalDeviceFeatures2(VkPhysicalDevice physicalDevice,
                                                        VkPhysicalDeviceFeatures2 *pFeatures) {
        auto handle = VulkanLayerImpl::getHandle(physicalDevice);
        auto *pTensorFeatures = findTypeMutable<VkPhysicalDeviceTensorFeaturesARM>(
            pFeatures->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSOR_FEATURES_ARM);
        handle->loader->vkGetPhysicalDeviceFeatures2(physicalDevice, pFeatures);
#ifdef VK_ARM_tensors2
        if (auto *features = findTypeMutable<VkPhysicalDeviceTensors2FeaturesARM>(
                pFeatures->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSORS_2_FEATURES_ARM)) {
            features->tensorFlexibleStrides = VK_FALSE;
            // Assumption: block-scaled storage does not imply shader/graph support;
            // those uses require separately advertised tensor format features.
            features->tensorBlockScaled = VK_TRUE;
        }
#endif
        if (pTensorFeatures) {
            // query buffer feature
            VkPhysicalDeviceVulkan12Features queryVulkan12Feature{};
            queryVulkan12Feature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
            queryVulkan12Feature.pNext = nullptr;
            VkPhysicalDeviceFeatures2 queryFeatures2{};
            queryFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            queryFeatures2.pNext = &queryVulkan12Feature;
            handle->loader->vkGetPhysicalDeviceFeatures2(physicalDevice, &queryFeatures2);

            pTensorFeatures->tensorNonPacked = VK_TRUE;
            pTensorFeatures->shaderTensorAccess = VK_TRUE;
            pTensorFeatures->shaderStorageTensorArrayDynamicIndexing =
                pFeatures->features.shaderUniformBufferArrayDynamicIndexing;
            pTensorFeatures->shaderStorageTensorArrayNonUniformIndexing =
                queryVulkan12Feature.shaderUniformBufferArrayNonUniformIndexing;
            pTensorFeatures->descriptorBindingStorageTensorUpdateAfterBind =
                queryVulkan12Feature.descriptorBindingStorageBufferUpdateAfterBind;
            pTensorFeatures->tensors = VK_TRUE;
        }
    }

    static void VKAPI_CALL vkGetPhysicalDeviceFeatures2KHR(VkPhysicalDevice physicalDevice,
                                                           VkPhysicalDeviceFeatures2 *pFeatures) {
        vkGetPhysicalDeviceFeatures2(physicalDevice, pFeatures);
    }

    static void VKAPI_CALL vkGetPhysicalDeviceProperties2(VkPhysicalDevice physicalDevice,
                                                          VkPhysicalDeviceProperties2 *pProperties) {

        auto handle = VulkanLayerImpl::getHandle(physicalDevice);
        auto tensorProps = findAndRemoveType<VkPhysicalDeviceTensorPropertiesARM>(
            pProperties, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSOR_PROPERTIES_ARM);

        handle->loader->vkGetPhysicalDeviceProperties2(physicalDevice, pProperties);

        const auto &limits = pProperties->properties.limits;

        if (tensorProps.current) {
            tensorProps.current->maxTensorDimensionCount = TensorLayout::MAX_DIMENSIONS;
            tensorProps.current->maxTensorElements = limits.maxStorageBufferRange;
            tensorProps.current->maxPerDimensionTensorElements = limits.maxStorageBufferRange;
            tensorProps.current->maxTensorStride = limits.maxStorageBufferRange;
            tensorProps.current->maxTensorSize = limits.maxStorageBufferRange;
            tensorProps.current->maxTensorShaderAccessArrayLength = std::numeric_limits<uint8_t>::max();
            tensorProps.current->maxTensorShaderAccessSize = limits.maxStorageBufferRange;
            tensorProps.current->maxDescriptorSetStorageTensors = limits.maxDescriptorSetUniformBuffers;
            tensorProps.current->maxPerStageDescriptorSetStorageTensors = limits.maxPerStageDescriptorUniformBuffers;
            tensorProps.current->maxDescriptorSetUpdateAfterBindStorageTensors = limits.maxDescriptorSetUniformBuffers;
            tensorProps.current->maxPerStageDescriptorUpdateAfterBindStorageTensors =
                limits.maxPerStageDescriptorUniformBuffers;
            tensorProps.current->shaderStorageTensorArrayNonUniformIndexingNative = false;
            tensorProps.current->shaderTensorSupportedStages =
                VkShaderStageFlagBits::VK_SHADER_STAGE_COMPUTE_BIT; // EL only supports tensors in compute shaders
            insertType(tensorProps);
        }
    }

    static void VKAPI_CALL vkGetPhysicalDeviceProperties2KHR(VkPhysicalDevice physicalDevice,
                                                             VkPhysicalDeviceProperties2 *pProperties) {
        vkGetPhysicalDeviceProperties2(physicalDevice, pProperties);
    }

    static VkResult VKAPI_CALL vkCreateDevice(VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo *createInfo,
                                              const VkAllocationCallbacks *allocator, VkDevice *device) {
#ifdef VK_ARM_tensors2
        if (const auto *features = findType<VkPhysicalDeviceTensors2FeaturesARM>(
                createInfo->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSORS_2_FEATURES_ARM);
            features && features->tensorFlexibleStrides) {
            tensorLog(Severity::Error) << "Tensor flexible strides are not supported" << std::endl;
            *device = VK_NULL_HANDLE;
            return VK_ERROR_FEATURE_NOT_PRESENT;
        }
#endif
        auto originCreateInfoChain = dumpVkStructureList(createInfo);

        VkDeviceCreateInfo newCreateInfo{*createInfo};
#ifdef VK_ARM_tensors2
        findAndRemoveType<VkPhysicalDeviceTensors2FeaturesARM>(
            &newCreateInfo, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSORS_2_FEATURES_ARM);
#endif
        const auto tensorFeatures = findAndRemoveType<VkPhysicalDeviceTensorFeaturesARM>(
            &newCreateInfo, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSOR_FEATURES_ARM);
        // Tensor descriptor arrays are represented by uniform-buffer descriptor arrays.
        // Enable their native indexing features without changing the application's feature flags.
        VkPhysicalDeviceFeatures features{};
        VkPhysicalDeviceFeatures2 features2{};
        if (tensorFeatures.current && tensorFeatures.current->shaderStorageTensorArrayDynamicIndexing) {
            if (findType<VkPhysicalDeviceFeatures2>(newCreateInfo.pNext,
                                                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2)) {
                replaceOrAppendFeature(&newCreateInfo, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, features2,
                                       [](VkPhysicalDeviceFeatures2 &value) {
                                           value.features.shaderUniformBufferArrayDynamicIndexing = VK_TRUE;
                                       });
            } else {
                if (createInfo->pEnabledFeatures) {
                    features = *createInfo->pEnabledFeatures;
                }
                features.shaderUniformBufferArrayDynamicIndexing = VK_TRUE;
                newCreateInfo.pEnabledFeatures = &features;
            }
        }
        VkPhysicalDeviceVulkan12Features vulkan12Features{};
        VkPhysicalDeviceDescriptorIndexingFeatures indexingFeatures{};
        if (tensorFeatures.current && tensorFeatures.current->shaderStorageTensorArrayNonUniformIndexing) {
            if (findType<VkPhysicalDeviceVulkan12Features>(newCreateInfo.pNext,
                                                           VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES)) {
                replaceOrAppendFeature(&newCreateInfo, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                                       vulkan12Features, [](VkPhysicalDeviceVulkan12Features &value) {
                                           value.shaderUniformBufferArrayNonUniformIndexing = VK_TRUE;
                                       });
            } else {
                replaceOrAppendFeature(&newCreateInfo, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES,
                                       indexingFeatures, [](VkPhysicalDeviceDescriptorIndexingFeatures &value) {
                                           value.shaderUniformBufferArrayNonUniformIndexing = VK_TRUE;
                                       });
            }
        }
        auto result = VulkanLayerImpl::vkCreateDevice(physicalDevice, &newCreateInfo, allocator, device);
        if (result == VK_SUCCESS) {
            auto handle = VulkanLayerImpl::getHandle(*device);
            handle->uniformBufferUpdateAfterBindEnabled = isUniformBufferUpdateAfterBindEnabled(createInfo);
        }

        loadVkStructureList(const_cast<VkDeviceCreateInfo *>(createInfo), originCreateInfoChain);
        return result;
    }

    static VkResult VKAPI_CALL vkSetDebugUtilsObjectNameEXT(VkDevice device,
                                                            const VkDebugUtilsObjectNameInfoEXT *pNameInfo) {
        auto handle = VulkanLayerImpl::getHandle(device);
        switch (pNameInfo->objectType) {
        case VK_OBJECT_TYPE_TENSOR_ARM: {
            const auto *tensorARM = reinterpret_cast<const TensorARM *>(pNameInfo->objectHandle);
            VkDebugUtilsObjectNameInfoEXT newNameInfo{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
                                                      pNameInfo->pNext, VK_OBJECT_TYPE_BUFFER, 0,
                                                      pNameInfo->pObjectName};
            for (const auto &plane : tensorARM->getPlanes()) {
                newNameInfo.objectHandle = reinterpret_cast<uint64_t>(plane.buffer);
                const VkResult result = handle->loader->vkSetDebugUtilsObjectNameEXT(device, &newNameInfo);
                if (result != VK_SUCCESS) {
                    return result;
                }
            }
        } break;
        case VK_OBJECT_TYPE_TENSOR_VIEW_ARM:
            break;
        default:
            return handle->loader->vkSetDebugUtilsObjectNameEXT(device, pNameInfo);
        }
        return VK_SUCCESS;
    }

    static VkResult VKAPI_CALL vkGetPhysicalDeviceToolPropertiesEXT(VkPhysicalDevice device, uint32_t *pToolCount,
                                                                    VkPhysicalDeviceToolProperties *pToolProperties) {
        auto handle = VulkanLayerImpl::getHandle(device);

        VkPhysicalDeviceToolProperties tool = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TOOL_PROPERTIES_EXT,
                                               nullptr,
                                               "Tensor Layer",
                                               "1.0",
                                               VK_TOOL_PURPOSE_ADDITIONAL_FEATURES_BIT,
                                               "Tensor Layer",
                                               "VK_LAYER_ML_Tensor_Emulation"};

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

    static inline std::map<std::pair<TensorLoweringMode, std::size_t>, std::vector<uint32_t>> spirvCache;

    static inline MemoryAliasing memoryAliasing;
    static bool isUniformBufferUpdateAfterBindEnabled(const VkDeviceCreateInfo *createInfo) {
        if (const auto *vulkan12Features = findType<VkPhysicalDeviceVulkan12Features>(
                createInfo, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES)) {
            if (vulkan12Features->descriptorBindingUniformBufferUpdateAfterBind) {
                return true;
            }
        }

        if (const auto *descriptorIndexingFeatures = findType<VkPhysicalDeviceDescriptorIndexingFeatures>(
                createInfo, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES)) {
            if (descriptorIndexingFeatures->descriptorBindingUniformBufferUpdateAfterBind) {
                return true;
            }
        }

        return false;
    }
};

} // namespace
} // namespace mlsdk::el::layer

/*******************************************************************************
 * External functions
 *******************************************************************************/

extern "C" {
using namespace mlsdk::el::layer;

MLEL_EXPORT PFN_vkVoidFunction VKAPI_CALL vk_layerGetPhysicalDeviceProcAddr(VkInstance instance, const char *pName) {
    return TensorLayer::vk_layerGetPhysicalDeviceProcAddr(instance, pName);
}

MLEL_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface *pNegotiateLayerInterface) {
    if (!pNegotiateLayerInterface || pNegotiateLayerInterface->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    if (pNegotiateLayerInterface->loaderLayerInterfaceVersion < 2) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    tensorLog(Severity::Info) << mlsdk::el::details::version << std::endl;

    pNegotiateLayerInterface->loaderLayerInterfaceVersion = 2;
    pNegotiateLayerInterface->pfnGetInstanceProcAddr = TensorLayer::vkGetInstanceProcAddr;
    pNegotiateLayerInterface->pfnGetDeviceProcAddr = TensorLayer::vkGetDeviceProcAddr;
    pNegotiateLayerInterface->pfnGetPhysicalDeviceProcAddr = TensorLayer::vk_layerGetPhysicalDeviceProcAddr;

    return VK_SUCCESS;
}

MLEL_EXPORT PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char *name) {
    return TensorLayer::vkGetInstanceProcAddr(instance, name);
}

MLEL_EXPORT PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char *name) {
    return TensorLayer::vkGetDeviceProcAddr(device, name);
}

MLEL_EXPORT VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(uint32_t *pPropertyCount,
                                                                   VkLayerProperties *pProperties) {
    return TensorLayer::vkEnumerateInstanceLayerProperties(pPropertyCount, pProperties);
}

#ifdef __ANDROID__
MLEL_EXPORT VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(const char *pLayerName, uint32_t *pPropertyCount,
                                                                       VkExtensionProperties *pProperties) {
    return TensorLayer::vkEnumerateInstanceExtensionProperties(pLayerName, pPropertyCount, pProperties);
}
#endif

MLEL_EXPORT VkResult VKAPI_CALL vkEnumerateDeviceLayerProperties(VkPhysicalDevice physicalDevice,
                                                                 uint32_t *pPropertyCount,
                                                                 VkLayerProperties *pProperties) {
    return TensorLayer::vkEnumerateDeviceLayerProperties(physicalDevice, pPropertyCount, pProperties);
}

MLEL_EXPORT VkResult VKAPI_CALL vkEnumerateDeviceExtensionProperties(VkPhysicalDevice physicalDevice,
                                                                     const char *pLayerName, uint32_t *pPropertyCount,
                                                                     VkExtensionProperties *pProperties) {
    return TensorLayer::vkEnumerateDeviceExtensionProperties(physicalDevice, pLayerName, pPropertyCount, pProperties);
}
}
