/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "mlel/vulkan_layer.hpp"

#include "tensor_log.hpp"

#include "descriptor_binding.hpp"
#include "tensor_ext.hpp"
#include "tensor_processor.hpp"
#include "tensor_view.hpp"
#include <limits>
#include <memory>

using namespace mlsdk::el::log;

/*******************************************************************************
 * Tensor layer
 *******************************************************************************/

namespace mlsdk::el::layer {

/*******************************************************************************
 * Instance
 *******************************************************************************/

class TensorInstance : public Instance {
  public:
    TensorInstance(VkInstance instance, PFN_vkGetInstanceProcAddr gipr, const VkAllocationCallbacks *callbacks)
        : Instance(instance, gipr, callbacks) {}
};

/*******************************************************************************
 * PhysicalDevice
 *******************************************************************************/

class TensorPhysicalDevice : public PhysicalDevice {
  public:
    TensorPhysicalDevice(const std::shared_ptr<Instance> &_instance, VkPhysicalDevice _physicalDevice)
        : PhysicalDevice(_instance, _physicalDevice) {}
};

/*******************************************************************************
 * Device
 *******************************************************************************/

class TensorDevice : public Device {
  public:
    TensorDevice(const std::shared_ptr<PhysicalDevice> &_physicalDevice, VkDevice _device,
                 PFN_vkGetInstanceProcAddr _gipr, PFN_vkGetDeviceProcAddr _gdpr,
                 const VkAllocationCallbacks *_callbacks)
        : Device(_physicalDevice, _device, _gipr, _gdpr, _callbacks) {}
};

/*******************************************************************************
 * DeviceMemory
 *******************************************************************************/

class DeviceMemory {
  public:
    VkImage boundImage = VK_NULL_HANDLE;
    VkTensorARM boundTensor = VK_NULL_HANDLE;
};

/*******************************************************************************
 * Hash calculation
 *******************************************************************************/
inline std::size_t spirvHash(const std::vector<uint32_t> &spirv) {
    std::size_t hash = spirv.size();
    for (auto &i : spirv) {
        hash ^= std::hash<uint32_t>()(i) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    return hash;
}

/*******************************************************************************
 * Layer
 *******************************************************************************/

constexpr std::array<const VkExtensionProperties, 1> extensions = {
    VkExtensionProperties{VK_ARM_TENSORS_EXTENSION_NAME, VK_ARM_TENSORS_SPEC_VERSION},
};
constexpr std::array<const VkExtensionProperties, 0> requiredExtensions = {};
constexpr VkLayerProperties layerProperties = {
    "VK_LAYER_ML_Tensor_Emulation",
    VK_MAKE_VERSION(1, 3, 0),
    VK_ARM_TENSORS_SPEC_VERSION,
    "ML Tensor Emulation Layer",
};

using VulkanLayerImpl =
    VulkanLayer<layerProperties, extensions, requiredExtensions, TensorInstance, TensorPhysicalDevice, TensorDevice>;

class TensorLayer : public VulkanLayerImpl {
  public:
    static PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char *name) {
        static std::map<std::string, PFN_vkVoidFunction> vtable = {
            // Device functions
            {"vkGetDeviceProcAddr", PFN_vkVoidFunction(vkGetDeviceProcAddr)},

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

            // Memory
            {"vkAllocateMemory", PFN_vkVoidFunction(vkAllocateMemory)},
            {"vkFreeMemory", PFN_vkVoidFunction(vkFreeMemory)}};

        auto it = vtable.find(name);
        if (it != vtable.end()) {
            return it->second;
        }

        return VulkanLayerImpl::vkGetDeviceProcAddr(device, name);
    }

    static PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char *name) {
        static std::map<std::string, PFN_vkVoidFunction> vtable = {
            {"vkGetInstanceProcAddr", PFN_vkVoidFunction(vkGetInstanceProcAddr)},
            // PhysicalDevice functions
            {"vkGetPhysicalDeviceProperties2", PFN_vkVoidFunction(vkGetPhysicalDeviceProperties2)},
            {"vkGetPhysicalDeviceFormatProperties2", PFN_vkVoidFunction(vkGetPhysicalDeviceFormatProperties2)},
            {"vkGetPhysicalDeviceFeatures2", PFN_vkVoidFunction(vkGetPhysicalDeviceFeatures2)},
            {"vkGetPhysicalDeviceFeatures2KHR", PFN_vkVoidFunction(vkGetPhysicalDeviceFeatures2KHR)},
            {"vkCreateDevice", PFN_vkVoidFunction(vkCreateDevice)},
            // Device functions
            {"vkSetDebugUtilsObjectNameEXT", PFN_vkVoidFunction(vkSetDebugUtilsObjectNameEXT)}};

        auto it = vtable.find(name);
        if (it != vtable.end()) {
            return it->second;
        }

        return VulkanLayerImpl::vkGetInstanceProcAddr(instance, name);
    }

    static VkResult VKAPI_CALL vkCreateTensorARM(VkDevice device, const VkTensorCreateInfoARM *createInfo,
                                                 const VkAllocationCallbacks *allocator, VkTensorARM *tensor) {
        auto tensorARM = allocateObject<TensorARM>(allocator);
        VkResult result = tensorARM->create(*VulkanLayerImpl::getHandle(device), *createInfo, allocator);
        if (result != VK_SUCCESS) {
            destroyObject(allocator, tensorARM);
            return result;
        }
        *tensor = reinterpret_cast<VkTensorARM>(tensorARM);
        return result;
    }

    static void VKAPI_CALL vkDestroyTensorARM(VkDevice device, VkTensorARM tensor,
                                              const VkAllocationCallbacks *allocator) {
        auto tensorARM = reinterpret_cast<TensorARM *>(tensor);
        tensorARM->destroy(*VulkanLayerImpl::getHandle(device), allocator);
        destroyObject(allocator, tensorARM);
    }

    static VkResult VKAPI_CALL vkCreateTensorViewARM(VkDevice device, const VkTensorViewCreateInfoARM *createInfo,
                                                     const VkAllocationCallbacks *allocator,
                                                     VkTensorViewARM *tensorView) {
        auto tensorViewARM = allocateObject<TensorViewARM>(allocator);
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
        auto tensorViewARM = reinterpret_cast<TensorViewARM *>(tensorView);
        tensorViewARM->destroy(*VulkanLayerImpl::getHandle(device), allocator);
        destroyObject(allocator, tensorViewARM);
    }

    static void VKAPI_CALL vkGetTensorMemoryRequirementsARM(VkDevice device,
                                                            const VkTensorMemoryRequirementsInfoARM *info,
                                                            VkMemoryRequirements2 *requirements) {
        auto tensor = reinterpret_cast<TensorARM *>(info->tensor);
        tensor->getMemoryRequirements(*VulkanLayerImpl::getHandle(device), &requirements->memoryRequirements);
    }

    static VkResult VKAPI_CALL vkBindTensorMemoryARM(VkDevice device, uint32_t bindInfoCount,
                                                     const VkBindTensorMemoryInfoARM *bindInfos) {
        VkResult result = VK_SUCCESS;
        for (uint32_t i = 0; i < bindInfoCount; i++) {
            auto tensor = reinterpret_cast<TensorARM *>(bindInfos[i].tensor);
            result = tensor->bindTensorMemory(*VulkanLayerImpl::getHandle(device), bindInfos[i].memory,
                                              bindInfos[i].memoryOffset);
            if (result == VK_SUCCESS) {
                auto deviceMemory = getHandle(bindInfos[i].memory);
                deviceMemory->boundTensor = bindInfos[i].tensor;
                if (deviceMemory->boundImage != VK_NULL_HANDLE) {
                    // update tensor info if tensor aliased with an image
                    tensor->updateAliasedTensorInfo(*VulkanLayerImpl::getHandle(device), deviceMemory->boundImage);
                }
            } else {
                break;
            }
        }
        return result;
    }

    static void VKAPI_CALL vkGetDeviceTensorMemoryRequirementsARM(VkDevice device,
                                                                  const VkDeviceTensorMemoryRequirementsARM *info,
                                                                  VkMemoryRequirements2 *requirements) {
        TensorARM::getDeviceTensorMemoryRequirements(*VulkanLayerImpl::getHandle(device), *info->pCreateInfo,
                                                     requirements);
    }

    static void VKAPI_CALL vkCmdCopyTensorARM(VkCommandBuffer commandBuffer,
                                              const VkCopyTensorInfoARM *copyTensorInfo) {
        assert(copyTensorInfo->regionCount == 1 && "Only support single region to copy tensor.");
        auto srcTensor = reinterpret_cast<TensorARM *>(copyTensorInfo->srcTensor);
        auto dstTensor = reinterpret_cast<TensorARM *>(copyTensorInfo->dstTensor);
        srcTensor->copyToTensor(*VulkanLayerImpl::getHandle(commandBuffer), *dstTensor);
    }

    static VkResult vkGetTensorViewOpaqueCaptureDescriptorDataARM(VkDevice device,
                                                                  const VkTensorViewCaptureDescriptorDataInfoARM *pInfo,
                                                                  void *pData) {
        auto tensorView = reinterpret_cast<TensorViewARM *>(pInfo->tensorView);
        return tensorView->getOpaqueCaptureDescriptorDataEXT(*VulkanLayerImpl::getHandle(device), pData);
    }

    static VkResult vkGetTensorOpaqueCaptureDescriptorDataARM(VkDevice device,
                                                              const VkTensorCaptureDescriptorDataInfoARM *pInfo,
                                                              void *pData) {
        auto tensor = reinterpret_cast<TensorARM *>(pInfo->tensor);
        return tensor->getOpaqueCaptureDescriptorDataEXT(*VulkanLayerImpl::getHandle(device), pData);
    }

    static VkResult vkCreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo *pCreateInfo,
                                         const VkAllocationCallbacks *pAllocator, VkShaderModule *pShaderModule) {
        auto handle = VulkanLayerImpl::getHandle(device);
        if (pCreateInfo != nullptr && pCreateInfo->pCode != nullptr && pCreateInfo->codeSize > 0) {
            std::vector<uint32_t> spirvSource = {pCreateInfo->pCode,
                                                 pCreateInfo->pCode + pCreateInfo->codeSize / sizeof(uint32_t)};
            const std::size_t hashCode = spirvHash(spirvSource);
            bool hasCacheEntry;
            std::size_t shaderModuleCodeSize = 0;
            const uint32_t *shaderModulepCode = nullptr;

            {
                scopedMutex l(globalMutex);
                const auto it = spirvCache.find(hashCode);
                hasCacheEntry = (it != spirvCache.end());
                if (hasCacheEntry) {
                    shaderModuleCodeSize = it->second.size() * sizeof(uint32_t);
                    shaderModulepCode = it->second.data();
                }
            }

            if (!hasCacheEntry) {
                TensorProcessor tensorProcessor(spirvSource);
                if (!tensorProcessor.isValidShader()) {
                    return VK_ERROR_UNKNOWN;
                }
                if (tensorProcessor.isTensorComputeShader()) {
                    scopedMutex l(globalMutex);
                    auto &spirvSourceNew = spirvCache[hashCode];
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

    static VkResult vkCreateComputePipelines(VkDevice device, VkPipelineCache pipelineCache, uint32_t createInfoCount,
                                             const VkComputePipelineCreateInfo *pCreateInfos,
                                             const VkAllocationCallbacks *pAllocator, VkPipeline *pPipelines) {
        auto handle = VulkanLayerImpl::getHandle(device);

        std::vector<VkComputePipelineCreateInfo> createInfosNew;
        std::map<const VkShaderModuleCreateInfo *, VkShaderModule>
            shaderCache; // avoid creating multiple shader modules for the same shader

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
            if (pShaderCreateInfo == nullptr || pShaderCreateInfo->pCode == nullptr ||
                pShaderCreateInfo->codeSize == 0) {
                continue;
            }
            // If a shaderModule has already been created for this shaderModuleCreateInfo, it can be reused for this
            // pipeline
            if (auto it = shaderCache.find(pShaderCreateInfo); it != shaderCache.end()) {
                createInfosNew[i].stage.module = it->second;
                continue;
            }
            // Check if the shader uses tensors
            std::vector<uint32_t> spirvSource = {
                pShaderCreateInfo->pCode, pShaderCreateInfo->pCode + pShaderCreateInfo->codeSize / sizeof(uint32_t)};
            TensorProcessor tensorProcessor(spirvSource);
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
                std::size_t hashCode = spirvHash(spirvSource);
                auto &spirvSourceNew = spirvCache[hashCode];
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
            // The incoming shader is provided via a "const *" pNext chain. To replace the old shader with the new one,
            // we would have to copy all structs in the chain to work around pNext being immutable.
            // Instead, we explicitly call vkCreateShaderModule and set ".module" in the copied
            // VkPipelineShaderStageCreateInfo, as it will take preference over the VkShaderModuleCreateInfo in the
            // pNext chain.
            VkShaderModule shaderModule;
            handle->loader->vkCreateShaderModule(device, &shaderModuleCreateInfo, pAllocator, &shaderModule);
            shaderCache[pShaderCreateInfo] = shaderModule;
            shaderStageCreateInfoNew.module = shaderModule;
        }
        const auto *pCreateInfosNew = createInfosNew.empty() ? pCreateInfos : createInfosNew.data();
        VkResult res = handle->loader->vkCreateComputePipelines(device, pipelineCache, createInfoCount, pCreateInfosNew,
                                                                pAllocator, pPipelines);
        for (auto it : shaderCache) {
            handle->loader->vkDestroyShaderModule(device, it.second, pAllocator);
        }
        return res;
    }

    static VkResult vkCreateDescriptorPool(VkDevice device, const VkDescriptorPoolCreateInfo *pCreateInfo,
                                           const VkAllocationCallbacks *pAllocator, VkDescriptorPool *pDescriptorPool) {
        auto handle = VulkanLayerImpl::getHandle(device);

        auto poolSizes = descriptor_binding::substituteTensorDescriptorPoolSizes(std::vector<VkDescriptorPoolSize>{
            pCreateInfo->pPoolSizes, pCreateInfo->pPoolSizes + pCreateInfo->poolSizeCount});

        VkDescriptorPoolCreateInfo newPoolInfo(*pCreateInfo);
        newPoolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        newPoolInfo.pPoolSizes = poolSizes.data();

        return handle->loader->vkCreateDescriptorPool(device, &newPoolInfo, pAllocator, pDescriptorPool);
    }

    static VkResult vkCreateDescriptorSetLayout(VkDevice device, const VkDescriptorSetLayoutCreateInfo *pCreateInfo,
                                                const VkAllocationCallbacks *pAllocator,
                                                VkDescriptorSetLayout *pSetLayout) {
        auto handle = VulkanLayerImpl::getHandle(device);

        auto bindingInfo = findType<VkDescriptorSetLayoutBindingFlagsCreateInfo>(
            pCreateInfo, VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO);

        const auto bindings =
            descriptor_binding::substituteTensorBinding(pCreateInfo->bindingCount, pCreateInfo->pBindings, bindingInfo);

        const VkDescriptorSetLayoutCreateInfo newCreateInfo{
            pCreateInfo->sType,        // type
            pCreateInfo->pNext,        // next
            pCreateInfo->flags,        // flags
            uint32_t(bindings.size()), // binding count
            bindings.data(),           // bindings
        };

        return handle->loader->vkCreateDescriptorSetLayout(device, &newCreateInfo, pAllocator, pSetLayout);
    }

    static void vkUpdateDescriptorSets(VkDevice device, uint32_t descriptorWriteCount,
                                       const VkWriteDescriptorSet *pDescriptorWrites, uint32_t descriptorCopyCount,
                                       const VkCopyDescriptorSet *pDescriptorCopies) {
        auto handle = VulkanLayerImpl::getHandle(device);

        auto [writes, _bufferInfos, _imageInfos] =
            descriptor_binding::substituteTensorWriteDescriptorSet(*handle, descriptorWriteCount, pDescriptorWrites);

        handle->loader->vkUpdateDescriptorSets(device, uint32_t(writes.size()), writes.data(), descriptorCopyCount,
                                               pDescriptorCopies);
    }

    static void vkCmdPushDescriptorSetKHR(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint,
                                          VkPipelineLayout layout, uint32_t set, uint32_t descriptorWriteCount,
                                          const VkWriteDescriptorSet *pDescriptorWrites) {
        auto handle = VulkanLayerImpl::getHandle(commandBuffer);

        auto [writes, _bufferInfos, _imageInfos] = descriptor_binding::substituteTensorWriteDescriptorSet(
            *handle->device, descriptorWriteCount, pDescriptorWrites);

        handle->loader->vkCmdPushDescriptorSetKHR(commandBuffer, pipelineBindPoint, layout, set,
                                                  static_cast<uint32_t>(writes.size()), writes.data());
    }

    static void vkCmdPipelineBarrier2(VkCommandBuffer commandBuffer, const VkDependencyInfo *pDependencyInfo) {
        auto handle = VulkanLayerImpl::getHandle(commandBuffer);

        auto tensorDependencyInfo =
            findType<VkTensorDependencyInfoARM>(pDependencyInfo->pNext, VK_STRUCTURE_TYPE_TENSOR_DEPENDENCY_INFO_ARM);
        auto tensorBarrier =
            findType<VkTensorMemoryBarrierARM>(pDependencyInfo->pNext, VK_STRUCTURE_TYPE_TENSOR_MEMORY_BARRIER_ARM);

        if (tensorDependencyInfo == nullptr && tensorBarrier == nullptr &&
            pDependencyInfo->pImageMemoryBarriers == nullptr) {
            return handle->loader->vkCmdPipelineBarrier2(commandBuffer, pDependencyInfo);
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
        if (tensorDependencyInfo != nullptr) {
            std::vector<VkTensorMemoryBarrierARM> tensorMemoryBarriers{
                tensorDependencyInfo->pTensorMemoryBarriers,
                tensorDependencyInfo->pTensorMemoryBarriers + tensorDependencyInfo->tensorMemoryBarrierCount};

            for (const auto &barrier : tensorMemoryBarriers) {
                auto tensorARM = reinterpret_cast<TensorARM *>(barrier.tensor);
                bufferMemoryBarriers.emplace_back(VkBufferMemoryBarrier2{
                    VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2, // sType
                    nullptr,                                   // pNext
                    barrier.srcStageMask,                      // srcStageMask
                    barrier.srcAccessMask,                     // srcAccessMask
                    barrier.dstStageMask,                      // dstStageMask
                    barrier.dstAccessMask,                     // dstAccessMask
                    barrier.srcQueueFamilyIndex,               // srcQueueFamilyIndex
                    barrier.dstQueueFamilyIndex,               // dstQueueFamilyIndex
                    tensorARM->getTensorBuffer(),              // buffer
                    0,                                         // offset
                    VK_WHOLE_SIZE                              // size
                });
            }
        } else if (tensorBarrier != nullptr) {
            auto tensorARM = reinterpret_cast<TensorARM *>(tensorBarrier->tensor);
            bufferMemoryBarriers.emplace_back(VkBufferMemoryBarrier2{
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2, // sType
                nullptr,                                   // pNext
                tensorBarrier->srcStageMask,               // srcStageMask
                tensorBarrier->srcAccessMask,              // srcAccessMask
                tensorBarrier->dstStageMask,               // dstStageMask
                tensorBarrier->dstAccessMask,              // dstAccessMask
                tensorBarrier->srcQueueFamilyIndex,        // srcQueueFamilyIndex
                tensorBarrier->dstQueueFamilyIndex,        // dstQueueFamilyIndex
                tensorARM->getTensorBuffer(),              // buffer
                0,                                         // offset
                VK_WHOLE_SIZE                              // size
            });
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
            return handle->loader->vkCmdPipelineBarrier(
                commandBuffer, srcStageMask, dstStageMask, dependencyFlags, memoryBarrierCount, pMemoryBarriers,
                bufferMemoryBarrierCount, pBufferMemoryBarriers, imageMemoryBarrierCount, pImageMemoryBarriers);
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

    static VkResult vkCreateImage(VkDevice device, const VkImageCreateInfo *pCreateInfo,
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

    static VkResult vkBindImageMemory(VkDevice device, VkImage image, VkDeviceMemory memory,
                                      VkDeviceSize memoryOffset) {
        auto handle = VulkanLayerImpl::getHandle(device);
        auto result = handle->loader->vkBindImageMemory(device, image, memory, memoryOffset);
        if (result == VK_SUCCESS) {
            auto deviceMemory = getHandle(memory);
            deviceMemory->boundImage = image;
            if (deviceMemory->boundTensor != VK_NULL_HANDLE) {
                // update tensor info if tensor is aliased with image
                auto tensor = reinterpret_cast<TensorARM *>(deviceMemory->boundTensor);
                tensor->updateAliasedTensorInfo(*handle, image);
            }
        }
        return result;
    }

    static VkResult vkBindImageMemory2(VkDevice device, uint32_t bindInfoCount,
                                       const VkBindImageMemoryInfo *pBindInfos) {
        auto handle = VulkanLayerImpl::getHandle(device);
        auto result = handle->loader->vkBindImageMemory2(device, bindInfoCount, pBindInfos);
        if (result == VK_SUCCESS) {
            for (uint32_t i = 0; i < bindInfoCount; i++) {
                auto deviceMemory = getHandle(pBindInfos[i].memory);
                deviceMemory->boundImage = pBindInfos[i].image;
                if (deviceMemory->boundTensor != VK_NULL_HANDLE) {
                    // update tensor info if tensor is aliased with image
                    auto tensor = reinterpret_cast<TensorARM *>(deviceMemory->boundTensor);
                    tensor->updateAliasedTensorInfo(*handle, deviceMemory->boundImage);
                }
            }
        }

        return result;
    }

    static VkResult vkAllocateMemory(VkDevice device, const VkMemoryAllocateInfo *pAllocateInfo,
                                     const VkAllocationCallbacks *pAllocator, VkDeviceMemory *pMemory) {
        auto handle = VulkanLayerImpl::getHandle(device);
        auto newAllocateFlagInfo =
            getType<VkMemoryAllocateFlagsInfo>(pAllocateInfo->pNext, VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
                                               VkMemoryAllocateFlagsInfo{
                                                   VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
                                                   nullptr,
                                                   VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
                                                   0,
                                               });
        newAllocateFlagInfo.flags |= VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        const VkMemoryAllocateInfo newAllocateInfo{
            pAllocateInfo->sType,
            &newAllocateFlagInfo,
            pAllocateInfo->allocationSize,
            pAllocateInfo->memoryTypeIndex,
        };
        auto result = handle->loader->vkAllocateMemory(device, &newAllocateInfo, pAllocator, pMemory);
        if (result == VK_SUCCESS) {
            scopedMutex l(globalMutex);
            deviceMemoryMap[*pMemory] = std::make_shared<DeviceMemory>();
        }
        return result;
    }

    static void vkFreeMemory(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks *pAllocator) {
        {
            scopedMutex l(globalMutex);
            deviceMemoryMap.erase(memory);
        }
        auto handle = VulkanLayerImpl::getHandle(device);
        return handle->loader->vkFreeMemory(device, memory, pAllocator);
    }

    static void vkGetPhysicalDeviceFormatProperties2(VkPhysicalDevice physicalDevice, VkFormat format,
                                                     VkFormatProperties2 *pFormatProperties) {
        auto handle = VulkanLayerImpl::getHandle(physicalDevice);
        auto pTensorFormatProp = const_cast<VkTensorFormatPropertiesARM *>(findType<VkTensorFormatPropertiesARM>(
            pFormatProperties->pNext, VK_STRUCTURE_TYPE_TENSOR_FORMAT_PROPERTIES_ARM));
        handle->loader->vkGetPhysicalDeviceFormatProperties2(physicalDevice, format, pFormatProperties);
        if (pTensorFormatProp) {
            pTensorFormatProp->optimalTilingTensorFeatures =
                VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT |
                VK_FORMAT_FEATURE_2_TENSOR_SHADER_BIT_ARM | VK_FORMAT_FEATURE_2_TENSOR_DATA_GRAPH_BIT_ARM;
            pTensorFormatProp->linearTilingTensorFeatures = pTensorFormatProp->optimalTilingTensorFeatures;
        }
    }

    static void vkGetPhysicalDeviceFeatures2(VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures2 *pFeatures) {
        auto handle = VulkanLayerImpl::getHandle(physicalDevice);
        auto pTensorFeatures =
            const_cast<VkPhysicalDeviceTensorFeaturesARM *>(findType<VkPhysicalDeviceTensorFeaturesARM>(
                pFeatures->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSOR_FEATURES_ARM));
        handle->loader->vkGetPhysicalDeviceFeatures2(physicalDevice, pFeatures);
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
                pFeatures->features.shaderStorageBufferArrayDynamicIndexing;
            pTensorFeatures->shaderStorageTensorArrayNonUniformIndexing =
                queryVulkan12Feature.shaderStorageBufferArrayNonUniformIndexing;
            pTensorFeatures->descriptorBindingStorageTensorUpdateAfterBind =
                queryVulkan12Feature.descriptorBindingStorageBufferUpdateAfterBind;
        }
    }

    static void vkGetPhysicalDeviceFeatures2KHR(VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures2 *pFeatures) {
        vkGetPhysicalDeviceFeatures2(physicalDevice, pFeatures);
    }

    static void vkGetPhysicalDeviceProperties2(VkPhysicalDevice physicalDevice,
                                               VkPhysicalDeviceProperties2 *pProperties) {

        auto handle = VulkanLayerImpl::getHandle(physicalDevice);
        auto tensorProps = findAndRemoveType<VkPhysicalDeviceTensorPropertiesARM>(
            pProperties, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSOR_PROPERTIES_ARM);

        handle->loader->vkGetPhysicalDeviceProperties2(physicalDevice, pProperties);

        if (tensorProps.current) {
            tensorProps.current->maxTensorDimensionCount = TensorARM::TENSOR_MAX_DIMENSIONS;
            tensorProps.current->maxTensorElements = pProperties->properties.limits.maxStorageBufferRange;
            tensorProps.current->maxTensorStride = pProperties->properties.limits.maxStorageBufferRange;
            tensorProps.current->maxDescriptorSetStorageTensors = std::numeric_limits<uint32_t>::max();
            tensorProps.current->maxPerStageDescriptorSetStorageTensors = std::numeric_limits<uint32_t>::max();
            tensorProps.current->maxDescriptorSetUpdateAfterBindStorageTensors = std::numeric_limits<uint32_t>::max();
            tensorProps.current->maxPerStageDescriptorUpdateAfterBindStorageTensors =
                std::numeric_limits<uint32_t>::max();
            tensorProps.current->shaderStorageTensorArrayNonUniformIndexingNative = false;
            tensorProps.current->shaderTensorSupportedStages =
                VkShaderStageFlagBits::VK_SHADER_STAGE_COMPUTE_BIT; // EL only supports tensors in compute shaders

            insertType(tensorProps);
        }
    }

    static VkResult VKAPI_CALL vkCreateDevice(VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo *createInfo,
                                              const VkAllocationCallbacks *allocator, VkDevice *device) {
        auto originCreateInfoChain = dumpVkStructureList(createInfo);

        VkDeviceCreateInfo newCreateInfo{*createInfo};
        findAndRemoveType<VkPhysicalDeviceTensorFeaturesARM>(&newCreateInfo,
                                                             VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSOR_FEATURES_ARM);
        auto result = VulkanLayerImpl::vkCreateDevice(physicalDevice, &newCreateInfo, allocator, device);

        loadVkStructureList(const_cast<VkDeviceCreateInfo *>(createInfo), originCreateInfoChain);
        return result;
    }

    static VkResult VKAPI_CALL vkSetDebugUtilsObjectNameEXT(VkDevice device,
                                                            const VkDebugUtilsObjectNameInfoEXT *pNameInfo) {
        auto handle = VulkanLayerImpl::getHandle(device);
        switch (pNameInfo->objectType) {
        case VK_OBJECT_TYPE_TENSOR_ARM: {
            auto tensorARM = reinterpret_cast<TensorARM *>(pNameInfo->objectHandle);
            VkDebugUtilsObjectNameInfoEXT newNameInfo{
                VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT, pNameInfo->pNext, VK_OBJECT_TYPE_BUFFER,
                reinterpret_cast<uint64_t>(tensorARM->getTensorBuffer()), pNameInfo->pObjectName};
            return handle->loader->vkSetDebugUtilsObjectNameEXT(device, &newNameInfo);
        } break;
        case VK_OBJECT_TYPE_TENSOR_VIEW_ARM:
            break;
        default:
            return handle->loader->vkSetDebugUtilsObjectNameEXT(device, pNameInfo);
        }
        return VK_SUCCESS;
    }

    static std::shared_ptr<DeviceMemory> getHandle(const VkDeviceMemory handle) {
        scopedMutex l(globalMutex);
        return deviceMemoryMap[handle];
    }

    static inline std::unordered_map<std::size_t, std::vector<uint32_t>> spirvCache;
    static inline std::unordered_map<VkDeviceMemory, std::shared_ptr<DeviceMemory>> deviceMemoryMap;
};
} // namespace mlsdk::el::layer

/*******************************************************************************
 * External functions
 *******************************************************************************/

extern "C" {
using namespace mlsdk::el::layer;

LAYER_EXPORT PFN_vkVoidFunction VKAPI_CALL tensorGetInstanceProcAddr(VkInstance instance, const char *name) {
    return TensorLayer::vkGetInstanceProcAddr(instance, name);
}

LAYER_EXPORT PFN_vkVoidFunction VKAPI_CALL tensorGetDeviceProcAddr(VkDevice device, const char *name) {
    return TensorLayer::vkGetDeviceProcAddr(device, name);
}

PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char *name) {
    return TensorLayer::vkGetInstanceProcAddr(instance, name);
}

PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char *name) {
    return TensorLayer::vkGetDeviceProcAddr(device, name);
}

VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(uint32_t *pPropertyCount, VkLayerProperties *pProperties) {
    return TensorLayer::vkEnumerateInstanceLayerProperties(pPropertyCount, pProperties);
}

VkResult VKAPI_CALL vkEnumerateDeviceLayerProperties(VkPhysicalDevice physicalDevice, uint32_t *pPropertyCount,
                                                     VkLayerProperties *pProperties) {
    return TensorLayer::vkEnumerateDeviceLayerProperties(physicalDevice, pPropertyCount, pProperties);
}

VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(const char *pLayerName, uint32_t *pPropertyCount,
                                                           VkExtensionProperties *pProperties) {
    return TensorLayer::vkEnumerateInstanceExtensionProperties(pLayerName, pPropertyCount, pProperties);
}

VkResult VKAPI_CALL vkEnumerateDeviceExtensionProperties(VkPhysicalDevice physicalDevice, const char *pLayerName,
                                                         uint32_t *pPropertyCount, VkExtensionProperties *pProperties) {
    return TensorLayer::vkEnumerateDeviceExtensionProperties(physicalDevice, pLayerName, pPropertyCount, pProperties);
}
}
