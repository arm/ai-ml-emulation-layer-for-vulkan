/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "mlel/log.hpp"
#include "mlel/vulkan_allocator.hpp"

#include <vulkan/vk_layer.h>
#include <vulkan/vulkan.hpp>

#include <algorithm>
#include <array>
#include <initializer_list>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

/*****************************************************************************
 * Vulkan Layer
 *****************************************************************************/

namespace mlsdk::el::layer {

namespace {

#if defined(_WIN64) || defined(_WIN32)
#    define LAYER_EXPORT _declspec(dllexport)
#else
#    define LAYER_EXPORT
#endif

mlsdk::el::log::Log layerLog("VMEL_COMMON_SEVERITY", "Layer");
} // namespace

template <typename T> struct LinkedType {
    LinkedType() : parent(nullptr), current(nullptr) {}
    LinkedType(VkBaseOutStructure *parent, T *current) : parent(parent), current(current) {}
    VkBaseOutStructure *parent;
    T *current;
};

template <class T, class U> static T *findTypeMutable(U *ptr, VkStructureType type) {
    auto p = reinterpret_cast<VkBaseOutStructure *>(ptr);

    while (p != nullptr) {
        if (p->sType == type) {
            return reinterpret_cast<T *>(p);
        }
        p = p->pNext;
    }

    return nullptr;
}

template <class T, class U> static const T *findType(const U *ptr, const VkStructureType type) {
    auto p = reinterpret_cast<const VkBaseInStructure *>(ptr);

    while (p != nullptr) {
        if (p->sType == type) {
            return reinterpret_cast<const T *>(p);
        }
        p = p->pNext;
    }

    return nullptr;
}

template <typename T, class U> static LinkedType<T> findLinkedType(U *ptr, const VkStructureType type) {
    auto p = reinterpret_cast<VkBaseOutStructure *>(ptr);

    VkBaseOutStructure *previous = nullptr;
    while (p != nullptr) {
        if (p->sType == type) {
            return {previous, reinterpret_cast<T *>(p)};
        }
        previous = p;
        p = p->pNext;
    }

    return {};
}

template <class T, class U> static T getType(const U *ptr, const VkStructureType type, const T &defaultType) {
    auto p = findType<T>(ptr, type);
    if (p) {
        return *p;
    } else {
        return defaultType;
    }
}

template <class T, class U> static const T *removeType(U *ptr, const VkStructureType type) {
    auto p = reinterpret_cast<VkBaseOutStructure *>(ptr);
    // remove VkStructureType from the pNext chain, header will not be checked
    while (p) {
        if (p->pNext && p->pNext->sType == type) {
            auto typeToRemove = reinterpret_cast<const T *>(p->pNext);
            p->pNext = p->pNext->pNext;
            return typeToRemove;
        }
        p = p->pNext;
    }
    return nullptr;
}

template <class T, class U> static void appendType(T *list, U *node) {
    auto p = reinterpret_cast<VkBaseOutStructure *>(list);
    while (p->pNext) {
        p = p->pNext;
    }
    p->pNext = reinterpret_cast<VkBaseOutStructure *>(node);
    node->pNext = nullptr;
}

template <typename T, class U> static LinkedType<T> findAndRemoveType(U *ptr, const VkStructureType type) {
    auto node = findLinkedType<T>(ptr, type);
    if (node.current == nullptr) {
        return {};
    }
    node.parent->pNext = reinterpret_cast<VkBaseOutStructure *>(node.current->pNext);
    return node;
}

template <typename T> static void insertType(LinkedType<T> &node) {
    if (node.parent != nullptr) {
        node.parent->pNext = reinterpret_cast<VkBaseOutStructure *>(node.current);
    }
}

template <class T> static std::vector<const VkBaseOutStructure *> dumpVkStructureList(const T *list) {
    std::vector<const VkBaseOutStructure *> vec;
    auto p = reinterpret_cast<const VkBaseOutStructure *>(list);
    while (p) {
        p = p->pNext;
        vec.emplace_back(p);
    }
    return vec;
}

template <class T> static void loadVkStructureList(T *list, const std::vector<const VkBaseOutStructure *> &vec) {
    auto tail = reinterpret_cast<VkBaseOutStructure *>(list);
    for (auto &pNode : vec) {
        tail->pNext = const_cast<VkBaseOutStructure *>(pNode);
        tail = tail->pNext;
    }
}

template <typename DispatchableType> void checkDispatchable(DispatchableType) {
    static_assert(
        std::is_same<DispatchableType, VkInstance>::value || std::is_same<DispatchableType, VkPhysicalDevice>::value ||
            std::is_same<DispatchableType, VkDevice>::value || std::is_same<DispatchableType, VkQueue>::value ||
            std::is_same<DispatchableType, VkCommandBuffer>::value,
        "unrecognized dispatchable type");
}

template <typename DispatchableType> void setDispatchTableKey(DispatchableType dispatchable, void *data) {
    checkDispatchable(dispatchable);
    *reinterpret_cast<void **>(dispatchable) = data;
}

template <typename DispatchableType> void *getDispatchTableKey(DispatchableType dispatchable) {
    checkDispatchable(dispatchable);
    return *reinterpret_cast<void **>(dispatchable);
}

/*****************************************************************************
 * Loader
 *****************************************************************************/

class Loader {
  public:
    Loader(const Loader &_loader) : loader{_loader.loader} {}

    explicit Loader(std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader) : loader{_loader} {}

    explicit Loader(const VkAllocationCallbacks *_callbacks, VkInstance _instance, PFN_vkGetInstanceProcAddr _gipr)
        : loader{allocateObject<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic>(_callbacks, _instance, _gipr),
                 [=](VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic *toDelete) {
                     destroyObject(_callbacks, toDelete);
                 }} {}

    explicit Loader(const VkAllocationCallbacks *_callbacks, VkInstance _instance, PFN_vkGetInstanceProcAddr _gipr,
                    VkDevice _device, PFN_vkGetDeviceProcAddr _gdpr)
        : Loader(_callbacks, _instance, _gipr) {
        loader->vkGetDeviceProcAddr = _gdpr;
        loader->init(VULKAN_HPP_NAMESPACE::Device(_device));
    }

    std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> loader;
};

/*****************************************************************************
 * Instance
 *****************************************************************************/

class Instance : public Loader {
  public:
    explicit Instance(VkInstance _instance, PFN_vkGetInstanceProcAddr _gipr, const VkAllocationCallbacks *_callbacks,
                      PFN_vkGetInstanceProcAddr _nextGetInstanceProcAddr,
                      PFN_GetPhysicalDeviceProcAddr _nextGetPhysicalDeviceProcAddr)
        : Loader(_callbacks, _instance, _gipr), instance{_instance}, callbacks{_callbacks},
          nextGetInstanceProcAddr{_nextGetInstanceProcAddr}, nextGetPhysicalDeviceProcAddr{
                                                                 _nextGetPhysicalDeviceProcAddr} {}

    VkInstance instance;
    const VkAllocationCallbacks *callbacks;
    PFN_vkGetInstanceProcAddr nextGetInstanceProcAddr;
    PFN_GetPhysicalDeviceProcAddr nextGetPhysicalDeviceProcAddr;
};

/*****************************************************************************
 * PhysicalDevice
 *****************************************************************************/

class PhysicalDevice : public Loader {
  public:
    explicit PhysicalDevice(std::shared_ptr<Instance> _instance, VkPhysicalDevice _physicalDevice)
        : Loader(*_instance), instance{_instance}, physicalDevice{_physicalDevice} {}

    std::shared_ptr<Instance> instance;
    VkPhysicalDevice physicalDevice;
};

/*****************************************************************************
 * Device
 *****************************************************************************/

class Device : public Loader {
  public:
    explicit Device(std::shared_ptr<PhysicalDevice> _physicalDevice, VkDevice _device, PFN_vkGetInstanceProcAddr _gipr,
                    PFN_vkGetDeviceProcAddr _gdpr, const VkAllocationCallbacks *_callbacks)
        : Loader(_callbacks, _physicalDevice->instance->instance, _gipr, _device, _gdpr),
          physicalDevice{_physicalDevice}, device{_device}, callbacks{_callbacks} {}

    std::shared_ptr<PhysicalDevice> physicalDevice;
    VkDevice device;
    const VkAllocationCallbacks *callbacks;
};

/*****************************************************************************
 * CommandBuffer
 *****************************************************************************/

class CommandBuffer : public Loader {
  public:
    explicit CommandBuffer(std::shared_ptr<Device> _device, VkCommandBuffer _commandBuffer, VkCommandPool _commandPool)
        : Loader(_device->loader), device{_device}, commandBuffer{_commandBuffer}, commandPool{_commandPool} {}

    virtual ~CommandBuffer() {
        if (secondaryCommandBuffer != VK_NULL_HANDLE) {
            loader->vkFreeCommandBuffers(device->device, commandPool, 1, &secondaryCommandBuffer);
        }
    }

    void beginSecondaryCommandBuffer() {
        if (secondaryCommandBuffer == VK_NULL_HANDLE) {
            secondaryCommandBuffer = createSecondaryCommandBuffer();
        }
        VkCommandBufferInheritanceInfo commandBufferInheritanceInfo{};
        commandBufferInheritanceInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
        commandBufferInheritanceInfo.pNext = nullptr;
        const VkCommandBufferBeginInfo beginInfo = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, // type
            nullptr,                                     // next
            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, // flag
            &commandBufferInheritanceInfo                // pInheritanceInfo
        };

        if (loader->vkBeginCommandBuffer(secondaryCommandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("Failed to begin recording secondary command buffer!");
        }
    }

    void endAndSubmitSecondaryCommandBuffer() {
        if (loader->vkEndCommandBuffer(secondaryCommandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to end secondary command buffer!");
        }
        loader->vkCmdExecuteCommands(commandBuffer, 1, &secondaryCommandBuffer);
    }

    std::shared_ptr<Device> device;
    VkCommandBuffer commandBuffer;
    VkCommandPool commandPool;
    VkCommandBuffer secondaryCommandBuffer = VK_NULL_HANDLE;

    VkPipelineLayout pipelineLayout = {};
    std::map<uint32_t, VkDescriptorSet> descriptorSets;

  private:
    VkCommandBuffer createSecondaryCommandBuffer() const {
        VkCommandBuffer cmd;
        const VkCommandBufferAllocateInfo allocInfo = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, // type
            nullptr,                                        // next
            commandPool,                                    // commandPool
            VK_COMMAND_BUFFER_LEVEL_SECONDARY,              // level
            1                                               // count
        };
        if (loader->vkAllocateCommandBuffers(device->device, &allocInfo, &cmd) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate secondary command buffer!");
        }
        // initialize internal created dispatchable command buffer with device dispatch table key
        setDispatchTableKey(cmd, getDispatchTableKey(device->device));
        return cmd;
    }
};

/*****************************************************************************
 * ShaderModule
 *****************************************************************************/

class ShaderModule {
  public:
    explicit ShaderModule(const VkShaderModuleCreateInfo *_info)
        : info{*_info}, code{_info->pCode, _info->pCode + _info->codeSize / sizeof(uint32_t)} {}

    const VkShaderModuleCreateInfo info;
    const std::vector<uint32_t> code;
};

/*****************************************************************************
 * DescriptorSetLayout
 *****************************************************************************/

class DescriptorSetLayout {
  public:
    explicit DescriptorSetLayout(const VkDescriptorSetLayoutCreateInfo *_info)
        : info{*_info}, bindings{createBindings()} {}

    const VkDescriptorSetLayoutCreateInfo info;
    const std::map<uint32_t, VkDescriptorSetLayoutBinding> bindings;

  private:
    std::map<uint32_t, VkDescriptorSetLayoutBinding> createBindings() const {
        std::map<uint32_t, VkDescriptorSetLayoutBinding> newBindings;

        for (uint32_t i = 0; i < info.bindingCount; i++) {
            newBindings[info.pBindings[i].binding] = info.pBindings[i];
        }

        return newBindings;
    }
};

/*****************************************************************************
 * DescriptorSet
 *****************************************************************************/

class DescriptorSet {
  public:
    explicit DescriptorSet(const std::shared_ptr<DescriptorSetLayout> &_descriptorSetLayout)
        : descriptorSetLayout{_descriptorSetLayout} {}

    std::shared_ptr<DescriptorSetLayout> descriptorSetLayout;
};

/**************************************************************************
 * PipelineLayout
 **************************************************************************/

class PipelineLayout {
  public:
    explicit PipelineLayout(const VkPipelineLayoutCreateInfo *_info,
                            const std::vector<std::shared_ptr<DescriptorSetLayout>> &descriptorSetLayouts)
        : info{*_info}, descriptorSetLayouts{descriptorSetLayouts}, pushConstants{_info->pPushConstantRanges,
                                                                                  _info->pPushConstantRanges +
                                                                                      _info->pushConstantRangeCount} {}

    const VkPipelineLayoutCreateInfo info;
    const std::vector<std::shared_ptr<DescriptorSetLayout>> descriptorSetLayouts;
    const std::vector<VkPushConstantRange> pushConstants;
};

/*****************************************************************************
 * VulkanLayer
 *****************************************************************************/

template <const auto &layerProperties, const auto &extensions, const auto &requiredExtensions,
          typename InstanceImpl = Instance, typename PhysicalDeviceImpl = PhysicalDevice, typename DeviceImpl = Device>
class VulkanLayer {
  public:
    VulkanLayer() = delete;

    // Enable heterogeneous lookup, e.g. const char*
    using vTable = std::map<std::string, PFN_vkVoidFunction, std::less<>>;

    /*******************************************************************************
     * GetProcAddress
     *******************************************************************************/

    static PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char *name) {
        static const vTable vtable = {
            // Instance functions
            {"vkGetInstanceProcAddr", PFN_vkVoidFunction(vkGetInstanceProcAddr)},
            {"vkCreateInstance", PFN_vkVoidFunction(vkCreateInstance)},
            {"vkDestroyInstance", PFN_vkVoidFunction(vkDestroyInstance)},
            {"vkEnumeratePhysicalDevices", PFN_vkVoidFunction(vkEnumeratePhysicalDevices)},

            // PhysicalDevice functions
            {"vkEnumerateDeviceExtensionProperties", PFN_vkVoidFunction(vkEnumerateDeviceExtensionProperties)},
            {"vkCreateDevice", PFN_vkVoidFunction(vkCreateDevice)},
            {"vk_layerGetPhysicalDeviceProcAddr", PFN_vkVoidFunction(vk_layerGetPhysicalDeviceProcAddr)}};

        if (auto it = vtable.find(name); it != vtable.end()) {
            return it->second;
        }

        auto handle = getHandle(instance);
        if (handle != nullptr) {
            return handle->loader->vkGetInstanceProcAddr(instance, name);
        }

        return nullptr;
    }

    static PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char *name) {
        static const vTable vtable = {
            // Device functions
            {"vkGetDeviceProcAddr", PFN_vkVoidFunction(vkGetDeviceProcAddr)},
            {"vkDestroyDevice", PFN_vkVoidFunction(vkDestroyDevice)},

            // DescriptorSetLayout
            {"vkCreateDescriptorSetLayout", PFN_vkVoidFunction(vkCreateDescriptorSetLayout)},
            {"vkDestroyDescriptorSetLayout", PFN_vkVoidFunction(vkDestroyDescriptorSetLayout)},

            // PipelineLayout
            {"vkCreatePipelineLayout", PFN_vkVoidFunction(vkCreatePipelineLayout)},
            {"vkDestroyPipelineLayout", PFN_vkVoidFunction(vkDestroyPipelineLayout)},

            // Device queue
            {"vkGetDeviceQueue", PFN_vkVoidFunction(vkGetDeviceQueue)},
            {"vkGetDeviceQueue2", PFN_vkVoidFunction(vkGetDeviceQueue2)},

            // Command buffers
            {"vkAllocateCommandBuffers", PFN_vkVoidFunction(vkAllocateCommandBuffers)},
            {"vkFreeCommandBuffers", PFN_vkVoidFunction(vkFreeCommandBuffers)}};

        if (auto it = vtable.find(name); it != vtable.end()) {
            return it->second;
        }

        auto handle = getHandle(device);
        if (handle != nullptr) {
            return handle->loader->vkGetDeviceProcAddr(device, name);
        }

        return nullptr;
    }

    static PFN_vkVoidFunction VKAPI_CALL vk_layerGetPhysicalDeviceProcAddr(VkInstance instance, const char *name) {
        static const vTable vtable = {
            {"vk_layerGetPhysicalDeviceProcAddr", PFN_vkVoidFunction(vk_layerGetPhysicalDeviceProcAddr)},
            // PhysicalDevice functions
            {"vkCreateDevice", PFN_vkVoidFunction(vkCreateDevice)}};

        if (auto it = vtable.find(name); it != vtable.end()) {
            return it->second;
        }

        if (instance == VK_NULL_HANDLE) {
            return nullptr;
        }

        if (auto handle = getHandle(instance)) {
            PFN_vkGetInstanceProcAddr getInstanceProcAddr = nullptr;
            PFN_GetPhysicalDeviceProcAddr nextPhysicalDeviceProcAddr = nullptr;
            if (handle)
                getInstanceProcAddr = handle->nextGetInstanceProcAddr;

            // Default to propagating through Instance Proc Address
            if (getInstanceProcAddr) {
                nextPhysicalDeviceProcAddr = reinterpret_cast<PFN_GetPhysicalDeviceProcAddr>(
                    getInstanceProcAddr(instance, "vk_layerGetPhysicalDeviceProcAddr"));
            }
            // Fallback to using stored next physical device proc address
            if (!nextPhysicalDeviceProcAddr) {
                nextPhysicalDeviceProcAddr = handle->nextGetPhysicalDeviceProcAddr;
            }

            // Block against recursive calls
            if (nextPhysicalDeviceProcAddr && nextPhysicalDeviceProcAddr != vk_layerGetPhysicalDeviceProcAddr) {
                return nextPhysicalDeviceProcAddr(instance, name);
            }
        }
        return nullptr;
    }

    /**************************************************************************
     * Introspection functions
     **************************************************************************/
    static VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(uint32_t *pPropertyCount,
                                                                  VkLayerProperties *pProperties) {
        if (pProperties == nullptr) {
            *pPropertyCount = 1;
            return VK_SUCCESS;
        }
        std::copy(&layerProperties, &layerProperties + 1, pProperties);
        *pPropertyCount = 1;
        return VK_SUCCESS;
    }

    static VkResult VKAPI_CALL vkEnumerateDeviceLayerProperties(VkPhysicalDevice, uint32_t *pPropertyCount,
                                                                VkLayerProperties *pProperties) {
        return vkEnumerateInstanceLayerProperties(pPropertyCount, pProperties);
    }

    static VkResult VKAPI_CALL vkEnumerateDeviceExtensionProperties(VkPhysicalDevice physicalDevice,
                                                                    const char *layerName, uint32_t *propertyCount,
                                                                    VkExtensionProperties *properties) {
        if (layerName && std::string(layerName) == layerProperties.layerName) {
            if (properties == nullptr || extensions.size() == 0) {
                *propertyCount = static_cast<uint32_t>(extensions.size());
                return VK_SUCCESS;
            }

            const uint32_t copySize = std::min(*propertyCount, static_cast<uint32_t>(extensions.size()));
            std::copy(extensions.begin(), extensions.begin() + copySize, properties);
            *propertyCount = copySize;
            if (copySize < extensions.size()) {
                return VK_INCOMPLETE;
            }

            return VK_SUCCESS;
        }

        auto handle = getHandle(physicalDevice);
        if (handle == nullptr) {
            return VK_ERROR_INVALID_EXTERNAL_HANDLE;
        }

        uint32_t lowerLayerPropertyCount = 0;
        auto res = handle->loader->vkEnumerateDeviceExtensionProperties(physicalDevice, layerName,
                                                                        &lowerLayerPropertyCount, nullptr);
        if (res != VK_SUCCESS) {
            return res;
        }
        if (!properties) {
            *propertyCount = lowerLayerPropertyCount + static_cast<uint32_t>(extensions.size());
        } else {
            res = handle->loader->vkEnumerateDeviceExtensionProperties(physicalDevice, layerName,
                                                                       &lowerLayerPropertyCount, properties);

            if (res != VK_SUCCESS) {
                return res;
            }
            for (size_t i = 0; i < extensions.size(); ++i) {
                properties[lowerLayerPropertyCount + i] = extensions[i];
            }
        }
        return VK_SUCCESS;
    }

    /*******************************************************************************
     * Instance
     *******************************************************************************/

    static VkResult VKAPI_CALL vkCreateInstance(const VkInstanceCreateInfo *createInfo,
                                                const VkAllocationCallbacks *allocator, VkInstance *instance) {
        auto layerCreateInfo = findInstanceCreateInfo(createInfo);
        if (layerCreateInfo == nullptr) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        auto getInstanceProcAddr = layerCreateInfo->u.pLayerInfo->pfnNextGetInstanceProcAddr;
        auto getNextPhysicalDeviceProcAddr = layerCreateInfo->u.pLayerInfo->pfnNextGetPhysicalDeviceProcAddr;

        layerCreateInfo->u.pLayerInfo = layerCreateInfo->u.pLayerInfo->pNext;

        auto createInstance = reinterpret_cast<PFN_vkCreateInstance>(getInstanceProcAddr(nullptr, "vkCreateInstance"));
        auto ret = createInstance(createInfo, allocator, instance);
        if (ret != VK_SUCCESS) {
            return ret;
        }

        {
            scopedMutex l(globalMutex);
            instanceMap[*instance] =
                std::allocate_shared<InstanceImpl>(Allocator<InstanceImpl>{allocator}, *instance, getInstanceProcAddr,
                                                   allocator, getInstanceProcAddr, getNextPhysicalDeviceProcAddr);
        }

        return VK_SUCCESS;
    }

    static void VKAPI_CALL vkDestroyInstance(VkInstance instance, const VkAllocationCallbacks *allocator) {
        auto handle = getHandle(instance);
        handle->loader->vkDestroyInstance(instance, allocator);

        {
            scopedMutex l(globalMutex);

            instanceMap.erase(instance);

            // Erase physical devices referencing handle
            for (auto it = physicalDeviceMap.begin(); it != physicalDeviceMap.end();) {
                if (it->second->instance == handle) {
                    it = physicalDeviceMap.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }

    static VkResult VKAPI_CALL vkEnumeratePhysicalDevices(VkInstance instance, uint32_t *physicalDeviceCount,
                                                          VkPhysicalDevice *physicalDevices) {
        auto handle = getHandle(instance);

        auto res = handle->loader->vkEnumeratePhysicalDevices(instance, physicalDeviceCount, physicalDevices);
        if (res != VK_SUCCESS) {
            return res;
        }

        {
            scopedMutex l(globalMutex);

            if (physicalDevices != nullptr) {
                for (uint32_t i = 0; i < *physicalDeviceCount; i++) {
                    physicalDeviceMap[physicalDevices[i]] = std::allocate_shared<PhysicalDeviceImpl>(
                        Allocator<PhysicalDeviceImpl>{handle->callbacks}, handle, physicalDevices[i]);
                }
            }
        }

        return VK_SUCCESS;
    }

    /*******************************************************************************
     * PhysicalDevice
     *******************************************************************************/

    static VkResult VKAPI_CALL vkCreateDevice(VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo *createInfo,
                                              const VkAllocationCallbacks *allocator, VkDevice *device) {
        auto handle = getHandle(physicalDevice);
        if (handle == nullptr) {
            return VK_ERROR_INVALID_EXTERNAL_HANDLE;
        }

        // Search if at least one extension from 'extensions' is present in 'createInfo->ppEnabledExtensionNames'
        std::vector<const char *> deviceExtensions{createInfo->ppEnabledExtensionNames,
                                                   createInfo->ppEnabledExtensionNames +
                                                       createInfo->enabledExtensionCount};

        const bool hasExtension =
            std::any_of(extensions.begin(), extensions.end(), [deviceExtensions](const auto &left) {
                return std::any_of(deviceExtensions.begin(), deviceExtensions.end(),
                                   [&left](const std::string &right) { return right == left.extensionName; });
            });

        if (hasExtension) {
            // Remove layer implemented extensions
            deviceExtensions.erase(
                std::remove_if(deviceExtensions.begin(), deviceExtensions.end(), [](const std::string &left) {
                    return std::any_of(extensions.begin(), extensions.end(),
                                       [left](const auto &right) { return left == right.extensionName; });
                }));
            VkPhysicalDeviceProperties physicalDeviceProperties;
            handle->loader->vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);
            if (physicalDeviceProperties.apiVersion < VK_MAKE_VERSION(1, 3, 0)) {
                layerLog(mlsdk::el::log::Severity::Error)
                    << "ML Emulation Layer for Vulkan requires at least API version 1.3" << std::endl;
                return VK_ERROR_INITIALIZATION_FAILED;
            }
        }

        // check layer required extensions and add to device create info.
        uint32_t count = 0;
        handle->loader->vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, nullptr);
        std::vector<VkExtensionProperties> supportedExtensions(count);
        handle->loader->vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count,
                                                             supportedExtensions.data());

        /* coverity[dead_error_begin] */
        for (const auto &requiredExt : requiredExtensions) {
            auto it = std::find_if(supportedExtensions.begin(), supportedExtensions.end(),
                                   [&requiredExt](const auto &supportExt) {
                                       return std::strcmp(supportExt.extensionName, requiredExt.extensionName) == 0;
                                   });
            if (it == supportedExtensions.end()) {
                layerLog(mlsdk::el::log::Severity::Error)
                    << "ML Emulation Layer for Vulkan requires extension: " << requiredExt.extensionName << std::endl;
                return VK_ERROR_FEATURE_NOT_PRESENT;
            }
            auto it2 =
                std::find_if(deviceExtensions.begin(), deviceExtensions.end(),
                             [&requiredExt](const std::string &ext) { return ext == requiredExt.extensionName; });
            if (it2 == deviceExtensions.end()) {
                deviceExtensions.emplace_back(requiredExt.extensionName);
            }
        }

        auto layerCreateInfo = findDeviceLayerCreateInfo(createInfo);
        if (layerCreateInfo == nullptr) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        // query device feature
        VkPhysicalDeviceVulkan11Features queryVulkan11Feature{};
        queryVulkan11Feature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        queryVulkan11Feature.pNext = nullptr;
        VkPhysicalDeviceVulkan12Features queryVulkan12Feature{};
        queryVulkan12Feature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        queryVulkan12Feature.pNext = &queryVulkan11Feature;
        VkPhysicalDeviceVulkan13Features queryVulkan13Feature{};
        queryVulkan13Feature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        queryVulkan13Feature.pNext = &queryVulkan12Feature;
        VkPhysicalDeviceFeatures2 queryFeatures2{};
        queryFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        queryFeatures2.pNext = &queryVulkan13Feature;
        handle->loader->vkGetPhysicalDeviceFeatures2(physicalDevice, &queryFeatures2);

        // store origin deviceCreateInfo chain and modify to enable features
        auto originCreateInfoChain = dumpVkStructureList(createInfo);
        VkDeviceCreateInfo newCreateInfo{
            createInfo->sType,
            createInfo->pNext,
            createInfo->flags,
            createInfo->queueCreateInfoCount,
            createInfo->pQueueCreateInfos,
            createInfo->enabledLayerCount,
            createInfo->ppEnabledLayerNames,
            static_cast<uint32_t>(deviceExtensions.size()), // enabledExtensionCount
            deviceExtensions.data(),                        // ppEnabledExtensionNames
            nullptr,                                        // pEnabledFeatures
        };

        auto pDeviceFeature2 =
            removeType<VkPhysicalDeviceFeatures2>(&newCreateInfo, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2);
        VkPhysicalDeviceFeatures deviceFeatures{};
        VkPhysicalDeviceFeatures2 layerPhysicalDeviceFeatures2{};
        if (pDeviceFeature2) {
            // Copy feature flags from VkPhysicalDeviceFeatures2 in createInfo->pNext chain
            layerPhysicalDeviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            layerPhysicalDeviceFeatures2 = *pDeviceFeature2;
            layerPhysicalDeviceFeatures2.pNext = nullptr;
            layerPhysicalDeviceFeatures2.features.shaderInt64 = queryFeatures2.features.shaderInt64;
            layerPhysicalDeviceFeatures2.features.shaderInt16 = queryFeatures2.features.shaderInt16;
            layerPhysicalDeviceFeatures2.features.shaderFloat64 = queryFeatures2.features.shaderFloat64;
            appendType(&newCreateInfo, &layerPhysicalDeviceFeatures2);
        } else if (createInfo->pEnabledFeatures) {
            // Copy feature flags from createInfo->pEnabledFeatures
            // Using an `else if` is fine because the vulkan specification states pEnableFeatures must be NULL if the
            // pNext chain includes a VkPhysicalDeviceFeatures2
            deviceFeatures = *(createInfo->pEnabledFeatures);
            deviceFeatures.shaderInt64 = queryFeatures2.features.shaderInt64;
            deviceFeatures.shaderInt16 = queryFeatures2.features.shaderInt16;
            deviceFeatures.shaderFloat64 = queryFeatures2.features.shaderFloat64;
            newCreateInfo.pEnabledFeatures = &deviceFeatures;
        }

        auto pDeviceFeature11 = removeType<VkPhysicalDeviceVulkan11Features>(
            &newCreateInfo, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES);
        VkPhysicalDeviceVulkan11Features layerVulkan11Feature{};
        layerVulkan11Feature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        if (pDeviceFeature11) {
            layerVulkan11Feature = *pDeviceFeature11;
            layerVulkan11Feature.pNext = nullptr;
        }
        layerVulkan11Feature.storageBuffer16BitAccess = queryVulkan11Feature.storageBuffer16BitAccess;
        appendType(&newCreateInfo, &layerVulkan11Feature);

        auto pDeviceFeature12 = removeType<VkPhysicalDeviceVulkan12Features>(
            &newCreateInfo, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES);
        VkPhysicalDeviceVulkan12Features layerVulkan12Feature{};
        layerVulkan12Feature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        if (pDeviceFeature12) {
            layerVulkan12Feature = *pDeviceFeature12;
            layerVulkan12Feature.pNext = nullptr;
        }
        layerVulkan12Feature.shaderInt8 = queryVulkan12Feature.shaderInt8;
        layerVulkan12Feature.shaderFloat16 = queryVulkan12Feature.shaderFloat16;
        layerVulkan12Feature.storageBuffer8BitAccess = queryVulkan12Feature.storageBuffer8BitAccess;
        layerVulkan12Feature.bufferDeviceAddress = queryVulkan12Feature.bufferDeviceAddress;
        layerVulkan12Feature.bufferDeviceAddressCaptureReplay = queryVulkan12Feature.bufferDeviceAddressCaptureReplay;
        layerVulkan12Feature.descriptorBindingStorageBufferUpdateAfterBind =
            queryVulkan12Feature.descriptorBindingStorageBufferUpdateAfterBind;
        appendType(&newCreateInfo, &layerVulkan12Feature);

        auto pDeviceFeature13 = removeType<VkPhysicalDeviceVulkan13Features>(
            &newCreateInfo, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES);
        VkPhysicalDeviceVulkan13Features layerVulkan13Feature{};
        layerVulkan13Feature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        if (pDeviceFeature13) {
            layerVulkan13Feature = *pDeviceFeature13;
            layerVulkan13Feature.pNext = nullptr;
        }
        layerVulkan13Feature.synchronization2 = queryVulkan13Feature.synchronization2;
        layerVulkan13Feature.maintenance4 = queryVulkan13Feature.maintenance4;
        appendType(&newCreateInfo, &layerVulkan13Feature);

        auto getInstanceProcAddr = layerCreateInfo->u.pLayerInfo->pfnNextGetInstanceProcAddr;
        auto getDeviceProcAddr = layerCreateInfo->u.pLayerInfo->pfnNextGetDeviceProcAddr;
        layerCreateInfo->u.pLayerInfo = layerCreateInfo->u.pLayerInfo->pNext;

        auto createDevice =
            reinterpret_cast<PFN_vkCreateDevice>(getInstanceProcAddr(handle->instance->instance, "vkCreateDevice"));
        if (createDevice == nullptr) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        auto res = createDevice(physicalDevice, &newCreateInfo, allocator, device);

        // recover deviceCreateInfo chain
        loadVkStructureList(const_cast<VkDeviceCreateInfo *>(createInfo), originCreateInfoChain);

        if (res != VK_SUCCESS) {
            return res;
        }

        {
            scopedMutex l(globalMutex);
            deviceMap[*device] = std::allocate_shared<DeviceImpl>(Allocator<DeviceImpl>{allocator}, handle, *device,
                                                                  getInstanceProcAddr, getDeviceProcAddr, allocator);
            bufferDeviceAddressCaptureReplayFeatMap[*device] = layerVulkan12Feature.bufferDeviceAddressCaptureReplay;
        }

        return VK_SUCCESS;
    }

    /*******************************************************************************
     * Device
     *******************************************************************************/

    static void VKAPI_CALL vkDestroyDevice(VkDevice device, const VkAllocationCallbacks *allocator) {
        auto handle = getHandle(device);
        handle->loader->vkDestroyDevice(device, allocator);

        {
            scopedMutex l(globalMutex);

            // Erase device map
            deviceMap.erase(device);

            // Erase queue maps
            for (auto it = queueMap.begin(); it != queueMap.end();) {
                if (it->second == handle) {
                    it = queueMap.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }

    /*******************************************************************************
     * DeviceQueue
     *******************************************************************************/

    static void VKAPI_CALL vkGetDeviceQueue(VkDevice device, uint32_t queueFamilyIndex, uint32_t queueIndex,
                                            VkQueue *queue) {
        auto handle = getHandle(device);
        handle->loader->vkGetDeviceQueue(device, queueFamilyIndex, queueIndex, queue);

        {
            scopedMutex l(globalMutex);
            queueMap[*queue] = std::move(handle);
        }
    }

    static void VKAPI_CALL vkGetDeviceQueue2(VkDevice device, const VkDeviceQueueInfo2 *queueInfo, VkQueue *queue) {
        auto handle = getHandle(device);
        handle->loader->vkGetDeviceQueue2(device, queueInfo, queue);

        {
            scopedMutex l(globalMutex);
            queueMap[*queue] = std::move(handle);
        }
    }

    /**************************************************************************
     * DescriptorSetLayout
     **************************************************************************/

    static VkResult VKAPI_CALL vkCreateDescriptorSetLayout(VkDevice device,
                                                           const VkDescriptorSetLayoutCreateInfo *createInfo,
                                                           const VkAllocationCallbacks *allocator,
                                                           VkDescriptorSetLayout *setLayout) {
        auto handle = getHandle(device);
        auto res = handle->loader->vkCreateDescriptorSetLayout(device, createInfo, allocator, setLayout);

        if (res == VK_SUCCESS) {
            scopedMutex l(globalMutex);
            descriptorSetLayoutMap[*setLayout] = std::make_shared<DescriptorSetLayout>(createInfo);
        }

        return res;
    }

    static void VKAPI_CALL vkDestroyDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout descriptorSetLayout,
                                                        const VkAllocationCallbacks *allocator) {
        auto handle = getHandle(device);
        handle->loader->vkDestroyDescriptorSetLayout(device, descriptorSetLayout, allocator);

        {
            scopedMutex l(globalMutex);
            descriptorSetLayoutMap.erase(descriptorSetLayout);
        }
    }

    /**************************************************************************
     * PipelineLayout
     **************************************************************************/

    static VkResult VKAPI_CALL vkCreatePipelineLayout(VkDevice device, const VkPipelineLayoutCreateInfo *createInfo,
                                                      const VkAllocationCallbacks *allocator,
                                                      VkPipelineLayout *pipelineLayout) {
        auto handle = getHandle(device);
        auto res = handle->loader->vkCreatePipelineLayout(device, createInfo, allocator, pipelineLayout);

        if (res == VK_SUCCESS) {
            scopedMutex l(globalMutex);
            pipelineLayoutMap[*pipelineLayout] = std::make_shared<PipelineLayout>(
                createInfo, getHandle(createInfo->pSetLayouts, createInfo->setLayoutCount));
        }

        return res;
    }

    static void VKAPI_CALL vkDestroyPipelineLayout(VkDevice device, VkPipelineLayout pipelineLayout,
                                                   const VkAllocationCallbacks *allocator) {
        auto handle = getHandle(device);
        handle->loader->vkDestroyPipelineLayout(device, pipelineLayout, allocator);

        {
            scopedMutex l(globalMutex);
            pipelineLayoutMap.erase(pipelineLayout);
        }
    }

    /*******************************************************************************
     * CommandBuffers
     *******************************************************************************/

    static VkResult VKAPI_CALL vkAllocateCommandBuffers(VkDevice device,
                                                        const VkCommandBufferAllocateInfo *allocateInfo,
                                                        VkCommandBuffer *commandBuffers) {
        auto handle = getHandle(device);
        auto result = handle->loader->vkAllocateCommandBuffers(device, allocateInfo, commandBuffers);
        if (result != VK_SUCCESS) {
            return result;
        }

        {
            scopedMutex l(globalMutex);

            for (unsigned int i = 0; i < allocateInfo->commandBufferCount; i++) {
                commandBufferMap[commandBuffers[i]] =
                    std::make_shared<CommandBuffer>(handle, commandBuffers[i], allocateInfo->commandPool);
            }
        }

        return VK_SUCCESS;
    }

    static void VKAPI_CALL vkFreeCommandBuffers(VkDevice device, VkCommandPool commandPool, uint32_t commandBufferCount,
                                                const VkCommandBuffer *commandBuffers) {
        auto handle = getHandle(device);
        handle->loader->vkFreeCommandBuffers(device, commandPool, commandBufferCount, commandBuffers);

        {
            scopedMutex l(globalMutex);

            for (unsigned int i = 0; i < commandBufferCount; i++) {
                commandBufferMap.erase(commandBuffers[i]);
            }
        }
    }

  protected:
    static std::shared_ptr<InstanceImpl> getHandle(const VkInstance handle) {
        scopedMutex l(globalMutex);
        return instanceMap[handle];
    }

    static std::shared_ptr<PhysicalDeviceImpl> getHandle(const VkPhysicalDevice handle) {
        scopedMutex l(globalMutex);
        return physicalDeviceMap[handle];
    }

    static std::shared_ptr<DeviceImpl> getHandle(const VkDevice handle) {
        scopedMutex l(globalMutex);
        return deviceMap[handle];
    }

    static std::shared_ptr<DescriptorSetLayout> getHandle(const VkDescriptorSetLayout handle) {
        scopedMutex l(globalMutex);
        return descriptorSetLayoutMap[handle];
    }

    static std::vector<std::shared_ptr<DescriptorSetLayout>> getHandle(const VkDescriptorSetLayout *handle,
                                                                       const uint32_t count) {
        scopedMutex l(globalMutex);

        std::vector<std::shared_ptr<DescriptorSetLayout>> descriptorSetLayouts;
        for (uint32_t i = 0; i < count; i++) {
            descriptorSetLayouts.emplace_back(descriptorSetLayoutMap[handle[i]]);
        }

        return descriptorSetLayouts;
    }

    static std::shared_ptr<PipelineLayout> getHandle(const VkPipelineLayout handle) {
        scopedMutex l(globalMutex);
        return pipelineLayoutMap[handle];
    }

    static std::shared_ptr<DeviceImpl> getHandle(const VkQueue handle) {
        scopedMutex l(globalMutex);
        return queueMap[handle];
    }

    static std::shared_ptr<CommandBuffer> getHandle(const VkCommandBuffer handle) {
        scopedMutex l(globalMutex);
        return commandBufferMap[handle];
    }

    static VkBool32 getBufferDeviceAddressCaptureReplayFeat(const VkDevice handle) {
        return bufferDeviceAddressCaptureReplayFeatMap[handle];
    }

    static VkLayerInstanceCreateInfo *findInstanceCreateInfo(const VkInstanceCreateInfo *createInfo) {
        auto info = reinterpret_cast<const VkLayerInstanceCreateInfo *>(createInfo->pNext);
        while (info != nullptr) {
            if (info->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO && info->function == VK_LAYER_LINK_INFO) {
                return const_cast<VkLayerInstanceCreateInfo *>(info);
            }

            info = reinterpret_cast<const VkLayerInstanceCreateInfo *>(info->pNext);
        }

        return nullptr;
    }

    static VkLayerDeviceCreateInfo *findDeviceLayerCreateInfo(const VkDeviceCreateInfo *createInfo) {
        auto info = reinterpret_cast<const VkLayerDeviceCreateInfo *>(createInfo->pNext);
        while (info != nullptr) {
            if (info->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO && info->function == VK_LAYER_LINK_INFO) {
                return const_cast<VkLayerDeviceCreateInfo *>(info);
            }

            info = reinterpret_cast<const VkLayerDeviceCreateInfo *>(info->pNext);
        }

        return nullptr;
    }

    static std::recursive_mutex globalMutex;
    using scopedMutex = std::lock_guard<std::recursive_mutex>;

    static std::map<VkInstance, std::shared_ptr<InstanceImpl>> instanceMap;
    static std::map<VkPhysicalDevice, std::shared_ptr<PhysicalDeviceImpl>> physicalDeviceMap;
    static std::map<VkDevice, std::shared_ptr<DeviceImpl>> deviceMap;
    static std::map<VkDescriptorSetLayout, std::shared_ptr<DescriptorSetLayout>> descriptorSetLayoutMap;
    static std::map<VkPipelineLayout, std::shared_ptr<PipelineLayout>> pipelineLayoutMap;
    static std::map<VkQueue, std::shared_ptr<DeviceImpl>> queueMap;
    static std::map<VkCommandBuffer, std::shared_ptr<CommandBuffer>> commandBufferMap;
    static std::map<VkDevice, VkBool32> bufferDeviceAddressCaptureReplayFeatMap;
};

template <const auto &layerProperties, const auto &extensions, const auto &requiredExtensions, typename InstanceImpl,
          typename PhysicalDeviceImpl, typename DeviceImpl>
std::recursive_mutex VulkanLayer<layerProperties, extensions, requiredExtensions, InstanceImpl, PhysicalDeviceImpl,
                                 DeviceImpl>::globalMutex;

template <const auto &layerProperties, const auto &extensions, const auto &requiredExtensions, typename InstanceImpl,
          typename PhysicalDeviceImpl, typename DeviceImpl>
std::map<VkInstance, std::shared_ptr<InstanceImpl>> VulkanLayer<
    layerProperties, extensions, requiredExtensions, InstanceImpl, PhysicalDeviceImpl, DeviceImpl>::instanceMap;

template <const auto &layerProperties, const auto &extensions, const auto &requiredExtensions, typename InstanceImpl,
          typename PhysicalDeviceImpl, typename DeviceImpl>
std::map<VkPhysicalDevice, std::shared_ptr<PhysicalDeviceImpl>> VulkanLayer<
    layerProperties, extensions, requiredExtensions, InstanceImpl, PhysicalDeviceImpl, DeviceImpl>::physicalDeviceMap;

template <const auto &layerProperties, const auto &extensions, const auto &requiredExtensions, typename InstanceImpl,
          typename PhysicalDeviceImpl, typename DeviceImpl>
std::map<VkDevice, std::shared_ptr<DeviceImpl>> VulkanLayer<layerProperties, extensions, requiredExtensions,
                                                            InstanceImpl, PhysicalDeviceImpl, DeviceImpl>::deviceMap;

template <const auto &layerProperties, const auto &extensions, const auto &requiredExtensions, typename InstanceImpl,
          typename PhysicalDeviceImpl, typename DeviceImpl>
std::map<VkDescriptorSetLayout, std::shared_ptr<DescriptorSetLayout>>
    VulkanLayer<layerProperties, extensions, requiredExtensions, InstanceImpl, PhysicalDeviceImpl,
                DeviceImpl>::descriptorSetLayoutMap;

template <const auto &layerProperties, const auto &extensions, const auto &requiredExtensions, typename InstanceImpl,
          typename PhysicalDeviceImpl, typename DeviceImpl>
std::map<VkPipelineLayout, std::shared_ptr<PipelineLayout>> VulkanLayer<
    layerProperties, extensions, requiredExtensions, InstanceImpl, PhysicalDeviceImpl, DeviceImpl>::pipelineLayoutMap;

template <const auto &layerProperties, const auto &extensions, const auto &requiredExtensions, typename InstanceImpl,
          typename PhysicalDeviceImpl, typename DeviceImpl>
std::map<VkQueue, std::shared_ptr<DeviceImpl>> VulkanLayer<layerProperties, extensions, requiredExtensions,
                                                           InstanceImpl, PhysicalDeviceImpl, DeviceImpl>::queueMap;

template <const auto &layerProperties, const auto &extensions, const auto &requiredExtensions, typename InstanceImpl,
          typename PhysicalDeviceImpl, typename DeviceImpl>
std::map<VkCommandBuffer, std::shared_ptr<CommandBuffer>> VulkanLayer<
    layerProperties, extensions, requiredExtensions, InstanceImpl, PhysicalDeviceImpl, DeviceImpl>::commandBufferMap;

template <const auto &layerProperties, const auto &extensions, const auto &requiredExtensions, typename InstanceImpl,
          typename PhysicalDeviceImpl, typename DeviceImpl>
std::map<VkDevice, VkBool32> VulkanLayer<layerProperties, extensions, requiredExtensions, InstanceImpl,
                                         PhysicalDeviceImpl, DeviceImpl>::bufferDeviceAddressCaptureReplayFeatMap;

} // namespace mlsdk::el::layer
