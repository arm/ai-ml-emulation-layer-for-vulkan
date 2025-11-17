/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "mlel/device.hpp"

#include <algorithm>
#include <map>
#include <vector>

namespace mlsdk::el::utilities {

/*******************************************************************************
 * Instance
 *******************************************************************************/

Instance::Instance(const std::shared_ptr<vk::raii::Context> &ctx, const std::vector<const char *> &instanceLayers)
    : context{ctx}, instance{createInstance(instanceLayers, {"VK_EXT_debug_utils"})} {}

vk::raii::Instance const &Instance::operator&() const { return instance; }

vk::raii::Instance Instance::createInstance(const std::vector<const char *> &layers,
                                            const std::vector<const char *> &extensions) const {
    const vk::ApplicationInfo applicationInfo{
        "ML Emulation Layer",     // application name
        VK_MAKE_VERSION(1, 3, 0), // application version
        "ML Emulation Layer",     // engine name
        VK_MAKE_VERSION(1, 3, 0), // engine version
        VK_MAKE_VERSION(1, 3, 0), // api version
    };
    std::vector<const char *> exts = extensions;
    vk::InstanceCreateFlags flags;
#ifdef EXPERIMENTAL_MOLTEN_VK_SUPPORT
    exts.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    flags = vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
#endif
    const vk::InstanceCreateInfo instanceCreateInfo{
        flags,                                // flags
        &applicationInfo,                     // application info
        static_cast<uint32_t>(layers.size()), // enabled layer count
        layers.data(),                        // enabled layers
        static_cast<uint32_t>(exts.size()),   // enabled extension count
        exts.data(),                          // enabled extensions
    };

    vk::raii::Instance instance(*context, instanceCreateInfo);
    return instance;
}

/*******************************************************************************
 * PhysicalDevice
 *******************************************************************************/

PhysicalDevice::PhysicalDevice(const std::shared_ptr<Instance> &_instance,
                               const std::vector<const char *> &deviceExtensions)
    : instance(_instance), physicalDevice{createPhysicalDevice(deviceExtensions)} {}

vk::raii::PhysicalDevice const &PhysicalDevice::operator&() const { return physicalDevice; }

const std::shared_ptr<Instance> &PhysicalDevice::getInstance() const { return instance; }

std::vector<vk::DeviceQueueCreateInfo> PhysicalDevice::getQueueCreateInfo(const vk::QueueFlags flags) const {
    return getQueueCreateInfo(physicalDevice, queuePriorities, flags);
}

uint32_t PhysicalDevice::getComputeFamilyIndex() const {
    auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

    for (uint32_t i = 0; i < queueFamilyProperties.size(); i++) {
        auto &property = queueFamilyProperties[i];

        if (property.queueFlags & vk::QueueFlagBits::eCompute) {
            return i;
        }
    }

    return uint32_t(-1);
}

std::vector<uint32_t> PhysicalDevice::getMemoryTypeIndices(const vk::MemoryPropertyFlags memoryPropertyFlags,
                                                           const uint32_t memoryTypeBits) const {
    vk::PhysicalDeviceMemoryProperties memoryProperties{physicalDevice.getMemoryProperties()};
    std::vector<uint32_t> memoryTypeIndices;

    // Compile a list of memory allocation infos
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
        // Exclude memory types that are not part of mask
        if (((memoryTypeBits >> i) & 1) == 0) {
            continue;
        }

        const auto &memoryType = memoryProperties.memoryTypes[i];

        // Check that all required memory properties are supported
        if ((memoryType.propertyFlags & memoryPropertyFlags) != memoryPropertyFlags) {
            continue;
        }

        // Add memory type
        memoryTypeIndices.emplace_back(i);
    }

    // Sort infos in priority order
    std::sort(memoryTypeIndices.begin(), memoryTypeIndices.end(),
              [&memoryProperties](const auto &leftIndex, const auto &rightIndex) {
                  const auto &leftMemoryType = memoryProperties.memoryTypes[leftIndex];
                  const auto &rightMemoryType = memoryProperties.memoryTypes[rightIndex];

                  const auto &leftHeap = memoryProperties.memoryHeaps[leftMemoryType.heapIndex];
                  const auto &rightHeap = memoryProperties.memoryHeaps[rightMemoryType.heapIndex];

                  // Prioritize device local memory, it is likely faster
                  const auto leftDeviceLocal = leftMemoryType.propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal;
                  const auto rightDeviceLocal =
                      rightMemoryType.propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal;
                  if (leftDeviceLocal != rightDeviceLocal) {
                      return leftDeviceLocal > rightDeviceLocal;
                  }

                  // Select the larger heap
                  return leftHeap.size > rightHeap.size;
              });

    return memoryTypeIndices;
}

bool PhysicalDevice::hasExtensionProperties(const vk::raii::PhysicalDevice &physicalDevice,
                                            const std::vector<const char *> &extensions) const {
    const auto extensionProperties = physicalDevice.enumerateDeviceExtensionProperties();

    for (auto &extension : extensions) {
        auto it = std::find_if(extensionProperties.begin(), extensionProperties.end(), [&](const auto &property) {
            return std::strcmp(property.extensionName, extension) == 0;
        });
        if (it == extensionProperties.end()) {
            return false;
        }
    }

    return true;
}

std::vector<vk::raii::PhysicalDevice> PhysicalDevice::enumeratePhysicalDevices() const {
    vk::raii::PhysicalDevices physicalDevices(&(*instance));
    return physicalDevices;
}

vk::raii::PhysicalDevice PhysicalDevice::createPhysicalDevice(const std::vector<const char *> &extensions) const {
    std::vector<vk::raii::PhysicalDevice> physicalDevices;

    for (auto currentPhysicalDevice : enumeratePhysicalDevices()) {
        // Verify that device supports compute queues
        auto queueCreateInfo = getQueueCreateInfo(currentPhysicalDevice, queuePriorities, vk::QueueFlagBits::eCompute);
        if (queueCreateInfo.size() == 0) {
            continue;
        }

        // Verify that device supports all enabled extensions
        if (!hasExtensionProperties(currentPhysicalDevice, extensions)) {
            continue;
        }

        physicalDevices.push_back(currentPhysicalDevice);
    }

    std::sort(physicalDevices.begin(), physicalDevices.end(), [this](const auto &left, const auto &right) {
        // Select discrete GPU
        if (left.getProperties().deviceType != right.getProperties().deviceType) {
            std::map<vk::PhysicalDeviceType, int> priority = {
                {vk::PhysicalDeviceType::eDiscreteGpu, 5}, {vk::PhysicalDeviceType::eIntegratedGpu, 4},
                {vk::PhysicalDeviceType::eVirtualGpu, 3},  {vk::PhysicalDeviceType::eCpu, 2},
                {vk::PhysicalDeviceType::eOther, 1},
            };

            return priority[left.getProperties().deviceType] < priority[right.getProperties().deviceType];
        }

        auto getMemoryHeapsMaxSize = [](const vk::raii::PhysicalDevice &physicalDevice) {
            auto memoryHeaps = physicalDevice.getMemoryProperties().memoryHeaps;
            return std::max_element(memoryHeaps.begin(),
                                    memoryHeaps.begin() + physicalDevice.getMemoryProperties().memoryHeapCount,
                                    [](const auto &leftMemHeap, const auto &rightMemHeap) {
                                        return leftMemHeap.size < rightMemHeap.size;
                                    })
                ->size;
        };

        return getMemoryHeapsMaxSize(left) < getMemoryHeapsMaxSize(right);
    });

    if (physicalDevices.empty()) {
        throw std::runtime_error("Failed to find physical device");
    }

    return physicalDevices.back();
}

std::vector<vk::DeviceQueueCreateInfo>
PhysicalDevice::getQueueCreateInfo(const vk::raii::PhysicalDevice &physicalDevice,
                                   const std::array<const float, 16> &queuePriorities, const vk::QueueFlags flags) {
    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfo;

    auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
    for (uint32_t i = 0; i < queueFamilyProperties.size(); i++) {
        auto &property = queueFamilyProperties[i];

        if (property.queueFlags & flags)
            queueCreateInfo.push_back(vk::DeviceQueueCreateInfo{
                {},                     // flags
                i,                      // queue family index
                property.queueCount,    // queue count
                queuePriorities.data(), // queue priorities
            });
    }

    return queueCreateInfo;
}

/*******************************************************************************
 * Device
 *******************************************************************************/

Device::Device(const std::shared_ptr<PhysicalDevice> &_physicalDevice,
               const std::vector<const char *> &deviceExtensions, const void *deviceFeatures)
    : physicalDevice(_physicalDevice), device{createDevice({}, deviceExtensions, deviceFeatures)} {}

vk::raii::Device const &Device::operator&() const { return device; }

const std::shared_ptr<PhysicalDevice> &Device::getPhysicalDevice() const { return physicalDevice; };

const std::shared_ptr<vk::raii::DeviceMemory>
Device::allocateDeviceMemory(const vk::DeviceSize size, const vk::MemoryPropertyFlags memoryPropertyFlags,
                             const uint32_t memoryTypeBits) const {
    auto memoryTypeIndices = physicalDevice->getMemoryTypeIndices(memoryPropertyFlags, memoryTypeBits);

    for (auto index : memoryTypeIndices) {
        try {
            const vk::MemoryAllocateInfo memoryAllocateInfo{size, index};
            return std::make_shared<vk::raii::DeviceMemory>(device, memoryAllocateInfo);
        } catch (const vk::OutOfDeviceMemoryError &) {
            // Ignore exception and try next memory index
        }
    }

    throw vk::OutOfDeviceMemoryError("Failed to allocate device memory of size " + std::to_string(size));
}

vk::raii::Device Device::createDevice(const std::vector<const char *> &layers,
                                      const std::vector<const char *> &extensions, const void *deviceFeatures) const {
    auto queueCreateInfo = physicalDevice->getQueueCreateInfo(vk::QueueFlagBits::eCompute);
    std::vector<const char *> exts = extensions;
#ifdef EXPERIMENTAL_MOLTEN_VK_SUPPORT
    exts.push_back("VK_KHR_portability_subset");
#endif
    const vk::DeviceCreateInfo deviceCreateInfo{
        {},                                            // flags
        static_cast<uint32_t>(queueCreateInfo.size()), // queue create info count
        queueCreateInfo.data(),                        // queue create infos
        static_cast<uint32_t>(layers.size()),          // enabled layer count
        layers.data(),                                 // enabled layers
        static_cast<uint32_t>(exts.size()),            // enabled extension count
        exts.data(),                                   // enabled extensions
        nullptr,                                       // Don't set pEnabledFeatures here!
        deviceFeatures                                 // Attach deviceFeatures via pNext
    };
    vk::raii::Device device(&(*physicalDevice), deviceCreateInfo);

    return device;
}

/*******************************************************************************
 * MakeDevice
 *******************************************************************************/

std::shared_ptr<Device> makeDevice(const std::vector<const char *> &instanceLayers,
                                   const std::vector<const char *> &deviceExtensions, const void *deviceFeatures) {
    auto context = std::make_shared<vk::raii::Context>();
    auto instance = std::make_shared<Instance>(context, instanceLayers);
    auto physicalDevice = std::make_shared<PhysicalDevice>(instance, deviceExtensions);

    return std::make_shared<Device>(physicalDevice, deviceExtensions, deviceFeatures);
};

} // namespace mlsdk::el::utilities
