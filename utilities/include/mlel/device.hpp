/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include <vulkan/vulkan_raii.hpp>

#include <memory>
#include <vector>

namespace mlsdk::el::utilities {

/*******************************************************************************
 * Instance
 *******************************************************************************/

class Instance {
  public:
    Instance(const std::shared_ptr<vk::raii::Context> &ctx, const std::vector<const char *> &instanceLayers);

    vk::raii::Instance const &operator&() const;

  private:
    vk::raii::Instance createInstance(const std::vector<const char *> &layers,
                                      const std::vector<const char *> &extensions) const;

    std::shared_ptr<vk::raii::Context> context;
    vk::raii::Instance instance;
};

/*******************************************************************************
 * PhysicalDevice
 *******************************************************************************/

class PhysicalDevice {
  public:
    PhysicalDevice(const std::shared_ptr<Instance> &_instance, const std::vector<const char *> &deviceExtensions);

    vk::raii::PhysicalDevice const &operator&() const;
    const std::shared_ptr<Instance> &getInstance() const;
    std::vector<vk::DeviceQueueCreateInfo> getQueueCreateInfo(const vk::QueueFlags flags) const;
    uint32_t getComputeFamilyIndex() const;
    std::vector<uint32_t> getMemoryTypeIndices(const vk::MemoryPropertyFlags memoryPropertyFlags = {},
                                               const uint32_t memoryTypeBits = 0xffffffff) const;

  private:
    bool hasExtensionProperties(const vk::raii::PhysicalDevice &physicalDevice,
                                const std::vector<const char *> &extensions) const;

    std::vector<vk::raii::PhysicalDevice> enumeratePhysicalDevices() const;
    vk::raii::PhysicalDevice createPhysicalDevice(const std::vector<const char *> &extensions) const;
    static std::vector<vk::DeviceQueueCreateInfo> getQueueCreateInfo(const vk::raii::PhysicalDevice &physicalDevice,
                                                                     const std::array<const float, 16> &queuePriorities,
                                                                     const vk::QueueFlags flags);

    std::shared_ptr<Instance> instance;
    vk::raii::PhysicalDevice physicalDevice;
    std::array<const float, 16> queuePriorities = {1.0f};
};

/*******************************************************************************
 * Device
 *******************************************************************************/

class Device {
  public:
    Device(const std::shared_ptr<PhysicalDevice> &_physicalDevice, const std::vector<const char *> &deviceExtensions,
           const void *deviceFeatures);

    vk::raii::Device const &operator&() const;
    const std::shared_ptr<PhysicalDevice> &getPhysicalDevice() const;
    const std::shared_ptr<vk::raii::DeviceMemory>
    allocateDeviceMemory(const vk::DeviceSize size, const vk::MemoryPropertyFlags memoryPropertyFlags = {},
                         const uint32_t memoryTypeBits = 0xffffffff) const;

  private:
    vk::raii::Device createDevice(const std::vector<const char *> &layers, const std::vector<const char *> &extensions,
                                  const void *deviceFeatures) const;

    std::shared_ptr<PhysicalDevice> physicalDevice;
    vk::raii::Device device;
};

/*******************************************************************************
 * MakeDevice
 *******************************************************************************/

std::shared_ptr<Device> makeDevice(const std::vector<const char *> &instanceLayers,
                                   const std::vector<const char *> &deviceExtensions, const void *deviceFeatures);

} // namespace mlsdk::el::utilities
