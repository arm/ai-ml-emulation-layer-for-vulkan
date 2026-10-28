/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

#include <vulkan/vulkan_core.h>

#include <map>
#include <variant>
#include <vector>

namespace mlsdk::el::layer {

// This class manages aliasing of images and tensors to device memory and offsets.
class MemoryAliasing {
    using Resource = std::variant<VkImage, VkTensorARM>;
    struct DeviceMemoryAndOffset {
        VkDeviceMemory deviceMemory;
        VkDeviceSize offset;
        bool operator==(const DeviceMemoryAndOffset &other) const noexcept {
            return deviceMemory == other.deviceMemory && offset == other.offset;
        }
    };

    std::map<Resource, DeviceMemoryAndOffset> resourceMap;

  public:
    template <typename T>
    void addAliasingResource(const VkDeviceMemory &memory, const VkDeviceSize &offset, T resource) {
        resourceMap[resource] = {memory, offset};
    }

    template <typename T>
    std::vector<T> getAliasingResources(const VkDeviceMemory &memory, const VkDeviceSize &offset) const {
        std::vector<T> resources;
        for (const auto &[resource, memAndOffset] : resourceMap) {
            if (std::holds_alternative<T>(resource) && memAndOffset == DeviceMemoryAndOffset{memory, offset}) {
                resources.push_back(std::get<T>(resource));
            }
        }
        return resources;
    }

    template <typename T> void removeAliasingResource(T resource) { resourceMap.erase(Resource(resource)); }

    void removeAliasingMemory(const VkDeviceMemory &mem) {
        std::vector<Resource> resourcesToRemove;
        for (const auto &[resource, memory] : resourceMap) {
            if (memory.deviceMemory == mem) {
                resourcesToRemove.push_back(resource);
            }
        }
        for (auto resource : resourcesToRemove) {
            resourceMap.erase(resource);
        }
    }
};

} // namespace mlsdk::el::layer
