/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <vulkan/vulkan.h>

#include <map>

namespace mlsdk::el::layer {

// Inline shader replacements belong to one pipeline-creation call, including
// any return taken before pipeline creation reaches the downstream driver.
class TemporaryShaderModules {
  public:
    TemporaryShaderModules(VkDevice device, const VkAllocationCallbacks *allocator, PFN_vkCreateShaderModule create,
                           PFN_vkDestroyShaderModule destroy)
        : _device(device), _allocator(allocator), _create(create), _destroy(destroy) {}
    TemporaryShaderModules(const TemporaryShaderModules &) = delete;
    TemporaryShaderModules &operator=(const TemporaryShaderModules &) = delete;

    ~TemporaryShaderModules() {
        for (const auto &[_, module] : _modules) {
            _destroy(_device, module, _allocator);
        }
    }

    VkShaderModule find(const VkShaderModuleCreateInfo *key) const {
        const auto it = _modules.find(key);
        return it == _modules.end() ? VK_NULL_HANDLE : it->second;
    }

    VkResult create(const VkShaderModuleCreateInfo *key, const VkShaderModuleCreateInfo &info, VkShaderModule &module) {
        if (const VkShaderModule cached = find(key); cached != VK_NULL_HANDLE) {
            module = cached;
            return VK_SUCCESS;
        }
        const auto entry = _modules.emplace(key, VK_NULL_HANDLE).first;
        module = VK_NULL_HANDLE;
        const auto result = _create(_device, &info, _allocator, &module);
        if (result != VK_SUCCESS) {
            _modules.erase(entry);
            return result;
        }
        entry->second = module;
        return VK_SUCCESS;
    }

  private:
    VkDevice _device;
    const VkAllocationCallbacks *_allocator;
    PFN_vkCreateShaderModule _create;
    PFN_vkDestroyShaderModule _destroy;
    std::map<const VkShaderModuleCreateInfo *, VkShaderModule> _modules;
};

} // namespace mlsdk::el::layer
