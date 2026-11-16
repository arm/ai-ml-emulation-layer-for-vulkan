/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "temporary_shader_modules.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <type_traits>
#include <vector>

namespace {
template <typename Handle> Handle fakeHandle(uintptr_t value) {
    if constexpr (std::is_pointer_v<Handle>) {
        return reinterpret_cast<Handle>(value);
    } else {
        return static_cast<Handle>(value);
    }
}
struct ShaderCalls {
    unsigned creates = 0;
    unsigned failAt = 0;
    std::vector<VkShaderModule> destroyed;
    VkAllocationCallbacks allocator{};
    bool useAllocator = false;
};
thread_local ShaderCalls calls;

VKAPI_ATTR VkResult VKAPI_CALL createShader(VkDevice, const VkShaderModuleCreateInfo *,
                                            const VkAllocationCallbacks *allocator, VkShaderModule *module) {
    EXPECT_EQ(allocator, calls.useAllocator ? &calls.allocator : nullptr);
    ++calls.creates;
    if (calls.creates == calls.failAt) {
        // Model a driver which does not write the output on failure.
        EXPECT_EQ(*module, VK_NULL_HANDLE);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    *module = fakeHandle<VkShaderModule>(calls.creates);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroyShader(VkDevice, VkShaderModule module, const VkAllocationCallbacks *allocator) {
    EXPECT_EQ(allocator, calls.useAllocator ? &calls.allocator : nullptr);
    calls.destroyed.push_back(module);
}

TEST(TemporaryShaderModules, CreationFailurePropagatesAndCleansEarlierModules) {
    for (const unsigned failAt : {1u, 2u}) {
        calls = {};
        auto &state = calls;
        state.failAt = failAt;
        state.useAllocator = true;
        const VkShaderModuleCreateInfo infos[2]{};
        const auto result = [&] {
            mlsdk::el::layer::TemporaryShaderModules modules(VK_NULL_HANDLE, &state.allocator, createShader,
                                                             destroyShader);
            for (const auto &info : infos) {
                VkShaderModule module = VK_NULL_HANDLE;
                const auto status = modules.create(&info, info, module);
                if (status != VK_SUCCESS) {
                    return status;
                }
            }
            return VK_SUCCESS;
        }();
        EXPECT_EQ(result, VK_ERROR_OUT_OF_HOST_MEMORY);
        EXPECT_EQ(state.creates, failAt);
        EXPECT_EQ(state.destroyed.size(), failAt - 1);
        if (failAt == 2) {
            EXPECT_EQ(state.destroyed.front(), fakeHandle<VkShaderModule>(1));
        }
    }
}

TEST(TemporaryShaderModules, FailedCreationCanBeRetried) {
    calls = {};
    auto &state = calls;
    state.failAt = 1;
    VkShaderModuleCreateInfo info{};
    {
        mlsdk::el::layer::TemporaryShaderModules modules(VK_NULL_HANDLE, nullptr, createShader, destroyShader);
        VkShaderModule module = VK_NULL_HANDLE;
        EXPECT_EQ(modules.create(&info, info, module), VK_ERROR_OUT_OF_HOST_MEMORY);
        EXPECT_EQ(modules.find(&info), VK_NULL_HANDLE);
        EXPECT_TRUE(state.destroyed.empty());
        EXPECT_EQ(modules.create(&info, info, module), VK_SUCCESS);
        EXPECT_EQ(module, fakeHandle<VkShaderModule>(2));
        EXPECT_EQ(modules.find(&info), module);
        EXPECT_EQ(state.creates, 2u);
    }
    ASSERT_EQ(state.destroyed.size(), 1u);
    EXPECT_EQ(state.destroyed.front(), fakeHandle<VkShaderModule>(2));
}

TEST(TemporaryShaderModules, ReuseDestroysOnceOnEveryExit) {
    for (const VkResult result : {VK_SUCCESS, VK_ERROR_UNKNOWN, VK_ERROR_DEVICE_LOST}) {
        calls = {};
        const auto &state = calls;
        VkShaderModuleCreateInfo info{};
        const auto run = [&] {
            mlsdk::el::layer::TemporaryShaderModules modules(VK_NULL_HANDLE, nullptr, createShader, destroyShader);
            VkShaderModule first = VK_NULL_HANDLE;
            VkShaderModule reused = VK_NULL_HANDLE;
            EXPECT_EQ(modules.create(&info, info, first), VK_SUCCESS);
            EXPECT_EQ(modules.create(&info, info, reused), VK_SUCCESS);
            EXPECT_EQ(first, reused);
            EXPECT_EQ(modules.find(&info), first);
            EXPECT_TRUE(state.destroyed.empty());
            return result;
        }();
        EXPECT_EQ(run, result);
        EXPECT_EQ(state.creates, 1u);
        ASSERT_EQ(state.destroyed.size(), 1u);
    }
}
} // namespace
