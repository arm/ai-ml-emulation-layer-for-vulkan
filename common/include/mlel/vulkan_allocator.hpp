/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include <vulkan/vulkan.hpp>

/*******************************************************************************
 * Allocator
 *******************************************************************************/

namespace mlsdk::el::layer {

template <class T> struct Allocator {
    using value_type = T;

    template <class U> constexpr explicit Allocator(const Allocator<U> &) noexcept {}

    explicit Allocator(const VkAllocationCallbacks *_callbacks) : callbacks(_callbacks) {}

    [[nodiscard]] T *allocate(std::size_t n) const {
        if (callbacks != nullptr) {
            if (auto pointer =
                    static_cast<T *>(callbacks->pfnAllocation(callbacks->pUserData, n * sizeof(T), alignment, scope))) {
                return pointer;
            }
        } else {
            if (auto pointer = static_cast<T *>(::malloc(n * sizeof(T)))) {
                return pointer;
            }
        }

        throw std::bad_alloc();
    }

    void deallocate(T *pointer, std::size_t) const noexcept {
        if (callbacks != nullptr) {
            callbacks->pfnFree(callbacks->pUserData, pointer);
        } else {
            ::free(pointer);
        }
    }

  private:
    const VkAllocationCallbacks *callbacks = nullptr;
    static constexpr VkSystemAllocationScope scope = VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE;
    static constexpr size_t alignment = sizeof(uint64_t);
};

template <class T, class... Args> T *allocateObject(const VkAllocationCallbacks *callbacks, Args &&...args) {
    auto object = static_cast<T *>((Allocator<T>{callbacks}).allocate(1));
    new (object) T(args...);
    return object;
}

template <class T> void destroyObject(const VkAllocationCallbacks *callbacks, T *object) {
    object->~T();
    (Allocator<T>{callbacks}).deallocate(object, 1);
}

} // namespace mlsdk::el::layer
