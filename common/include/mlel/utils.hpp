/*
 * SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

/*******************************************************************************
 * Includes
 *******************************************************************************/
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <vulkan/vulkan.hpp>

#ifdef EXPERIMENTAL_MOLTEN_VK_SUPPORT
namespace mlsdk::el {
constexpr int EXPERIMENTAL_MVK_BUFFER_BINDING_OFFSET = 1000;
}
#endif

namespace mlsdk::el::utils {

template <typename T> T roundUp(const T data, size_t multiple) {
    return static_cast<T>(((data + multiple - 1) / multiple) * multiple);
}

inline uint32_t divideRoundUp(const uint32_t value, const uint32_t divide) { return (value + divide - 1) / divide; }

inline void replaceAll(std::string &str, std::string_view pattern, std::string_view replacement) {
    if (pattern.empty()) {
        return;
    }

    for (size_t pos = str.find(pattern.data(), 0, pattern.size()); pos != std::string::npos;
         pos = str.find(pattern.data(), pos, pattern.size())) {
        str.replace(pos, pattern.size(), replacement.data(), replacement.size());
        pos += replacement.size();
    }
}

/// Gets the total number of elements in a tensor given its dimensions, throws if result is negative.
size_t getElementCount(const std::vector<int64_t> &dimensions);

std::vector<uint32_t> glslToSpirv(const std::string &glsl);

struct FormatInfo {
    bool isInteger;
    bool isSigned;
    std::string_view lowest;
    std::string_view max;
    std::string_view glslType;
    std::string_view typeId;
    std::string_view compType;
};

const FormatInfo *getFormatInfo(VkFormat format);

const FormatInfo *getFormatInfo(VkFormat format, bool isUnsigned);

template <typename T> class Span {
  private:
    const T *m_data;
    const size_t m_size;

  public:
    Span(const T *_data, const size_t _size) : m_data{_data}, m_size{_size} {}
    const T *data() const { return m_data; }
    size_t size() const { return m_size; }
};

void setDebugUtilsObjectName(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &loader,
                             VkDevice device, VkObjectType type, uint64_t handle, const std::string &name);

} // namespace mlsdk::el::utils
