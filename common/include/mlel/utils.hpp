/*
 * SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

/*******************************************************************************
 * Includes
 *******************************************************************************/
#include <memory>
#include <string>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace mlsdk::el::utils {

namespace {
template <typename T> inline T roundUp(const T data, size_t multiple) {
    return ((data + multiple - 1) / multiple) * multiple;
}

inline uint32_t divideRoundUp(const uint32_t value, const uint32_t divide) { return (value + divide - 1) / divide; }
} // namespace

std::vector<uint32_t> glslToSpirv(const std::string &glsl);

class FormatBase {
  public:
    virtual ~FormatBase() {}

    virtual bool isInteger() const = 0;
    virtual bool isSigned() const = 0;
    virtual std::string lowest() const = 0;
    virtual std::string max() const = 0;
    virtual std::string glslType() const = 0;
    virtual std::string toInt() const = 0;
};

std::shared_ptr<FormatBase> makeFormat(const VkFormat format);

std::shared_ptr<FormatBase> makeFormat(const VkFormat format, const bool isUnsigned);

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
