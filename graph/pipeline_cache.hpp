/*
 * SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "mlel/utils.hpp"

#include <vulkan/vulkan.hpp>

#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>

namespace mlsdk::el::compute {

/*******************************************************************************
 * PipelineCache
 *******************************************************************************/

using SpirvBinary = utils::Span<uint32_t>;

class PipelineCache {
  public:
    using KeyList = std::initializer_list<std::string_view>;

    PipelineCache(const void *data, size_t size, VkPipelineCache _pipelineCache);
    ~PipelineCache() = default;

    // Embedded SPIR-V remains valid for the layer's lifetime; missing variants throw.
    SpirvBinary lookup(std::string_view shaderName, const KeyList &keys);
    VkPipelineCache getPipelineCache() const;

  private:
    VkPipelineCache pipelineCache;

    static std::string makeKey(std::string_view shaderName, const KeyList &keys);
};

} // namespace mlsdk::el::compute
