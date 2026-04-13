/*
 * SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
#include <map>
#include <string>
#include <utility>
#include <vector>

using namespace mlsdk::el::utils;

namespace mlsdk::el::compute {

/*******************************************************************************
 * PipelineCache
 *******************************************************************************/

using SpirvBinary = Span<uint32_t>;

class PipelineCache {
  public:
    using KeyList = std::initializer_list<std::string_view>;
    using ReplaceList = std::initializer_list<std::pair<std::string_view, std::string_view>>;

    PipelineCache(const void *data, size_t size, VkPipelineCache _pipelineCache);
    ~PipelineCache() = default;

    SpirvBinary lookup(std::string_view shaderName, const KeyList &keys, const ReplaceList &repl);
    VkPipelineCache getPipelineCache() const;

  private:
    using Entry = std::pair<std::vector<uint32_t>, uint32_t>;

    VkPipelineCache pipelineCache;
    std::map<std::string, Entry> cache{};

    static std::string makeKey(std::string_view shaderName, const KeyList &keys);
    static std::vector<uint32_t> replaceCompileGlsl(std::string_view glslSource, const ReplaceList &replaceList);
};

} // namespace mlsdk::el::compute
