/*
 * SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "pipeline_cache.hpp"
#include "shaders/precompiled_shaders.hpp"

#include <numeric>
#include <stdexcept>
#include <string>

namespace mlsdk::el::compute {

PipelineCache::PipelineCache([[maybe_unused]] const void *data, [[maybe_unused]] const size_t size,
                             VkPipelineCache _pipelineCache)
    : pipelineCache{_pipelineCache} {};

SpirvBinary PipelineCache::lookup(std::string_view shaderName, const KeyList &keys) {
    const auto key = makeKey(shaderName, keys);
    if (auto it = precompiledSpirvModules.find(key); it != precompiledSpirvModules.end()) {
        auto [data, size] = it->second;
        return {data, size};
    }
    throw std::runtime_error("Missing precompiled shader: " + key);
}

VkPipelineCache PipelineCache::getPipelineCache() const { return pipelineCache; }

std::string PipelineCache::makeKey(std::string_view shaderName, const KeyList &keys) {
    return std::accumulate(
        keys.begin(), keys.end(), std::string(shaderName),
        [](const std::string &acc, const std::string_view &key) { return acc + '_' + std::string(key); });
}

} // namespace mlsdk::el::compute
