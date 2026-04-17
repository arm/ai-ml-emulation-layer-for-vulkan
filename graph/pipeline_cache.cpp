/*
 * SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "pipeline_cache.hpp"
#include "shaders/shaders.hpp.inc"
#include "shaders/shaders_spv.hpp.inc"

#include <vulkan/vulkan.hpp>

#include <array>
#include <map>
#include <numeric>
#include <regex>
#include <set>
#include <string>
#include <vector>

using namespace mlsdk::el::utils;

namespace mlsdk::el::compute {

namespace {

std::string_view getGlslSource(std::string_view shaderName) {
    if (auto it = glslMap.find(shaderName); it != glslMap.end()) {
        return it->second;
    }
    return ""; // no match
}

uint32_t crc32(std::string_view input) {
    static constexpr auto lut = []() {
        constexpr uint32_t polynomial = 0xedb88320;
        std::array<uint32_t, 256> result{};
        uint32_t n = 0;
        for (auto &it : result) {
            uint32_t crc = n++;
            for (int i = 0; i < 8; i++) {
                crc = (crc >> 1) ^ ((crc & 1) ? polynomial : 0);
            }
            it = crc;
        }

        return result;
    }();

    return ~std::accumulate(input.begin(), input.end(), ~0u, [](uint32_t crc, char val) {
        return lut[(crc ^ static_cast<uint32_t>(val)) & 0xff] ^ (crc >> 8);
    });
}

} // namespace

/*******************************************************************************
 * PipelineCache
 *******************************************************************************/

PipelineCache::PipelineCache([[maybe_unused]] const void *data, [[maybe_unused]] const size_t size,
                             VkPipelineCache _pipelineCache)
    : pipelineCache{_pipelineCache} {};

SpirvBinary PipelineCache::lookup(std::string_view shaderName, const KeyList &keys, const ReplaceList &repl) {
    // Find precompiled shader
    const auto key = makeKey(shaderName, keys);

    if (auto it = precompiledSpirvModules.find(key); it != precompiledSpirvModules.end()) {
        auto [data, size] = it->second;
        return {data, size};
    }

    // Find source code
    auto glslSource = getGlslSource(shaderName);

    if (glslSource.empty()) {
        // No match found for GLSL source code
        throw std::runtime_error("Failed to find GLSL source code for operator " + std::string(shaderName) +
                                 " with key " + key);
    }

    const auto srcHash = crc32(glslSource);

    if (auto it = cache.find(key); it != cache.end()) {
        // Cache entry exists
        const auto &[spirv, oldHash] = it->second;

        if (oldHash == srcHash) {
            // Cache entry is up to date
            return {spirv.data(), spirv.size()};
        }
    }

    // Cache entry is missing or out of date; compile source and add to cache
    cache[key] = {replaceCompileGlsl(glslSource, repl), srcHash};

    return {cache[key].first.data(), cache[key].first.size()};
}

VkPipelineCache PipelineCache::getPipelineCache() const { return pipelineCache; }

std::string PipelineCache::makeKey(std::string_view shaderName, const KeyList &keys) {
    return std::accumulate(
        keys.begin(), keys.end(), std::string(shaderName),
        [](const std::string &acc, const std::string_view &key) { return acc + '_' + std::string(key); });
}

std::vector<uint32_t> PipelineCache::replaceCompileGlsl(std::string_view glslSource, const ReplaceList &replaceList) {
    auto tmp = std::string(glslSource);

    for (const auto &[pattern, repl] : replaceList) {
        tmp = std::regex_replace(tmp, std::regex(std::string(pattern)), std::string(repl));
    }

    return glslToSpirv(tmp);
}

} // namespace mlsdk::el::compute
