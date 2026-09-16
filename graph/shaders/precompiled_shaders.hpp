/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string_view>
#include <tuple>

namespace mlsdk::el::compute {

extern const std::map<std::string_view, std::tuple<const uint32_t *const, const std::size_t>> precompiledSpirvModules;

} // namespace mlsdk::el::compute
