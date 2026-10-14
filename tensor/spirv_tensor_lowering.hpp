/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

#include <spirv-tools/libspirv.hpp>

#include <cstdint>
#include <vector>

namespace mlsdk::el::layer {

// Returns success, preserving unchanged modules exactly and clearing SPIR-V on failure.
// Diagnostics are delivered through the message consumer; the compute flag is set only after successful lowering.
bool lowerTensorSpirv(std::vector<uint32_t> &spirv, bool &loweredTensorCompute,
                      const spvtools::MessageConsumer &messageConsumer);

} // namespace mlsdk::el::layer
