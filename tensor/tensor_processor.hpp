/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once
#include "spirv_glsl_tensor_buffer.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace mlsdk::el::layer {

enum class TensorLoweringMode { GlslToSpirv, DirectSpirv };

// Set VMEL_EXPERIMENTAL_TENSOR_LOWERING_MODE before starting the application. Only "1"
// selects direct SPIR-V lowering; unset, "0", and other values retain the default.
TensorLoweringMode getTensorLoweringMode();

class TensorProcessor {
  public:
    explicit TensorProcessor(std::vector<uint32_t> spirv_, TensorLoweringMode mode = getTensorLoweringMode());
    bool isTensorComputeShader() const;
    bool isValidShader() const;
    std::vector<uint32_t> getNewSpirv() const;

  private:
    std::vector<uint32_t> m_spirv;
    std::unique_ptr<CompilerTensorAsBuffer> m_spirvCompiler;
    bool m_isTensorCompute = false;
    bool m_isValid = false;
};
} // namespace mlsdk::el::layer
