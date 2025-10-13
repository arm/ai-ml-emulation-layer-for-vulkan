/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once
#include "spirv_glsl_tensor_buffer.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace mlsdk::el::layer {
class TensorProcessor {
  public:
    static constexpr size_t TENSOR_MAX_ACCESS_BYTES = 32;

    explicit TensorProcessor(const std::vector<uint32_t> &spirv_);
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
