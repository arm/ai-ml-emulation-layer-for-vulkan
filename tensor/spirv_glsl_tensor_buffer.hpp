/*
 * SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

#include <map>
#include <string>

#include <spirv_glsl.hpp>
#include <tuple>

namespace mlsdk::el::layer {

class CompilerTensorAsBuffer : public spirv_cross::CompilerGLSL {
  public:
    explicit CompilerTensorAsBuffer(std::vector<uint32_t> spirv_);
    bool hasTensors() const { return !tensorVariables.empty() || !tensorArrayVariables.empty(); }
    bool isCompute() const { return get_execution_model() == spv::ExecutionModelGLCompute; }

  protected:
    // overridden functions
    void emit_header() override;
    void emit_instruction(const spirv_cross::Instruction &instr) override;
    void emit_entry_point_declarations() override;

  private:
    // Largest allowed tensor rank
    static const int MAX_RANK = 6;
    // GLSL constanst and macros used to emulate tensor function, loaded from `shaders/tensor.glsl`
    static const std::string tensorDefines;
    // Used to keep track of SPIRV-Cross variables across `CompilerGLSL` function calls
    std::vector<std::pair<uint32_t, uint32_t>> tensorVariables;
    std::vector<uint32_t> tensorArrayVariables;

    // Tensor structs use uint64_t to store tensor description's shape and stride which are both uint64_t arrays
    // To create the tensor struct SPIRType, we therefore need SPIRTypes for uint64_t arrays
    std::tuple<uint32_t, uint32_t> tensorDimTypeIds;
    std::tuple<uint32_t, uint32_t> getTensorDimTypeIds();

    // Functions to set up SPIRV-Cross internal types and variables for emulating tensors
    uint32_t createTensorBDA(uint32_t tensorTypeId);
    void createTensorStruct(uint32_t tensorTypeId, uint32_t bufferPtrId);
    void createTensorPtr(uint32_t tensorPtrId);
    void createTensorInterface(uint32_t tensorVarId);
};

} // namespace mlsdk::el::layer
