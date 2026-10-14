/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

#include "source/opt/ir_builder.h"

#include <cstdint>
#include <map>
#include <memory>
#include <tuple>

namespace mlsdk::el::layer::tensor_spirv {

using spvtools::opt::Instruction;
using spvtools::opt::InstructionBuilder;
using spvtools::opt::IRContext;
using spvtools::opt::Operand;

inline Operand idOperand(uint32_t id) { return {SPV_OPERAND_TYPE_ID, {id}}; }

inline std::unique_ptr<Instruction> makeInstruction(IRContext &context, spv::Op opcode, uint32_t type, uint32_t result,
                                                    const Instruction::OperandList &operands = {}) {
    return std::make_unique<Instruction>(&context, opcode, type, result, operands);
}

struct TensorType {
    uint32_t elementType;
    uint32_t rank;
};

struct AccessState {
    explicit AccessState(IRContext &context);

    IRContext &context;
    uint32_t uint64Type;
    uint32_t descriptorType;
    uint32_t descriptorPointerType;
    uint32_t interfacePointerType;
    // Operation, element type, coordinate-array type, and rank identify a helper.
    std::map<std::tuple<spv::Op, uint32_t, uint32_t, uint32_t>, uint32_t> helpers;
};

uint32_t getOrCreateReadHelper(AccessState &state, const TensorType &tensor, uint32_t coordinatesType);
uint32_t getOrCreateWriteHelper(AccessState &state, const TensorType &tensor, uint32_t coordinatesType);
uint32_t emitSizeQuery(AccessState &state, InstructionBuilder &builder, uint32_t descriptor, uint32_t dimension);

} // namespace mlsdk::el::layer::tensor_spirv
