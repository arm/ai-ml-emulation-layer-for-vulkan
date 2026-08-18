/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "graph_motion_engine_1_0_ext_inst.hpp"
#include "graph_ext_inst_context.hpp"
#include "graph_log.hpp"

#include <spirv/unified1/ArmMotionEngine.100.h>
#include <unordered_map>

using namespace mlsdk::el::log;
using namespace mlsdk::el::compute;

/*******************************************************************************
 * GraphPass extended instruction sets
 *******************************************************************************/
namespace spvtools::opt {

void GraphMotionEngine10ExtInst::handleOp(const Instruction *opExtInst) const {
    const auto &motionEngine = ArmMotionEngineInstructions(opExtInst->GetInOperand(1).words[0]);

    // Verify that this is a Motion Engine external instruction
    static const std::unordered_map<ArmMotionEngineInstructions, std::string> opNameMap = {
        {ArmMotionEngineMIN_SAD, "MIN_SAD"},
        {ArmMotionEngineMIN_SAD_COST, "MIN_SAD_COST"},
        {ArmMotionEngineRAW_SAD, "RAW_SAD"},
    };
    std::string debugName =
        context.debugName(opExtInst, opNameMap.count(motionEngine) ? opNameMap.at(motionEngine) : "UNKNOWN");

    switch (motionEngine) {
    case ArmMotionEngineMIN_SAD:
        handleMinSad(opExtInst, debugName);
        break;
    case ArmMotionEngineMIN_SAD_COST:
        handleMinSadCost(opExtInst, debugName);
        break;
    case ArmMotionEngineRAW_SAD:
        handleRawSad(opExtInst, debugName);
        break;
    default:
        throw std::runtime_error(std::string("Unsupported ArmMotionEngine operand ") + std::to_string(motionEngine));
    }
}

void GraphMotionEngine10ExtInst::handleMinSad(const Instruction *opExtInst, const std::string &debugName) const {
    // OpExtInst <result id> <OpExtInstImport id> MIN_SAD kernel_sizes search_window_sizes input_strides
    // window_strides window_offsets padding search_pattern input0 input1
    assert(opExtInst->NumInOperands() == 11);

    const auto &resultId = opExtInst->result_id();
    const auto &kernelSizes = context.getConstVector<uint32_t>(opExtInst->GetInOperand(2));
    const auto &searchWindowSizes = context.getConstVector<uint32_t>(opExtInst->GetInOperand(3));
    const auto &inputStrides = context.getConstVector<uint32_t>(opExtInst->GetInOperand(4));
    const auto &windowStrides = context.getConstVector<uint32_t>(opExtInst->GetInOperand(5));
    const auto &windowOffsets = context.getConstVector<uint32_t>(opExtInst->GetInOperand(6));
    const auto &padding = context.getConstVector<uint32_t>(opExtInst->GetInOperand(7));
    const auto &searchPattern = context.getConstScalar<uint32_t>(opExtInst->GetInOperand(8));
    const auto &input0Id = opExtInst->GetInOperand(9);
    const auto &input1Id = opExtInst->GetInOperand(10);

    graphLog(Severity::Info) << "OpExtInst result=" << resultId << ", " << debugName << ", kernelSizes=" << kernelSizes
                             << ", searchWindowSizes=" << searchWindowSizes << ", inputStrides=" << inputStrides
                             << ", windowStrides=" << windowStrides << ", windowOffsets=" << windowOffsets
                             << ", padding=" << padding << ", searchPattern=" << searchPattern << ", input0=%"
                             << input0Id.AsId() << ", input1=%" << input1Id.AsId() << std::endl;

    context.pipeline().makeMinSad(context.getTensor(input0Id), context.getTensor(input1Id),
                                  context.getTensor(*opExtInst), kernelSizes, searchWindowSizes, inputStrides,
                                  windowStrides, windowOffsets, padding, searchPattern, debugName);
}

void GraphMotionEngine10ExtInst::handleMinSadCost(const Instruction *opExtInst, const std::string &debugName) const {
    // OpExtInst <result id> <OpExtInstImport id> MIN_SAD_COST kernel_sizes search_window_sizes input_strides
    // window_strides window_offsets padding search_pattern input0 input1
    assert(opExtInst->NumInOperands() == 11);

    const auto &resultId = opExtInst->result_id();
    const auto &kernelSizes = context.getConstVector<uint32_t>(opExtInst->GetInOperand(2));
    const auto &searchWindowSizes = context.getConstVector<uint32_t>(opExtInst->GetInOperand(3));
    const auto &inputStrides = context.getConstVector<uint32_t>(opExtInst->GetInOperand(4));
    const auto &windowStrides = context.getConstVector<uint32_t>(opExtInst->GetInOperand(5));
    const auto &windowOffsets = context.getConstVector<uint32_t>(opExtInst->GetInOperand(6));
    const auto &padding = context.getConstVector<uint32_t>(opExtInst->GetInOperand(7));
    const auto &searchPattern = context.getConstScalar<uint32_t>(opExtInst->GetInOperand(8));
    const auto &input0Id = opExtInst->GetInOperand(9);
    const auto &input1Id = opExtInst->GetInOperand(10);

    graphLog(Severity::Info) << "OpExtInst result=" << resultId << ", " << debugName << ", kernelSizes=" << kernelSizes
                             << ", searchWindowSizes=" << searchWindowSizes << ", inputStrides=" << inputStrides
                             << ", windowStrides=" << windowStrides << ", windowOffsets=" << windowOffsets
                             << ", padding=" << padding << ", searchPattern=" << searchPattern << ", input0=%"
                             << input0Id.AsId() << ", input1=%" << input1Id.AsId() << std::endl;

    context.pipeline().makeMinSadCost(context.getTensor(input0Id), context.getTensor(input1Id),
                                      context.getTensor(*opExtInst, 0), context.getTensor(*opExtInst, 1), kernelSizes,
                                      searchWindowSizes, inputStrides, windowStrides, windowOffsets, padding,
                                      searchPattern, debugName);
}

void GraphMotionEngine10ExtInst::handleRawSad(const Instruction *opExtInst, const std::string &debugName) const {
    // OpExtInst <result id> <OpExtInstImport id> RAW_SAD kernel_sizes search_window_sizes input_strides
    // window_strides window_offsets padding input0 input1
    assert(opExtInst->NumInOperands() == 10);

    const auto &resultId = opExtInst->result_id();
    const auto &kernelSizes = context.getConstVector<uint32_t>(opExtInst->GetInOperand(2));
    const auto &searchWindowSizes = context.getConstVector<uint32_t>(opExtInst->GetInOperand(3));
    const auto &inputStrides = context.getConstVector<uint32_t>(opExtInst->GetInOperand(4));
    const auto &windowStrides = context.getConstVector<uint32_t>(opExtInst->GetInOperand(5));
    const auto &windowOffsets = context.getConstVector<uint32_t>(opExtInst->GetInOperand(6));
    const auto &padding = context.getConstVector<uint32_t>(opExtInst->GetInOperand(7));
    const auto &input0Id = opExtInst->GetInOperand(8);
    const auto &input1Id = opExtInst->GetInOperand(9);

    graphLog(Severity::Info) << "OpExtInst result=" << resultId << ", " << debugName << ", kernelSizes=" << kernelSizes
                             << ", searchWindowSizes=" << searchWindowSizes << ", inputStrides=" << inputStrides
                             << ", windowStrides=" << windowStrides << ", windowOffsets=" << windowOffsets
                             << ", padding=" << padding << ", input0=%" << input0Id.AsId() << ", input1=%"
                             << input1Id.AsId() << std::endl;

    context.pipeline().makeRawSad(context.getTensor(input0Id), context.getTensor(input1Id),
                                  context.getTensor(*opExtInst), kernelSizes, searchWindowSizes, inputStrides,
                                  windowStrides, windowOffsets, padding, debugName);
}

} // namespace spvtools::opt
