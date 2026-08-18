/*
 * SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "graph_tosa_1_0_ext_inst.hpp"
#include "graph_ext_inst_context.hpp"
#include "graph_log.hpp"

#include <cstring>
#include <spirv/unified1/TOSA.001000.1.h>
#include <unordered_map>

using namespace mlsdk::el::log;
using namespace mlsdk::el::compute;

/*******************************************************************************
 * GraphPass extended instruction sets
 *******************************************************************************/
namespace spvtools::opt {

namespace {

enum RoundingMode {
    SingleRound = 1,
    InexactRound = 2,
    DoubleRound = 3,
};

} // namespace

void GraphTosa10ExtInst::handleOp(const Instruction *opExtInst) const {
    const auto &tosa = TOSAInstructions(opExtInst->GetInOperand(1).words[0]);

    // Verify that this is a TOSA external instruction
    static const std::unordered_map<TOSAInstructions, std::string> opNameMap = {
        {TOSAABS, "ABS"},
        {TOSAADD, "ADD"},
        {TOSAARGMAX, "ARGMAX"},
        {TOSAARITHMETIC_RIGHT_SHIFT, "ARITHMETIC_RIGHT_SHIFT"},
        {TOSAAVG_POOL2D, "AVG_POOL2D"},
        {TOSABITWISE_AND, "BITWISE_AND"},
        {TOSABITWISE_NOT, "BITWISE_NOT"},
        {TOSABITWISE_OR, "BITWISE_OR"},
        {TOSABITWISE_XOR, "BITWISE_XOR"},
        {TOSACAST, "CAST"},
        {TOSACEIL, "CEIL"},
        {TOSACLAMP, "CLAMP"},
        {TOSACLZ, "CLZ"},
        {TOSACONCAT, "CONCAT"},
        {TOSACONV2D, "CONV2D"},
        {TOSACONV3D, "CONV3D"},
        {TOSACOS, "COS"},
        {TOSADEPTHWISE_CONV2D, "DEPTHWISE_CONV2D"},
        {TOSAEQUAL, "EQUAL"},
        {TOSAERF, "ERF"},
        {TOSAEXP, "EXP"},
        {TOSAFFT2D, "FFT2D"},
        {TOSAFLOOR, "FLOOR"},
        {TOSAGATHER, "GATHER"},
        {TOSAGREATER, "GREATER"},
        {TOSAGREATER_EQUAL, "GREATER_EQUAL"},
        {TOSAINTDIV, "INTDIV"},
        {TOSALOG, "LOG"},
        {TOSALOGICAL_AND, "LOGICAL_AND"},
        {TOSALOGICAL_LEFT_SHIFT, "LOGICAL_LEFT_SHIFT"},
        {TOSALOGICAL_NOT, "LOGICAL_NOT"},
        {TOSALOGICAL_OR, "LOGICAL_OR"},
        {TOSALOGICAL_RIGHT_SHIFT, "LOGICAL_RIGHT_SHIFT"},
        {TOSALOGICAL_XOR, "LOGICAL_XOR"},
        {TOSAMATMUL, "MATMUL"},
        {TOSAMAX_POOL2D, "MAX_POOL2D"},
        {TOSAMAXIMUM, "MAXIMUM"},
        {TOSAMINIMUM, "MINIMUM"},
        {TOSAMUL, "MUL"},
        {TOSANEGATE, "NEGATE"},
        {TOSAPAD, "PAD"},
        {TOSAPOW, "POW"},
        {TOSARECIPROCAL, "RECIPROCAL"},
        {TOSAREDUCE_ALL, "REDUCE_ALL"},
        {TOSAREDUCE_ANY, "REDUCE_ANY"},
        {TOSAREDUCE_MAX, "REDUCE_MAX"},
        {TOSAREDUCE_MIN, "REDUCE_MIN"},
        {TOSAREDUCE_PRODUCT, "REDUCE_PRODUCT"},
        {TOSAREDUCE_SUM, "REDUCE_SUM"},
        {TOSARESCALE, "RESCALE"},
        {TOSARESHAPE, "RESHAPE"},
        {TOSARESIZE, "RESIZE"},
        {TOSAREVERSE, "REVERSE"},
        {TOSARFFT2D, "RFFT2D"},
        {TOSARSQRT, "RSQRT"},
        {TOSASCATTER, "SCATTER"},
        {TOSASELECT, "SELECT"},
        {TOSASIGMOID, "SIGMOID"},
        {TOSASIN, "SIN"},
        {TOSASLICE, "SLICE"},
        {TOSASUB, "SUB"},
        {TOSATABLE, "TABLE"},
        {TOSATANH, "TANH"},
        {TOSATILE, "TILE"},
        {TOSATRANSPOSE, "TRANSPOSE"},
        {TOSATRANSPOSE_CONV2D, "TRANSPOSE_CONV2D"},
    };
    std::string debugName = context.debugName(opExtInst, opNameMap.count(tosa) ? opNameMap.at(tosa) : "UNKNOWN");

    switch (tosa) {
    case TOSAABS:
        handleElementwiseUnary(opExtInst, debugName, &GraphPipeline::makeAbs);
        break;
    case TOSAADD:
        handleElementwiseBinary(opExtInst, debugName, &GraphPipeline::makeAdd);
        break;
    case TOSAARGMAX:
        handleArgmax(opExtInst, debugName);
        break;
    case TOSAARITHMETIC_RIGHT_SHIFT:
        handleArithmeticRightShift(opExtInst, debugName);
        break;
    case TOSAAVG_POOL2D:
        handleAvgPool2D(opExtInst, debugName);
        break;
    case TOSABITWISE_AND:
        handleElementwiseBinary(opExtInst, debugName, &GraphPipeline::makeBitwiseAnd);
        break;
    case TOSABITWISE_NOT:
        handleElementwiseUnary(opExtInst, debugName, &GraphPipeline::makeBitwiseNot);
        break;
    case TOSABITWISE_OR:
        handleElementwiseBinary(opExtInst, debugName, &GraphPipeline::makeBitwiseOr);
        break;
    case TOSABITWISE_XOR:
        handleElementwiseBinary(opExtInst, debugName, &GraphPipeline::makeBitwiseXor);
        break;
    case TOSACAST:
        handleCast(opExtInst, debugName);
        break;
    case TOSACEIL:
        handleElementwiseUnary(opExtInst, debugName, &GraphPipeline::makeCeil);
        break;
    case TOSACLAMP:
        handleClamp(opExtInst, debugName);
        break;
    case TOSACLZ:
        handleElementwiseUnary(opExtInst, debugName, &GraphPipeline::makeClz);
        break;
    case TOSACONCAT:
        handleConcat(opExtInst, debugName);
        break;
    case TOSACONV2D:
        handleConv2D(opExtInst, debugName);
        break;
    case TOSACONV3D:
        handleConv3D(opExtInst, debugName);
        break;
    case TOSACOS:
        handleElementwiseUnary(opExtInst, debugName, &GraphPipeline::makeCos);
        break;
    case TOSADEPTHWISE_CONV2D:
        handleDepthwiseConv2D(opExtInst, debugName);
        break;
    case TOSAEQUAL:
        handleElementwiseBinary(opExtInst, debugName, &GraphPipeline::makeEqual);
        break;
    case TOSAERF:
        handleElementwiseUnary(opExtInst, debugName, &GraphPipeline::makeErf);
        break;
    case TOSAEXP:
        handleElementwiseUnary(opExtInst, debugName, &GraphPipeline::makeExp);
        break;
    case TOSAFFT2D:
        handleFft2D(opExtInst, debugName);
        break;
    case TOSAFLOOR:
        handleElementwiseUnary(opExtInst, debugName, &GraphPipeline::makeFloor);
        break;
    case TOSAGATHER:
        handleGather(opExtInst, debugName);
        break;
    case TOSAGREATER:
        handleElementwiseBinary(opExtInst, debugName, &GraphPipeline::makeGreater);
        break;
    case TOSAGREATER_EQUAL:
        handleElementwiseBinary(opExtInst, debugName, &GraphPipeline::makeGreaterEqual);
        break;
    case TOSAINTDIV:
        handleElementwiseBinary(opExtInst, debugName, &GraphPipeline::makeIntdiv);
        break;
    case TOSALOG:
        handleElementwiseUnary(opExtInst, debugName, &GraphPipeline::makeLog);
        break;
    case TOSALOGICAL_AND:
        handleElementwiseBinary(opExtInst, debugName, &GraphPipeline::makeLogicalAnd);
        break;
    case TOSALOGICAL_LEFT_SHIFT:
        handleElementwiseBinary(opExtInst, debugName, &GraphPipeline::makeLogicalLeftShift);
        break;
    case TOSALOGICAL_NOT:
        handleElementwiseUnary(opExtInst, debugName, &GraphPipeline::makeLogicalNot);
        break;
    case TOSALOGICAL_OR:
        handleElementwiseBinary(opExtInst, debugName, &GraphPipeline::makeLogicalOr);
        break;
    case TOSALOGICAL_RIGHT_SHIFT:
        handleElementwiseBinary(opExtInst, debugName, &GraphPipeline::makeLogicalRightShift);
        break;
    case TOSALOGICAL_XOR:
        handleElementwiseBinary(opExtInst, debugName, &GraphPipeline::makeLogicalXor);
        break;
    case TOSAMATMUL:
        handleMatmul(opExtInst, debugName);
        break;
    case TOSAMAX_POOL2D:
        handleMaxPool2D(opExtInst, debugName);
        break;
    case TOSAMAXIMUM:
        handleMaximum(opExtInst, debugName);
        break;
    case TOSAMINIMUM:
        handleMinimum(opExtInst, debugName);
        break;
    case TOSAMUL:
        handleMul(opExtInst, debugName);
        break;
    case TOSANEGATE:
        handleNegate(opExtInst, debugName);
        break;
    case TOSAPAD:
        handlePad(opExtInst, debugName);
        break;
    case TOSAPOW:
        handleElementwiseBinary(opExtInst, debugName, &GraphPipeline::makePow);
        break;
    case TOSARECIPROCAL:
        handleElementwiseUnary(opExtInst, debugName, &GraphPipeline::makeReciprocal);
        break;
    case TOSAREDUCE_ALL:
        handleReduce(opExtInst, debugName, &GraphPipeline::makeReduceAll);
        break;
    case TOSAREDUCE_ANY:
        handleReduce(opExtInst, debugName, &GraphPipeline::makeReduceAny);
        break;
    case TOSAREDUCE_MAX:
        handleReduceMax(opExtInst, debugName);
        break;
    case TOSAREDUCE_MIN:
        handleReduceMin(opExtInst, debugName);
        break;
    case TOSAREDUCE_PRODUCT:
        handleReduce(opExtInst, debugName, &GraphPipeline::makeReduceProduct);
        break;
    case TOSAREDUCE_SUM:
        handleReduce(opExtInst, debugName, &GraphPipeline::makeReduceSum);
        break;
    case TOSARESCALE:
        handleRescale(opExtInst, debugName);
        break;
    case TOSARESHAPE:
        handleReshape(opExtInst, debugName);
        break;
    case TOSARESIZE:
        handleResize(opExtInst, debugName);
        break;
    case TOSAREVERSE:
        handleReverse(opExtInst, debugName);
        break;
    case TOSARFFT2D:
        handleRfft2D(opExtInst, debugName);
        break;
    case TOSARSQRT:
        handleElementwiseUnary(opExtInst, debugName, &GraphPipeline::makeRsqrt);
        break;
    case TOSASCATTER:
        handleScatter(opExtInst, debugName);
        break;
    case TOSASELECT:
        handleSelect(opExtInst, debugName);
        break;
    case TOSASIGMOID:
        handleElementwiseUnary(opExtInst, debugName, &GraphPipeline::makeSigmoid);
        break;
    case TOSASIN:
        handleElementwiseUnary(opExtInst, debugName, &GraphPipeline::makeSin);
        break;
    case TOSASLICE:
        handleSlice(opExtInst, debugName);
        break;
    case TOSASUB:
        handleElementwiseBinary(opExtInst, debugName, &GraphPipeline::makeSub);
        break;
    case TOSATABLE:
        handleTable(opExtInst, debugName);
        break;
    case TOSATANH:
        handleElementwiseUnary(opExtInst, debugName, &GraphPipeline::makeTanh);
        break;
    case TOSATILE:
        handleTile(opExtInst, debugName);
        break;
    case TOSATRANSPOSE:
        handleTranspose(opExtInst, debugName);
        break;
    case TOSATRANSPOSE_CONV2D:
        handleTransposeConv2D(opExtInst, debugName);
        break;
    default:
        throw std::runtime_error(std::string("Unsupported TOSA.001000.1 operand ") + std::to_string(tosa));
    }
}

void GraphTosa10ExtInst::handleArgmax(const Instruction *opExtInst, const std::string &debugName) const {
    // OpExtInst <result id> <OpExtInstImport id> ARGMAX axis nanMode input
    assert(opExtInst->NumInOperands() == 5);

    const auto &resultId = opExtInst->result_id();
    const auto &axis = context.getConstScalar<uint32_t>(opExtInst->GetInOperand(2));
    const auto &nanMode = context.getConstScalar<uint32_t>(opExtInst->GetInOperand(3));
    const auto &inputId = opExtInst->GetInOperand(4);

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << ',' << debugName << ", axis=" << axis
                             << ", nanMode=" << nanMode << ", input=%" << inputId.AsId() << std::endl;

    context.pipeline().makeArgmax(context.getTensor(inputId), context.getTensor(*opExtInst), axis, nanMode, debugName);
}

void GraphTosa10ExtInst::handleArithmeticRightShift(const Instruction *opExtInst, const std::string &debugName) const {
    // OpExtInst <result id> <OpExtInstImport id> ARITHMETIC_RIGHT_SHIFT round input1 input2
    assert(opExtInst->NumInOperands() == 5);

    const auto &resultId = opExtInst->result_id();
    const auto &round = context.getBoolConstant(opExtInst->GetInOperand(2));
    const auto &inputId1 = opExtInst->GetInOperand(3);
    const auto &inputId2 = opExtInst->GetInOperand(4);

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << ',' << debugName << ", round=" << round
                             << ", input1=%" << inputId1.AsId() << ", input2=%" << inputId2.AsId() << std::endl;

    context.pipeline().makeArithmeticRightShift(context.getTensor(inputId1), context.getTensor(inputId2),
                                                context.getTensor(*opExtInst), round, debugName);
}

void GraphTosa10ExtInst::handleAvgPool2D(const Instruction *opExtInst, const std::string &debugName) const {
    // OpExtInst <result id> <OpExtInstImport id> AVG_POOL2D kernel stride pad accType input inputZeroPoint
    // outputZeroPoint
    assert(opExtInst->NumInOperands() == 9);

    const auto &resultId = opExtInst->result_id();
    const auto &kernel = context.getConstVector<int32_t>(opExtInst->GetInOperand(2));
    const auto &stride = context.getConstVector<int32_t>(opExtInst->GetInOperand(3));
    const auto &pad = context.getConstVector<int32_t>(opExtInst->GetInOperand(4));
    const auto &accType = context.getConstScalar<uint32_t>(opExtInst->GetInOperand(5));
    const auto &inputId = opExtInst->GetInOperand(6);
    const auto &inputZeroPoint = context.getConstVector<int8_t>(opExtInst->GetInOperand(7));
    const auto &outputZeroPoint = context.getConstVector<int8_t>(opExtInst->GetInOperand(8));

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << ", " << debugName << ", kernel=" << kernel
                             << ", stride=" << stride << ", pad=" << pad << ", accType=" << accType
                             << ", inputZeroPoint=" << inputZeroPoint << ", outputZeroPoint=" << outputZeroPoint
                             << ", input=%" << inputId.AsId() << std::endl;

    context.pipeline().makeAvgPool2D(context.getTensor(inputId), context.getTensor(*opExtInst), kernel, stride, pad,
                                     accType, inputZeroPoint[0], outputZeroPoint[0], debugName);
}

void GraphTosa10ExtInst::handleCast(const Instruction *opExtInst, const std::string &debugName) const {
    // OpExtInst <result id> <OpExtInstImport id> CAST input
    assert(opExtInst->NumInOperands() == 3);

    const auto &resultId = opExtInst->result_id();
    const auto &inputId = opExtInst->GetInOperand(2);

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << ',' << debugName << ", input=%" << inputId.AsId()
                             << std::endl;

    context.pipeline().makeCast(context.getTensor(inputId), context.getTensor(*opExtInst), debugName);
}

void GraphTosa10ExtInst::handleClamp(const Instruction *opExtInst, const std::string &debugName) const {
    // OpExtInst <result id> <OpExtInstImport id> CLAMP minVal maxVal nanMode input
    assert(opExtInst->NumInOperands() == 6);

    auto getClampBound = [&](const Operand &operand) {
        const auto *constant = context.findConstant(operand.AsId());
        const auto *floatConstant = constant->AsFloatConstant();
        if (floatConstant != nullptr) {
            const auto *type = floatConstant->type()->AsFloat();
            if (isBFloat16(type)) {
                const uint32_t bits = uint32_t(uint16_t(floatConstant->words()[0])) << 16;
                float value = 0.0F;
                std::memcpy(&value, &bits, sizeof(value));
                return real_t(value);
            }
        }

        return context.getConstScalar<real_t>(constant);
    };

    const auto &resultId = opExtInst->result_id();
    const auto minVal = getClampBound(opExtInst->GetInOperand(2));
    const auto maxVal = getClampBound(opExtInst->GetInOperand(3));
    const auto nanMode = context.getConstScalar<uint32_t>(opExtInst->GetInOperand(4));
    const auto &inputId = opExtInst->GetInOperand(5);

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << ',' << debugName << ", minVal=" << minVal
                             << ", maxVal=" << maxVal << ", nanMode=" << nanMode << ", input=%" << inputId.AsId()
                             << std::endl;

    context.pipeline().makeClamp(context.getTensor(inputId), context.getTensor(*opExtInst), minVal, maxVal, nanMode,
                                 debugName);
}

void GraphTosa10ExtInst::handleConcat(const Instruction *opExtInst, const std::string &debugName) const {
    // OpExtInst <result id> <OpExtInstImport id> CONCAT axis [inputs]
    assert(opExtInst->NumInOperands() > 2);

    const auto &resultId = opExtInst->result_id();
    const auto &axis = context.getConstScalar<uint32_t>(opExtInst->GetInOperand(2));

    std::vector<std::shared_ptr<TensorDescriptor>> inputs;
    std::string inputsStr;
    for (uint32_t i = 3; i < opExtInst->NumInOperands(); i++) {
        inputs.push_back(context.getTensor(opExtInst->GetInOperand(i)));
        inputsStr += ", input" + std::to_string(i - 3) + "=%" + std::to_string(opExtInst->GetInOperand(i).AsId());
    }

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << ',' << debugName << ", axis=" << axis << inputsStr
                             << std::endl;

    context.pipeline().makeConcat(inputs, context.getTensor(*opExtInst), axis, debugName);
}

void GraphTosa10ExtInst::handleConv2D(const Instruction *opExtInst, const std::string &debugName) const {
    // OpExtInst <result id> <OpExtInstImport id> CONV2D pad stride dilation accType localBound input weight bias
    // inputZeroPoint weightZeroPoint
    assert(opExtInst->NumInOperands() == 12);

    const auto &resultId = opExtInst->result_id();
    const auto &pad = context.getConstVector<int32_t>(opExtInst->GetInOperand(2));
    const auto &stride = context.getConstVector<int32_t>(opExtInst->GetInOperand(3));
    const auto &dilation = context.getConstVector<int32_t>(opExtInst->GetInOperand(4));
    const auto &accType = context.getConstScalar<uint32_t>(opExtInst->GetInOperand(5));
    const auto &localBound = context.getBoolConstant(opExtInst->GetInOperand(6));
    const auto &inputId = opExtInst->GetInOperand(7);
    const auto &weightId = opExtInst->GetInOperand(8);
    const auto &biasId = opExtInst->GetInOperand(9);
    const auto &inputZeroPoint = context.getConstVector<int8_t>(opExtInst->GetInOperand(10));
    const auto &weightZeroPoint = context.getConstVector<int8_t>(opExtInst->GetInOperand(11));

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << ',' << debugName << ", pad=" << pad
                             << ", stride=" << stride << ", dilation=" << dilation << ", accType=" << accType
                             << ", localBound=" << localBound << ", input=%" << inputId.AsId() << ", weight=%"
                             << weightId.AsId() << ", bias=%" << biasId.AsId() << ", inputZeroPoint=" << inputZeroPoint
                             << ", weightZeroPoint=" << weightZeroPoint << std::endl;

    context.pipeline().makeConv2D(context.getTensor(inputId), context.getTensor(*opExtInst),
                                  context.getTensor(weightId), context.getTensor(biasId), pad, stride, dilation,
                                  inputZeroPoint[0], weightZeroPoint[0], accType, debugName);
}

void GraphTosa10ExtInst::handleConv3D(const Instruction *opExtInst, const std::string &debugName) const {
    // OpExtInst <result id> <OpExtInstImport id> CONV3D pad stride dilation accType localBound input weight bias
    // inputZeroPoint weightZeroPoint
    assert(opExtInst->NumInOperands() == 12);

    const auto &resultId = opExtInst->result_id();
    const auto &pad = context.getConstVector<int32_t>(opExtInst->GetInOperand(2));
    const auto &stride = context.getConstVector<int32_t>(opExtInst->GetInOperand(3));
    const auto &dilation = context.getConstVector<int32_t>(opExtInst->GetInOperand(4));
    const auto &accType = context.getConstScalar<uint32_t>(opExtInst->GetInOperand(5));
    const auto &localBound = context.getBoolConstant(opExtInst->GetInOperand(6));
    const auto &inputId = opExtInst->GetInOperand(7);
    const auto &weightId = opExtInst->GetInOperand(8);
    const auto &biasId = opExtInst->GetInOperand(9);
    const auto &inputZeroPoint = context.getConstVector<int8_t>(opExtInst->GetInOperand(10));
    const auto &weightZeroPoint = context.getConstVector<int8_t>(opExtInst->GetInOperand(11));

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << ',' << debugName << ", pad=" << pad
                             << ", stride=" << stride << ", dilation=" << dilation << ", accType=" << accType
                             << ", localBound=" << localBound << ", input=%" << inputId.AsId() << ", weight=%"
                             << weightId.AsId() << ", bias=%" << biasId.AsId() << ", inputZeroPoint=" << inputZeroPoint
                             << ", weightZeroPoint=" << weightZeroPoint << std::endl;

    context.pipeline().makeConv3D(context.getTensor(inputId), context.getTensor(*opExtInst),
                                  context.getTensor(weightId), context.getTensor(biasId), pad, stride, dilation,
                                  inputZeroPoint[0], weightZeroPoint[0], accType, debugName);
}

void GraphTosa10ExtInst::handleDepthwiseConv2D(const Instruction *opExtInst, const std::string &debugName) const {
    // OpExtInst <result id> <OpExtInstImport id> DEPTHWISE_CONV2D pad stride dilation accType localBound input weight
    // bias inputZeroPoint weightZeroPoint
    assert(opExtInst->NumInOperands() == 12);

    const auto &resultId = opExtInst->result_id();
    const auto &pad = context.getConstVector<int32_t>(opExtInst->GetInOperand(2));
    const auto &stride = context.getConstVector<int32_t>(opExtInst->GetInOperand(3));
    const auto &dilation = context.getConstVector<int32_t>(opExtInst->GetInOperand(4));
    const auto &accType = context.getConstScalar<uint32_t>(opExtInst->GetInOperand(5));
    const auto &localBound = context.getBoolConstant(opExtInst->GetInOperand(6));
    const auto &inputId = opExtInst->GetInOperand(7);
    const auto &weightId = opExtInst->GetInOperand(8);
    const auto &biasId = opExtInst->GetInOperand(9);
    const auto &inputZeroPoint = context.getConstVector<int8_t>(opExtInst->GetInOperand(10));
    const auto &weightZeroPoint = context.getConstVector<int8_t>(opExtInst->GetInOperand(11));

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << ',' << debugName << ", pad=" << pad
                             << ", stride=" << stride << ", dilation=" << dilation << ", accType=" << accType
                             << ", localBound=" << localBound << ", input=%" << inputId.AsId() << ", weight=%"
                             << weightId.AsId() << ", bias=%" << biasId.AsId() << ", inputZeroPoint=" << inputZeroPoint
                             << ", weightZeroPoint=" << weightZeroPoint << std::endl;

    context.pipeline().makeDepthwiseConv2D(context.getTensor(inputId), context.getTensor(*opExtInst),
                                           context.getTensor(weightId), context.getTensor(biasId), pad, stride,
                                           dilation, inputZeroPoint[0], weightZeroPoint[0], accType, debugName);
}

void GraphTosa10ExtInst::handleElementwiseBinary(
    const Instruction *opExtInst, const std::string &debugName,
    const std::function<void(GraphPipeline *, const std::shared_ptr<TensorDescriptor> &,
                             const std::shared_ptr<TensorDescriptor> &, const std::shared_ptr<TensorDescriptor> &,
                             const std::string &)> &function) const {
    // OpExtInst <result id> <OpExtInstImport id> OPERATION input1 input2
    assert(opExtInst->NumInOperands() == 4);

    const auto &resultId = opExtInst->result_id();
    const auto &inputId1 = opExtInst->GetInOperand(2);
    const auto &inputId2 = opExtInst->GetInOperand(3);

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << ", " << debugName << ", input1=%" << inputId1.AsId()
                             << ", input2=%" << inputId2.AsId() << std::endl;

    std::invoke(function, &context.pipeline(), context.getTensor(inputId1), context.getTensor(inputId2),
                context.getTensor(*opExtInst), debugName);
}

void GraphTosa10ExtInst::handleElementwiseUnary(
    const Instruction *opExtInst, const std::string &debugName,
    const std::function<void(GraphPipeline *, const std::shared_ptr<TensorDescriptor> &,
                             const std::shared_ptr<TensorDescriptor> &, const std::string &)> &function) const {
    // OpExtInst <result id> <OpExtInstImport id> OPERATION input1
    assert(opExtInst->NumInOperands() == 3);

    const auto &resultId = opExtInst->result_id();
    const auto &inputId1 = opExtInst->GetInOperand(2);

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << ',' << debugName << ", input1=%" << inputId1.AsId()
                             << std::endl;

    std::invoke(function, &context.pipeline(), context.getTensor(inputId1), context.getTensor(*opExtInst), debugName);
}

void GraphTosa10ExtInst::handleFft2D(const Instruction *opExtInst, const std::string &debugName) const {
    // OpExtInst <result id> <OpExtInstImport id> FFT2D inverse localBound input_real input_imag
    assert(opExtInst->NumInOperands() == 6);

    const auto &resultId = opExtInst->result_id();
    const auto &inverse = context.getBoolConstant(opExtInst->GetInOperand(2));
    const auto &localBound = context.getBoolConstant(opExtInst->GetInOperand(3));
    const auto &inputRealId = opExtInst->GetInOperand(4);
    const auto &inputImagId = opExtInst->GetInOperand(5);

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << ',' << debugName << ", inverse=" << inverse
                             << ", localBound=" << localBound << ", inputReal=%" << inputRealId.AsId()
                             << ", inputImag=%" << inputImagId.AsId() << std::endl;

    context.pipeline().makeFft2D(context.getTensor(inputRealId), context.getTensor(inputImagId),
                                 context.getTensor(*opExtInst, 0), context.getTensor(*opExtInst, 1), inverse,
                                 debugName);
}

void GraphTosa10ExtInst::handleGather(const Instruction *opExtInst, const std::string &debugName) const {
    // OpExtInst <result id> <OpExtInstImport id> GATHER values indices
    assert(opExtInst->NumInOperands() == 4);

    const auto &resultId = opExtInst->result_id();
    const auto &valuesId = opExtInst->GetInOperand(2);
    const auto &indicesId = opExtInst->GetInOperand(3);

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << ',' << debugName << ", values=%" << valuesId.AsId()
                             << ", indices=%" << indicesId.AsId() << std::endl;

    context.pipeline().makeGather(context.getTensor(valuesId), context.getTensor(indicesId),
                                  context.getTensor(*opExtInst), debugName);
}

void GraphTosa10ExtInst::handleMatmul(const Instruction *opExtInst, const std::string &debugName) const {
    // OpExtInst <result id> <OpExtInstImport id> MATMUL input1 input2 input1ZeroPoint input2ZeroPoint
    assert(opExtInst->NumInOperands() == 6);

    const auto &resultId = opExtInst->result_id();
    const auto &inputId1 = opExtInst->GetInOperand(2);
    const auto &inputId2 = opExtInst->GetInOperand(3);
    const auto &input1ZeroPoint = context.getConstVector<int8_t>(opExtInst->GetInOperand(4));
    const auto &input2ZeroPoint = context.getConstVector<int8_t>(opExtInst->GetInOperand(5));

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << ',' << debugName << ", input1=%" << inputId1.AsId()
                             << ", input2=%" << inputId2.AsId() << ", input1ZeroPoint=" << input1ZeroPoint
                             << ", input2ZeroPoint=" << input2ZeroPoint << std::endl;

    context.pipeline().makeMatmul(context.getTensor(inputId1), context.getTensor(inputId2),
                                  context.getTensor(*opExtInst), input1ZeroPoint[0], input2ZeroPoint[0], debugName);
}

void GraphTosa10ExtInst::handleMaximum(const Instruction *opExtInst, const std::string &debugName) const {
    // OpExtInst <result id> <OpExtInstImport id> MAXIMUM nanMode input1 input2
    assert(opExtInst->NumInOperands() == 5);

    const auto &resultId = opExtInst->result_id();
    const auto &nanMode = context.getConstScalar<uint32_t>(opExtInst->GetInOperand(2));
    const auto &inputId1 = opExtInst->GetInOperand(3);
    const auto &inputId2 = opExtInst->GetInOperand(4);

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << ',' << debugName << ", nanMode=" << nanMode
                             << ", input1=%" << inputId1.AsId() << ", input2=%" << inputId2.AsId() << std::endl;

    context.pipeline().makeMaximum(context.getTensor(inputId1), context.getTensor(inputId2),
                                   context.getTensor(*opExtInst), nanMode, debugName);
}

void GraphTosa10ExtInst::handleMaxPool2D(const Instruction *opExtInst, const std::string &debugName) const {
    // OpExtInst <result id> <OpExtInstImport id> MAX_POOL2D kernel stride pad nanMode input
    assert(opExtInst->NumInOperands() == 7);

    const auto &resultId = opExtInst->result_id();
    const auto &kernel = context.getConstVector<int32_t>(opExtInst->GetInOperand(2));
    const auto &stride = context.getConstVector<int32_t>(opExtInst->GetInOperand(3));
    const auto &pad = context.getConstVector<int32_t>(opExtInst->GetInOperand(4));
    const auto &nanMode = context.getConstScalar<uint32_t>(opExtInst->GetInOperand(5));
    const auto &inputId = opExtInst->GetInOperand(6);

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << ',' << debugName << ", kernel=" << kernel
                             << ", stride=" << stride << ", pad=" << pad << ", nanMode=" << nanMode << ", input=%"
                             << inputId.AsId() << std::endl;

    context.pipeline().makeMaxPool2D(context.getTensor(inputId), context.getTensor(*opExtInst), kernel, stride, pad,
                                     nanMode, debugName);
}

void GraphTosa10ExtInst::handleMinimum(const Instruction *opExtInst, const std::string &debugName) const {
    // OpExtInst <result id> <OpExtInstImport id> MINIMUM nanMode input1 input2
    assert(opExtInst->NumInOperands() == 5);

    const auto &resultId = opExtInst->result_id();
    const auto &nanMode = context.getConstScalar<uint32_t>(opExtInst->GetInOperand(2));
    const auto &inputId1 = opExtInst->GetInOperand(3);
    const auto &inputId2 = opExtInst->GetInOperand(4);

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << ',' << debugName << ", nanMode=" << nanMode
                             << ", input1=%" << inputId1.AsId() << ", input2=%" << inputId2.AsId() << std::endl;

    context.pipeline().makeMinimum(context.getTensor(inputId1), context.getTensor(inputId2),
                                   context.getTensor(*opExtInst), nanMode, debugName);
}

void GraphTosa10ExtInst::handleMul(const Instruction *opExtInst, const std::string &debugName) const {
    // OpExtInst <result id> <OpExtInstImport id> MUL input1 input2 shift
    assert(opExtInst->NumInOperands() == 5);

    const auto &resultId = opExtInst->result_id();
    const auto &inputId1 = opExtInst->GetInOperand(2);
    const auto &inputId2 = opExtInst->GetInOperand(3);
    const auto &shift = context.getConstVector<uint8_t>(opExtInst->GetInOperand(4));

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << ',' << debugName << ", input1=%" << inputId1.AsId()
                             << ", input2=%" << inputId2.AsId() << ", shift=" << shift << std::endl;

    context.pipeline().makeMul(context.getTensor(inputId1), context.getTensor(inputId2), context.getTensor(*opExtInst),
                               shift[0], debugName);
}

void GraphTosa10ExtInst::handleNegate(const Instruction *opExtInst, const std::string &debugName) const {
    // OpExtInst <result id> <OpExtInstImport id> NEGATE input inputZeroPoint outputZeroPoint
    assert(opExtInst->NumInOperands() == 5);

    const auto &resultId = opExtInst->result_id();
    const auto &inputId = opExtInst->GetInOperand(2);
    const auto &inputZeroPoint = context.getConstVector<int32_t>(opExtInst->GetInOperand(3));
    const auto &outputZeroPoint = context.getConstVector<int32_t>(opExtInst->GetInOperand(4));

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << ',' << debugName << ", input=%" << inputId.AsId()
                             << ", inputZeroPoint=" << inputZeroPoint << ", outputZeroPoint=" << outputZeroPoint
                             << std::endl;

    context.pipeline().makeNegate(context.getTensor(inputId), context.getTensor(*opExtInst), inputZeroPoint[0],
                                  outputZeroPoint[0], debugName);
}

void GraphTosa10ExtInst::handlePad(const Instruction *opExtInst, const std::string &debugName) const {
    // OpExtInst <result id> <OpExtInstImport id> PAD input padding padConst
    assert(opExtInst->NumInOperands() == 5);

    const auto &resultId = opExtInst->result_id();
    const auto &inputId = opExtInst->GetInOperand(2);
    const auto output = context.getTensor(*opExtInst);
    const auto &padding = context.getOrMakeCompositeTensor(opExtInst->GetInOperand(3).AsId());
    real_t padConst = 0.0;
    int32_t padConstInt = 0;

    const auto vkFormat = output->getFormat();
    // Reduced-float constants are stored as raw payload bits.
    if (vkFormat == VK_FORMAT_R16_SFLOAT_FPENCODING_BFLOAT16_ARM ||
        vkFormat == VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E5M2_ARM ||
        vkFormat == VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E4M3_ARM) {
        const auto *constant = context.findConstant(opExtInst->GetInOperand(4).AsId());
        const auto *scalar = constant;

        const auto *composite = constant->AsCompositeConstant();
        if (composite != nullptr) {
            assert(composite->GetComponents().size() == 1);
            scalar = composite->GetComponents()[0];
        }

        if (scalar->AsNullConstant() != nullptr) {
            padConst = 0.0;
        } else {
            const auto *floatConstant = scalar->AsFloatConstant();
            if (floatConstant == nullptr) {
                throw std::runtime_error(
                    "Unsupported PAD constant encoding, expected scalar, null, or composite constant. Format: " +
                    std::to_string(vkFormat) + ", is composite: " + (composite != nullptr ? "true" : "false"));
            }
            const auto *floatType = floatConstant->type()->AsFloat();
            if (vkFormat == VK_FORMAT_R16_SFLOAT_FPENCODING_BFLOAT16_ARM) {
                if (!isBFloat16(floatType)) {
                    throw std::runtime_error("Unsupported BF16 PAD constant encoding, floatType: " +
                                             std::string(floatType->str()));
                }

                const auto bf16 = uint16_t(floatConstant->words()[0]);
                const uint32_t fp32Bits = uint32_t(bf16) << 16;
                float fp32Value = 0.0f;
                std::memcpy(&fp32Value, &fp32Bits, sizeof(fp32Bits));
                padConst = real_t(fp32Value);
            } else if (vkFormat == VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E5M2_ARM) {
                if (!isFloat8E5M2(floatType)) {
                    throw std::runtime_error("Unsupported FLOAT8E5M2 PAD constant encoding, floatType: " +
                                             std::string(floatType->str()));
                }

                const auto f8 = uint8_t(floatConstant->words()[0]);
                const auto &fp = reinterpret_cast<const float8_e5m2 &>(f8);
                padConst = real_t(fp);
            } else if (vkFormat == VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E4M3_ARM) {
                if (!isFloat8E4M3(floatType)) {
                    throw std::runtime_error("Unsupported FLOAT8E4M3 PAD constant encoding, floatType: " +
                                             std::string(floatType->str()));
                }

                const auto f8 = uint8_t(floatConstant->words()[0]);
                const auto &fp = reinterpret_cast<const float8_e4m3 &>(f8);
                padConst = real_t(fp);
            }
        }
    } else if (vkFormat == VK_FORMAT_R32_SINT) {
        const auto &padConstVector = context.getConstVector<int32_t>(opExtInst->GetInOperand(4));
        padConstInt = padConstVector[0];
        padConst = real_t(padConstInt);
    } else {
        const auto &padConstVector = context.getConstVector<real_t>(opExtInst->GetInOperand(4));
        padConst = padConstVector[0];
        padConstInt = int32_t(padConst);
    }

    graphLog(Severity::Info) << "OpExtInst result=" << resultId << ',' << debugName << ", padding=" << padding
                             << ", padConst=" << std::fixed << std::setprecision(0) << padConst << ", input=%"
                             << inputId.AsId() << std::endl;

    context.pipeline().makePad(context.getTensor(inputId), output, padding, padConst, padConstInt, debugName);
}

void GraphTosa10ExtInst::handleRescale(const Instruction *opExtInst, const std::string &debugName) const {
    // OpExtInst <result id> <OpExtInstImport id> RESCALE scale32 roundingMode perChannel inputUnsigned
    // outputUnsigned input multiplier shift inputZeroPoint outputZeroPoint inputUnsigned outputUnsigned input
    assert(opExtInst->NumInOperands() == 12);

    const auto &resultId = opExtInst->result_id();
    const auto &scale32 = context.getBoolConstant(opExtInst->GetInOperand(2));
    const auto &roundingMode = context.getConstScalar<uint32_t>(opExtInst->GetInOperand(3));
    const auto &perChannel = context.getBoolConstant(opExtInst->GetInOperand(4));
    const auto &inputUnsigned = context.getBoolConstant(opExtInst->GetInOperand(5));
    const auto &outputUnsigned = context.getBoolConstant(opExtInst->GetInOperand(6));
    const auto &inputId = opExtInst->GetInOperand(7);
    const auto &multiplier = context.getOrMakeCompositeTensor(opExtInst->GetInOperand(8).AsId());
    const auto &shift = context.getOrMakeCompositeTensor(opExtInst->GetInOperand(9).AsId());
    const auto &inputZeroPoint = context.getConstVector<int32_t>(opExtInst->GetInOperand(10));
    const auto &outputZeroPoint = context.getConstVector<int32_t>(opExtInst->GetInOperand(11));

    graphLog(Severity::Info) << "OpExtInst result=" << resultId << ',' << debugName << ", scale32=" << scale32
                             << ", roundingRound=" << roundingMode << ", perChannel=" << perChannel
                             << ", inputUnsigned=" << inputUnsigned << ", outputUnsigned=" << outputUnsigned
                             << ", input=%" << inputId.AsId() << ", multiplier=" << multiplier << ", shift=" << shift
                             << ", inputZeroPoint=" << inputZeroPoint << ", outputZeroPoint=" << outputZeroPoint
                             << std::endl;

    const bool doubleRound = (roundingMode == RoundingMode::DoubleRound);

    context.pipeline().makeRescale(context.getTensor(inputId), context.getTensor(*opExtInst), inputZeroPoint[0],
                                   outputZeroPoint[0], multiplier, shift, scale32, doubleRound, perChannel,
                                   inputUnsigned, outputUnsigned, debugName);
}

void GraphTosa10ExtInst::handleReduce(
    const Instruction *opExtInst, const std::string &debugName,
    const std::function<void(GraphPipeline *, const std::shared_ptr<TensorDescriptor> &,
                             const std::shared_ptr<TensorDescriptor> &, const uint32_t, const std::string &)> &function)
    const {
    // OpExtInst <result id> <OpExtInstImport id> REDUCE_* axis input
    assert(opExtInst->NumInOperands() == 4);

    const auto &resultId = opExtInst->result_id();
    const auto &axis = context.getConstScalar<uint32_t>(opExtInst->GetInOperand(2));
    const auto &inputId = opExtInst->GetInOperand(3);

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << ", " << debugName << ", axis=" << axis
                             << ", input=%" << inputId.AsId() << std::endl;

    std::invoke(function, &context.pipeline(), context.getTensor(inputId), context.getTensor(*opExtInst), axis,
                debugName);
}

void GraphTosa10ExtInst::handleReduceMax(const Instruction *opExtInst, const std::string &debugName) const {
    // OpExtInst <result id> <OpExtInstImport id> REDUCE_MAX axis nanMode input
    assert(opExtInst->NumInOperands() == 5);

    const auto &resultId = opExtInst->result_id();
    const auto &axis = context.getConstScalar<uint32_t>(opExtInst->GetInOperand(2));
    const auto &nanMode = context.getConstScalar<uint32_t>(opExtInst->GetInOperand(3));
    const auto &inputId = opExtInst->GetInOperand(4);

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << ',' << debugName << ", axis=" << axis
                             << ", nanMode=" << nanMode << ", input=%" << inputId.AsId() << std::endl;

    context.pipeline().makeReduceMax(context.getTensor(inputId), context.getTensor(*opExtInst), axis, nanMode,
                                     debugName);
}

void GraphTosa10ExtInst::handleReduceMin(const Instruction *opExtInst, const std::string &debugName) const {
    // OpExtInst <result id> <OpExtInstImport id> REDUCE_MIN axis nanMode input
    assert(opExtInst->NumInOperands() == 5);

    const auto &resultId = opExtInst->result_id();
    const auto &axis = context.getConstScalar<uint32_t>(opExtInst->GetInOperand(2));
    const auto &nanMode = context.getConstScalar<uint32_t>(opExtInst->GetInOperand(3));
    const auto &inputId = opExtInst->GetInOperand(4);

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << ',' << debugName << ", axis=" << axis
                             << ", nanMode=" << nanMode << ", input=%" << inputId.AsId() << std::endl;

    context.pipeline().makeReduceMin(context.getTensor(inputId), context.getTensor(*opExtInst), axis, nanMode,
                                     debugName);
}

void GraphTosa10ExtInst::handleReshape(const Instruction *opExtInst, const std::string &debugName) const {
    // OpExtInst <result id> <OpExtInstImport id> RESHAPE input shape
    assert(opExtInst->NumInOperands() == 4);

    const auto &resultId = opExtInst->result_id();
    const auto &inputId = opExtInst->GetInOperand(2);
    const auto &shape = context.getConstVector<uint32_t>(opExtInst->GetInOperand(3));

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << ',' << debugName << ", input=%" << inputId.AsId()
                             << ", shape=" << shape << std::endl;

    context.pipeline().makeReshape(context.getTensor(inputId), context.getTensor(*opExtInst), debugName);
}

void GraphTosa10ExtInst::handleResize(const Instruction *opExtInst, const std::string &debugName) const {
    // OpExtInst <result id> <OpExtInstImport id> RESIZE mode input scale offset border
    assert(opExtInst->NumInOperands() == 7);

    const auto &resultId = opExtInst->result_id();
    const auto &mode = context.getConstScalar<uint32_t>(opExtInst->GetInOperand(2));
    const auto &inputId = opExtInst->GetInOperand(3);
    const auto &scale = context.getConstVector<int32_t>(opExtInst->GetInOperand(4));
    const auto &offset = context.getConstVector<int32_t>(opExtInst->GetInOperand(5));
    const auto &border = context.getConstVector<int32_t>(opExtInst->GetInOperand(6));

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << ',' << debugName << ", scale=" << scale
                             << ", offset=" << offset << ", border=" << border << ", mode=" << mode << ", input=%"
                             << inputId.AsId() << std::endl;

    context.pipeline().makeResize(context.getTensor(inputId), context.getTensor(*opExtInst), scale, offset, border,
                                  mode, debugName);
}

void GraphTosa10ExtInst::handleReverse(const Instruction *opExtInst, const std::string &debugName) const {
    // OpExtInst <result id> <OpExtInstImport id> REVERSE axis input
    assert(opExtInst->NumInOperands() == 4);

    const auto &resultId = opExtInst->result_id();
    const auto &axis = context.getConstScalar<uint32_t>(opExtInst->GetInOperand(2));
    const auto &inputId = opExtInst->GetInOperand(3);

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << ',' << debugName << ", axis=" << axis << ", input=%"
                             << inputId.AsId() << std::endl;

    context.pipeline().makeReverse(context.getTensor(inputId), context.getTensor(*opExtInst), axis, debugName);
}

void GraphTosa10ExtInst::handleRfft2D(const Instruction *opExtInst, const std::string &debugName) const {
    // OpExtInst <result id> <OpExtInstImport id> RFFT2D localBound input
    assert(opExtInst->NumInOperands() == 4);

    const auto &resultId = opExtInst->result_id();
    const auto &localBound = context.getBoolConstant(opExtInst->GetInOperand(2));
    const auto &inputId = opExtInst->GetInOperand(3);

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << ',' << debugName << ", localBound=" << localBound
                             << ", input=%" << inputId.AsId() << std::endl;

    context.pipeline().makeRfft2D(context.getTensor(inputId), context.getTensor(*opExtInst, 0),
                                  context.getTensor(*opExtInst, 1), debugName);
}

void GraphTosa10ExtInst::handleScatter(const Instruction *opExtInst, const std::string &debugName) const {
    // OpExtInst <result id> <OpExtInstImport id> SCATTER valuesIn indices input
    assert(opExtInst->NumInOperands() == 5);

    const auto &resultId = opExtInst->result_id();
    const auto &valuesInId = opExtInst->GetInOperand(2);
    const auto &indicesId = opExtInst->GetInOperand(3);
    const auto &inputId = opExtInst->GetInOperand(4);

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << ',' << debugName << ", valuesIn=%" << inputId.AsId()
                             << ", indices=%" << indicesId.AsId() << ", input=%" << inputId.AsId() << std::endl;

    context.pipeline().makeScatter(context.getTensor(inputId), context.getTensor(valuesInId),
                                   context.getTensor(indicesId), context.getTensor(*opExtInst), debugName);
}

void GraphTosa10ExtInst::handleSelect(const Instruction *opExtInst, const std::string &debugName) const {
    // OpExtInst <result id> <OpExtInstImport id> SELECT input1 input2 input3
    assert(opExtInst->NumInOperands() == 5);

    const auto &resultId = opExtInst->result_id();
    const auto &inputId1 = opExtInst->GetInOperand(2);
    const auto &inputId2 = opExtInst->GetInOperand(3);
    const auto &inputId3 = opExtInst->GetInOperand(4);

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << ',' << debugName << ", input1=%" << inputId1.AsId()
                             << ", input2=%" << inputId2.AsId() << ", input3=%" << inputId3.AsId() << std::endl;

    context.pipeline().makeSelect(context.getTensor(inputId1), context.getTensor(inputId2), context.getTensor(inputId3),
                                  context.getTensor(*opExtInst), debugName);
}

void GraphTosa10ExtInst::handleSlice(const Instruction *opExtInst, const std::string &debugName) const {
    // OpExtInst <result id> <OpExtInstImport id> SLICE start size input
    assert(opExtInst->NumInOperands() == 5);

    const auto &resultId = opExtInst->result_id();
    const auto &inputId = opExtInst->GetInOperand(2);
    const auto &start = context.getConstVector<uint32_t>(opExtInst->GetInOperand(3));
    const auto &size = context.getConstVector<uint32_t>(opExtInst->GetInOperand(4));

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << ',' << debugName << " , input=%" << inputId.AsId()
                             << ", start=" << start << ", size=" << size << std::endl;

    context.pipeline().makeSlice(context.getTensor(inputId), context.getTensor(*opExtInst), start, debugName);
}

void GraphTosa10ExtInst::handleTable(const Instruction *opExtInst, const std::string &debugName) const {
    // OpExtInst <result id> <OpExtInstImport id> TABLE %input table
    assert(opExtInst->NumInOperands() == 4);

    const auto &resultId = opExtInst->result_id();
    const auto &inputId = opExtInst->GetInOperand(2);
    const auto &table = context.getOrMakeCompositeTensor(opExtInst->GetInOperand(3).AsId());

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << ',' << debugName << ", input=%" << inputId.AsId()
                             << ", table=" << table << std::endl;

    context.pipeline().makeTable(context.getTensor(inputId), context.getTensor(*opExtInst), table, debugName);
}

void GraphTosa10ExtInst::handleTile(const Instruction *opExtInst, const std::string &debugName) const {
    // OpExtInst <result id> <OpExtInstImport id> TILE input multiplies
    assert(opExtInst->NumInOperands() == 4);

    const auto &resultId = opExtInst->result_id();
    const auto &inputId = opExtInst->GetInOperand(2);
    const auto &multiples = context.getConstVector<uint32_t>(opExtInst->GetInOperand(3));

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << ',' << debugName << ", input=%" << inputId.AsId()
                             << ", multiples=" << multiples << std::endl;

    context.pipeline().makeTile(context.getTensor(inputId), context.getTensor(*opExtInst), debugName);
}

void GraphTosa10ExtInst::handleTranspose(const Instruction *opExtInst, const std::string &debugName) const {
    // OpExtInst <result id> <OpExtInstImport id> TRANSPOSE perms input
    assert(opExtInst->NumInOperands() == 4);

    const auto &resultId = opExtInst->result_id();
    const auto &perms = context.getConstVector<uint32_t>(opExtInst->GetInOperand(2));
    const auto &inputId = opExtInst->GetInOperand(3);

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << ',' << debugName << ", perms=" << perms
                             << ", input=%" << inputId.AsId() << std::endl;

    context.pipeline().makeTranspose(context.getTensor(inputId), context.getTensor(*opExtInst), perms, debugName);
}

void GraphTosa10ExtInst::handleTransposeConv2D(const Instruction *opExtInst, const std::string &debugName) const {
    // OpExtInst <result id> <OpExtInstImport id> TRANSPOSE_CONV2D outPad stride accType localBound input weight
    // bias inputZeroPoint weightZeroPoint
    assert(opExtInst->NumInOperands() == 11);

    const auto &resultId = opExtInst->result_id();
    const auto &outPad = context.getConstVector<int32_t>(opExtInst->GetInOperand(2));
    const auto &stride = context.getConstVector<int32_t>(opExtInst->GetInOperand(3));
    const auto &accType = context.getConstScalar<uint32_t>(opExtInst->GetInOperand(4));
    const auto &localBound = context.getBoolConstant(opExtInst->GetInOperand(5));
    const auto &inputId = opExtInst->GetInOperand(6);
    const auto &weightId = opExtInst->GetInOperand(7);
    const auto &biasId = opExtInst->GetInOperand(8);
    const auto &inputZeroPoint = context.getConstVector<int8_t>(opExtInst->GetInOperand(9));
    const auto &weightZeroPoint = context.getConstVector<int8_t>(opExtInst->GetInOperand(10));

    graphLog(Severity::Info) << "OpExtInst result=" << resultId << ',' << debugName << " , outPad=" << outPad
                             << ", stride=" << stride << ", accType=" << accType << ", localBound=" << localBound
                             << ", input=%" << inputId.AsId() << ", weight=%" << weightId.AsId() << ", bias=%"
                             << biasId.AsId() << ", inputZeroPoint=" << inputZeroPoint
                             << ", weightZeroPoint=" << weightZeroPoint << std::endl;

    context.pipeline().makeTransposeConv2D(context.getTensor(inputId), context.getTensor(*opExtInst),
                                           context.getTensor(weightId), context.getTensor(biasId), outPad, stride,
                                           inputZeroPoint[0], weightZeroPoint[0], accType, debugName);
}

} // namespace spvtools::opt
