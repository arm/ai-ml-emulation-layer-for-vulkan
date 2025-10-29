/*
 * SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "spirv_pass_tosaspv_v100.hpp"
#include "graph_log.hpp"

#include <spirv/unified1/ArmMotionEngine.100.h>
#include <spirv/unified1/TOSA.001000.1.h>

using namespace mlsdk::el::log;
using namespace mlsdk::el::compute;

/*******************************************************************************
 * GraphPass TosaSpv100
 *******************************************************************************/
namespace spvtools::opt {

void GraphPassTosaSpv100::handleGraph(const Graph *graph) {
    // Iterate over instructions in the graph
    for (const auto &opExtInst : graph->instructions()) {

        // OpExtInst <result id> <OpExtInstImport id> <tosa operation> [arguments]
        switch (opExtInst->opcode()) {
        case spv::Op::OpExtInst:
            break;
        case spv::Op::OpCompositeExtract:
            continue;
        default:
            throw std::runtime_error(std::string("Unsupported graph instruction ") +
                                     std::to_string(static_cast<unsigned>(opExtInst->opcode())));
        }

        const auto &resultId = opExtInst->GetInOperand(0);
        const auto &importInstr = get_def_use_mgr()->GetDef(resultId.AsId());
        const auto &importName = importInstr->GetInOperand(0).AsString();

        if (importName == tosaSpv100)
            handleTosaInst(&*opExtInst);
        else if (importName == motionEngine100)
            handleMotionEngineInst(&*opExtInst);
        else {
            throw std::runtime_error(std::string("Unsupported extension ") + importName);
        }
    }
}

void GraphPassTosaSpv100::handleTosaInst(const Instruction *opExtInst) {
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
    std::string debugName = extractDebugInfoFromSPV(opExtInst, opNameMap.count(tosa) ? opNameMap.at(tosa) : "UNKNOWN");

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

void GraphPassTosaSpv100::handleMotionEngineInst(const Instruction *opExtInst) {
    const auto &motionEngine = ArmMotionEngineInstructions(opExtInst->GetInOperand(1).words[0]);

    // Verify that this is a Motion Engine external instruction
    static const std::unordered_map<ArmMotionEngineInstructions, std::string> opNameMap = {
        {ArmMotionEngineMIN_SAD, "MIN_SAD"},
        {ArmMotionEngineMIN_SAD_COST, "MIN_SAD_COST"},
        {ArmMotionEngineRAW_SAD, "RAW_SAD"},
    };
    std::string debugName =
        extractDebugInfoFromSPV(opExtInst, opNameMap.count(motionEngine) ? opNameMap.at(motionEngine) : "UNKNOWN");

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

void GraphPassTosaSpv100::handleArgmax(const Instruction *opExtInst, const std::string &debugName) {
    // OpExtInst <result id> <OpExtInstImport id> ARGMAX axis nanMode input
    assert(opExtInst->NumInOperands() == 5);

    const auto &resultId = opExtInst->result_id();
    const auto &axis = getConstant<uint32_t>(opExtInst->GetInOperand(2));
    const auto &nanMode = getConstant<uint32_t>(opExtInst->GetInOperand(3));
    const auto &inputId = opExtInst->GetInOperand(4);

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << "," << debugName << ", axis=" << axis
                             << ", nanMode=" << nanMode << ", input=%" << inputId.AsId() << std::endl;

    graphPipeline.makeArgmax(getTensor(inputId), getTensor(*opExtInst), axis, nanMode, debugName);
}

void GraphPassTosaSpv100::handleArithmeticRightShift(const Instruction *opExtInst, const std::string &debugName) {
    // OpExtInst <result id> <OpExtInstImport id> ARITHMETIC_RIGHT_SHIFT round input1 input2
    assert(opExtInst->NumInOperands() == 5);

    const auto &resultId = opExtInst->result_id();
    const auto &round = getBoolConstant(opExtInst->GetInOperand(2));
    const auto &inputId1 = opExtInst->GetInOperand(3);
    const auto &inputId2 = opExtInst->GetInOperand(4);

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << "," << debugName << ", round=" << round
                             << ", input1=%" << inputId1.AsId() << ", input2=%" << inputId2.AsId() << std::endl;

    graphPipeline.makeArithmeticRightShift(getTensor(inputId1), getTensor(inputId2), getTensor(*opExtInst), round,
                                           debugName);
}

void GraphPassTosaSpv100::handleAvgPool2D(const Instruction *opExtInst, const std::string &debugName) {
    // OpExtInst <result id> <OpExtInstImport id> AVG_POOL2D kernel stride pad accType input inputZeroPoint
    // outputZeroPoint
    assert(opExtInst->NumInOperands() == 9);

    const auto &resultId = opExtInst->result_id();
    const auto &kernel = getConstVector(opExtInst->GetInOperand(2));
    const auto &stride = getConstVector(opExtInst->GetInOperand(3));
    const auto &pad = getConstVector(opExtInst->GetInOperand(4));
    const auto &accType = getConstant<uint32_t>(opExtInst->GetInOperand(5));
    const auto &inputId = opExtInst->GetInOperand(6);
    const auto &inputZeroPoint = getConstVector<int8_t>(opExtInst->GetInOperand(7));
    const auto &outputZeroPoint = getConstVector<int8_t>(opExtInst->GetInOperand(8));

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << ", " << debugName << ", kernel=" << kernel
                             << ", stride=" << stride << ", pad=" << pad << ", accType=" << accType
                             << ", inputZeroPoint=" << inputZeroPoint << ", outputZeroPoint=" << outputZeroPoint
                             << ", input=%" << inputId.AsId() << std::endl;

    graphPipeline.makeAvgPool2D(getTensor(inputId), getTensor(*opExtInst), kernel, stride, pad, accType,
                                inputZeroPoint[0], outputZeroPoint[0], debugName);
}

void GraphPassTosaSpv100::handleCast(const Instruction *opExtInst, const std::string &debugName) {
    // OpExtInst <result id> <OpExtInstImport id> CAST input
    assert(opExtInst->NumInOperands() == 3);

    const auto &resultId = opExtInst->result_id();
    const auto &inputId = opExtInst->GetInOperand(2);

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << "," << debugName << ", input=%" << inputId.AsId()
                             << std::endl;

    graphPipeline.makeCast(getTensor(inputId), getTensor(*opExtInst), debugName);
}

void GraphPassTosaSpv100::handleClamp(const Instruction *opExtInst, const std::string &debugName) {
    // OpExtInst <result id> <OpExtInstImport id> CLAMP minVal maxVal nanMode input
    assert(opExtInst->NumInOperands() == 6);

    const auto &resultId = opExtInst->result_id();
    const auto &minVal = getConstant<double>(opExtInst->GetInOperand(2));
    const auto &maxVal = getConstant<double>(opExtInst->GetInOperand(3));
    const auto &nanMode = getConstant<uint32_t>(opExtInst->GetInOperand(4));
    const auto &inputId = opExtInst->GetInOperand(5);

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << "," << debugName << ", minVal=" << minVal
                             << ", maxVal=" << maxVal << ", nanMode=" << nanMode << ", input=%" << inputId.AsId()
                             << std::endl;

    graphPipeline.makeClamp(getTensor(inputId), getTensor(*opExtInst), minVal, maxVal, nanMode, debugName);
}

void GraphPassTosaSpv100::handleConcat(const Instruction *opExtInst, const std::string &debugName) {
    // OpExtInst <result id> <OpExtInstImport id> CONCAT axis [inputs]
    assert(opExtInst->NumInOperands() > 2);

    const auto &resultId = opExtInst->result_id();
    const auto &axis = getConstant<uint32_t>(opExtInst->GetInOperand(2));

    std::vector<std::shared_ptr<TensorDescriptor>> inputs;
    std::string inputsStr;
    for (uint32_t i = 3; i < opExtInst->NumInOperands(); i++) {
        inputs.push_back(getTensor(opExtInst->GetInOperand(i)));
        inputsStr += ", input" + std::to_string(i - 3) + "=%" + std::to_string(opExtInst->GetInOperand(i).AsId());
    }

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << "," << debugName << ", axis=" << axis << inputsStr
                             << std::endl;

    graphPipeline.makeConcat(inputs, getTensor(*opExtInst), axis, debugName);
}

void GraphPassTosaSpv100::handleConv2D(const Instruction *opExtInst, const std::string &debugName) {
    // OpExtInst <result id> <OpExtInstImport id> CONV2D pad stride dilation accType localBound input weight bias
    // inputZeroPoint weightZeroPoint
    assert(opExtInst->NumInOperands() == 12);

    const auto &resultId = opExtInst->result_id();
    const auto &pad = getConstVector(opExtInst->GetInOperand(2));
    const auto &stride = getConstVector(opExtInst->GetInOperand(3));
    const auto &dilation = getConstVector(opExtInst->GetInOperand(4));
    const auto &accType = getConstant<uint32_t>(opExtInst->GetInOperand(5));
    const auto &localBound = getBoolConstant(opExtInst->GetInOperand(6));
    const auto &inputId = opExtInst->GetInOperand(7);
    const auto &weightId = opExtInst->GetInOperand(8);
    const auto &biasId = opExtInst->GetInOperand(9);
    const auto &inputZeroPoint = getConstVector<int8_t>(opExtInst->GetInOperand(10));
    const auto &weightZeroPoint = getConstVector<int8_t>(opExtInst->GetInOperand(11));

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << "," << debugName << ", pad=" << pad
                             << ", stride=" << stride << ", dilation=" << dilation << ", accType=" << accType
                             << ", localBound=" << localBound << ", input=%" << inputId.AsId() << ", weight=%"
                             << weightId.AsId() << ", bias=%" << biasId.AsId() << ", inputZeroPoint=" << inputZeroPoint
                             << ", weightZeroPoint=" << weightZeroPoint << std::endl;

    graphPipeline.makeConv2D(getTensor(inputId), getTensor(*opExtInst), getTensor(weightId), getTensor(biasId), pad,
                             stride, dilation, inputZeroPoint[0], weightZeroPoint[0], accType, debugName);
}

void GraphPassTosaSpv100::handleConv3D(const Instruction *opExtInst, const std::string &debugName) {
    // OpExtInst <result id> <OpExtInstImport id> CONV3D pad stride dilation accType localBound input weight bias
    // inputZeroPoint weightZeroPoint
    assert(opExtInst->NumInOperands() == 12);

    const auto &resultId = opExtInst->result_id();
    const auto &pad = getConstVector(opExtInst->GetInOperand(2));
    const auto &stride = getConstVector(opExtInst->GetInOperand(3));
    const auto &dilation = getConstVector(opExtInst->GetInOperand(4));
    const auto &accType = getConstant<uint32_t>(opExtInst->GetInOperand(5));
    const auto &localBound = getBoolConstant(opExtInst->GetInOperand(6));
    const auto &inputId = opExtInst->GetInOperand(7);
    const auto &weightId = opExtInst->GetInOperand(8);
    const auto &biasId = opExtInst->GetInOperand(9);
    const auto &inputZeroPoint = getConstVector<int8_t>(opExtInst->GetInOperand(10));
    const auto &weightZeroPoint = getConstVector<int8_t>(opExtInst->GetInOperand(11));

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << "," << debugName << ", pad=" << pad
                             << ", stride=" << stride << ", dilation=" << dilation << ", accType=" << accType
                             << ", localBound=" << localBound << ", input=%" << inputId.AsId() << ", weight=%"
                             << weightId.AsId() << ", bias=%" << biasId.AsId() << ", inputZeroPoint=" << inputZeroPoint
                             << ", weightZeroPoint=" << weightZeroPoint << std::endl;

    graphPipeline.makeConv3D(getTensor(inputId), getTensor(*opExtInst), getTensor(weightId), getTensor(biasId), pad,
                             stride, dilation, inputZeroPoint[0], weightZeroPoint[0], accType, debugName);
}

void GraphPassTosaSpv100::handleDepthwiseConv2D(const Instruction *opExtInst, const std::string &debugName) {
    // OpExtInst <result id> <OpExtInstImport id> DEPTHWISE_CONV2D pad stride dilation accType localBound input weight
    // bias inputZeroPoint weightZeroPoint
    assert(opExtInst->NumInOperands() == 12);

    const auto &resultId = opExtInst->result_id();
    const auto &pad = getConstVector(opExtInst->GetInOperand(2));
    const auto &stride = getConstVector(opExtInst->GetInOperand(3));
    const auto &dilation = getConstVector(opExtInst->GetInOperand(4));
    const auto &accType = getConstant<uint32_t>(opExtInst->GetInOperand(5));
    const auto &localBound = getBoolConstant(opExtInst->GetInOperand(6));
    const auto &inputId = opExtInst->GetInOperand(7);
    const auto &weightId = opExtInst->GetInOperand(8);
    const auto &biasId = opExtInst->GetInOperand(9);
    const auto &inputZeroPoint = getConstVector<int8_t>(opExtInst->GetInOperand(10));
    const auto &weightZeroPoint = getConstVector<int8_t>(opExtInst->GetInOperand(11));

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << "," << debugName << ", pad=" << pad
                             << ", stride=" << stride << ", dilation=" << dilation << ", accType=" << accType
                             << ", localBound=" << localBound << ", input=%" << inputId.AsId() << ", weight=%"
                             << weightId.AsId() << ", bias=%" << biasId.AsId() << ", inputZeroPoint=" << inputZeroPoint
                             << ", weightZeroPoint=" << weightZeroPoint << std::endl;

    graphPipeline.makeDepthwiseConv2D(getTensor(inputId), getTensor(*opExtInst), getTensor(weightId), getTensor(biasId),
                                      pad, stride, dilation, inputZeroPoint[0], weightZeroPoint[0], accType, debugName);
}

void GraphPassTosaSpv100::handleElementwiseBinary(
    const Instruction *opExtInst, const std::string &debugName,
    std::function<void(GraphPipeline *, const std::shared_ptr<TensorDescriptor> &,
                       const std::shared_ptr<TensorDescriptor> &, const std::shared_ptr<TensorDescriptor> &,
                       const std::string &)>
        function) {
    // OpExtInst <result id> <OpExtInstImport id> OPERATION input1 input2
    assert(opExtInst->NumInOperands() == 4);

    const auto &resultId = opExtInst->result_id();
    const auto &inputId1 = opExtInst->GetInOperand(2);
    const auto &inputId2 = opExtInst->GetInOperand(3);

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << ", " << debugName << ", input1=%" << inputId1.AsId()
                             << ", input2=%" << inputId2.AsId() << std::endl;

    std::invoke(function, &graphPipeline, getTensor(inputId1), getTensor(inputId2), getTensor(*opExtInst), debugName);
}

void GraphPassTosaSpv100::handleElementwiseUnary(
    const Instruction *opExtInst, const std::string &debugName,
    std::function<void(GraphPipeline *, const std::shared_ptr<TensorDescriptor> &,
                       const std::shared_ptr<TensorDescriptor> &, const std::string &)>
        function) {
    // OpExtInst <result id> <OpExtInstImport id> OPERATION input1
    assert(opExtInst->NumInOperands() == 3);

    const auto &resultId = opExtInst->result_id();
    const auto &inputId1 = opExtInst->GetInOperand(2);

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << "," << debugName << ", input1=%" << inputId1.AsId()
                             << std::endl;

    std::invoke(function, &graphPipeline, getTensor(inputId1), getTensor(*opExtInst), debugName);
}

void GraphPassTosaSpv100::handleFft2D(const Instruction *opExtInst, const std::string &debugName) {
    // OpExtInst <result id> <OpExtInstImport id> FFT2D inverse localBound input_real input_imag
    assert(opExtInst->NumInOperands() == 6);

    const auto &resultId = opExtInst->result_id();
    const auto &inverse = getBoolConstant(opExtInst->GetInOperand(2));
    const auto &localBound = getBoolConstant(opExtInst->GetInOperand(3));
    const auto &inputRealId = opExtInst->GetInOperand(4);
    const auto &inputImagId = opExtInst->GetInOperand(5);

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << "," << debugName << ", inverse=" << inverse
                             << ", localBound=" << localBound << ", inputReal=%" << inputRealId.AsId()
                             << ", inputImag=%" << inputImagId.AsId() << std::endl;

    graphPipeline.makeFft2D(getTensor(inputRealId), getTensor(inputImagId), getTensor(*opExtInst, 0),
                            getTensor(*opExtInst, 1), inverse, debugName);
}

void GraphPassTosaSpv100::handleGather(const Instruction *opExtInst, const std::string &debugName) {
    // OpExtInst <result id> <OpExtInstImport id> GATHER values indices
    assert(opExtInst->NumInOperands() == 4);

    const auto &resultId = opExtInst->result_id();
    const auto &valuesId = opExtInst->GetInOperand(2);
    const auto &indicesId = opExtInst->GetInOperand(3);

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << "," << debugName << ", values=%" << valuesId.AsId()
                             << ", indices=%" << indicesId.AsId() << std::endl;

    graphPipeline.makeGather(getTensor(valuesId), getTensor(indicesId), getTensor(*opExtInst), debugName);
}

void GraphPassTosaSpv100::handleMatmul(const Instruction *opExtInst, const std::string &debugName) {
    // OpExtInst <result id> <OpExtInstImport id> MATMUL input1 input2 input1ZeroPoint input2ZeroPoint
    assert(opExtInst->NumInOperands() == 6);

    const auto &resultId = opExtInst->result_id();
    const auto &inputId1 = opExtInst->GetInOperand(2);
    const auto &inputId2 = opExtInst->GetInOperand(3);
    const auto &input1ZeroPoint = getConstVector<int8_t>(opExtInst->GetInOperand(4));
    const auto &input2ZeroPoint = getConstVector<int8_t>(opExtInst->GetInOperand(5));

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << "," << debugName << ", input1=%" << inputId1.AsId()
                             << ", input2=%" << inputId2.AsId() << ", input1ZeroPoint=" << input1ZeroPoint
                             << ", input2ZeroPoint=" << input2ZeroPoint << std::endl;

    graphPipeline.makeMatmul(getTensor(inputId1), getTensor(inputId2), getTensor(*opExtInst), input1ZeroPoint[0],
                             input2ZeroPoint[0], debugName);
}

void GraphPassTosaSpv100::handleMaximum(const Instruction *opExtInst, const std::string &debugName) {
    // OpExtInst <result id> <OpExtInstImport id> MAXIMUM nanMode input1 input2
    assert(opExtInst->NumInOperands() == 5);

    const auto &resultId = opExtInst->result_id();
    const auto &nanMode = getConstant<uint32_t>(opExtInst->GetInOperand(2));
    const auto &inputId1 = opExtInst->GetInOperand(3);
    const auto &inputId2 = opExtInst->GetInOperand(4);

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << "," << debugName << ", nanMode=" << nanMode
                             << ", input1=%" << inputId1.AsId() << ", input2=%" << inputId2.AsId() << std::endl;

    graphPipeline.makeMaximum(getTensor(inputId1), getTensor(inputId2), getTensor(*opExtInst), nanMode, debugName);
}

void GraphPassTosaSpv100::handleMaxPool2D(const Instruction *opExtInst, const std::string &debugName) {
    // OpExtInst <result id> <OpExtInstImport id> MAX_POOL2D kernel stride pad nanMode input
    assert(opExtInst->NumInOperands() == 7);

    const auto &resultId = opExtInst->result_id();
    const auto &kernel = getConstVector(opExtInst->GetInOperand(2));
    const auto &stride = getConstVector(opExtInst->GetInOperand(3));
    const auto &pad = getConstVector(opExtInst->GetInOperand(4));
    const auto &nanMode = getConstant<uint32_t>(opExtInst->GetInOperand(5));
    const auto &inputId = opExtInst->GetInOperand(6);

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << "," << debugName << ", kernel=" << kernel
                             << ", stride=" << stride << ", pad=" << pad << ", nanMode=" << nanMode << ", input=%"
                             << inputId.AsId() << std::endl;

    graphPipeline.makeMaxPool2D(getTensor(inputId), getTensor(*opExtInst), kernel, stride, pad, nanMode, debugName);
}

void GraphPassTosaSpv100::handleMinimum(const Instruction *opExtInst, const std::string &debugName) {
    // OpExtInst <result id> <OpExtInstImport id> MINIMUM nanMode input1 input2
    assert(opExtInst->NumInOperands() == 5);

    const auto &resultId = opExtInst->result_id();
    const auto &nanMode = getConstant<uint32_t>(opExtInst->GetInOperand(2));
    const auto &inputId1 = opExtInst->GetInOperand(3);
    const auto &inputId2 = opExtInst->GetInOperand(4);

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << "," << debugName << ", nanMode=" << nanMode
                             << ", input1=%" << inputId1.AsId() << ", input2=%" << inputId2.AsId() << std::endl;

    graphPipeline.makeMinimum(getTensor(inputId1), getTensor(inputId2), getTensor(*opExtInst), nanMode, debugName);
}

void GraphPassTosaSpv100::handleMul(const Instruction *opExtInst, const std::string &debugName) {
    // OpExtInst <result id> <OpExtInstImport id> MUL input1 input2 shift
    assert(opExtInst->NumInOperands() == 5);

    const auto &resultId = opExtInst->result_id();
    const auto &inputId1 = opExtInst->GetInOperand(2);
    const auto &inputId2 = opExtInst->GetInOperand(3);
    const auto &shift = getConstVector<uint8_t>(opExtInst->GetInOperand(4));

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << "," << debugName << ", input1=%" << inputId1.AsId()
                             << ", input2=%" << inputId2.AsId() << ", shift=" << shift << std::endl;

    graphPipeline.makeMul(getTensor(inputId1), getTensor(inputId2), getTensor(*opExtInst), shift[0], debugName);
}

void GraphPassTosaSpv100::handleNegate(const Instruction *opExtInst, const std::string &debugName) {
    // OpExtInst <result id> <OpExtInstImport id> NEGATE input inputZeroPoint outputZeroPoint
    assert(opExtInst->NumInOperands() == 5);

    const auto &resultId = opExtInst->result_id();
    const auto &inputId = opExtInst->GetInOperand(2);
    const auto &inputZeroPoint = getConstVector<int32_t>(opExtInst->GetInOperand(3));
    const auto &outputZeroPoint = getConstVector<int32_t>(opExtInst->GetInOperand(4));

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << "," << debugName << ", input=%" << inputId.AsId()
                             << ", inputZeroPoint=" << inputZeroPoint << ", outputZeroPoint=" << outputZeroPoint
                             << std::endl;

    graphPipeline.makeNegate(getTensor(inputId), getTensor(*opExtInst), inputZeroPoint[0], outputZeroPoint[0],
                             debugName);
}

void GraphPassTosaSpv100::handlePad(const Instruction *opExtInst, const std::string &debugName) {
    // OpExtInst <result id> <OpExtInstImport id> PAD input padding padConst
    assert(opExtInst->NumInOperands() == 5);

    const auto &resultId = opExtInst->result_id();
    const auto &inputId = opExtInst->GetInOperand(2);
    const auto &padding = getOrMakeCompositeTensor(opExtInst->GetInOperand(3).AsId());
    const auto &padConst = getConstVector<double>(opExtInst->GetInOperand(4));

    graphLog(Severity::Info) << "OpExtInst result=" << resultId << "," << debugName << ", padding=" << padding
                             << ", padConst=" << std::fixed << std::setprecision(0) << padConst << ", input=%"
                             << inputId.AsId() << std::endl;

    graphPipeline.makePad(getTensor(inputId), getTensor(*opExtInst), padding, padConst[0], debugName);
}

void GraphPassTosaSpv100::handleRescale(const Instruction *opExtInst, const std::string &debugName) {
    // OpExtInst <result id> <OpExtInstImport id> RESCALE scale32 roundingMode perChannel inputUnsigned
    // outputUnsigned input multiplier shift inputZeroPoint outputZeroPoint inputUnsigned outputUnsigned input
    assert(opExtInst->NumInOperands() == 12);

    const auto &resultId = opExtInst->result_id();
    const auto &scale32 = getBoolConstant(opExtInst->GetInOperand(2));
    const auto &roundingMode = getConstant<uint32_t>(opExtInst->GetInOperand(3));
    const auto &perChannel = getBoolConstant(opExtInst->GetInOperand(4));
    const auto &inputUnsigned = getBoolConstant(opExtInst->GetInOperand(5));
    const auto &outputUnsigned = getBoolConstant(opExtInst->GetInOperand(6));
    const auto &inputId = opExtInst->GetInOperand(7);
    const auto &multiplier = getOrMakeCompositeTensor(opExtInst->GetInOperand(8).AsId());
    const auto &shift = getOrMakeCompositeTensor(opExtInst->GetInOperand(9).AsId());
    const auto &inputZeroPoint = getConstVector<int32_t>(opExtInst->GetInOperand(10));
    const auto &outputZeroPoint = getConstVector<int32_t>(opExtInst->GetInOperand(11));

    graphLog(Severity::Info) << "OpExtInst result=" << resultId << "," << debugName << ", scale32=" << scale32
                             << ", roundingRound=" << roundingMode << ", perChannel=" << perChannel
                             << ", inputUnsigned=" << inputUnsigned << ", outputUnsigned=" << outputUnsigned
                             << ", input=%" << inputId.AsId() << ", multiplier=" << multiplier << ", shift=" << shift
                             << ", inputZeroPoint=" << inputZeroPoint << ", outputZeroPoint=" << outputZeroPoint
                             << std::endl;

    const bool doubleRound = (roundingMode == RoundingMode::DoubleRound);

    graphPipeline.makeRescale(getTensor(inputId), getTensor(*opExtInst), inputZeroPoint[0], outputZeroPoint[0],
                              multiplier, shift, scale32, doubleRound, perChannel, inputUnsigned, outputUnsigned,
                              debugName);
}

void GraphPassTosaSpv100::handleReduce(
    const Instruction *opExtInst, const std::string &debugName,
    std::function<void(GraphPipeline *, const std::shared_ptr<TensorDescriptor> &,
                       const std::shared_ptr<TensorDescriptor> &, const uint32_t, const std::string &)>
        function) {
    // OpExtInst <result id> <OpExtInstImport id> REDUCE_* axis input
    assert(opExtInst->NumInOperands() == 4);

    const auto &resultId = opExtInst->result_id();
    const auto &axis = getConstant<uint32_t>(opExtInst->GetInOperand(2));
    const auto &inputId = opExtInst->GetInOperand(3);

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << ", " << debugName << ", axis=" << axis
                             << ", input=%" << inputId.AsId() << std::endl;

    std::invoke(function, &graphPipeline, getTensor(inputId), getTensor(*opExtInst), axis, debugName);
}

void GraphPassTosaSpv100::handleReduceMax(const Instruction *opExtInst, const std::string &debugName) {
    // OpExtInst <result id> <OpExtInstImport id> REDUCE_MAX axis nanMode input
    assert(opExtInst->NumInOperands() == 5);

    const auto &resultId = opExtInst->result_id();
    const auto &axis = getConstant<uint32_t>(opExtInst->GetInOperand(2));
    const auto &nanMode = getConstant<uint32_t>(opExtInst->GetInOperand(3));
    const auto &inputId = opExtInst->GetInOperand(4);

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << "," << debugName << ", axis=" << axis
                             << ", nanMode=" << nanMode << ", input=%" << inputId.AsId() << std::endl;

    graphPipeline.makeReduceMax(getTensor(inputId), getTensor(*opExtInst), axis, nanMode, debugName);
}

void GraphPassTosaSpv100::handleReduceMin(const Instruction *opExtInst, const std::string &debugName) {
    // OpExtInst <result id> <OpExtInstImport id> REDUCE_MIN axis nanMode input
    assert(opExtInst->NumInOperands() == 5);

    const auto &resultId = opExtInst->result_id();
    const auto &axis = getConstant<uint32_t>(opExtInst->GetInOperand(2));
    const auto &nanMode = getConstant<uint32_t>(opExtInst->GetInOperand(3));
    const auto &inputId = opExtInst->GetInOperand(4);

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << "," << debugName << ", axis=" << axis
                             << ", nanMode=" << nanMode << ", input=%" << inputId.AsId() << std::endl;

    graphPipeline.makeReduceMin(getTensor(inputId), getTensor(*opExtInst), axis, nanMode, debugName);
}

void GraphPassTosaSpv100::handleReshape(const Instruction *opExtInst, const std::string &debugName) {
    // OpExtInst <result id> <OpExtInstImport id> RESHAPE input shape
    assert(opExtInst->NumInOperands() == 4);

    const auto &resultId = opExtInst->result_id();
    const auto &inputId = opExtInst->GetInOperand(2);
    const auto &shape = getConstVector(opExtInst->GetInOperand(3));

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << "," << debugName << ", input=%" << inputId.AsId()
                             << ", shape=" << shape << std::endl;

    graphPipeline.makeReshape(getTensor(inputId), getTensor(*opExtInst), debugName);
}

void GraphPassTosaSpv100::handleResize(const Instruction *opExtInst, const std::string &debugName) {
    // OpExtInst <result id> <OpExtInstImport id> RESIZE mode input scale offset border
    assert(opExtInst->NumInOperands() == 7);

    const auto &resultId = opExtInst->result_id();
    const auto &mode = getConstant<uint32_t>(opExtInst->GetInOperand(2));
    const auto &inputId = opExtInst->GetInOperand(3);
    const auto &scale = getConstVector<int32_t>(opExtInst->GetInOperand(4));
    const auto &offset = getConstVector<int32_t>(opExtInst->GetInOperand(5));
    const auto &border = getConstVector<int32_t>(opExtInst->GetInOperand(6));

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << "," << debugName << ", scale=" << scale
                             << ", offset=" << offset << ", border=" << border << ", mode=" << mode << ", input=%"
                             << inputId.AsId() << std::endl;

    graphPipeline.makeResize(getTensor(inputId), getTensor(*opExtInst), scale, offset, border, mode, debugName);
}

void GraphPassTosaSpv100::handleReverse(const Instruction *opExtInst, const std::string &debugName) {
    // OpExtInst <result id> <OpExtInstImport id> REVERSE axis input
    assert(opExtInst->NumInOperands() == 4);

    const auto &resultId = opExtInst->result_id();
    const auto &axis = getConstant<uint32_t>(opExtInst->GetInOperand(2));
    const auto &inputId = opExtInst->GetInOperand(3);

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << "," << debugName << ", axis=" << axis << ", input=%"
                             << inputId.AsId() << std::endl;

    graphPipeline.makeReverse(getTensor(inputId), getTensor(*opExtInst), axis, debugName);
}

void GraphPassTosaSpv100::handleRfft2D(const Instruction *opExtInst, const std::string &debugName) {
    // OpExtInst <result id> <OpExtInstImport id> RFFT2D localBound input
    assert(opExtInst->NumInOperands() == 4);

    const auto &resultId = opExtInst->result_id();
    const auto &localBound = getBoolConstant(opExtInst->GetInOperand(2));
    const auto &inputId = opExtInst->GetInOperand(3);

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << "," << debugName << ", localBound=" << localBound
                             << ", input=%" << inputId.AsId() << std::endl;

    graphPipeline.makeRfft2D(getTensor(inputId), getTensor(*opExtInst, 0), getTensor(*opExtInst, 1), debugName);
}

void GraphPassTosaSpv100::handleScatter(const Instruction *opExtInst, const std::string &debugName) {
    // OpExtInst <result id> <OpExtInstImport id> SCATTER valuesIn indices input
    assert(opExtInst->NumInOperands() == 5);

    const auto &resultId = opExtInst->result_id();
    const auto &valuesInId = opExtInst->GetInOperand(2);
    const auto &indicesId = opExtInst->GetInOperand(3);
    const auto &inputId = opExtInst->GetInOperand(4);

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << "," << debugName << ", valuesIn=%" << inputId.AsId()
                             << ", indices=%" << indicesId.AsId() << ", input=%" << inputId.AsId() << std::endl;

    graphPipeline.makeScatter(getTensor(inputId), getTensor(valuesInId), getTensor(indicesId), getTensor(*opExtInst),
                              debugName);
}

void GraphPassTosaSpv100::handleSelect(const Instruction *opExtInst, const std::string &debugName) {
    // OpExtInst <result id> <OpExtInstImport id> SELECT input1 input2 input3
    assert(opExtInst->NumInOperands() == 5);

    const auto &resultId = opExtInst->result_id();
    const auto &inputId1 = opExtInst->GetInOperand(2);
    const auto &inputId2 = opExtInst->GetInOperand(3);
    const auto &inputId3 = opExtInst->GetInOperand(4);

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << "," << debugName << ", input1=%" << inputId1.AsId()
                             << ", input2=%" << inputId2.AsId() << ", input3=%" << inputId3.AsId() << std::endl;

    graphPipeline.makeSelect(getTensor(inputId1), getTensor(inputId2), getTensor(inputId3), getTensor(*opExtInst),
                             debugName);
}

void GraphPassTosaSpv100::handleSlice(const Instruction *opExtInst, const std::string &debugName) {
    // OpExtInst <result id> <OpExtInstImport id> SLICE start size input
    assert(opExtInst->NumInOperands() == 5);

    const auto &resultId = opExtInst->result_id();
    const auto &inputId = opExtInst->GetInOperand(2);
    const auto &start = getConstVector(opExtInst->GetInOperand(3));
    const auto &size = getConstVector(opExtInst->GetInOperand(4));

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << "," << debugName << " , input=%" << inputId.AsId()
                             << ", start=" << start << ", size=" << size << std::endl;

    graphPipeline.makeSlice(getTensor(inputId), getTensor(*opExtInst), start, debugName);
}

void GraphPassTosaSpv100::handleTable(const Instruction *opExtInst, const std::string &debugName) {
    // OpExtInst <result id> <OpExtInstImport id> TABLE %input table
    assert(opExtInst->NumInOperands() == 4);

    const auto &resultId = opExtInst->result_id();
    const auto &inputId = opExtInst->GetInOperand(2);
    const auto &table = getOrMakeCompositeTensor(opExtInst->GetInOperand(3).AsId());

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << "," << debugName << ", input=%" << inputId.AsId()
                             << ", table=" << table << std::endl;

    graphPipeline.makeTable(getTensor(inputId), getTensor(*opExtInst), table, debugName);
}

void GraphPassTosaSpv100::handleTile(const Instruction *opExtInst, const std::string &debugName) {
    // OpExtInst <result id> <OpExtInstImport id> TILE input multiplies
    assert(opExtInst->NumInOperands() == 4);

    const auto &resultId = opExtInst->result_id();
    const auto &inputId = opExtInst->GetInOperand(2);
    const auto &multiples = getConstVector(opExtInst->GetInOperand(3));

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << "," << debugName << ", input=%" << inputId.AsId()
                             << ", multiples=" << multiples << std::endl;

    graphPipeline.makeTile(getTensor(inputId), getTensor(*opExtInst), debugName);
}

void GraphPassTosaSpv100::handleTranspose(const Instruction *opExtInst, const std::string &debugName) {
    // OpExtInst <result id> <OpExtInstImport id> TRANSPOSE perms input
    assert(opExtInst->NumInOperands() == 4);

    const auto &resultId = opExtInst->result_id();
    const auto &perms = getConstVector(opExtInst->GetInOperand(2));
    const auto &inputId = opExtInst->GetInOperand(3);

    graphLog(Severity::Info) << "OpExtInst result=%" << resultId << "," << debugName << ", perms=" << perms
                             << ", input=%" << inputId.AsId() << std::endl;

    graphPipeline.makeTranspose(getTensor(inputId), getTensor(*opExtInst), perms, debugName);
}

void GraphPassTosaSpv100::handleTransposeConv2D(const Instruction *opExtInst, const std::string &debugName) {
    // OpExtInst <result id> <OpExtInstImport id> TRANSPOSE_CONV2D outPad stride accType localBound input weight
    // bias inputZeroPoint weightZeroPoint
    assert(opExtInst->NumInOperands() == 11);

    const auto &resultId = opExtInst->result_id();
    const auto &outPad = getConstVector(opExtInst->GetInOperand(2));
    const auto &stride = getConstVector(opExtInst->GetInOperand(3));
    const auto &accType = getConstant<uint32_t>(opExtInst->GetInOperand(4));
    const auto &localBound = getBoolConstant(opExtInst->GetInOperand(5));
    const auto &inputId = opExtInst->GetInOperand(6);
    const auto &weightId = opExtInst->GetInOperand(7);
    const auto &biasId = opExtInst->GetInOperand(8);
    const auto &inputZeroPoint = getConstVector<int8_t>(opExtInst->GetInOperand(9));
    const auto &weightZeroPoint = getConstVector<int8_t>(opExtInst->GetInOperand(10));

    graphLog(Severity::Info) << "OpExtInst result=" << resultId << "," << debugName << " , outPad=" << outPad
                             << ", stride=" << stride << ", accType=" << accType << ", localBound=" << localBound
                             << ", input=%" << inputId.AsId() << ", weight=%" << weightId.AsId() << ", bias=%"
                             << biasId.AsId() << ", inputZeroPoint=" << inputZeroPoint
                             << ", weightZeroPoint=" << weightZeroPoint << std::endl;

    graphPipeline.makeTransposeConv2D(getTensor(inputId), getTensor(*opExtInst), getTensor(weightId), getTensor(biasId),
                                      outPad, stride, inputZeroPoint[0], weightZeroPoint[0], accType, debugName);
}

void GraphPassTosaSpv100::handleMinSad(const Instruction *opExtInst, const std::string &debugName) {
    // OpExtInst <result id> <OpExtInstImport id> MIN_SAD kernel_sizes search_window_sizes input_strides
    // window_strides window_offsets padding search_pattern input0 input1
    assert(opExtInst->NumInOperands() == 11);

    const auto &resultId = opExtInst->result_id();
    const auto &kernelSizes = getConstVector(opExtInst->GetInOperand(2));
    const auto &searchWindowSizes = getConstVector(opExtInst->GetInOperand(3));
    const auto &inputStrides = getConstVector(opExtInst->GetInOperand(4));
    const auto &windowStrides = getConstVector(opExtInst->GetInOperand(5));
    const auto &windowOffsets = getConstVector(opExtInst->GetInOperand(6));
    const auto &padding = getConstVector(opExtInst->GetInOperand(7));
    const auto &searchPattern = getConstant<uint32_t>(opExtInst->GetInOperand(8));
    const auto &input0Id = opExtInst->GetInOperand(9);
    const auto &input1Id = opExtInst->GetInOperand(10);

    graphLog(Severity::Info) << "OpExtInst result=" << resultId << ", " << debugName << ", kernelSizes=" << kernelSizes
                             << ", searchWindowSizes=" << searchWindowSizes << ", inputStrides=" << inputStrides
                             << ", windowStrides=" << windowStrides << ", windowOffsets=" << windowOffsets
                             << ", padding=" << padding << ", searchPattern=" << searchPattern << ", input0=%"
                             << input0Id.AsId() << ", input1=%" << input1Id.AsId() << std::endl;

    graphPipeline.makeMinSad(getTensor(input0Id), getTensor(input1Id), getTensor(*opExtInst), kernelSizes,
                             searchWindowSizes, inputStrides, windowStrides, windowOffsets, padding, searchPattern,
                             debugName);
}

void GraphPassTosaSpv100::handleMinSadCost(const Instruction *opExtInst, const std::string &debugName) {
    // OpExtInst <result id> <OpExtInstImport id> MIN_SAD_COST kernel_sizes search_window_sizes input_strides
    // window_strides window_offsets padding search_pattern input0 input1
    assert(opExtInst->NumInOperands() == 11);

    const auto &resultId = opExtInst->result_id();
    const auto &kernelSizes = getConstVector(opExtInst->GetInOperand(2));
    const auto &searchWindowSizes = getConstVector(opExtInst->GetInOperand(3));
    const auto &inputStrides = getConstVector(opExtInst->GetInOperand(4));
    const auto &windowStrides = getConstVector(opExtInst->GetInOperand(5));
    const auto &windowOffsets = getConstVector(opExtInst->GetInOperand(6));
    const auto &padding = getConstVector(opExtInst->GetInOperand(7));
    const auto &searchPattern = getConstant<uint32_t>(opExtInst->GetInOperand(8));
    const auto &input0Id = opExtInst->GetInOperand(9);
    const auto &input1Id = opExtInst->GetInOperand(10);

    graphLog(Severity::Info) << "OpExtInst result=" << resultId << ", " << debugName << ", kernelSizes=" << kernelSizes
                             << ", searchWindowSizes=" << searchWindowSizes << ", inputStrides=" << inputStrides
                             << ", windowStrides=" << windowStrides << ", windowOffsets=" << windowOffsets
                             << ", padding=" << padding << ", searchPattern=" << searchPattern << ", input0=%"
                             << input0Id.AsId() << ", input1=%" << input1Id.AsId() << std::endl;

    graphPipeline.makeMinSadCost(getTensor(input0Id), getTensor(input1Id), getTensor(*opExtInst, 0),
                                 getTensor(*opExtInst, 1), kernelSizes, searchWindowSizes, inputStrides, windowStrides,
                                 windowOffsets, padding, searchPattern, debugName);
}

void GraphPassTosaSpv100::handleRawSad(const Instruction *opExtInst, const std::string &debugName) {
    // OpExtInst <result id> <OpExtInstImport id> RAW_SAD kernel_sizes search_window_sizes input_strides
    // window_strides window_offsets padding input0 input1
    assert(opExtInst->NumInOperands() == 10);

    const auto &resultId = opExtInst->result_id();
    const auto &kernelSizes = getConstVector(opExtInst->GetInOperand(2));
    const auto &searchWindowSizes = getConstVector(opExtInst->GetInOperand(3));
    const auto &inputStrides = getConstVector(opExtInst->GetInOperand(4));
    const auto &windowStrides = getConstVector(opExtInst->GetInOperand(5));
    const auto &windowOffsets = getConstVector(opExtInst->GetInOperand(6));
    const auto &padding = getConstVector(opExtInst->GetInOperand(7));
    const auto &input0Id = opExtInst->GetInOperand(8);
    const auto &input1Id = opExtInst->GetInOperand(9);

    graphLog(Severity::Info) << "OpExtInst result=" << resultId << ", " << debugName << ", kernelSizes=" << kernelSizes
                             << ", searchWindowSizes=" << searchWindowSizes << ", inputStrides=" << inputStrides
                             << ", windowStrides=" << windowStrides << ", windowOffsets=" << windowOffsets
                             << ", padding=" << padding << ", input0=%" << input0Id.AsId() << ", input1=%"
                             << input1Id.AsId() << std::endl;

    graphPipeline.makeRawSad(getTensor(input0Id), getTensor(input1Id), getTensor(*opExtInst), kernelSizes,
                             searchWindowSizes, inputStrides, windowStrides, windowOffsets, padding, debugName);
}

} // namespace spvtools::opt
