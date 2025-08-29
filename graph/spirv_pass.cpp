/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "spirv_pass.hpp"
#include "graph_log.hpp"

using namespace mlsdk::el::log;
using namespace mlsdk::el::compute;

/*******************************************************************************
 * Base Graph Pass
 *******************************************************************************/

namespace spvtools::opt {

Pass::Status GraphPassBase::Process() {
    handleGraphs();
    return Status::SuccessWithChange;
}

void GraphPassBase::handleGraphConstants() {
    for (const auto &instruction : get_module()->types_values()) {
        switch (instruction.opcode()) {
        case spv::Op::OpGraphConstantARM: {
            const auto resultId = instruction.result_id();
            const auto constantId = static_cast<uint32_t>(instruction.GetOperand(2).AsLiteralUint64());

            if (tensorMap.find(resultId) == tensorMap.end()) {
                auto &tensors = tensorMap[resultId];
                tensors[0] = graphPipeline.getConstTensor(constantId);
                graphLog(Severity::Info) << "%" << resultId << ": constId=" << constantId << ", tensor=" << tensors[0]
                                         << ", " << *tensors[0] << std::endl;
            }
            break;
        }
        default:
            break;
        }
    }
}

void GraphPassBase::handleGraphs() {
    const auto &module = *get_module();

    // Iterate over graph entry points
    for (const auto &graphEntry : module.graph_entry_points()) {
        graphLog(Severity::Info) << graphEntry << std::endl;

        // OpGraphEntryPointARM <graph id> <name> [input tensors] [output tensors]
        // auto op = graphEntry.begin();
        // auto &graphId = *(op++);
        // auto &graphName = *(op++);

        // Find OpGraphARM graph entry
        const auto *graph = getGraphById(graphEntry.GetOperand(0));
        assert(graph != nullptr);

        handleGraphConstants();
        handleInputsAndOutputs(graphEntry);
        handleGraph(graph);
    }
}

void GraphPassBase::handleInputsAndOutputs(const Instruction &opGraphEntryPoint) {
    // OpGraphEntryPointARM <graph id> <name> [inputTensorId:s] [outputTensorId:s]
    const auto *graph = getGraphById(opGraphEntryPoint.GetOperand(0));

    // Input- and output operators
    const auto &inputs = opGraphEntryPoint.begin() + 2;
    const auto &outputs = inputs + uint32_t(graph->inputs().size());

    // Connect OpGraphInputARM result id:s with input tensors listed by OpGraphEntryPointARM
    // OpGraphInputARM <result type> <result id> <input index> [<array index>]
    for (const auto &opGraphInputARM : graph->inputs()) {
        // The result id inside the local graph
        const auto resultId = opGraphInputARM->result_id();

        // External id from the graph entry point
        const auto &inputIndex = getConstant(opGraphInputARM->GetOperand(2));
        const uint32_t arrayIndex =
            opGraphInputARM->NumOperands() > 3 ? getConstant<uint32_t>(opGraphInputARM->GetOperand(3)) : 0;
        const auto &inputTensor = getTensor(inputs[inputIndex], arrayIndex);

        // Map result id to external tensor
        tensorMap[resultId][0] = inputTensor;
        graphLog(Severity::Info) << "%" << resultId << ": tensor=" << inputTensor << std::endl;

        graphPipeline.makeInput(inputTensor);
    }

    // Create and connect output tensors
    for (const auto &opGraphSetOutputARM : graph->outputs()) {
        // OpGraphSetOutputARM <value id> <output index id> [<array index id>]

        assert(opGraphSetOutputARM->opcode() == spv::Op::OpGraphSetOutputARM);

        // Tensor that shall be bound to the output
        const auto &instruction = get_def_use_mgr()->GetDef(opGraphSetOutputARM->GetOperand(0).AsId());

        // The external tensor id from the graph entry point
        const auto &outputIndex = getConstant(opGraphSetOutputARM->GetOperand(1));
        const uint32_t arrayIndex =
            opGraphSetOutputARM->NumOperands() > 2 ? getConstant<uint32_t>(opGraphSetOutputARM->GetOperand(2)) : 0;
        const auto &outputTensor = getTensor(outputs[outputIndex], arrayIndex);

        graphPipeline.makeOutput(outputTensor);

        switch (instruction->opcode()) {
        case spv::Op::OpConstantComposite:
        case spv::Op::OpConstantCompositeReplicateEXT:
        case spv::Op::OpGraphInputARM:
        case spv::Op::OpGraphConstantARM: {
            const auto &inputTensor = getTensor(*instruction);
            graphPipeline.makeCast(inputTensor, outputTensor,
                                   extractDebugInfoFromSPV(instruction, &opGraphEntryPoint, "CAST"));
            break;
        }
        case spv::Op::OpCompositeExtract: {
            const auto &compositeId = instruction->GetOperand(2);
            const auto &compositeIndex = instruction->GetOperand(3).AsLiteralUint64();

            tensorMap[compositeId.AsId()][compositeIndex] = outputTensor;
            graphLog(Severity::Info) << "%" << compositeId.AsId() << "[" << compositeIndex
                                     << "]: tensor=" << outputTensor << std::endl;
            break;
        }
        default: {
            tensorMap[instruction->result_id()][0] = outputTensor;
            graphLog(Severity::Info) << "%" << instruction->result_id() << ": tensor=" << outputTensor << std::endl;
        }
        }
    }
}

const Graph *GraphPassBase::getGraphById(const Operand &operand) {
    // OpGraphARM <OpTypeGraphARM id>
    const auto &opGraphARM = get_def_use_mgr()->GetDef(operand.AsId());
    const auto &graphs = get_module()->graphs();
    const auto found =
        std::find_if(graphs.begin(), graphs.end(), [&](auto &graph) { return graph->DefInst() == *opGraphARM; });
    if (found != graphs.end()) {
        return (*found).get();
    }
    return nullptr;
}

std::tuple<std::vector<analysis::TensorARM *>, std::vector<analysis::TensorARM *>>
GraphPassBase::getGraphType(const Operand &operand) {
    // <return id> = OpTypeGraphARM <number of inputs> [inputs] [outputs]
    const auto &opTypeGraphARM = get_def_use_mgr()->GetDef(operand.AsId());
    assert(opTypeGraphARM->opcode() == spv::Op::OpTypeGraphARM);
    auto op = opTypeGraphARM->begin();

    auto numInputs = (op++)->AsLiteralUint64();

    // Inputs
    std::vector<analysis::TensorARM *> inputs;
    while (numInputs-- > 0) {
        inputs.push_back(getTensorType(*op++));
    }

    // Outputs
    std::vector<analysis::TensorARM *> outputs;
    while (op != opTypeGraphARM->end()) {
        outputs.push_back(getTensorType(*op++));
    }

    return {inputs, outputs};
}

analysis::TensorARM *GraphPassBase::getTensorType(const Operand &operand, const uint32_t index) const {
    return getTensorType(operand.AsId(), index);
}

analysis::TensorARM *GraphPassBase::getTensorType(uint32_t id, const uint32_t index) const {
    const auto &instruction = get_def_use_mgr()->GetDef(id);

    switch (instruction->opcode()) {
    case spv::Op::OpTypeTensorARM:
        break;
    case spv::Op::OpExtInst:
    case spv::Op::OpGraphInputARM:
    case spv::Op::OpGraphConstantARM:
    case spv::Op::OpConstantComposite:
    case spv::Op::OpSpecConstantComposite:
    case spv::Op::OpConstantCompositeReplicateEXT:
    case spv::Op::OpSpecConstantCompositeReplicateEXT:
        id = instruction->GetOperand(0).AsId();
        break;
    case spv::Op::OpTypeStruct:
        id = instruction->GetInOperand(index).AsId();
        break;
    default:
        return nullptr;
    }

    const auto &type = context()->get_type_mgr()->GetType(id);
    assert(type);
    const auto &tensorType = type->AsTensorARM();
    assert(tensorType);

    return tensorType;
}

std::tuple<uint64_t, uint64_t> GraphPassBase::getDescriptorSetAndBinding(const Operand &operand) {
    uint64_t descriptorSet = std::numeric_limits<uint64_t>::max();
    uint64_t binding = 0;

    for (const auto &decoration : get_decoration_mgr()->GetDecorationsFor(operand.AsId(), false)) {
        switch (static_cast<spv::Decoration>(decoration->GetSingleWordInOperand(1))) {
        case spv::Decoration::DescriptorSet:
            descriptorSet = decoration->GetOperand(2).AsLiteralUint64();
            break;
        case spv::Decoration::Binding:
            binding = decoration->GetOperand(2).AsLiteralUint64();
            break;
        default:
            break;
        }
    }

    return std::make_tuple(descriptorSet, binding);
}

std::tuple<uint64_t, uint64_t, std::shared_ptr<TensorDescriptor>>
GraphPassBase::getTensorByDecoration(const Operand &operand, const uint32_t arrayIndex) {
    const auto &[descriptorSet, binding] = getDescriptorSetAndBinding(operand);

    if (descriptorSet == std::numeric_limits<uint64_t>::max()) {
        return std::make_tuple(descriptorSet, binding, nullptr);
    }

    auto tensor =
        graphPipeline.getTensor(static_cast<uint32_t>(descriptorSet), static_cast<uint32_t>(binding), arrayIndex);
    return std::make_tuple(descriptorSet, binding, std::move(tensor));
}

void GraphPassBase::mapTensorByDecoration(uint32_t resultId, const Operand &operand, const uint32_t arrayIndex) {
    // If the tensor has already been mapped we don't want to map it again
    if (tensorMap[resultId][0] == nullptr) {
        const auto &[descriptorSet, binding, tensor] = getTensorByDecoration(operand, arrayIndex);
        tensorMap[resultId][0] = tensor;
        graphLog(Severity::Info) << "%" << resultId << "[" << arrayIndex << "]: set=" << descriptorSet
                                 << ", binding=" << binding << ", tensor=" << tensorMap[resultId][0] << ", "
                                 << *tensorMap[resultId][0] << std::endl;
    }
}

std::shared_ptr<TensorDescriptor> GraphPassBase::getTensor(const Instruction &instruction, const uint32_t arrayIndex) {
    if (tensorMap[instruction.result_id()][arrayIndex] != nullptr) {
        return tensorMap[instruction.result_id()][arrayIndex];
    }

    switch (instruction.opcode()) {
    case spv::Op::OpCompositeExtract: {
        const auto &compositeId = instruction.GetOperand(2);
        const auto index = static_cast<uint32_t>(instruction.GetOperand(3).AsLiteralUint64());
        return getTensor(compositeId, index);
    }
    case spv::Op::OpConstantComposite:
    case spv::Op::OpConstantCompositeReplicateEXT:
    case spv::Op::OpConstantNull: {
        auto tensor = makeCompositeTensor(instruction.result_id());
        tensorMap[instruction.result_id()][arrayIndex] = tensor;
        return tensor;
    }
    case spv::Op::OpExtInst: {
        auto tensor = makeTensor(getTensorType(instruction.GetOperand(1)));
        tensorMap[instruction.result_id()][arrayIndex] = tensor;

        graphLog(Severity::Info) << "%" << instruction.result_id() << "[" << arrayIndex << "]: tensor=" << tensor
                                 << ", " << *tensor << std::endl;

        return tensor;
    }
    case spv::Op::OpGraphConstantARM: {
        const auto constantId = static_cast<uint32_t>(instruction.GetOperand(2).AsLiteralUint64());
        return graphPipeline.getConstTensor(constantId);
    }
    case spv::Op::OpVariable: {
        const auto &[set, binding, tensor] = getTensorByDecoration(instruction.GetOperand(1), arrayIndex);
        graphLog(Severity::Info) << "%" << instruction.result_id() << "[" << arrayIndex << "]: set=" << set
                                 << ", binding=" << binding << ", tensor=" << tensor << ", " << *tensor << std::endl;
        return tensor;
    }
    default:
        throw std::runtime_error("Unsupported instruction type in getTensor: " +
                                 std::to_string(int(instruction.opcode())));
    }
}

std::shared_ptr<TensorDescriptor> GraphPassBase::getTensor(const Operand &operand, const uint32_t arrayIndex) {
    const auto &instruction = get_def_use_mgr()->GetDef(operand.AsId());
    return getTensor(*instruction, arrayIndex);
}

std::shared_ptr<TensorDescriptor> GraphPassBase::makeTensor(const analysis::TensorARM *tensor) const {
    const VkFormat format = getVkFormat(tensor->element_type());
    const std::vector<int64_t> dimensions =
        tensor->is_shaped() ? getConstVector<int64_t>(tensor->shape_id()) : std::vector<int64_t>{};

    return graphPipeline.makeTensor(format, dimensions);
}

std::shared_ptr<TensorDescriptor> GraphPassBase::getOrMakeCompositeTensor(const uint32_t id) const {
    const auto &instruction = get_def_use_mgr()->GetDef(id);
    if (instruction->opcode() == spv::Op::OpGraphConstantARM) {
        const auto constantId = static_cast<uint32_t>(instruction->GetOperand(2).AsLiteralUint64());
        return graphPipeline.getConstTensor(constantId);
    }
    return makeCompositeTensor(instruction->result_id());
}

std::shared_ptr<TensorDescriptor> GraphPassBase::makeCompositeTensor(const uint32_t id) const {
    const auto &instruction = get_def_use_mgr()->GetDef(id);
    const auto &tensorType = getTensorType(instruction->type_id());
    const auto &format = getVkFormat(tensorType->element_type());
    const auto &dimensions = getConstVector<int64_t>(tensorType->shape_id());

    switch (format) {
    case VK_FORMAT_R8_BOOL_ARM:
    case VK_FORMAT_R8_SINT:
        return graphPipeline.makeConstCompositeTensor(format, dimensions,
                                                      getConstVector<int8_t>(instruction->result_id()).data());
    case VK_FORMAT_R16_SINT:
        return graphPipeline.makeConstCompositeTensor(format, dimensions,
                                                      getConstVector<int16_t>(instruction->result_id()).data());
    case VK_FORMAT_R32_SINT:
        return graphPipeline.makeConstCompositeTensor(format, dimensions,
                                                      getConstVector<int32_t>(instruction->result_id()).data());
    case VK_FORMAT_R64_SINT:
        return graphPipeline.makeConstCompositeTensor(format, dimensions,
                                                      getConstVector<int64_t>(instruction->result_id()).data());
    case VK_FORMAT_R16_SFLOAT:
        return graphPipeline.makeConstCompositeTensor(format, dimensions,
                                                      getConstVector<float16>(instruction->result_id()).data());
    case VK_FORMAT_R32_SFLOAT:
        return graphPipeline.makeConstCompositeTensor(format, dimensions,
                                                      getConstVector<float>(instruction->result_id()).data());
    case VK_FORMAT_R64_SFLOAT:
        return graphPipeline.makeConstCompositeTensor(format, dimensions,
                                                      getConstVector<double>(instruction->result_id()).data());
    default:
        throw std::runtime_error(std::string("Unsupported composite tensor format: " + std::to_string(format)));
    }

    return nullptr;
}

VkFormat GraphPassBase::getVkFormat(const analysis::Type *type) const {
    const auto &integerType = type->AsInteger();
    if (integerType) {
        switch (integerType->width()) {
        case 8:
            return VK_FORMAT_R8_SINT;
        case 16:
            return VK_FORMAT_R16_SINT;
        case 32:
            return VK_FORMAT_R32_SINT;
        case 64:
            return VK_FORMAT_R64_SINT;
        default:
            throw std::runtime_error(std::string("Unsupported integer tensor format: " + type->str()));
        }
    }

    const auto &boolType = type->AsBool();
    if (boolType) {
        return VK_FORMAT_R8_BOOL_ARM;
    }

    const auto &floatType = type->AsFloat();
    if (floatType) {
        switch (floatType->width()) {
        case 16:
            return VK_FORMAT_R16_SFLOAT;
        case 32:
            return VK_FORMAT_R32_SFLOAT;
        case 64:
            return VK_FORMAT_R64_SFLOAT;
        default:
            throw std::runtime_error(std::string("Unsupported float tensor format: " + type->str()));
        }
    }

    throw std::runtime_error(std::string("Unsupported tensor format: " + type->str()));
}

bool GraphPassBase::getBoolConstant(const Operand &operand) {
    const auto &constant = context()->get_constant_mgr()->FindDeclaredConstant(operand.AsId());
    return constant->AsBoolConstant()->value();
}

std::string GraphPassBase::extractDebugInfoFromSPV(const Instruction *opExtInst, const Instruction *,
                                                   const std::string &defaultName) {
    if (!opExtInst) {
        return defaultName;
    }

    graphLog(Severity::Debug) << "[TRACE] extractDebugInfoFromSPV called with result_id: " << opExtInst->result_id()
                              << std::endl;

    bool hasDebugInfoExtension = false;
    for (const auto &inst : get_module()->extensions()) {
        if (inst.opcode() == spv::Op::OpExtension && inst.GetOperand(0).AsString() == "SPV_KHR_non_semantic_info") {
            hasDebugInfoExtension = true;
            break;
        }
    }

    if (!hasDebugInfoExtension) {
        return defaultName;
    }

    // TODO: extend with other non-semantic info decoration options

    return defaultName;
}
} // namespace spvtools::opt
