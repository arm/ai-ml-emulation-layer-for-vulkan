/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "spirv_pass.hpp"
#include "graph_log.hpp"

#include "mlel/utils.hpp"

using namespace mlsdk::el::log;
using namespace mlsdk::el::compute;

namespace {

template <typename T, spv::FPEncoding encoding> bool isFloatEncoding(const spvtools::opt::analysis::Float *f) {
    return f && f->width() == (8 * sizeof(T)) && f->encoding() == encoding;
}

bool isBFloat16Impl(const spvtools::opt::analysis::Float *f) {
    return isFloatEncoding<uint16_t, spv::FPEncoding::BFloat16KHR>(f);
}

bool isFloat8E5M2Impl(const spvtools::opt::analysis::Float *f) {
    return isFloatEncoding<uint8_t, spv::FPEncoding::Float8E5M2EXT>(f);
}

bool isFloat8E4M3Impl(const spvtools::opt::analysis::Float *f) {
    return isFloatEncoding<uint8_t, spv::FPEncoding::Float8E4M3EXT>(f);
}

template <typename T, spv::FPEncoding encoding>
void flattenFloatComposite(const spvtools::opt::analysis::CompositeConstant *composite, std::vector<T> &values) {
    const auto &components = composite->GetComponents();
    values.reserve(values.size() + components.size());
    for (const auto &component : components) {
        if (const auto *innerComposite = component->AsCompositeConstant()) {
            flattenFloatComposite<T, encoding>(innerComposite, values);
            continue;
        }

        const auto *floatConstant = component->AsFloatConstant();
        if (floatConstant == nullptr || !isFloatEncoding<T, encoding>(floatConstant->type()->AsFloat())) {
            throw std::runtime_error("Unsupported float constant encoding in composite constant");
        }

        // Raw payload extraction
        values.push_back(T(floatConstant->words()[0]));
    }
}

} // namespace

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
        const auto inputIndex = getConstScalar(opGraphInputARM->GetOperand(2));
        const uint32_t arrayIndex =
            opGraphInputARM->NumOperands() > 3 ? getConstScalar<uint32_t>(opGraphInputARM->GetOperand(3)) : 0;
        auto inputTensor = getTensor(inputs[inputIndex], arrayIndex);

        // Seed OpGraphInputARM in the local cache before visiting graph ops.
        // Unlike OpVariable and OpGraphConstantARM, graph inputs do not have a GraphPipeline lookup path.
        graphLog(Severity::Info) << '%' << resultId << ": tensor=" << inputTensor << std::endl;
        graphPipeline.makeInput(inputTensor);
        tensorMap[resultId][0] = std::move(inputTensor);
    }

    // Create and connect output tensors
    for (const auto &opGraphSetOutputARM : graph->outputs()) {
        // OpGraphSetOutputARM <value id> <output index id> [<array index id>]

        assert(opGraphSetOutputARM->opcode() == spv::Op::OpGraphSetOutputARM);

        // Tensor that shall be bound to the output
        const auto *instruction = get_def_use_mgr()->GetDef(opGraphSetOutputARM->GetOperand(0).AsId());

        // The external tensor id from the graph entry point
        const auto outputIndex = getConstScalar(opGraphSetOutputARM->GetOperand(1));
        const uint32_t arrayIndex =
            opGraphSetOutputARM->NumOperands() > 2 ? getConstScalar<uint32_t>(opGraphSetOutputARM->GetOperand(2)) : 0;
        auto outputTensor = getTensor(outputs[outputIndex], arrayIndex);

        graphPipeline.makeOutput(outputTensor);

        switch (instruction->opcode()) {
        case spv::Op::OpConstantComposite:
        case spv::Op::OpConstantCompositeReplicateEXT:
        case spv::Op::OpGraphInputARM:
        case spv::Op::OpGraphConstantARM: {
            const auto inputTensor = getTensor(*instruction);
            graphPipeline.makeCast(inputTensor, outputTensor, extractDebugInfoFromSPV(instruction, "CAST"));
            break;
        }
        case spv::Op::OpCompositeExtract: {
            const auto &compositeId = instruction->GetOperand(2);
            const auto compositeIndex = instruction->GetOperand(3).AsLiteralUint64();

            graphLog(Severity::Info) << '%' << compositeId.AsId() << '[' << compositeIndex
                                     << "]: tensor=" << outputTensor << std::endl;
            tensorMap[compositeId.AsId()][compositeIndex] = std::move(outputTensor);
            break;
        }
        default: {
            graphLog(Severity::Info) << '%' << instruction->result_id() << ": tensor=" << outputTensor << std::endl;
            tensorMap[instruction->result_id()][0] = std::move(outputTensor);
        }
        }
    }
}

const Graph *GraphPassBase::getGraphById(const Operand &operand) {
    // OpGraphARM <OpTypeGraphARM id>
    const auto *opGraphARM = get_def_use_mgr()->GetDef(operand.AsId());
    const auto &graphs = get_module()->graphs();
    const auto found =
        std::find_if(graphs.begin(), graphs.end(), [&](auto &graph) { return graph->DefInst() == *opGraphARM; });
    if (found != graphs.end()) {
        return found->get();
    }
    return nullptr;
}

const analysis::TensorARM *GraphPassBase::getTensorType(const Operand &operand) const {
    return getTensorType(operand.AsId());
}

const analysis::TensorARM *GraphPassBase::getTensorType(uint32_t id) const {
    const auto *instruction = get_def_use_mgr()->GetDef(id);

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
        id = instruction->GetInOperand(0).AsId();
        break;
    default:
        return nullptr;
    }

    const auto *type = context()->get_type_mgr()->GetType(id);
    assert(type);
    const auto *tensorType = type->AsTensorARM();
    assert(tensorType);

    return tensorType;
}

std::tuple<uint64_t, uint64_t> GraphPassBase::getDescriptorSetAndBinding(const Operand &operand) {
    uint64_t descriptorSet = std::numeric_limits<uint64_t>::max();
    uint64_t binding = 0;

    for (const auto *decoration : get_decoration_mgr()->GetDecorationsFor(operand.AsId(), false)) {
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
    case spv::Op::OpGraphInputARM:
        throw std::runtime_error("OpGraphInputARM tensor requested before graph inputs were connected");
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

        graphLog(Severity::Info) << '%' << instruction.result_id() << '[' << arrayIndex << "]: tensor=" << tensor
                                 << ", " << *tensor << std::endl;

        return tensor;
    }
    case spv::Op::OpGraphConstantARM: {
        const auto constantId = static_cast<uint32_t>(instruction.GetOperand(2).AsLiteralUint64());
        // Graph constants are canonicalized in GraphPipeline, so resolve them there and mirror the
        // descriptor in the local cache for later result-id lookups.
        auto tensor = graphPipeline.getConstTensor(constantId);
        tensorMap[instruction.result_id()][arrayIndex] = tensor;
        return tensor;
    }
    case spv::Op::OpVariable: {
        const auto &[set, binding, tensor] = getTensorByDecoration(instruction.GetOperand(1), arrayIndex);
        // Descriptor-backed variables are owned by GraphPipeline's set/binding cache.
        // Store the descriptor locally as well so subsequent uses resolve through one code path.
        tensorMap[instruction.result_id()][arrayIndex] = tensor;
        graphLog(Severity::Info) << '%' << instruction.result_id() << '[' << arrayIndex << "]: set=" << set
                                 << ", binding=" << binding << ", tensor=" << tensor << ", " << *tensor << std::endl;
        return tensor;
    }
    default:
        throw std::runtime_error("Unsupported instruction type in getTensor: " +
                                 std::to_string(int(instruction.opcode())));
    }
}

std::shared_ptr<TensorDescriptor> GraphPassBase::getTensor(const Operand &operand, const uint32_t arrayIndex) {
    const auto *instruction = get_def_use_mgr()->GetDef(operand.AsId());
    return getTensor(*instruction, arrayIndex);
}

std::shared_ptr<TensorDescriptor> GraphPassBase::makeTensor(const analysis::TensorARM *tensor) const {
    const VkFormat format = getVkFormat(tensor->element_type());
    auto dimensions = tensor->is_shaped() ? getConstVector<int64_t>(tensor->shape_id()) : std::vector<int64_t>{};

    return graphPipeline.makeTensor(format, std::move(dimensions));
}

std::shared_ptr<TensorDescriptor> GraphPassBase::getOrMakeCompositeTensor(const uint32_t id) {
    auto tensor = tensorMap[id][0];
    if (tensor != nullptr) {
        return tensor;
    }

    const auto *instruction = get_def_use_mgr()->GetDef(id);
    if (instruction->opcode() == spv::Op::OpGraphConstantARM) {
        const auto constantId = static_cast<uint32_t>(instruction->GetOperand(2).AsLiteralUint64());
        tensor = graphPipeline.getConstTensor(constantId);
    } else {
        tensor = makeCompositeTensor(instruction->result_id());
    }
    tensorMap[id][0] = tensor;
    return tensor;
}

std::shared_ptr<TensorDescriptor> GraphPassBase::makeCompositeTensor(const uint32_t id) const {
    const auto *instruction = get_def_use_mgr()->GetDef(id);
    const auto *tensorType = getTensorType(instruction->type_id());
    const auto format = getVkFormat(tensorType->element_type());
    auto dimensions = getConstVector<int64_t>(tensorType->shape_id());

    switch (format) {
    case VK_FORMAT_R8_BOOL_ARM:
    case VK_FORMAT_R8_SINT:
        return graphPipeline.makeConstCompositeTensor(format, std::move(dimensions),
                                                      getConstVector<int8_t>(instruction->result_id()).data());
    case VK_FORMAT_R16_SINT:
        return graphPipeline.makeConstCompositeTensor(format, std::move(dimensions),
                                                      getConstVector<int16_t>(instruction->result_id()).data());
    case VK_FORMAT_R32_SINT:
        return graphPipeline.makeConstCompositeTensor(format, std::move(dimensions),
                                                      getConstVector<int32_t>(instruction->result_id()).data());
    case VK_FORMAT_R64_SINT:
        return graphPipeline.makeConstCompositeTensor(format, std::move(dimensions),
                                                      getConstVector<int64_t>(instruction->result_id()).data());
    case VK_FORMAT_R16_SFLOAT:
        return graphPipeline.makeConstCompositeTensor(format, std::move(dimensions),
                                                      getConstVector<float16>(instruction->result_id()).data());
    case VK_FORMAT_R16_SFLOAT_FPENCODING_BFLOAT16_ARM: {
        const auto bf16Values = getConstVector<uint16_t, spv::FPEncoding::BFloat16KHR>(instruction, dimensions);
        return graphPipeline.makeConstCompositeTensor(format, std::move(dimensions), bf16Values.data());
    }
    case VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E5M2_ARM: {
        const auto f8Values = getConstVector<uint8_t, spv::FPEncoding::Float8E5M2EXT>(instruction, dimensions);
        return graphPipeline.makeConstCompositeTensor(format, std::move(dimensions), f8Values.data());
    }
    case VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E4M3_ARM: {
        const auto f8Values = getConstVector<uint8_t, spv::FPEncoding::Float8E4M3EXT>(instruction, dimensions);
        return graphPipeline.makeConstCompositeTensor(format, std::move(dimensions), f8Values.data());
    }
    case VK_FORMAT_R32_SFLOAT:
        return graphPipeline.makeConstCompositeTensor(format, std::move(dimensions),
                                                      getConstVector<float>(instruction->result_id()).data());
    case VK_FORMAT_R64_SFLOAT:
        return graphPipeline.makeConstCompositeTensor(format, std::move(dimensions),
                                                      getConstVector<double>(instruction->result_id()).data());
    default:
        throw std::runtime_error(std::string("Unsupported composite tensor format: " + std::to_string(format)));
    }
}

VkFormat GraphPassBase::getVkFormat(const analysis::Type *type) const {
    const auto *integerType = type->AsInteger();
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

    const auto *boolType = type->AsBool();
    if (boolType) {
        return VK_FORMAT_R8_BOOL_ARM;
    }

    const auto *floatType = type->AsFloat();
    if (floatType) {
        if (isFloat8E5M2Impl(floatType)) {
            return VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E5M2_ARM;
        }
        if (isFloat8E4M3Impl(floatType)) {
            return VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E4M3_ARM;
        }
        switch (floatType->width()) {
        case 16:
            if (isBFloat16Impl(floatType)) {
                return VK_FORMAT_R16_SFLOAT_FPENCODING_BFLOAT16_ARM;
            } else {
                return VK_FORMAT_R16_SFLOAT;
            }
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
    const auto *constant = context()->get_constant_mgr()->FindDeclaredConstant(operand.AsId());
    return constant->AsBoolConstant()->value();
}

// Disabled until needed
// std::string GraphPassBase::extractDebugInfoFromSPV(const Instruction *opExtInst, const std::string &defaultName) {
//     if (!opExtInst) {
//         return defaultName;
//     }

//     graphLog(Severity::Debug) << "[TRACE] extractDebugInfoFromSPV called with result_id: " << opExtInst->result_id()
//                               << std::endl;

//     bool hasDebugInfoExtension = false;
//     for (const auto &inst : get_module()->extensions()) {
//         if (inst.opcode() == spv::Op::OpExtension && inst.GetOperand(0).AsString() == "SPV_KHR_non_semantic_info") {
//             hasDebugInfoExtension = true;
//             break;
//         }
//     }

//     if (!hasDebugInfoExtension) {
//         return defaultName;
//     }

//     // TODO: extend with other non-semantic info decoration options

//     return defaultName;
// }

bool GraphPassBase::isBFloat16(const spvtools::opt::analysis::Float *f) { return isBFloat16Impl(f); }
bool GraphPassBase::isFloat8E5M2(const spvtools::opt::analysis::Float *f) { return isFloat8E5M2Impl(f); }
bool GraphPassBase::isFloat8E4M3(const spvtools::opt::analysis::Float *f) { return isFloat8E4M3Impl(f); }

size_t GraphPassBase::getElementCount(const uint32_t id) const {
    const auto dimensions = getConstVector<int64_t>(id);
    return mlsdk::el::utils::getElementCount(dimensions);
}

bool GraphPassBase::isCompositeReplicateConstantOpcode(const spv::Op opcode) {
    return opcode == spv::Op::OpConstantCompositeReplicateEXT || opcode == spv::Op::OpSpecConstantCompositeReplicateEXT;
}

template <typename T, spv::FPEncoding fpEncoding>
std::vector<T> GraphPassBase::getConstVector(const spvtools::opt::Instruction *instruction,
                                             const std::vector<int64_t> &dimensions) const {
    std::vector<T> values;
    const auto *constant = context()->get_constant_mgr()->FindDeclaredConstant(instruction->result_id());
    if (const auto *composite = constant->AsCompositeConstant()) {
        const bool isSplat = isCompositeReplicateConstantOpcode(instruction->opcode());
        flattenFloatComposite<T, fpEncoding>(composite, values);

        if (isSplat) {
            assert(values.size() == 1);
            const auto compositeCount = mlsdk::el::utils::getElementCount(dimensions);
            values.resize(compositeCount, values.front());
        }
    } else if (constant->AsNullConstant()) {
        const auto compositeCount = mlsdk::el::utils::getElementCount(dimensions);
        values.resize(compositeCount, 0);
    } else {
        throw std::runtime_error("Unsupported format " + std::to_string(int(fpEncoding)) +
                                 " constant kind for composite tensor");
    }

    return values;
}

} // namespace spvtools::opt
