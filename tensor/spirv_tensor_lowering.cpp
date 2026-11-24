/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include "spirv_tensor_lowering.hpp"
#include "spirv_tensor_access.hpp"
#include "tensor_descriptor_layout.hpp"

#include "source/opt/pass.h"

#include <spirv-tools/optimizer.hpp>

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace mlsdk::el::layer {
namespace {

using namespace tensor_spirv;

class TensorLoweringPass final : public spvtools::opt::Pass {
  public:
    explicit TensorLoweringPass(bool &loweredTensorCompute) : loweredTensorCompute(loweredTensorCompute) {}
    const char *name() const override { return "tensor-lowering"; }

  private:
    Status Process() override;

    bool &loweredTensorCompute;
};

bool isTensorAccess(spv::Op opcode) {
    return opcode == spv::Op::OpTensorReadARM || opcode == spv::Op::OpTensorWriteARM ||
           opcode == spv::Op::OpTensorQuerySizeARM;
}

Instruction &definition(IRContext &context, uint32_t id) {
    auto *inst = context.get_def_use_mgr()->GetDef(id);
    if (!inst) {
        throw std::invalid_argument("Undefined TensorARM operand " + std::to_string(id));
    }
    return *inst;
}

std::optional<uint64_t> integerConstant(IRContext &context, uint32_t id) {
    const auto &inst = definition(context, id);
    if (definition(context, inst.type_id()).opcode() != spv::Op::OpTypeInt) {
        throw std::invalid_argument("Expected an integer TensorARM operand");
    }
    if (inst.opcode() == spv::Op::OpConstantNull) {
        return 0;
    }
    if (inst.opcode() == spv::Op::OpSpecConstant || inst.opcode() == spv::Op::OpSpecConstantOp) {
        return std::nullopt;
    }
    if (inst.opcode() != spv::Op::OpConstant) {
        throw std::invalid_argument("Expected a constant TensorARM operand");
    }
    const auto &words = inst.GetInOperand(0).words;
    return uint64_t(words[0]) | (words.size() == 2 ? uint64_t(words[1]) << 32 : 0);
}

void requireUses(IRContext &context, uint32_t id, std::initializer_list<spv::Op> supported) {
    context.get_def_use_mgr()->ForEachUser(id, [&](Instruction *user) {
        const auto opcode = user->opcode();
        if (opcode != spv::Op::OpName && opcode != spv::Op::OpDecorate &&
            std::find(supported.begin(), supported.end(), opcode) == supported.end()) {
            throw std::runtime_error(std::string("Unsupported TensorARM use: Op") + spvOpcodeString(uint32_t(opcode)));
        }
    });
}

void requireUniformDescriptor(IRContext &context, const Instruction &inst) {
    const auto check = [&](const uint32_t *id) {
        if (context.get_decoration_mgr()->HasDecoration(*id, spv::Decoration::NonUniform)) {
            // TODO: Translate non-uniform tensor descriptor indexing.
            throw std::runtime_error("Unsupported non-uniform TensorARM descriptor indexing");
        }
    };
    inst.ForEachId(check);
}

struct TensorModule {
    std::map<uint32_t, TensorType> types;
    std::vector<Instruction *> arrayTypes;
    std::vector<Instruction *> pointerTypes;
    std::vector<Instruction *> variables;
    std::vector<Instruction *> accessChains;
    std::vector<Instruction *> loads;
    std::vector<std::pair<Instruction *, TensorType>> accesses;
    bool dynamicIndexing = false;
};

TensorModule collectTensors(IRContext &context) {
    TensorModule tensors;
    for (auto &inst : context.types_values()) {
        if (inst.opcode() != spv::Op::OpTypeTensorARM) {
            continue;
        }
        if (inst.NumInOperands() != 2) {
            throw std::runtime_error("TensorARM types require an element type and rank");
        }
        const auto rankId = inst.GetSingleWordInOperand(1);
        const auto rank = integerConstant(context, rankId);
        if (rank && (*rank == 0 || *rank > maxTensorRank)) {
            throw std::runtime_error("TensorARM rank must be between 1 and 6");
        }
        const auto element = inst.GetSingleWordInOperand(0);
        const auto &elementType = definition(context, element);
        if (elementType.opcode() != spv::Op::OpTypeInt && elementType.opcode() != spv::Op::OpTypeFloat &&
            elementType.opcode() != spv::Op::OpTypeBool) {
            throw std::runtime_error("Unsupported TensorARM scalar storage type");
        }
        const auto literalRank = rank ? std::optional<uint32_t>{static_cast<uint32_t>(*rank)} : std::nullopt;
        tensors.types.emplace(inst.result_id(), TensorType{element, rankId, literalRank});
        // TODO: Transform tensor function parameters.
        requireUses(context, inst.result_id(), {spv::Op::OpTypePointer, spv::Op::OpTypeArray, spv::Op::OpLoad});
    }
    std::set<uint32_t> tensorAndArrayTypeIds;
    for (const auto &[id, type] : tensors.types) {
        tensorAndArrayTypeIds.insert(id);
    }
    for (auto &inst : context.types_values()) {
        if (inst.opcode() != spv::Op::OpTypeArray) {
            continue;
        }
        const auto elementTypeId = inst.GetSingleWordInOperand(0);
        if (tensorAndArrayTypeIds.count(elementTypeId)) {
            requireUses(context, inst.result_id(), {spv::Op::OpTypePointer});
            tensorAndArrayTypeIds.insert(inst.result_id());
            tensors.arrayTypes.push_back(&inst);
        }
    }
    std::set<uint32_t> pointerTypeIds;
    for (auto &inst : context.types_values()) {
        if (inst.opcode() != spv::Op::OpTypePointer) {
            continue;
        }
        const auto pointeeTypeId = inst.GetSingleWordInOperand(1);
        if (tensorAndArrayTypeIds.count(pointeeTypeId) == 0) {
            continue;
        }
        if (inst.GetSingleWordInOperand(0) != static_cast<uint32_t>(spv::StorageClass::UniformConstant)) {
            throw std::runtime_error("TensorARM pointers must refer to uniform tensor descriptors");
        }
        requireUses(context, inst.result_id(),
                    {spv::Op::OpVariable, spv::Op::OpAccessChain, spv::Op::OpInBoundsAccessChain});
        pointerTypeIds.insert(inst.result_id());
        tensors.pointerTypes.push_back(&inst);
    }
    for (auto *pointer : tensors.pointerTypes) {
        context.get_def_use_mgr()->ForEachUser(pointer, [&](Instruction *inst) {
            if (inst->opcode() == spv::Op::OpVariable) {
                requireUses(
                    context, inst->result_id(),
                    {spv::Op::OpEntryPoint, spv::Op::OpLoad, spv::Op::OpAccessChain, spv::Op::OpInBoundsAccessChain});
                tensors.variables.push_back(inst);
            }
        });
    }
    for (auto &function : *context.module()) {
        function.ForEachInst([&](Instruction *inst) {
            if (pointerTypeIds.count(inst->type_id()) &&
                (inst->opcode() == spv::Op::OpAccessChain || inst->opcode() == spv::Op::OpInBoundsAccessChain)) {
                requireUses(context, inst->result_id(),
                            {spv::Op::OpLoad, spv::Op::OpAccessChain, spv::Op::OpInBoundsAccessChain});
                requireUniformDescriptor(context, *inst);
                for (uint32_t i = 1; i < inst->NumInOperands(); ++i) {
                    const auto opcode = definition(context, inst->GetSingleWordInOperand(i)).opcode();
                    tensors.dynamicIndexing |= opcode != spv::Op::OpConstant && opcode != spv::Op::OpConstantNull &&
                                               opcode != spv::Op::OpSpecConstant && opcode != spv::Op::OpSpecConstantOp;
                }
                tensors.accessChains.push_back(inst);
            }
            if (inst->opcode() == spv::Op::OpLoad && tensors.types.count(inst->type_id()) != 0) {
                requireUniformDescriptor(context, *inst);
                const auto *pointer = &definition(context, inst->GetSingleWordInOperand(0));
                const bool indexed =
                    pointerTypeIds.count(pointer->type_id()) && (pointer->opcode() == spv::Op::OpAccessChain ||
                                                                 pointer->opcode() == spv::Op::OpInBoundsAccessChain);
                if (!indexed &&
                    std::find(tensors.variables.begin(), tensors.variables.end(), pointer) == tensors.variables.end()) {
                    throw std::runtime_error("TensorARM load must refer to a uniform tensor descriptor");
                }
                requireUses(context, inst->result_id(),
                            {spv::Op::OpTensorReadARM, spv::Op::OpTensorWriteARM, spv::Op::OpTensorQuerySizeARM});
                tensors.loads.push_back(inst);
            } else if (isTensorAccess(inst->opcode())) {
                const auto &tensor = definition(context, inst->GetSingleWordInOperand(0));
                const auto it = tensors.types.find(tensor.type_id());
                if (it == tensors.types.end() || tensor.opcode() != spv::Op::OpLoad) {
                    throw std::runtime_error("TensorARM access must use a loaded tensor descriptor");
                }
                tensors.accesses.emplace_back(inst, it->second);
            }
        });
    }
    return tensors;
}

std::map<uint32_t, uint32_t> createReplacementTypes(AccessState &state, const TensorModule &tensors) {
    auto &context = state.context;
    auto *types = context.get_type_mgr();
    const auto metadataBlockTypeId = definition(context, state.interfacePointerType).GetSingleWordInOperand(1);
    std::map<uint32_t, uint32_t> replacementTypeIds;
    for (const auto &[id, type] : tensors.types) {
        replacementTypeIds.emplace(id, metadataBlockTypeId);
    }
    // Each array element is a separate descriptor buffer, not a packed metadata
    // entry within one buffer. Keep array lengths and selection indices intact.
    for (auto *array : tensors.arrayTypes) {
        const auto elementTypeId = array->GetSingleWordInOperand(0);
        const auto replacementElementTypeId = replacementTypeIds.at(elementTypeId);
        const auto *arrayType = types->GetType(array->result_id())->AsArray();
        const spvtools::opt::analysis::Array replacement(types->GetType(replacementElementTypeId),
                                                         arrayType->length_info());
        replacementTypeIds.emplace(array->result_id(), types->GetTypeInstruction(&replacement));
    }
    for (auto *pointer : tensors.pointerTypes) {
        const auto pointeeTypeId = pointer->GetSingleWordInOperand(1);
        const auto replacementPointeeTypeId = replacementTypeIds.at(pointeeTypeId);
        const auto replacementPointerTypeId =
            types->FindPointerToType(replacementPointeeTypeId, spv::StorageClass::Uniform);
        replacementTypeIds.emplace(pointer->result_id(), replacementPointerTypeId);
    }
    return replacementTypeIds;
}

void transformInterfaces(AccessState &state, const TensorModule &tensors) {
    auto &context = state.context;
    auto *decorations = context.get_decoration_mgr();
    const auto replacementTypeIds = createReplacementTypes(state, tensors);

    // Replace the tensor variables while preserving their descriptor set and binding.
    for (auto *variable : tensors.variables) {
        for (const auto *decoration : decorations->GetDecorationsFor(variable->result_id(), false)) {
            switch (static_cast<spv::Decoration>(decoration->GetSingleWordInOperand(1))) {
            case spv::Decoration::DescriptorSet:
            case spv::Decoration::Binding:
            case spv::Decoration::NonReadable:
            case spv::Decoration::NonWritable:
                break;
            default:
                // TODO: Translate tensor memory qualifiers onto element accesses.
                throw std::runtime_error("Unsupported TensorARM variable decoration");
            }
        }
        const auto replacementVariableId = context.TakeNextId();
        const auto replacementPointerTypeId = replacementTypeIds.at(variable->type_id());
        context.AddGlobalValue(
            makeInstruction(context, spv::Op::OpVariable, replacementPointerTypeId, replacementVariableId,
                            {{SPV_OPERAND_TYPE_STORAGE_CLASS, {static_cast<uint32_t>(spv::StorageClass::Uniform)}}}));
        // Read/write restrictions apply to tensor data; descriptor metadata is always readable.
        decorations->CloneDecorations(variable->result_id(), replacementVariableId,
                                      {spv::Decoration::DescriptorSet, spv::Decoration::Binding});
        context.ReplaceAllUsesWithPredicate(variable->result_id(), replacementVariableId,
                                            [](Instruction *use) { return use->opcode() != spv::Op::OpDecorate; });
        context.KillInst(variable);
    }

    // Array selections keep their indices and acquire the replacement pointer types.
    for (auto *selection : tensors.accessChains) {
        selection->SetResultType(replacementTypeIds.at(selection->type_id()));
        context.UpdateDefUse(selection);
    }
    if (tensors.dynamicIndexing) {
        context.AddCapability(spv::Capability::UniformBufferArrayDynamicIndexing);
    }
    // Each selected block contains its descriptor metadata at member zero.
    const auto descriptorMember = context.get_constant_mgr()->GetUIntConstId(0);
    for (auto *load : tensors.loads) {
        InstructionBuilder builder(&context, load, IRContext::kAnalysisDefUse);
        const auto selectedBlockPointer = load->GetSingleWordInOperand(0);
        const auto descriptorMemberPointer =
            builder.AddAccessChain(state.descriptorPointerType, selectedBlockPointer, {descriptorMember})->result_id();
        load->SetResultType(state.descriptorType);
        load->SetInOperand(0, {descriptorMemberPointer});
        context.UpdateDefUse(load);
    }
}

void lowerAccesses(AccessState &state, const TensorModule &tensors) {
    auto &context = state.context;
    const auto uintType = context.get_type_mgr()->GetUIntTypeId();
    for (const auto &[inst, tensor] : tensors.accesses) {
        InstructionBuilder builder(&context, inst, IRContext::kAnalysisDefUse);
        const auto descriptor = inst->GetSingleWordInOperand(0);
        uint32_t result;
        if (inst->opcode() == spv::Op::OpTensorQuerySizeARM) {
            if (inst->type_id() != uintType) {
                throw std::runtime_error("TensorARM size queries must return uint32");
            }
            const auto dimension = integerConstant(context, inst->GetSingleWordInOperand(1));
            if (!dimension) {
                // TODO: Support specialised query dimensions.
                throw std::runtime_error("TensorARM query dimension must be a literal constant");
            }
            if (*dimension >= tensor.literalRank.value_or(maxTensorRank)) {
                throw std::runtime_error("TensorARM query dimension exceeds the tensor rank");
            }
            result = emitSizeQuery(state, builder, descriptor, static_cast<uint32_t>(*dimension));
        } else {
            const bool write = inst->opcode() == spv::Op::OpTensorWriteARM;
            const uint32_t maskIndex = write ? 3 : 2;
            if (inst->NumInOperands() > maskIndex && inst->GetSingleWordInOperand(maskIndex) != 0) {
                // TODO: Support explicit bounds values and tensor memory operands.
                throw std::runtime_error("Unsupported TensorARM access operands");
            }
            const auto coordinates = inst->GetSingleWordInOperand(1);
            const auto coordinatesType = definition(context, coordinates).type_id();
            const auto &array = definition(context, coordinatesType);
            if (array.opcode() != spv::Op::OpTypeArray || array.GetSingleWordInOperand(0) != uintType) {
                throw std::runtime_error("TensorARM coordinates must be a uint32 array");
            }
            const auto count = integerConstant(context, array.GetSingleWordInOperand(1));
            if (count && tensor.literalRank && *count != *tensor.literalRank) {
                throw std::runtime_error("TensorARM coordinate count must match the rank");
            }
            const auto valueType =
                write ? definition(context, inst->GetSingleWordInOperand(2)).type_id() : inst->type_id();
            uint32_t elementCount = 1;
            if (valueType != tensor.elementType) {
                const auto &valueArray = definition(context, valueType);
                if (valueArray.opcode() != spv::Op::OpTypeArray ||
                    valueArray.GetSingleWordInOperand(0) != tensor.elementType) {
                    throw std::runtime_error("TensorARM access must use matching scalar or array elements");
                }
                const auto length = integerConstant(context, valueArray.GetSingleWordInOperand(1));
                if (!length) {
                    // TODO: Support specialisation-controlled access array lengths.
                    throw std::runtime_error("TensorARM access array length must be a literal constant");
                }
                if (*length == 0 || *length > std::numeric_limits<uint8_t>::max()) {
                    throw std::runtime_error("TensorARM access array length exceeds the supported limit");
                }
                elementCount = static_cast<uint32_t>(*length);
            }
            const bool scalar = valueType == tensor.elementType;
            if (scalar) {
                std::vector<uint32_t> arguments{descriptor, coordinates};
                if (write) {
                    arguments.push_back(inst->GetSingleWordInOperand(2));
                    const auto helper = getOrCreateWriteHelper(state, tensor, coordinatesType, elementCount);
                    result = builder.AddFunctionCall(context.get_type_mgr()->GetVoidTypeId(), helper, arguments)
                                 ->result_id();
                } else {
                    const auto helper = getOrCreateReadHelper(state, tensor, coordinatesType, elementCount);
                    result = builder.AddFunctionCall(tensor.elementType, helper, arguments)->result_id();
                }
            } else if (write) {
                const auto value = inst->GetSingleWordInOperand(2);
                for (uint32_t element = 0; element < elementCount; ++element) {
                    const auto elementValue =
                        builder.AddCompositeExtract(tensor.elementType, value, {element})->result_id();
                    const auto helper = getOrCreateWriteHelper(state, tensor, coordinatesType, elementCount, element);
                    builder.AddFunctionCall(context.get_type_mgr()->GetVoidTypeId(), helper,
                                            {descriptor, coordinates, elementValue});
                }
                result = 0;
            } else {
                std::vector<uint32_t> elements;
                for (uint32_t element = 0; element < elementCount; ++element) {
                    const auto helper = getOrCreateReadHelper(state, tensor, coordinatesType, elementCount, element);
                    elements.push_back(
                        builder.AddFunctionCall(tensor.elementType, helper, {descriptor, coordinates})->result_id());
                }
                result = builder.AddCompositeConstruct(inst->type_id(), elements)->result_id();
            }
        }
        if (inst->result_id() != 0) {
            context.ReplaceAllUsesWith(inst->result_id(), result);
        }
        context.KillInst(inst);
    }
}

void removeTensorDeclarations(IRContext &context, const TensorModule &tensors) {
    for (auto *pointer : tensors.pointerTypes) {
        context.KillInst(pointer);
    }
    for (auto it = tensors.arrayTypes.rbegin(); it != tensors.arrayTypes.rend(); ++it) {
        context.KillInst(*it);
    }
    for (const auto &[id, type] : tensors.types) {
        context.KillInst(&definition(context, id));
    }
    context.RemoveCapability(spv::Capability::TensorsARM);
    context.RemoveCapability(spv::Capability::StorageTensorArrayDynamicIndexingARM);
    context.RemoveCapability(spv::Capability::StorageTensorArrayNonUniformIndexingARM);
    context.RemoveExtension(spvtools::Extension::kSPV_ARM_tensors);
}

spvtools::opt::Pass::Status TensorLoweringPass::Process() {
    const auto *features = context()->get_feature_mgr();
    if (!features->HasCapability(spv::Capability::TensorsARM) &&
        !features->HasExtension(spvtools::Extension::kSPV_ARM_tensors)) {
        return Status::SuccessWithoutChange;
    }

    try {
        if (!context()->GetMemoryModel()) {
            throw std::invalid_argument("missing OpMemoryModel");
        }
        const auto tensors = collectTensors(*context());
        if (!tensors.types.empty()) {
            AccessState state(*context());
            transformInterfaces(state, tensors);
            lowerAccesses(state, tensors);
        }
        removeTensorDeclarations(*context(), tensors);
        loweredTensorCompute = true;
        return Status::SuccessWithChange;
    } catch (const std::invalid_argument &error) {
        if (consumer()) {
            const auto diagnostic = std::string("Malformed SPIR-V: ") + error.what();
            consumer()(SPV_MSG_ERROR, nullptr, {}, diagnostic.c_str());
        }
        return Status::Failure;
    } catch (const std::runtime_error &error) {
        if (consumer()) {
            consumer()(SPV_MSG_ERROR, nullptr, {}, error.what());
        }
        return Status::Failure;
    }
}

} // namespace

bool lowerTensorSpirv(std::vector<uint32_t> &spirv, bool &loweredTensorCompute,
                      const spvtools::MessageConsumer &messageConsumer) {
    loweredTensorCompute = false;
    bool lowered = false;
    spvtools::Optimizer optimizer{SPV_ENV_VULKAN_1_3};
    optimizer.SetMessageConsumer(messageConsumer);
    optimizer.RegisterPass(spvtools::Optimizer::PassToken{std::make_unique<TensorLoweringPass>(lowered)});

    spvtools::OptimizerOptions options;
    // Debug builds validate input before optimizer analyses, which assume valid SPIR-V.
#ifdef NDEBUG
    options.set_run_validator(false);
#endif
    std::vector<uint32_t> output;
    if (!optimizer.Run(spirv.data(), spirv.size(), &output, options)) {
        spirv.clear();
        return false;
    }
    loweredTensorCompute = lowered;
    // Optimizer serialization can omit OpNop; preserve unchanged modules exactly.
    if (loweredTensorCompute) {
        spirv = std::move(output);
    }
    return true;
}

} // namespace mlsdk::el::layer
