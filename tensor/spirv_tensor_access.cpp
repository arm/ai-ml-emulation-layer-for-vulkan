/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include "spirv_tensor_access.hpp"
#include "tensor_descriptor_layout.hpp"

#include <cstddef>
#include <initializer_list>
#include <utility>

namespace mlsdk::el::layer::tensor_spirv {
namespace {

using Descriptor = TensorDescriptorLayout;
constexpr uint32_t maxRank = maxTensorRank;
constexpr uint32_t metadataStride = sizeof(Descriptor::dimensions[0]);
static_assert(offsetof(Descriptor, address) == 0);
static_assert(offsetof(Descriptor, dimensions) == 16);
static_assert(offsetof(Descriptor, strides) == 112);
static_assert(metadataStride == 16 && sizeof(Descriptor) == 208);

uint32_t nullConstant(IRContext &context, uint32_t type) {
    return context.get_constant_mgr()->GetNullConstId(context.get_type_mgr()->GetType(type));
}

struct BoundsAndOffset {
    uint32_t inBounds;
    uint32_t byteOffset;
};

BoundsAndOffset emitDimensionBoundsAndByteOffset(AccessState &state, InstructionBuilder &builder, uint32_t descriptor,
                                                 uint32_t coordinates, uint32_t axis) {
    auto &context = state.context;
    const auto boolType = context.get_type_mgr()->GetBoolTypeId();
    const auto coordinateType = context.get_type_mgr()->GetUIntTypeId();
    const auto coord = builder.AddCompositeExtract(coordinateType, coordinates, {axis})->result_id();
    const auto wideCoord = builder.AddUnaryOp(state.uint64Type, spv::Op::OpUConvert, coord)->result_id();
    const auto dimension = builder.AddCompositeExtract(state.uint64Type, descriptor, {1, axis})->result_id();
    // The advertised per-dimension limit fits in uint32_t; only address arithmetic needs 64 bits.
    const auto narrowDimension = builder.AddUnaryOp(coordinateType, spv::Op::OpUConvert, dimension)->result_id();
    const auto stride = builder.AddCompositeExtract(state.uint64Type, descriptor, {2, axis})->result_id();
    const auto inBounds = builder.AddBinaryOp(boolType, spv::Op::OpULessThan, coord, narrowDimension)->result_id();
    const auto byteOffset = builder.AddBinaryOp(state.uint64Type, spv::Op::OpIMul, wideCoord, stride)->result_id();
    return {inBounds, byteOffset};
}

struct ElementPointer {
    uint32_t id;
    uint32_t alignment;
};

ElementPointer emitElementPointer(AccessState &state, InstructionBuilder &builder, const TensorType &tensor,
                                  uint32_t descriptor, uint32_t byteOffset) {
    const auto base = builder.AddCompositeExtract(state.uint64Type, descriptor, {0})->result_id();
    const auto address = builder.AddBinaryOp(state.uint64Type, spv::Op::OpIAdd, base, byteOffset)->result_id();
    const auto pointerType =
        state.context.get_type_mgr()->FindPointerToType(tensor.elementType, spv::StorageClass::PhysicalStorageBuffer);
    const auto pointer = builder.AddUnaryOp(pointerType, spv::Op::OpConvertUToPtr, address)->result_id();
    const auto *element = state.context.get_def_use_mgr()->GetDef(tensor.elementType);
    const uint32_t alignment = element->GetSingleWordInOperand(0) / 8;
    if (alignment == 1) {
        state.context.AddCapability(spv::Capability::StorageBuffer8BitAccess);
    } else if (alignment == 2) {
        state.context.AddCapability(spv::Capability::StorageBuffer16BitAccess);
    }
    return {pointer, alignment};
}

ElementPointer emitReadElementPointer(AccessState &state, InstructionBuilder &builder, const TensorType &tensor,
                                      uint32_t descriptor, uint32_t elementOffset) {
    auto *types = state.context.get_type_mgr();
    const auto *element = state.context.get_def_use_mgr()->GetDef(tensor.elementType);
    const uint32_t alignment = element->GetSingleWordInOperand(0) / 8;
    spvtools::opt::analysis::RuntimeArray data(types->GetType(tensor.elementType));
    data.AddDecoration({static_cast<uint32_t>(spv::Decoration::ArrayStride), alignment});
    spvtools::opt::analysis::Struct buffer({types->GetRegisteredType(&data)});
    buffer.AddDecoration({static_cast<uint32_t>(spv::Decoration::Block)});
    buffer.AddMemberDecoration(0, {static_cast<uint32_t>(spv::Decoration::Offset), 0});
    const auto bufferPointerType =
        types->FindPointerToType(types->GetTypeInstruction(&buffer), spv::StorageClass::PhysicalStorageBuffer);
    const auto elementPointerType =
        types->FindPointerToType(tensor.elementType, spv::StorageClass::PhysicalStorageBuffer);
    const auto base = builder.AddCompositeExtract(state.uint64Type, descriptor, {0})->result_id();
    const auto bufferPointer = builder.AddUnaryOp(bufferPointerType, spv::Op::OpConvertUToPtr, base)->result_id();
    const auto signedIndex =
        builder.AddUnaryOp(types->GetSIntTypeId(), spv::Op::OpSConvert, elementOffset)->result_id();
    const auto index = builder.AddUnaryOp(types->GetUIntTypeId(), spv::Op::OpBitcast, signedIndex)->result_id();
    const auto zero = state.context.get_constant_mgr()->GetUIntConstId(0);
    const auto pointer = builder.AddAccessChain(elementPointerType, bufferPointer, {zero, index})->result_id();
    if (alignment == 1) {
        state.context.AddCapability(spv::Capability::StorageBuffer8BitAccess);
    } else if (alignment == 2) {
        state.context.AddCapability(spv::Capability::StorageBuffer16BitAccess);
    }
    return {pointer, alignment};
}

struct AccessFunction {
    std::unique_ptr<spvtools::opt::Function> function;
    std::unique_ptr<spvtools::opt::BasicBlock> entry;
    std::unique_ptr<spvtools::opt::BasicBlock> access;
    std::unique_ptr<spvtools::opt::BasicBlock> merge;
};

struct FunctionParameter {
    uint32_t type;
    uint32_t id;
};

AccessFunction createAccessFunction(IRContext &context, uint32_t returnType,
                                    std::initializer_list<FunctionParameter> parameters) {
    Instruction::OperandList signature{idOperand(returnType)};
    for (const FunctionParameter &parameter : parameters) {
        signature.push_back(idOperand(parameter.type));
    }
    const auto functionType = context.TakeNextId();
    context.AddType(makeInstruction(context, spv::Op::OpTypeFunction, 0, functionType, signature));
    const auto functionId = context.TakeNextId();
    AccessFunction helper;
    helper.function = std::make_unique<spvtools::opt::Function>(
        makeInstruction(context, spv::Op::OpFunction, returnType, functionId,
                        {{SPV_OPERAND_TYPE_FUNCTION_CONTROL, {0}}, idOperand(functionType)}));
    for (const FunctionParameter &parameter : parameters) {
        helper.function->AddParameter(
            makeInstruction(context, spv::Op::OpFunctionParameter, parameter.type, parameter.id));
    }
    helper.entry = std::make_unique<spvtools::opt::BasicBlock>(
        makeInstruction(context, spv::Op::OpLabel, 0, context.TakeNextId()));
    helper.access = std::make_unique<spvtools::opt::BasicBlock>(
        makeInstruction(context, spv::Op::OpLabel, 0, context.TakeNextId()));
    helper.merge = std::make_unique<spvtools::opt::BasicBlock>(
        makeInstruction(context, spv::Op::OpLabel, 0, context.TakeNextId()));
    return helper;
}

uint32_t finishAccessFunction(IRContext &context, AccessFunction helper) {
    const auto functionId = helper.function->result_id();
    helper.function->AddBasicBlock(std::move(helper.entry));
    helper.function->AddBasicBlock(std::move(helper.access));
    helper.function->AddBasicBlock(std::move(helper.merge));
    helper.function->SetFunctionEnd(makeInstruction(context, spv::Op::OpFunctionEnd, 0, 0));
    auto *defUse = context.get_def_use_mgr();
    // Register all definitions before uses so branches can reference labels in later blocks.
    helper.function->ForEachInst([&](Instruction *inst) { defUse->AnalyzeInstDef(inst); });
    helper.function->ForEachInst([&](Instruction *inst) { defUse->AnalyzeInstUse(inst); });
    context.AddFunction(std::move(helper.function));
    return functionId;
}

} // namespace

AccessState::AccessState(IRContext &ir) : context(ir) {
    auto *types = context.get_type_mgr();
    const auto *wideType = types->GetIntType(64, false);
    uint64Type = types->GetTypeInstruction(wideType);
    const auto length = context.get_constant_mgr()->GetUIntConstId(maxRank);
    spvtools::opt::analysis::Array metadata(wideType, {length, {0, maxRank}});
    metadata.AddDecoration({static_cast<uint32_t>(spv::Decoration::ArrayStride), metadataStride});
    const auto *metadataType = types->GetRegisteredType(&metadata);
    spvtools::opt::analysis::Struct descriptor({wideType, metadataType, metadataType});
    descriptor.AddMemberDecoration(0, {static_cast<uint32_t>(spv::Decoration::Offset), offsetof(Descriptor, address)});
    descriptor.AddMemberDecoration(1,
                                   {static_cast<uint32_t>(spv::Decoration::Offset), offsetof(Descriptor, dimensions)});
    descriptor.AddMemberDecoration(2, {static_cast<uint32_t>(spv::Decoration::Offset), offsetof(Descriptor, strides)});
    descriptorType = types->GetTypeInstruction(&descriptor);
    descriptorPointerType = types->FindPointerToType(descriptorType, spv::StorageClass::Uniform);
    spvtools::opt::analysis::Struct interface({types->GetType(descriptorType)});
    interface.AddDecoration({static_cast<uint32_t>(spv::Decoration::Block)});
    interface.AddMemberDecoration(0, {static_cast<uint32_t>(spv::Decoration::Offset), 0});
    interfacePointerType = types->FindPointerToType(types->GetTypeInstruction(&interface), spv::StorageClass::Uniform);
    context.AddCapability(spv::Capability::Int64);
    context.AddCapability(spv::Capability::PhysicalStorageBufferAddresses);
    if (!context.get_feature_mgr()->HasExtension(spvtools::Extension::kSPV_KHR_physical_storage_buffer)) {
        context.AddExtension("SPV_KHR_physical_storage_buffer");
    }
    auto *memoryModel = context.module()->GetMemoryModel();
    memoryModel->SetInOperand(0, {static_cast<uint32_t>(spv::AddressingModel::PhysicalStorageBuffer64)});
}

uint32_t getOrCreateReadHelper(AccessState &state, const TensorType &tensor, uint32_t coordinatesType) {
    const auto key = std::make_tuple(spv::Op::OpTensorReadARM, tensor.elementType, coordinatesType, tensor.rank);
    if (const auto it = state.helpers.find(key); it != state.helpers.end()) {
        return it->second;
    }
    auto &context = state.context;
    const auto descriptor = context.TakeNextId();
    const auto coordinates = context.TakeNextId();
    auto helper = createAccessFunction(context, tensor.elementType,
                                       {{state.descriptorType, descriptor}, {coordinatesType, coordinates}});
    InstructionBuilder builder(&context, helper.entry.get());
    const auto boolType = context.get_type_mgr()->GetBoolTypeId();
    const auto zero = nullConstant(context, state.uint64Type);
    BoundsAndOffset bounds{0, zero};
    // Stop accumulating after the first invalid coordinate, preserving the valid prefix.
    for (uint32_t axis = 0; axis < tensor.rank; ++axis) {
        const auto dimension = emitDimensionBoundsAndByteOffset(state, builder, descriptor, coordinates, axis);
        bounds.inBounds =
            axis == 0 ? dimension.inBounds
                      : builder.AddBinaryOp(boolType, spv::Op::OpLogicalAnd, bounds.inBounds, dimension.inBounds)
                            ->result_id();
        const auto contribution =
            builder.AddSelect(state.uint64Type, bounds.inBounds, dimension.byteOffset, zero)->result_id();
        bounds.byteOffset =
            builder.AddBinaryOp(state.uint64Type, spv::Op::OpIAdd, bounds.byteOffset, contribution)->result_id();
    }
    // Convert byte offsets to signed element offsets before branching, then use a
    // uint32_t index into the typed buffer. Keep this sequence intact: simplifying
    // it to base + byteOffset let Mesa 25.2.8 discard the guarded offset and flatten
    // an out-of-bounds read into an unconditional load.
    auto *const signedType = context.get_type_mgr()->GetIntType(64, true);
    const auto signedTypeId = context.get_type_mgr()->GetTypeInstruction(signedType);
    const auto *element = context.get_def_use_mgr()->GetDef(tensor.elementType);
    const auto elementSize = element->GetSingleWordInOperand(0) / 8;
    const auto *sizeConstant = context.get_constant_mgr()->GetConstant(signedType, {elementSize, 0});
    const auto size = context.get_constant_mgr()->GetDefiningInstruction(sizeConstant)->result_id();
    const auto signedOffset = builder.AddUnaryOp(signedTypeId, spv::Op::OpBitcast, bounds.byteOffset)->result_id();
    const auto elementOffset = builder.AddBinaryOp(signedTypeId, spv::Op::OpSDiv, signedOffset, size)->result_id();
    builder.AddConditionalBranch(bounds.inBounds, helper.access->id(), helper.merge->id(), helper.merge->id());

    builder.SetInsertPoint(helper.access.get());
    const auto pointer = emitReadElementPointer(state, builder, tensor, descriptor, elementOffset);
    const auto value = builder.AddLoad(tensor.elementType, pointer.id, pointer.alignment)->result_id();
    builder.AddBranch(helper.merge->id());

    builder.SetInsertPoint(helper.merge.get());
    auto *const result = builder.AddPhi(tensor.elementType, {nullConstant(context, tensor.elementType),
                                                             helper.entry->id(), value, helper.access->id()});
    builder.AddUnaryOp(0, spv::Op::OpReturnValue, result->result_id());
    const auto functionId = finishAccessFunction(context, std::move(helper));
    state.helpers.emplace(key, functionId);
    return functionId;
}

uint32_t getOrCreateWriteHelper(AccessState &state, const TensorType &tensor, uint32_t coordinatesType) {
    const auto key = std::make_tuple(spv::Op::OpTensorWriteARM, tensor.elementType, coordinatesType, tensor.rank);
    if (const auto it = state.helpers.find(key); it != state.helpers.end()) {
        return it->second;
    }
    auto &context = state.context;
    const auto descriptor = context.TakeNextId();
    const auto coordinates = context.TakeNextId();
    const auto value = context.TakeNextId();
    auto helper = createAccessFunction(
        context, context.get_type_mgr()->GetVoidTypeId(),
        {{state.descriptorType, descriptor}, {coordinatesType, coordinates}, {tensor.elementType, value}});
    InstructionBuilder builder(&context, helper.entry.get());
    const auto boolType = context.get_type_mgr()->GetBoolTypeId();
    BoundsAndOffset bounds{0, nullConstant(context, state.uint64Type)};
    // Accumulate the full offset; the final bounds check controls whether the store executes.
    for (uint32_t axis = 0; axis < tensor.rank; ++axis) {
        const auto dimension = emitDimensionBoundsAndByteOffset(state, builder, descriptor, coordinates, axis);
        bounds.inBounds =
            axis == 0 ? dimension.inBounds
                      : builder.AddBinaryOp(boolType, spv::Op::OpLogicalAnd, bounds.inBounds, dimension.inBounds)
                            ->result_id();
        bounds.byteOffset =
            builder.AddBinaryOp(state.uint64Type, spv::Op::OpIAdd, bounds.byteOffset, dimension.byteOffset)
                ->result_id();
    }
    builder.AddConditionalBranch(bounds.inBounds, helper.access->id(), helper.merge->id(), helper.merge->id());

    builder.SetInsertPoint(helper.access.get());
    const auto pointer = emitElementPointer(state, builder, tensor, descriptor, bounds.byteOffset);
    builder.AddInstruction(
        makeInstruction(context, spv::Op::OpStore, 0, 0,
                        {idOperand(pointer.id),
                         idOperand(value),
                         {SPV_OPERAND_TYPE_MEMORY_ACCESS, {static_cast<uint32_t>(spv::MemoryAccessMask::Aligned)}},
                         {SPV_OPERAND_TYPE_LITERAL_INTEGER, {pointer.alignment}}}));
    builder.AddBranch(helper.merge->id());

    builder.SetInsertPoint(helper.merge.get());
    builder.AddNullaryOp(0, spv::Op::OpReturn);
    const auto functionId = finishAccessFunction(context, std::move(helper));
    state.helpers.emplace(key, functionId);
    return functionId;
}

uint32_t emitSizeQuery(AccessState &state, InstructionBuilder &builder, uint32_t descriptor, uint32_t dimension) {
    const auto size = builder.AddCompositeExtract(state.uint64Type, descriptor, {1, dimension})->result_id();
    return builder.AddUnaryOp(state.context.get_type_mgr()->GetUIntTypeId(), spv::Op::OpUConvert, size)->result_id();
}

} // namespace mlsdk::el::layer::tensor_spirv
