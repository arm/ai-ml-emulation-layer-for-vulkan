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
#include <vector>

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
    uint32_t block;
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
    return {inBounds, byteOffset, 0};
}

struct AccessFunction {
    std::unique_ptr<spvtools::opt::Function> function;
    std::unique_ptr<spvtools::opt::BasicBlock> entry;
    std::vector<std::unique_ptr<spvtools::opt::BasicBlock>> boundsBlocks;
    std::unique_ptr<spvtools::opt::BasicBlock> access;
    std::unique_ptr<spvtools::opt::BasicBlock> merge;
};

std::unique_ptr<spvtools::opt::BasicBlock> createBlock(IRContext &context) {
    return std::make_unique<spvtools::opt::BasicBlock>(
        makeInstruction(context, spv::Op::OpLabel, 0, context.TakeNextId()));
}

struct AccessArrays {
    uint32_t coordinates;
    uint32_t dimensions;
    uint32_t strides;
    // Function-storage pointers to individual array elements, used by OpAccessChain.
    uint32_t coordinatePointerType;
    uint32_t metadataPointerType;
};

// OpCompositeExtract takes literal indices. Copy the by-value helper arguments
// into function-local arrays so the generated loop can index them with its axis ID.
// Call this at the start of the entry block: OpVariable must precede other instructions.
AccessArrays prepareAccessArrays(AccessState &state, InstructionBuilder &builder, uint32_t descriptor,
                                 uint32_t coordinates, uint32_t coordinatesType) {
    auto &context = state.context;
    auto *types = context.get_type_mgr();
    const auto uintType = types->GetUIntTypeId();
    const auto coordsPointer = types->FindPointerToType(coordinatesType, spv::StorageClass::Function);
    const auto length = context.get_constant_mgr()->GetUIntConstId(maxRank);
    const spvtools::opt::analysis::Array metadataType(types->GetType(state.uint64Type), {length, {0, maxRank}});
    const auto metadataTypeId = types->GetTypeInstruction(&metadataType);
    const auto metadataPointer = types->FindPointerToType(metadataTypeId, spv::StorageClass::Function);
    const auto coordinatePointer = types->FindPointerToType(uintType, spv::StorageClass::Function);
    const auto widePointer = types->FindPointerToType(state.uint64Type, spv::StorageClass::Function);
    const auto storageClass = static_cast<uint32_t>(spv::StorageClass::Function);
    const auto coords = builder.AddVariable(coordsPointer, storageClass)->result_id();
    const auto dimensions = builder.AddVariable(metadataPointer, storageClass)->result_id();
    const auto strides = builder.AddVariable(metadataPointer, storageClass)->result_id();
    builder.AddStore(coords, coordinates);
    // The descriptor's buffer-layout decorations are invalid on function-local
    // arrays. Reconstruct plain arrays from its six slots, preserving the values.
    std::vector<uint32_t> dimensionValues;
    std::vector<uint32_t> strideValues;
    for (uint32_t axis = 0; axis < maxRank; ++axis) {
        dimensionValues.push_back(builder.AddCompositeExtract(state.uint64Type, descriptor, {1, axis})->result_id());
        strideValues.push_back(builder.AddCompositeExtract(state.uint64Type, descriptor, {2, axis})->result_id());
    }
    builder.AddStore(dimensions, builder.AddCompositeConstruct(metadataTypeId, dimensionValues)->result_id());
    builder.AddStore(strides, builder.AddCompositeConstruct(metadataTypeId, strideValues)->result_id());
    return {coords, dimensions, strides, coordinatePointer, widePointer};
}

// SPIR-V value IDs carried between iterations of the dimension loop.
struct DimensionValues {
    uint32_t axis;
    uint32_t byteOffset;
    uint32_t inBounds;
};

// Emit one axis of the shared read/write calculation:
//   next.inBounds = current.inBounds && coords[axis] < dimensions[axis];
// The caller accumulates the byte offset, preserving the read/write access policy.
// The caller reserves the next-value IDs before constructing the loop's OpPhi nodes.
template <typename EmitOffset>
void emitDimensionStep(AccessState &state, InstructionBuilder &builder, const AccessArrays &arrays,
                       const DimensionValues &current, const DimensionValues &next, const EmitOffset &emitOffset) {
    auto &context = state.context;
    const auto uintType = context.get_type_mgr()->GetUIntTypeId();
    const auto boolType = context.get_type_mgr()->GetBoolTypeId();
    const auto coordAddress =
        builder.AddAccessChain(arrays.coordinatePointerType, arrays.coordinates, {current.axis})->result_id();
    const auto coord = builder.AddLoad(uintType, coordAddress)->result_id();
    const auto wideCoord = builder.AddUnaryOp(state.uint64Type, spv::Op::OpUConvert, coord)->result_id();
    const auto dimensionAddress =
        builder.AddAccessChain(arrays.metadataPointerType, arrays.dimensions, {current.axis})->result_id();
    const auto strideAddress =
        builder.AddAccessChain(arrays.metadataPointerType, arrays.strides, {current.axis})->result_id();
    const auto dimension = builder.AddLoad(state.uint64Type, dimensionAddress)->result_id();
    // The advertised per-dimension limit fits in uint32_t; only address arithmetic needs 64 bits.
    const auto narrowDimension = builder.AddUnaryOp(uintType, spv::Op::OpUConvert, dimension)->result_id();
    const auto stride = builder.AddLoad(state.uint64Type, strideAddress)->result_id();
    const auto valid = builder.AddBinaryOp(boolType, spv::Op::OpULessThan, coord, narrowDimension)->result_id();
    builder.AddNaryOp(boolType, spv::Op::OpLogicalAnd, {current.inBounds, valid}, next.inBounds);
    const auto contribution = builder.AddBinaryOp(state.uint64Type, spv::Op::OpIMul, wideCoord, stride)->result_id();
    emitOffset(current.byteOffset, next.inBounds, contribution, next.byteOffset);
}

// Build for (axis = 0; axis < rank; ++axis) inside the read/write helper.
// rank is a SPIR-V value ID; its specialisation value remains for the pipeline compiler.
template <typename EmitOffset>
BoundsAndOffset emitDimensionLoop(AccessState &state, AccessFunction &helper, InstructionBuilder &builder,
                                  const AccessArrays &arrays, uint32_t rank, uint32_t boolType,
                                  const EmitOffset &emitOffset) {
    auto &context = state.context;
    auto *types = context.get_type_mgr();
    const auto uintType = types->GetUIntTypeId();
    auto header = createBlock(context);
    auto body = createBlock(context);
    auto next = createBlock(context);
    auto done = createBlock(context);
    const DimensionValues nextValues{context.TakeNextId(), context.TakeNextId(), context.TakeNextId()};
    const auto zero = context.get_constant_mgr()->GetUIntConstId(0);
    const auto one = context.get_constant_mgr()->GetUIntConstId(1);
    const auto trueValue =
        context.get_constant_mgr()
            ->GetDefiningInstruction(context.get_constant_mgr()->GetConstant(types->GetType(boolType), {1}))
            ->result_id();
    builder.AddBranch(header->id());
    builder.SetInsertPoint(header.get());

    // On entry use axis=0, offset=0 and inBounds=true. On the back edge use the
    // preceding iteration's results. OpPhi refers to IDs defined later in the loop.
    const auto axis = builder.AddPhi(uintType, {zero, helper.entry->id(), nextValues.axis, next->id()})->result_id();
    const auto offset = builder
                            .AddPhi(state.uint64Type, {nullConstant(context, state.uint64Type), helper.entry->id(),
                                                       nextValues.byteOffset, next->id()})
                            ->result_id();
    const auto inBounds =
        builder.AddPhi(boolType, {trueValue, helper.entry->id(), nextValues.inBounds, next->id()})->result_id();
    const auto remaining = builder.AddBinaryOp(boolType, spv::Op::OpULessThan, axis, rank)->result_id();
    // Structured control flow identifies the loop exit and its continue block.
    builder.AddLoopMerge(done->id(), next->id());
    builder.AddConditionalBranch(remaining, body->id(), done->id());

    builder.SetInsertPoint(body.get());
    emitDimensionStep(state, builder, arrays, {axis, offset, inBounds}, nextValues, emitOffset);
    builder.AddBranch(next->id());
    builder.SetInsertPoint(next.get());
    builder.AddNaryOp(uintType, spv::Op::OpIAdd, {axis, one}, nextValues.axis);
    builder.AddBranch(header->id());

    // Leave insertion at the loop exit, where the caller emits its guarded load
    // or store. Reads also need this block ID for the zero-result OpPhi input.
    builder.SetInsertPoint(done.get());
    const auto exit = done->id();
    helper.boundsBlocks.push_back(std::move(header));
    helper.boundsBlocks.push_back(std::move(body));
    helper.boundsBlocks.push_back(std::move(next));
    helper.boundsBlocks.push_back(std::move(done));
    return {inBounds, offset, exit};
}

template <typename EmitOffset>
BoundsAndOffset emitSpecializedBoundsAndByteOffset(AccessState &state, AccessFunction &helper,
                                                   InstructionBuilder &builder, const TensorType &tensor,
                                                   uint32_t descriptor, uint32_t coordinates, uint32_t coordinatesType,
                                                   const EmitOffset &emitOffset) {
    auto &context = state.context;
    const auto uintType = context.get_type_mgr()->GetUIntTypeId();
    const auto boolType = context.get_type_mgr()->GetBoolTypeId();
    // These C++ helpers emit the array setup and loop directly into the access
    // function. They introduce no additional SPIR-V function calls.
    const auto arrays = prepareAccessArrays(state, builder, descriptor, coordinates, coordinatesType);

    // Preserve the rank's original ID, including any specialisation expression.
    // Only adapt its integer type to the uint32 loop counter when necessary.
    auto rank = tensor.rankId;
    const auto rankType = context.get_def_use_mgr()->GetDef(rank)->type_id();
    if (rankType != uintType) {
        const auto width = context.get_def_use_mgr()->GetDef(rankType)->GetSingleWordInOperand(0);
        rank = builder.AddUnaryOp(uintType, width == 32 ? spv::Op::OpBitcast : spv::Op::OpUConvert, rank)->result_id();
    }
    return emitDimensionLoop(state, helper, builder, arrays, rank, boolType, emitOffset);
}

// EmitOffset supplies only the access-specific accumulation. The bounds checks
// and the literal/specialised rank traversal are shared by reads and writes.
template <typename EmitOffset>
BoundsAndOffset emitBoundsAndByteOffset(AccessState &state, AccessFunction &helper, InstructionBuilder &builder,
                                        const TensorType &tensor, uint32_t descriptor, uint32_t coordinates,
                                        uint32_t coordinatesType, const EmitOffset &emitOffset) {
    if (!tensor.literalRank) {
        return emitSpecializedBoundsAndByteOffset(state, helper, builder, tensor, descriptor, coordinates,
                                                  coordinatesType, emitOffset);
    }
    auto &context = state.context;
    const auto boolType = context.get_type_mgr()->GetBoolTypeId();
    BoundsAndOffset bounds{0, nullConstant(context, state.uint64Type), helper.entry->id()};
    for (uint32_t axis = 0; axis < *tensor.literalRank; ++axis) {
        const auto dimension = emitDimensionBoundsAndByteOffset(state, builder, descriptor, coordinates, axis);
        bounds.inBounds =
            axis == 0 ? dimension.inBounds
                      : builder.AddBinaryOp(boolType, spv::Op::OpLogicalAnd, bounds.inBounds, dimension.inBounds)
                            ->result_id();
        bounds.byteOffset = emitOffset(bounds.byteOffset, bounds.inBounds, dimension.byteOffset, 0);
    }
    return bounds;
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
    helper.entry = createBlock(context);
    helper.access = createBlock(context);
    helper.merge = createBlock(context);
    return helper;
}

uint32_t finishAccessFunction(IRContext &context, AccessFunction helper) {
    const auto functionId = helper.function->result_id();
    helper.function->AddBasicBlock(std::move(helper.entry));
    for (auto &block : helper.boundsBlocks) {
        helper.function->AddBasicBlock(std::move(block));
    }
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
    const auto key = std::make_tuple(spv::Op::OpTensorReadARM, tensor.elementType, coordinatesType, tensor.rankId);
    if (const auto it = state.helpers.find(key); it != state.helpers.end()) {
        return it->second;
    }
    auto &context = state.context;
    const auto descriptor = context.TakeNextId();
    const auto coordinates = context.TakeNextId();
    auto helper = createAccessFunction(context, tensor.elementType,
                                       {{state.descriptorType, descriptor}, {coordinatesType, coordinates}});
    InstructionBuilder builder(&context, helper.entry.get());
    const auto zero = nullConstant(context, state.uint64Type);
    // Stop accumulating after the first invalid coordinate, preserving the valid prefix.
    const auto emitOffset = [&](uint32_t offset, uint32_t inBounds, uint32_t contribution, uint32_t resultId) {
        const auto guarded = builder.AddSelect(state.uint64Type, inBounds, contribution, zero)->result_id();
        return builder.AddNaryOp(state.uint64Type, spv::Op::OpIAdd, {offset, guarded}, resultId)->result_id();
    };
    const auto bounds =
        emitBoundsAndByteOffset(state, helper, builder, tensor, descriptor, coordinates, coordinatesType, emitOffset);
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
    auto *const result = builder.AddPhi(
        tensor.elementType, {nullConstant(context, tensor.elementType), bounds.block, value, helper.access->id()});
    builder.AddUnaryOp(0, spv::Op::OpReturnValue, result->result_id());
    const auto functionId = finishAccessFunction(context, std::move(helper));
    state.helpers.emplace(key, functionId);
    return functionId;
}

uint32_t getOrCreateWriteHelper(AccessState &state, const TensorType &tensor, uint32_t coordinatesType) {
    const auto key = std::make_tuple(spv::Op::OpTensorWriteARM, tensor.elementType, coordinatesType, tensor.rankId);
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
    // Accumulate the full offset; the final bounds check controls whether the store executes.
    const auto emitOffset = [&](uint32_t offset, uint32_t, uint32_t contribution, uint32_t resultId) {
        return builder.AddNaryOp(state.uint64Type, spv::Op::OpIAdd, {offset, contribution}, resultId)->result_id();
    };
    const auto bounds =
        emitBoundsAndByteOffset(state, helper, builder, tensor, descriptor, coordinates, coordinatesType, emitOffset);
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
