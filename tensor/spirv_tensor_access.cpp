/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include "spirv_tensor_access.hpp"
#include "tensor_descriptor_layout.hpp"

#include <cstddef>
#include <initializer_list>
#include <optional>
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

// Examples below are shader pseudocode; C++ uint32_t variables hold SPIR-V IDs.
struct ArrayAccessInfo {
    uint32_t remainingElements;
    uint32_t innermostStride;
};

struct BoundsAndOffset {
    uint32_t inBounds;
    uint32_t byteOffset;
    uint32_t block;
    // Multi-element accesses need both IDs to advance along the innermost dimension.
    std::optional<ArrayAccessInfo> arrayAccess = std::nullopt;
};

/* Equivalent shader logic:

    uint coordinate = coordinates[axis];
    uint dimensionSize = uint(descriptor.dimensions[axis]);
    bool inBounds = coordinate < dimensionSize;
    uint64_t byteOffset = uint64_t(coordinate) * descriptor.strides[axis];

    // For multi-element accesses on the innermost axis, also emit:
    uint64_t remainingElements = descriptor.dimensions[axis] - uint64_t(coordinate);
    uint64_t innermostStride = descriptor.strides[axis];

*/
BoundsAndOffset emitDimensionBoundsAndByteOffset(AccessState &state, InstructionBuilder &builder, uint32_t descriptor,
                                                 uint32_t coordinates, uint32_t axis, uint32_t axisAccessElementCount) {
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
    BoundsAndOffset bounds{inBounds, byteOffset, 0};
    if (axisAccessElementCount > 1) {
        // Only array accesses on the last axis need per-lane bounds and offsets.
        const auto remainingElements =
            builder.AddBinaryOp(state.uint64Type, spv::Op::OpISub, dimension, wideCoord)->result_id();
        bounds.arrayAccess = ArrayAccessInfo{remainingElements, stride};
    }
    return bounds;
}

struct AccessFunction {
    std::unique_ptr<spvtools::opt::Function> function;
    std::unique_ptr<spvtools::opt::BasicBlock> entry;
    std::vector<std::unique_ptr<spvtools::opt::BasicBlock>> blocks;
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

/* Equivalent shader logic:

    uint coords[RANK] = coordinates;
    uint64_t dimensions[6] = descriptor.dimensions;
    uint64_t strides[6] = descriptor.strides;

*/
// Local arrays allow dynamic indexing; OpCompositeExtract requires literal indices.
// Reconstruct metadata arrays to remove buffer-layout decorations. Emit their
// OpVariable declarations first in the entry block.
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

/* Equivalent shader logic (choose the read or write accumulation):

    uint coordinate = coords[current.axis];
    uint dimensionSize = uint(dimensions[current.axis]);
    next.inBounds = current.inBounds && coordinate < dimensionSize;
    uint64_t contribution = uint64_t(coordinate) * strides[current.axis];

    // Read:
    next.byteOffset = current.byteOffset + (next.inBounds ? contribution : 0);
    // Write:
    next.byteOffset = current.byteOffset + contribution;

*/
// Reserve next-value IDs before constructing the loop's OpPhi nodes.
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

/* Equivalent shader logic:

    bool inBounds = true;
    uint64_t byteOffset = 0;
    for (uint axis = 0; axis < rank; ++axis) {
        // Code emitted by emitDimensionStep updates inBounds and byteOffset here.
    }

*/
// OpPhi nodes carry axis, inBounds and byteOffset between iterations.
// rank retains its specialisation value for the pipeline compiler.
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
    helper.blocks.push_back(std::move(header));
    helper.blocks.push_back(std::move(body));
    helper.blocks.push_back(std::move(next));
    helper.blocks.push_back(std::move(done));
    return {inBounds, offset, exit};
}

/* Equivalent shader logic:

    // Code emitted by prepareAccessArrays creates coords, dimensions and strides.
    // Code emitted by emitDimensionLoop computes bounds.inBounds and bounds.byteOffset.

    // For multi-element accesses, also emit after the loop:
    uint last = rank - 1;
    bounds.arrayAccess.remainingElements = dimensions[last] - uint64_t(coords[last]);
    bounds.arrayAccess.innermostStride = strides[last];

*/
// rank retains its specialisation value for the pipeline compiler.
template <typename EmitOffset>
BoundsAndOffset emitSpecializedBoundsAndByteOffset(AccessState &state, AccessFunction &helper,
                                                   InstructionBuilder &builder, const TensorType &tensor,
                                                   uint32_t descriptor, uint32_t coordinates, uint32_t coordinatesType,
                                                   uint32_t elementCount, const EmitOffset &emitOffset) {
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
    auto bounds = emitDimensionLoop(state, helper, builder, arrays, rank, boolType, emitOffset);
    if (elementCount > 1) {
        // The last axis is also selected by the preserved rank specialisation.
        const auto last =
            builder.AddBinaryOp(uintType, spv::Op::OpISub, rank, context.get_constant_mgr()->GetUIntConstId(1))
                ->result_id();
        const auto coordAddress =
            builder.AddAccessChain(arrays.coordinatePointerType, arrays.coordinates, {last})->result_id();
        const auto coord = builder.AddLoad(uintType, coordAddress)->result_id();
        const auto wideCoord = builder.AddUnaryOp(state.uint64Type, spv::Op::OpUConvert, coord)->result_id();
        const auto dimensionAddress =
            builder.AddAccessChain(arrays.metadataPointerType, arrays.dimensions, {last})->result_id();
        const auto dimension = builder.AddLoad(state.uint64Type, dimensionAddress)->result_id();
        const auto remainingElements =
            builder.AddBinaryOp(state.uint64Type, spv::Op::OpISub, dimension, wideCoord)->result_id();
        const auto strideAddress =
            builder.AddAccessChain(arrays.metadataPointerType, arrays.strides, {last})->result_id();
        const auto innermostStride = builder.AddLoad(state.uint64Type, strideAddress)->result_id();
        bounds.arrayAccess = ArrayAccessInfo{remainingElements, innermostStride};
    }
    return bounds;
}

// Emit each axis using literal-index extracts. This C++ loop produces straight-line
// shader code; EmitOffset selects the read/write accumulation shown above.
template <typename EmitOffset>
BoundsAndOffset emitLiteralBoundsAndByteOffset(AccessState &state, AccessFunction &helper, InstructionBuilder &builder,
                                               uint32_t literalRank, uint32_t descriptor, uint32_t coordinates,
                                               uint32_t elementCount, const EmitOffset &emitOffset) {
    auto &context = state.context;
    const auto boolType = context.get_type_mgr()->GetBoolTypeId();
    BoundsAndOffset bounds{0, nullConstant(context, state.uint64Type), helper.entry->id()};
    for (uint32_t axis = 0; axis < literalRank; ++axis) {
        const auto dimension = emitDimensionBoundsAndByteOffset(state, builder, descriptor, coordinates, axis,
                                                                axis + 1 == literalRank ? elementCount : 1);
        bounds.inBounds =
            axis == 0 ? dimension.inBounds
                      : builder.AddBinaryOp(boolType, spv::Op::OpLogicalAnd, bounds.inBounds, dimension.inBounds)
                            ->result_id();
        bounds.byteOffset = emitOffset(bounds.byteOffset, bounds.inBounds, dimension.byteOffset, 0);
        if (axis + 1 == literalRank) {
            bounds.arrayAccess = dimension.arrayAccess;
        }
    }
    return bounds;
}

// Select straight-line code for a literal rank or a shader loop for a specialised rank.
template <typename EmitOffset>
BoundsAndOffset emitBoundsAndByteOffset(AccessState &state, AccessFunction &helper, InstructionBuilder &builder,
                                        const TensorType &tensor, uint32_t descriptor, uint32_t coordinates,
                                        uint32_t coordinatesType, uint32_t elementCount, const EmitOffset &emitOffset) {
    if (tensor.literalRank) {
        return emitLiteralBoundsAndByteOffset(state, helper, builder, *tensor.literalRank, descriptor, coordinates,
                                              elementCount, emitOffset);
    }
    return emitSpecializedBoundsAndByteOffset(state, helper, builder, tensor, descriptor, coordinates, coordinatesType,
                                              elementCount, emitOffset);
}

/* Equivalent shader logic for element > 0 (element 0 reuses bounds):

    uint64_t lane = uint64_t(element);
    bool inBounds = bounds.inBounds && lane < bounds.arrayAccess.remainingElements;
    uint64_t byteOffset = bounds.byteOffset + lane * bounds.arrayAccess.innermostStride;

*/
// Each lane stays on the starting row. Use 64-bit arithmetic to avoid coordinate
// wraparound. remainingElements may underflow for an invalid starting coordinate;
// bounds.inBounds still rejects all lanes of that access.
BoundsAndOffset emitElementBounds(AccessState &state, InstructionBuilder &builder, const BoundsAndOffset &bounds,
                                  uint32_t element, uint32_t block) {
    if (element == 0) {
        return bounds;
    }
    // Lanes after the first require metadata emitted for a multi-element access.
    const auto &arrayAccess = bounds.arrayAccess.value();
    auto &context = state.context;
    auto *constants = context.get_constant_mgr();
    const auto lane = constants
                          ->GetDefiningInstruction(
                              constants->GetConstant(context.get_type_mgr()->GetType(state.uint64Type), {element, 0}))
                          ->result_id();
    const auto boolType = context.get_type_mgr()->GetBoolTypeId();
    const auto withinRow =
        builder.AddBinaryOp(boolType, spv::Op::OpULessThan, lane, arrayAccess.remainingElements)->result_id();
    const auto valid = builder.AddBinaryOp(boolType, spv::Op::OpLogicalAnd, bounds.inBounds, withinRow)->result_id();
    const auto distance =
        builder.AddBinaryOp(state.uint64Type, spv::Op::OpIMul, lane, arrayAccess.innermostStride)->result_id();
    const auto offset =
        builder.AddBinaryOp(state.uint64Type, spv::Op::OpIAdd, bounds.byteOffset, distance)->result_id();
    return {valid, offset, block};
}

struct ElementPointer {
    uint32_t id;
    uint32_t storageType;
    uint32_t alignment;
};

uint32_t getStorageType(AccessState &state, const TensorType &tensor) {
    auto *types = state.context.get_type_mgr();
    auto storageType = tensor.elementType;
    if (types->GetType(storageType)->AsBool()) {
        // Tensor booleans occupy one byte. Keep bool in shader values and helper
        // signatures, converting only at the backing-memory load or store.
        storageType = types->GetTypeInstruction(types->GetIntType(8, true));
    }
    return storageType;
}

// StorageT is the tensor element's backing-memory type.
/* Equivalent pointer calculation (no memory access):

    uint64_t address = descriptor.address + byteOffset;
    StorageT* pointer = StorageT*(address);

*/
ElementPointer emitElementPointer(AccessState &state, InstructionBuilder &builder, const TensorType &tensor,
                                  uint32_t descriptor, uint32_t byteOffset) {
    const auto base = builder.AddCompositeExtract(state.uint64Type, descriptor, {0})->result_id();
    const auto address = builder.AddBinaryOp(state.uint64Type, spv::Op::OpIAdd, base, byteOffset)->result_id();
    auto *types = state.context.get_type_mgr();
    const auto storageType = getStorageType(state, tensor);
    const auto pointerType = types->FindPointerToType(storageType, spv::StorageClass::PhysicalStorageBuffer);
    const auto pointer = builder.AddUnaryOp(pointerType, spv::Op::OpConvertUToPtr, address)->result_id();
    const auto *element = state.context.get_def_use_mgr()->GetDef(storageType);
    const uint32_t alignment = element->GetSingleWordInOperand(0) / 8;
    if (alignment == 1) {
        state.context.AddCapability(spv::Capability::StorageBuffer8BitAccess);
    } else if (alignment == 2) {
        state.context.AddCapability(spv::Capability::StorageBuffer16BitAccess);
    }
    return {pointer, storageType, alignment};
}

/* Equivalent pointer calculation (no memory access):

    Buffer<StorageT> buffer = bufferAt(descriptor.address);
    uint index = uint(int(elementOffset));
    StorageT* pointer = &buffer.elements[index];

*/
// Buffer elements have stride sizeof(StorageT). Keep this indexing form with the
// signed 64-bit elementOffset; see the Mesa workaround in getOrCreateReadHelper.
ElementPointer emitReadElementPointer(AccessState &state, InstructionBuilder &builder, const TensorType &tensor,
                                      uint32_t descriptor, uint32_t elementOffset) {
    auto *types = state.context.get_type_mgr();
    const auto storageType = getStorageType(state, tensor);
    const auto *element = state.context.get_def_use_mgr()->GetDef(storageType);
    const uint32_t alignment = element->GetSingleWordInOperand(0) / 8;
    spvtools::opt::analysis::RuntimeArray data(types->GetType(storageType));
    data.AddDecoration({static_cast<uint32_t>(spv::Decoration::ArrayStride), alignment});
    spvtools::opt::analysis::Struct buffer({types->GetRegisteredType(&data)});
    buffer.AddDecoration({static_cast<uint32_t>(spv::Decoration::Block)});
    buffer.AddMemberDecoration(0, {static_cast<uint32_t>(spv::Decoration::Offset), 0});
    const auto bufferPointerType =
        types->FindPointerToType(types->GetTypeInstruction(&buffer), spv::StorageClass::PhysicalStorageBuffer);
    const auto elementPointerType = types->FindPointerToType(storageType, spv::StorageClass::PhysicalStorageBuffer);
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
    return {pointer, storageType, alignment};
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
    return helper;
}

uint32_t finishAccessFunction(IRContext &context, AccessFunction helper) {
    const auto functionId = helper.function->result_id();
    helper.function->AddBasicBlock(std::move(helper.entry));
    for (auto &block : helper.blocks) {
        helper.function->AddBasicBlock(std::move(block));
    }
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

// T is the shader element type; LENGTH is the literal access element count.
/* Equivalent shader logic:

    Bounds bounds;
    // Code emitted by emitBoundsAndByteOffset sets bounds here,
    // using guarded accumulation for reads.
    T values[LENGTH];
    for (uint element = 0; element < LENGTH; ++element) {
        Bounds elementBounds;
        // Code emitted by emitElementBounds sets elementBounds for this lane.
        int64_t elementOffset = int64_t(elementBounds.byteOffset) / int64_t(sizeof(StorageT));
        if (elementBounds.inBounds) {
            StorageT value = load(bufferAt(descriptor.address), uint(int(elementOffset)));
            values[element] = value; // For bool tensors: int(value) != 0.
        } else {
            values[element] = T(0); // false for bool tensors.
        }
    }
    return values;

*/
// C++ emits each lane separately, without a shader loop. Scalar reads return the
// sole element; an array of length one still returns an array. load is a typed
// memory instruction inside the valid branch.
uint32_t getOrCreateReadHelper(AccessState &state, const TensorType &tensor, uint32_t coordinatesType,
                               const AccessValueInfo &access) {
    const AccessFunctionKey key{spv::Op::OpTensorReadARM, access.valueType, coordinatesType, tensor.rankId};
    if (const auto it = state.helpers.find(key); it != state.helpers.end()) {
        return it->second;
    }
    auto &context = state.context;
    const auto descriptor = context.TakeNextId();
    const auto coordinates = context.TakeNextId();
    auto helper = createAccessFunction(context, access.valueType,
                                       {{state.descriptorType, descriptor}, {coordinatesType, coordinates}});
    InstructionBuilder builder(&context, helper.entry.get());
    const auto zero = nullConstant(context, state.uint64Type);
    // Stop accumulating after the first invalid coordinate, preserving the valid prefix.
    const auto emitOffset = [&](uint32_t offset, uint32_t inBounds, uint32_t contribution, uint32_t resultId) {
        const auto guarded = builder.AddSelect(state.uint64Type, inBounds, contribution, zero)->result_id();
        return builder.AddNaryOp(state.uint64Type, spv::Op::OpIAdd, {offset, guarded}, resultId)->result_id();
    };
    // Convert byte offsets to signed element offsets before branching, then use a
    // uint32_t index into the typed buffer. Keep this sequence intact: simplifying
    // it to base + byteOffset let Mesa 25.2.8 discard the guarded offset and flatten
    // an out-of-bounds read into an unconditional load.
    auto *const signedType = context.get_type_mgr()->GetIntType(64, true);
    const auto signedTypeId = context.get_type_mgr()->GetTypeInstruction(signedType);
    const auto *storageType = context.get_def_use_mgr()->GetDef(getStorageType(state, tensor));
    const auto elementSize = storageType->GetSingleWordInOperand(0) / 8;
    const auto *sizeConstant = context.get_constant_mgr()->GetConstant(signedType, {elementSize, 0});
    const auto size = context.get_constant_mgr()->GetDefiningInstruction(sizeConstant)->result_id();
    const auto bounds = emitBoundsAndByteOffset(state, helper, builder, tensor, descriptor, coordinates,
                                                coordinatesType, access.elementCount, emitOffset);
    auto block = bounds.block;
    std::vector<uint32_t> elements;
    for (uint32_t element = 0; element < access.elementCount; ++element) {
        const auto elementBounds = emitElementBounds(state, builder, bounds, element, block);
        auto load = createBlock(context);
        auto merge = createBlock(context);
        const auto signedOffset =
            builder.AddUnaryOp(signedTypeId, spv::Op::OpBitcast, elementBounds.byteOffset)->result_id();
        const auto elementOffset = builder.AddBinaryOp(signedTypeId, spv::Op::OpSDiv, signedOffset, size)->result_id();
        builder.AddConditionalBranch(elementBounds.inBounds, load->id(), merge->id(), merge->id());

        builder.SetInsertPoint(load.get());
        const auto pointer = emitReadElementPointer(state, builder, tensor, descriptor, elementOffset);
        auto value = builder.AddLoad(pointer.storageType, pointer.id, pointer.alignment)->result_id();
        if (context.get_type_mgr()->GetType(tensor.elementType)->AsBool()) {
            // Widen before comparing so byte storage does not require Int8 arithmetic.
            // Every nonzero byte represents true.
            const auto integerType = context.get_type_mgr()->GetSIntTypeId();
            value = builder.AddUnaryOp(integerType, spv::Op::OpSConvert, value)->result_id();
            value =
                builder
                    .AddBinaryOp(tensor.elementType, spv::Op::OpINotEqual, value, nullConstant(context, integerType))
                    ->result_id();
        }
        builder.AddBranch(merge->id());

        builder.SetInsertPoint(merge.get());
        elements.push_back(builder
                               .AddPhi(tensor.elementType, {nullConstant(context, tensor.elementType),
                                                            elementBounds.block, value, load->id()})
                               ->result_id());
        block = merge->id();
        helper.blocks.push_back(std::move(load));
        helper.blocks.push_back(std::move(merge));
    }
    // Even an array of length one keeps its composite type at the call site.
    const auto result = access.valueType == tensor.elementType
                            ? elements.front()
                            : builder.AddCompositeConstruct(access.valueType, elements)->result_id();
    builder.AddUnaryOp(0, spv::Op::OpReturnValue, result);
    const auto functionId = finishAccessFunction(context, std::move(helper));
    state.helpers.emplace(key, functionId);
    return functionId;
}

/* Equivalent shader logic:

    Bounds bounds;
    // Code emitted by emitBoundsAndByteOffset sets bounds here,
    // accumulating the full offset for writes.
    for (uint element = 0; element < LENGTH; ++element) {
        Bounds elementBounds;
        // Code emitted by emitElementBounds sets elementBounds for this lane.
        if (elementBounds.inBounds) {
            store(descriptor.address + elementBounds.byteOffset, values[element]);
            // For bool tensors, store int8_t(values[element] ? 1 : 0) instead.
        }
    }

*/
// C++ emits each lane separately, without a shader loop. Scalar writes use the
// scalar argument directly. store is a typed memory instruction; bool selection
// uses 32-bit integers before narrowing.
uint32_t getOrCreateWriteHelper(AccessState &state, const TensorType &tensor, uint32_t coordinatesType,
                                const AccessValueInfo &access) {
    const AccessFunctionKey key{spv::Op::OpTensorWriteARM, access.valueType, coordinatesType, tensor.rankId};
    if (const auto it = state.helpers.find(key); it != state.helpers.end()) {
        return it->second;
    }
    auto &context = state.context;
    const auto descriptor = context.TakeNextId();
    const auto coordinates = context.TakeNextId();
    const auto elements = context.TakeNextId();
    auto helper = createAccessFunction(
        context, context.get_type_mgr()->GetVoidTypeId(),
        {{state.descriptorType, descriptor}, {coordinatesType, coordinates}, {access.valueType, elements}});
    InstructionBuilder builder(&context, helper.entry.get());
    // Accumulate the full offset; the final bounds check controls whether the store executes.
    const auto emitOffset = [&](uint32_t offset, uint32_t, uint32_t contribution, uint32_t resultId) {
        return builder.AddNaryOp(state.uint64Type, spv::Op::OpIAdd, {offset, contribution}, resultId)->result_id();
    };
    const auto bounds = emitBoundsAndByteOffset(state, helper, builder, tensor, descriptor, coordinates,
                                                coordinatesType, access.elementCount, emitOffset);
    auto block = bounds.block;
    for (uint32_t element = 0; element < access.elementCount; ++element) {
        const auto elementBounds = emitElementBounds(state, builder, bounds, element, block);
        auto store = createBlock(context);
        auto merge = createBlock(context);
        builder.AddConditionalBranch(elementBounds.inBounds, store->id(), merge->id(), merge->id());

        builder.SetInsertPoint(store.get());
        const auto pointer = emitElementPointer(state, builder, tensor, descriptor, elementBounds.byteOffset);
        auto storedValue = access.valueType == tensor.elementType
                               ? elements
                               : builder.AddCompositeExtract(tensor.elementType, elements, {element})->result_id();
        if (context.get_type_mgr()->GetType(tensor.elementType)->AsBool()) {
            const auto integerType = context.get_type_mgr()->GetSIntTypeId();
            auto *constants = context.get_constant_mgr();
            const auto one =
                constants
                    ->GetDefiningInstruction(constants->GetConstant(context.get_type_mgr()->GetType(integerType), {1}))
                    ->result_id();
            // Select canonical 0/1 in 32 bits, then narrow only for the byte store.
            storedValue =
                builder.AddSelect(integerType, storedValue, one, nullConstant(context, integerType))->result_id();
            storedValue = builder.AddUnaryOp(pointer.storageType, spv::Op::OpSConvert, storedValue)->result_id();
        }
        builder.AddInstruction(
            makeInstruction(context, spv::Op::OpStore, 0, 0,
                            {idOperand(pointer.id),
                             idOperand(storedValue),
                             {SPV_OPERAND_TYPE_MEMORY_ACCESS, {static_cast<uint32_t>(spv::MemoryAccessMask::Aligned)}},
                             {SPV_OPERAND_TYPE_LITERAL_INTEGER, {pointer.alignment}}}));
        builder.AddBranch(merge->id());

        builder.SetInsertPoint(merge.get());
        block = merge->id();
        helper.blocks.push_back(std::move(store));
        helper.blocks.push_back(std::move(merge));
    }
    builder.AddNullaryOp(0, spv::Op::OpReturn);
    const auto functionId = finishAccessFunction(context, std::move(helper));
    state.helpers.emplace(key, functionId);
    return functionId;
}

/* Equivalent shader logic:

    uint size = uint(descriptor.dimensions[dimension]);

*/
// dimension is a literal index; the size is runtime descriptor metadata.
uint32_t emitSizeQuery(AccessState &state, InstructionBuilder &builder, uint32_t descriptor, uint32_t dimension) {
    const auto size = builder.AddCompositeExtract(state.uint64Type, descriptor, {1, dimension})->result_id();
    return builder.AddUnaryOp(state.context.get_type_mgr()->GetUIntTypeId(), spv::Op::OpUConvert, size)->result_id();
}

} // namespace mlsdk::el::layer::tensor_spirv
