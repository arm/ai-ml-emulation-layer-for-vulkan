/*
 * SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include "spirv_glsl_tensor_buffer.hpp"
#include <exception>
#include <set>

using namespace spirv_cross;

namespace mlsdk::el::layer {

const std::string CompilerTensorAsBuffer::tensorDefines =
#include "shaders/tensor.glsl"
    ;

CompilerTensorAsBuffer::CompilerTensorAsBuffer(std::vector<uint32_t> spirv_) : CompilerGLSL(std::move(spirv_)) {
    /* The CompilerGLSL constructor parses SPIRV to the SPIRV-Cross internal representation
     This constructor finds any tensors in the parsed data and replaces them with structs pointing to uniform buffers
     The end result is that a TensorARM variable (where TYPE is e.g. uint16_t and 0<RANK<=6)

     layout (set = SET, binding = BIND) uniform TensorARM<TYPE, RANK> VAR;
     layout (set = SET, binding = BIND) uniform TensorARM<TYPE, RANK> ARRAY[X];

     is replaced by a combination of types

     layout(buffer_reference, std430) buffer _Tensor_Buffer_TYPE
     {
        TYPE data[];
     };
     struct _Tensor_Descriptor_TYPE
     {
        _Tensor_Buffer_TYPE address;
        uint64_t rank;
        uint64_t shape[6];
        uint64_t stride[6];
     };
     layout(set = SET, binding = BIND) uniform _Tensor_Interface_VAR
     {
        _Tensor_Descriptor_TYPE descriptor;
     } _tensor_interface_VAR;
     layout(set = SET, binding = BIND) uniform _Tensor_Interface_ARRAY
     {
        _Tensor_Descriptor_TYPE descriptor;
     } _tensor_interface_ARRAY[X];

     Finally, before entering the `main` function, a global variable of name `VAR` and type `_Tensor_Descriptor_TYPE` is
     declared, and can be used wherever the original TensorARM variable was used.
     The variable is initialized at the start of the `main` function to point to the corresponding `descriptor` member.
     Note, this is for non-array tensors only. Arrays of tensors are handled by replacing the tensor access with an
     indexed access to the descriptor.

     _Tensor_Descriptor_TYPE VAR;

     void main(){
        VAR = _tensor_interface_VAR.descriptor;

        // Example of how code of tensorWriteARM is updated
        tensorWriteARM(_tensor_interface_ARRAY[X].descriptor, ...)

        ...
     }
     */

    // Set required options
    options.version = 460;
    options.es = false;
    options.vulkan_semantics = true;

    // Addressing model must be able to handle physical storage buffers
    ir.addressing_model = spv::AddressingModelPhysicalStorageBuffer64EXT;

    // The macros we define require support at least for 64-bit int types.
    // Other basic types are identified and extensions added by SPIRV-Cross if needed.
    require_extension_internal("GL_ARB_gpu_shader_int64");

    // Can't add new IR entries while iterating through existing entries, so we iterate and collect tensor
    // variables/types before converting them.
    std::set<uint32_t> tensorTypeIds;
    std::set<uint32_t> tensorPtrIds;
    std::vector<uint32_t> tensorVarIds;
    ir.for_each_typed_id<SPIRVariable>([&](uint32_t self, SPIRVariable &var) {
        const auto &tensorPtr = get_type_from_variable(self);
        if (tensorPtr.basetype != SPIRType::Tensor) {
            return;
        }
        tensorVarIds.push_back(self);
        tensorTypeIds.insert(tensorPtr.self);
        tensorPtrIds.insert(var.basetype);
    });

    // Keep track of which types of tensors we have made descriptors for.
    // The incoming SPIRV has different types for each tensor dimension, but we only care about the element type.
    std::map<uint32_t, uint32_t> typeMap{};
    for (auto tensorTypeId : tensorTypeIds) {
        // Get tensor element type id
        uint32_t elementTypeId = get<SPIRType>(tensorTypeId).tensor.type;

        if (auto tensorStructDoneId = typeMap[elementTypeId]; tensorStructDoneId != 0) {
            auto &tensorStructDone = get<SPIRType>(tensorStructDoneId);
            auto &tensorStructNew = get<SPIRType>(tensorTypeId);
            // Instead of constructing a new buffer type and making the struct from scratch,
            // simply copy the already existing tensor descriptor.
            tensorStructNew = tensorStructDone;
            // Setting `type_alias` tells SPIRV-Cross to only output one instance of this type.
            tensorStructNew.type_alias = tensorStructDoneId;
            // Struct member byte offsets need to be copied separately.
            ir.meta[tensorTypeId].members = ir.meta[typeMap[elementTypeId]].members;
            continue;
        }

        // Store mapping from the element type to this tensor struct ID.
        typeMap[elementTypeId] = tensorTypeId;

        // Get the GLSL string of the tensor element type
        const std::string elementTypeGlsl = type_to_glsl(get<SPIRType>(elementTypeId));
        // Create storage buffer struct
        uint32_t bufferPtrId = createTensorBuffer(tensorTypeId);
        // Create tensor struct, containing a pointer to the storage buffer
        createTensorStruct(tensorTypeId, bufferPtrId);

        // Set legible names for easier debugging
        // Note that TensorARM<bool> will produce a buffer called "_Tensor_Buffer_bool", but containing an "int8_t" data
        // member. If these names are already used for some reason, SPIRV-Cross will produce new names
        ir.set_name(get_pointee_type_id(bufferPtrId), join("_Tensor_Buffer_", elementTypeGlsl));
        ir.set_name(tensorTypeId, join("_Tensor_Descriptor_", elementTypeGlsl));
    }

    for (auto tensorPtrId : tensorPtrIds) {
        // TensorARM variables have types of "Pointer", referring down to the base Tensor type.
        // Once the base Tensor type has been updated, these pointers need to inherit some information from the updated
        // base. This also handles arrays of tensors
        createTensorPtr(tensorPtrId);
    }

    for (uint32_t tensorVarId : tensorVarIds) {
        // Layout decorations of a Tensor variable are moved to a uniform block containing a single member
        if (interface_variable_exists_in_entry_point(tensorVarId)) {
            createTensorInterface(tensorVarId);
        }
    }
}

void CompilerTensorAsBuffer::emit_header() {
    // Append definition of tensorSizeARM, tensorReadARM and tensorWriteARM macros
    // after the parent defined header
    CompilerGLSL::emit_header();
    statement(tensorDefines);
}

void CompilerTensorAsBuffer::emit_entry_point_declarations() {
    // This function is empty in the parent class, but is called right at the start of emitting the glsl "main"
    // function. Therefore it is incredibly convenient for handling variables without relying on other parent class
    // mechanisms. Each tensorARM variable in the incoming SPIRV has been converted and declared globally. Here we
    // manually initialize it to point to the uniform interface we've set up. SPIRV-Cross sometimes restarts compilation
    // and this function is called every time. Type names might change between calls due to identifier sanitizing, so
    // we construct the initializer from scratch every time.
    for (const auto &[tensorVarId, tensorInterfaceVarId] : tensorVariables) {
        const auto &tensorInterfaceVar = get<SPIRVariable>(tensorInterfaceVarId);
        const auto &tensorInterfacePtr = get<SPIRType>(tensorInterfaceVar.basetype);
        statement(to_name(tensorVarId), " = ", to_name(tensorInterfaceVarId), ".",
                  get_member_name(tensorInterfacePtr.self, 0), ";");
    }
    CompilerGLSL::emit_entry_point_declarations();
}

void CompilerTensorAsBuffer::emit_instruction(const Instruction &instruction) {
    // Whenever emit_instruction is called, this overrides the parent definition.
    // If the instruction concerns a tensor operator, custom output is produced.
    // Otherwise, the parent function is called.
    auto ops = stream(instruction);
    auto opcode = static_cast<spv::Op>(instruction.op);
    uint32_t length = instruction.length;

    opcode = get_remapped_spirv_op(opcode);
    switch (opcode) {
    case spv::OpTensorQuerySizeARM: {
        flush_variable_declaration(ops[1]);
        emit_binary_func_op(ops[0], ops[1], ops[2], ops[3], "tensorSizeARM");
        break;
    }
    case spv::OpTensorReadARM: {
        flush_variable_declaration(ops[1]);
        emit_uninitialized_temporary_expression(ops[0], ops[1]);

        const auto &outType = expression_type(ops[1]);

        std::string operands = "0";
        std::string operandArg = "0";
        if (length > 4) {
            operands = convert_to_string(ops[4]);
        }
        if (length > 5) {
            operandArg = to_expression(ops[5]);
        }
        if (length > 6) {
            throw std::runtime_error("The emulation layer doesn't support more than 5 arguments to tensorReadARM.");
        }

        const std::string macroName =
            is_array(outType) ? "_emu_GL_ARM_tensors_read_array" : "_emu_GL_ARM_tensors_read_scalar";
        statement(macroName, "(",                            // Macro name
                  to_expression(ops[2]), ", ",               // tensor
                  to_expression(ops[3]), ", ",               // coordinates
                  to_expression(ops[1]), ", ",               // out value
                  operands, ", ",                            // tensor operands
                  operandArg, ", ",                          // out of bounds value
                  type_to_glsl(get<SPIRType>(outType.self)), // glsl type of output var
                  ");");
        break;
    }
    case spv::OpTensorWriteARM: {
        flush_variable_declaration(ops[0]);

        const auto &outType = expression_type(ops[2]);
        auto elementType = get<SPIRType>(outType.self);

        // tensorWrite needs the tensor buffer type to properly cast input when writing
        if (elementType.basetype == SPIRType::Boolean) {
            elementType.basetype = SPIRType::SByte;
            elementType.width = 8;
        }

        std::string operands = "0";
        if (length > 3) {
            operands = convert_to_string(ops[3]);
        }
        if (length > 4) {
            throw std::runtime_error("The emulation layer doesn't support more than 4 arguments to tensorReadARM.");
        }

        const std::string macroName =
            is_array(outType) ? "_emu_GL_ARM_tensors_write_array" : "_emu_GL_ARM_tensors_write_scalar";
        statement(macroName, "(",              // Macro name
                  to_expression(ops[0]), ", ", // tensor
                  to_expression(ops[1]), ", ", // coordinates
                  to_expression(ops[2]), ", ", // in value
                  operands, ", ",              // tensor operands, currently not used
                  type_to_glsl(elementType),   // glsl type of tensor buffer
                  ");");

        break;
    }
    case spv::OpAccessChain: {
        // Tensor arrays are accessed here, when called with []
        const auto resultId = ops[0];     // ID to be used downstream
        const auto resultTypeId = ops[1]; // The pointer type
        const auto varId = ops[2];        // base variable, tensor
        const auto indexId = ops[3];      // index expression
        std::string varName = get_name(varId);

        if (std::any_of(tensorArrayVariables.begin(), tensorArrayVariables.end(),
                        [varId](const auto tensorVarId) { return varId == tensorVarId; })) {
            // Convert index ID into a GLSL expression
            // Inject a new GLSL expression:
            std::string newExpr = varName + '[' + to_expression(indexId) + "].descriptor";

            set<SPIRExpression>(resultTypeId, SPIRExpression(newExpr, resultId, true));
            break; // Done, don't emit normal code for this Op
        } else {
            CompilerGLSL::emit_instruction(instruction);
        }
        break;
    }
    default:
        CompilerGLSL::emit_instruction(instruction);
    }
}

std::tuple<uint32_t, uint32_t> CompilerTensorAsBuffer::getTensorDimTypeIds() {
    auto [arrayTypeId, elementTypeId] = tensorDimTypeIds;
    if (arrayTypeId != 0 && elementTypeId != 0) {
        return tensorDimTypeIds;
    }

    elementTypeId = ir.increase_bound_by(2);
    arrayTypeId = elementTypeId + 1;

    // Create a basic uint64_t SPIRType
    auto &elementType = set<SPIRType>(elementTypeId, spv::OpTypeInt);
    elementType.basetype = SPIRType::Int64;
    elementType.width = 64;

    // Create a uint64_t array SPIRType
    // Since `elementType` is a basic uint64_t type, it is used as a base to create the uint64_t array
    auto &arrayType = set<SPIRType>(arrayTypeId, elementType);
    arrayType.op = spv::OpTypeArray;
    arrayType.parent_type = elementTypeId;
    arrayType.self = elementType.self;
    arrayType.array.push_back(MAX_RANK);
    arrayType.array_size_literal.push_back(true);

    // Set array stride decoration
    ir.set_decoration(arrayTypeId, spv::DecorationArrayStride,
                      type_to_packed_array_stride(arrayType, Bitset{}, BufferPackingStd140));

    // return and store the IDs
    tensorDimTypeIds = {arrayTypeId, elementTypeId};
    return tensorDimTypeIds;
}

uint32_t CompilerTensorAsBuffer::createTensorBuffer(uint32_t tensorTypeId) {
    const auto &tensorType = get<SPIRType>(tensorTypeId);

    // Add new IDs to represent buffer types
    uint32_t bufferPtrId = ir.increase_bound_by(3);
    uint32_t bufferStructId = bufferPtrId + 1;
    uint32_t bufferDataId = bufferPtrId + 2;

    // Create buffer array type
    // Same as spirv_parse.cpp:parse `case OpTypeRuntimeArray`
    uint32_t elementTypeId = tensorType.tensor.type;
    auto *elementType = &get<SPIRType>(elementTypeId);
    if (elementType->basetype == SPIRType::Boolean) {
        // Bool tensors are strictly defined to use 8-bit bools, but the same is not true for buffers.
        // When converting to buffers we must replace bool elements with int8_t to ensure predictable bit widths.
        elementTypeId = ir.increase_bound_by(1);
        elementType = &set<SPIRType>(elementTypeId, spv::OpTypeInt);
        elementType->basetype = SPIRType::SByte;
        elementType->width = 8;
    }
    auto &bufferData = set<SPIRType>(bufferDataId, *elementType);
    bufferData.op = spv::OpTypeRuntimeArray;
    bufferData.array.push_back(0);
    bufferData.array_size_literal.push_back(true);
    bufferData.parent_type = elementTypeId;
    bufferData.self = elementType->self;
    ir.set_decoration(bufferDataId, spv::DecorationArrayStride,
                      type_to_packed_array_stride(bufferData, Bitset{}, BufferPackingStd430));

    // Create buffer struct
    // Same as spirv_parse.cpp:parse `case OpTypeStruct` (with type aliasing skipped)
    auto &bufferStruct = set<SPIRType>(bufferStructId, spv::OpTypeStruct);
    bufferStruct.basetype = SPIRType::Struct;
    bufferStruct.member_types.push_back(bufferDataId);

    ir.set_decoration(bufferStructId, spv::DecorationBlock);
    ir.set_member_name(bufferStructId, 0, "data");
    ir.set_member_decoration(bufferStructId, 0, spv::DecorationOffset, 0);

    // Create pointer to the buffer struct
    // Same as spirv_parse.cpp:parse `OpTypePointer` with base set to bufferData
    auto &bufferPtr = set<SPIRType>(bufferPtrId, spv::OpTypePointer);
    bufferPtr = bufferStruct;
    bufferPtr.op = spv::OpTypePointer;
    bufferPtr.pointer = true;
    bufferPtr.pointer_depth++;
    bufferPtr.storage = spv::StorageClassPhysicalStorageBuffer;
    bufferPtr.parent_type = bufferStructId;

    return bufferPtrId;
}

void CompilerTensorAsBuffer::createTensorStruct(uint32_t tensorTypeId, uint32_t bufferPtrId) {
    // Create uniform struct to replace the tensorARM
    auto [memberArrayTypeId, memberElementTypeId] = getTensorDimTypeIds();

    // Create a SPIRType for the struct
    auto &tensorStruct = set<SPIRType>(tensorTypeId, spv::OpTypeStruct);
    tensorStruct.basetype = SPIRType::Struct;
    tensorStruct.member_types.push_back(bufferPtrId);
    tensorStruct.member_types.push_back(memberElementTypeId);
    tensorStruct.member_types.push_back(memberArrayTypeId);
    tensorStruct.member_types.push_back(memberArrayTypeId);

    // Struct members need to be decorated with byte offsets.
    // Sum sizes of preceding members to get the offset.
    uint32_t offset = 0;
    for (uint32_t i = 0; i < 4; i++) {
        ir.set_member_decoration(tensorStruct.self, i, spv::DecorationOffset, offset);
        const auto &memberType = get<SPIRType>(tensorStruct.member_types[i]);
        offset += type_to_packed_size(memberType, Bitset{}, BufferPackingStd140);
    }

    ir.set_member_name(tensorStruct.self, 0, "address");
    ir.set_member_name(tensorStruct.self, 1, "rank");
    ir.set_member_name(tensorStruct.self, 2, "shape");
    ir.set_member_name(tensorStruct.self, 3, "stride");
}

void CompilerTensorAsBuffer::createTensorPtr(uint32_t tensorPtrId) {
    // The assumed type structure is
    // "Pointer" -> "Array_0" -> ... -> "Array_N" -> "Tensor Struct"
    // Pointers and arrays need to inherit some data from the updated struct

    // .self refers to the underlying Tensor type, stripping away any pointers/arrays
    auto &tensorPtr = get<SPIRType>(tensorPtrId);
    tensorPtr.storage = spv::StorageClassGeneric;
    const auto &tensorStruct = get<SPIRType>(tensorPtr.self);

    // Go through the hierarchy of types and apply changes that were made to the base Tensor type
    auto *fixType = &tensorPtr;
    while (fixType->parent_type) {
        fixType->member_types = tensorStruct.member_types;
        fixType->basetype = tensorStruct.basetype;
        fixType = &get<SPIRType>(fixType->parent_type);
    }
}

void CompilerTensorAsBuffer::createTensorInterface(uint32_t tensorVarId) {
    // Create a new uniform block to act as an interface to the tensor data.
    // The original tensorARM variable instead becomes a local variable.

    // Retrieve data about the original variable
    const auto &varName = ir.get_name(tensorVarId);
    auto &tensorVar = get<SPIRVariable>(tensorVarId);
    const auto &tensorType = get_type_from_variable(tensorVarId);

    // Add IDs for the new types and variable
    uint32_t tensorInterfaceId = ir.increase_bound_by(3);
    uint32_t tensorInterfacePtrId = tensorInterfaceId + 1;
    uint32_t tensorInterfaceVarId = tensorInterfacePtrId + 1;

    // Create a SPIRType for the struct
    auto &tensorInterface = set<SPIRType>(tensorInterfaceId, spv::OpTypeStruct);
    tensorInterface.basetype = SPIRType::Struct;
    tensorInterface.member_types.push_back(tensorType.self); // base type, no array

    ir.set_decoration(tensorInterfaceId, spv::DecorationBlock);
    ir.set_member_decoration(tensorInterfaceId, 0, spv::DecorationOffset, 0);
    ir.set_name(tensorInterfaceId, join("_Tensor_Interface_", varName));
    ir.set_member_name(tensorInterfaceId, 0, "descriptor");

    // Handle array case
    bool isTensorArray = false;
    if (!tensorType.array_size_literal.empty() && tensorType.array_size_literal.front()) {
        isTensorArray = true;
        tensorInterface.array.push_back(tensorType.array.front());
        tensorInterface.array_size_literal.push_back(true);
    }

    // Create a SPIRType for a pointer-to-struct
    auto &tensorInterfacePtr = set<SPIRType>(tensorInterfacePtrId, spv::OpTypePointer);
    tensorInterfacePtr = tensorInterface;
    tensorInterfacePtr.op = spv::OpTypePointer;
    tensorInterfacePtr.pointer = true;
    tensorInterfacePtr.pointer_depth++;
    tensorInterfacePtr.storage = spv::StorageClassUniform;
    tensorInterfacePtr.parent_type = tensorInterfaceId;

    // Create a new variable to represent the tensor interface
    set<SPIRVariable>(tensorInterfaceVarId, tensorInterfacePtrId, spv::StorageClassUniform);
    ir.set_name(tensorInterfaceVarId, join("_tensor_interface_", varName));
    auto &execution = get_entry_point();
    execution.interface_variables.push_back(tensorInterfaceVarId);

    // Retrieve binding and descriptor set from original variable
    uint32_t binding = ir.get_decoration(tensorVarId, spv::DecorationBinding);
    uint32_t descriptorSet = ir.get_decoration(tensorVarId, spv::DecorationDescriptorSet);
    ir.set_decoration(tensorInterfaceVarId, spv::DecorationBlock);
    ir.set_decoration(tensorInterfaceVarId, spv::DecorationBinding, binding);
    ir.set_decoration(tensorInterfaceVarId, spv::DecorationDescriptorSet, descriptorSet);

    if (isTensorArray) {
        // Change name of original variable
        ir.set_name(tensorVarId, ir.get_name(tensorInterfaceVarId));
        tensorArrayVariables.push_back(tensorVarId);
    } else {
        // Change original variable to be a "regular" global variable.
        tensorVar.storage = spv::StorageClassGeneric;
        ir.unset_decoration(tensorVarId, spv::DecorationBinding);
        ir.unset_decoration(tensorVarId, spv::DecorationDescriptorSet);
        global_variables.push_back(tensorVarId);

        // SPIRV-Cross will now declare the original variable globally, without initialization.
        // We will manually initialize it inside the GLSL "main" function by overriding
        // `CompilerGLSL::emit_entry_point_declarations`.
        tensorVariables.push_back({tensorVarId, tensorInterfaceVarId});
    }
}

} // namespace mlsdk::el::layer
