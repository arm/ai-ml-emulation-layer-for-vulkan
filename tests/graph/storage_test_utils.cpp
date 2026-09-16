/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include "storage_test_utils.hpp"
#include "mlel/pipeline.hpp"
#include "mlel/tensor.hpp"
#include "mlel/utils.hpp"
#include "vulkan_test_utils.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace mlsdk::el::tests {

using namespace utilities;

// These graphs keep TOSA's signless SPIR-V types fixed while varying the
// externally bound Vulkan storage formats independently.
std::vector<uint32_t> makeStorageConversionGraph(const std::string &inputType, const std::string &outputType,
                                                 uint32_t inputRank, uint32_t outputRank, bool binary,
                                                 const std::string &operation, const std::string &constants) {
    std::string source = R"(
OpCapability VulkanMemoryModel
OpCapability Shader
OpCapability GraphARM
OpCapability TensorsARM
OpCapability Int8
OpCapability Int16
OpCapability Int64
OpCapability Float16
OpCapability Float8EXT
OpCapability BFloat16TypeKHR
OpExtension "SPV_ARM_graph"
OpExtension "SPV_ARM_tensors"
OpExtension "SPV_EXT_float8"
OpExtension "SPV_KHR_bfloat16"
OpExtension "SPV_KHR_vulkan_memory_model"
%tosa = OpExtInstImport "TOSA.001000.1"
OpMemoryModel Logical Vulkan
OpDecorate %input_var DescriptorSet 0
OpDecorate %input_var Binding 0
OpDecorate %output_var DescriptorSet 0
)";
    source += "OpDecorate %output_var Binding " + std::to_string(binary ? 2 : 1) + "\n";
    if (binary) {
        source += "OpDecorate %second_var DescriptorSet 0\nOpDecorate %second_var Binding 1\n";
    }
    source += R"(
%i8 = OpTypeInt 8 0
%i16 = OpTypeInt 16 0
%i32 = OpTypeInt 32 0
%i64 = OpTypeInt 64 0
%f32 = OpTypeFloat 32
%f16 = OpTypeFloat 16
%bf16 = OpTypeFloat 16 BFloat16KHR
%e4 = OpTypeFloat 8 Float8E4M3EXT
%e5 = OpTypeFloat 8 Float8E5M2EXT
%bool = OpTypeBool
%false = OpConstantFalse %bool
%true = OpConstantTrue %bool
)";
    for (uint32_t i = 0; i <= 5; ++i) {
        source += "%u" + std::to_string(i) + " = OpConstant %i32 " + std::to_string(i) + "\n";
    }
    source += "%in_tensor = OpTypeTensorARM " + inputType + " %u" + std::to_string(inputRank) + "\n";
    source += "%out_tensor = OpTypeTensorARM " + outputType + " %u" + std::to_string(outputRank) + "\n";
    source += R"(
%shape_array = OpTypeArray %i32 %u1
%shape1 = OpConstantComposite %shape_array %u1
%shape2 = OpConstantComposite %shape_array %u2
%shape4 = OpConstantComposite %shape_array %u4
%const_i8 = OpTypeTensorARM %i8 %u1 %shape1
%const_i16 = OpTypeTensorARM %i16 %u1 %shape1
%const_i32 = OpTypeTensorARM %i32 %u1 %shape1
%const_pair = OpTypeTensorARM %i32 %u1 %shape2
%const_quad = OpTypeTensorARM %i32 %u1 %shape4
%zero8 = OpConstant %i8 0
%zero16 = OpConstant %i16 0
%zp8 = OpConstantComposite %const_i8 %zero8
%zp16 = OpConstantComposite %const_i16 %zero16
%zp32 = OpConstantComposite %const_i32 %u0
%kernel = OpConstantComposite %const_pair %u1 %u2
%stride = OpConstantComposite %const_pair %u1 %u1
%pad = OpConstantComposite %const_quad %u0 %u0 %u0 %u0
)";
    source += constants;
    source += R"(
%in_ptr = OpTypePointer UniformConstant %in_tensor
%out_ptr = OpTypePointer UniformConstant %out_tensor
%input_var = OpVariable %in_ptr UniformConstant
%output_var = OpVariable %out_ptr UniformConstant
)";
    if (binary) {
        source += "%second_var = OpVariable %in_ptr UniformConstant\n";
    }
    source += "%graph_type = OpTypeGraphARM " + std::string(binary ? "2 %in_tensor %in_tensor" : "1 %in_tensor") +
              " %out_tensor\n";
    source += "OpGraphEntryPointARM %graph \"storage_conversion\" %input_var " +
              std::string(binary ? "%second_var " : "") + "%output_var\n";
    source += "%graph = OpGraphARM %graph_type\n%a = OpGraphInputARM %in_tensor %u0\n";
    if (binary) {
        source += "%b = OpGraphInputARM %in_tensor %u1\n";
    }
    source +=
        "%result = OpExtInst %out_tensor %tosa " + operation + "\nOpGraphSetOutputARM %result %u0\nOpGraphEndARM\n";
    if (inputType == outputType && inputRank == outputRank) {
        mlsdk::el::utils::replaceAll(
            source, "%out_tensor = OpTypeTensorARM " + outputType + " %u" + std::to_string(outputRank) + "\n", "");
        mlsdk::el::utils::replaceAll(source, "%out_ptr = OpTypePointer UniformConstant %out_tensor\n", "");
        mlsdk::el::utils::replaceAll(source, "%out_tensor", "%in_tensor");
        mlsdk::el::utils::replaceAll(source, "%out_ptr", "%in_ptr");
    }
    return assembleSpirv(source);
}

} // namespace mlsdk::el::tests
