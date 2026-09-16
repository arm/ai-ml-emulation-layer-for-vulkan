/*
 * SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "mlel/utils.hpp"
#include "mlel/log.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>

#include <glslang/Include/glslang_c_interface.h>
#include <glslang/Public/resource_limits_c.h>

using namespace mlsdk::el::log;

namespace mlsdk::el::utils {

namespace {
Log layerLog("VMEL_COMMON_SEVERITY", "Layer");
} // namespace

bool hasExtension(const std::vector<vk::ExtensionProperties> &extensionProperties, std::string_view extension) {
    return std::any_of(extensionProperties.begin(), extensionProperties.end(), [&](const auto &property) {
        return std::string_view{property.extensionName.data()} == extension;
    });
}

size_t getElementCount(const std::vector<int64_t> &dimensions) {
    auto result = std::accumulate(dimensions.begin(), dimensions.end(), int64_t(1), std::multiplies<>());
    if (result < 0) {
        throw std::runtime_error("Tensor element count became negative: " + std::to_string(result));
    }
    return static_cast<size_t>(result);
}

std::vector<uint32_t> glslToSpirv(const std::string &glsl) {
    class Finally {
      public:
        explicit Finally(const std::function<void()> &_func) : func{_func} {}
        ~Finally() { func(); }
        Finally(const Finally &) = delete;
        Finally &operator=(const Finally &) = delete;
        Finally(Finally &&) = delete;
        Finally &operator=(Finally &&) = delete;

      private:
        std::function<void()> func;
    };

    glslang_initialize_process();
    Finally f1([]() { glslang_finalize_process(); });

    const glslang_input_t input = {
        GLSLANG_SOURCE_GLSL,        // language
        GLSLANG_STAGE_COMPUTE,      // stage
        GLSLANG_CLIENT_VULKAN,      // client
        GLSLANG_TARGET_VULKAN_1_3,  // client_version
        GLSLANG_TARGET_SPV,         // target_language
        GLSLANG_TARGET_SPV_1_6,     // target_language_version
        glsl.c_str(),               // code
        460,                        // default_version
        GLSLANG_CORE_PROFILE,       // default_profile
        true,                       // force_default_version_and_profile
        false,                      // forward_compatible
        GLSLANG_MSG_DEFAULT_BIT,    // messages
        glslang_default_resource(), // resource
        {},                         // callbacks
        {},                         // callbacks ctx
    };

    glslang_shader_t *shader = glslang_shader_create(&input);
    Finally f2([&shader]() { glslang_shader_delete(shader); });

#ifdef USE_FLOAT_AS_DOUBLE
    glslang_shader_set_preamble(shader, "#define USE_FLOAT_AS_DOUBLE\n");
#endif

    if (!glslang_shader_preprocess(shader, &input)) {
        layerLog(Severity::Error) << StringLineNumber(glsl);
        throw std::runtime_error(std::string("Failed to preprocess shader: ") + glslang_shader_get_info_log(shader));
    }

    if (!glslang_shader_parse(shader, &input)) {
        layerLog(Severity::Error) << StringLineNumber(glsl);
        throw std::runtime_error(std::string("Failed to parse shader: ") + glslang_shader_get_info_log(shader));
    }

    glslang_program_t *program = glslang_program_create();
    Finally f3([&program]() { glslang_program_delete(program); });

    glslang_program_add_shader(program, shader);

    if (!glslang_program_link(program, GLSLANG_MSG_SPV_RULES_BIT | GLSLANG_MSG_VULKAN_RULES_BIT)) {
        layerLog(Severity::Error) << StringLineNumber(glsl);
        throw std::runtime_error(std::string("Failed to link program: ") + glslang_shader_get_info_log(shader));
    }

    glslang_program_SPIRV_generate(program, input.stage);

    if (glslang_program_SPIRV_get_messages(program)) {
        layerLog(Severity::Error) << StringLineNumber(glsl);
        throw std::runtime_error(std::string("GLSLang returned messages: ") +
                                 glslang_program_SPIRV_get_messages(program));
    }

    std::vector<uint32_t> spirv{glslang_program_SPIRV_get_ptr(program),
                                glslang_program_SPIRV_get_ptr(program) + glslang_program_SPIRV_get_size(program)};

    return spirv;
}

namespace {
// Type tags are local shader constants defined in graph/shaders/graph_op/common.comp.
// They are encoded as two ASCII bytes: kind ('b', 'i', 'u', 'f') followed by byte size or reduced-float subtype.
constexpr ArithmeticTypeInfo int8Type{true, true, "-128", "127", "int8_t", "0x6931", "int8_t"};
constexpr ArithmeticTypeInfo uint8Type{true, false, "0u", "255u", "uint8_t", "0x7531", "uint8_t"};
constexpr ArithmeticTypeInfo boolType{true, false, "0", "1", "bool", "0x6231", "bool"};
constexpr ArithmeticTypeInfo int16Type{true, true, "-32768", "32767", "int16_t", "0x6932", "int16_t"};
constexpr ArithmeticTypeInfo uint16Type{true, false, "0u", "65535u", "uint16_t", "0x7532", "uint16_t"};
constexpr ArithmeticTypeInfo float16Type{false,       true,     "-65504.000000", "65504.000000",
                                         "float16_t", "0x6632", "float16_t"};
constexpr ArithmeticTypeInfo bfloat16Type{
    false, true, "-3.3895313892515355e+38", "3.3895313892515355e+38", "bfloat16_t", "0x6642", "float"};
constexpr ArithmeticTypeInfo float8e5m2Type{false, true, "-57344", "57344", "float8_e5m2_t", "0x664D", "float16_t"};
constexpr ArithmeticTypeInfo float8e4m3Type{false, true, "-448", "448", "float8_e4m3_t", "0x664E", "float16_t"};
constexpr ArithmeticTypeInfo int32Type{true, true, "-2147483648", "2147483647", "int", "0x6934", "int"};
constexpr ArithmeticTypeInfo uint32Type{true, false, "0u", "4294967295u", "uint32_t", "0x7534", "uint32_t"};
constexpr ArithmeticTypeInfo float32Type{false,
                                         true,
                                         "-340282346638528859811704183484516925440.000000",
                                         "340282346638528859811704183484516925440.000000",
                                         "float",
                                         "0x6634",
                                         "float"};
constexpr ArithmeticTypeInfo int64Type{true,     true,     "-9223372036854775808ll", "9223372036854775807ll", "int64_t",
                                       "0x6938", "int64_t"};
constexpr ArithmeticTypeInfo uint64Type{true,       false,    "0ull",    "18446744073709551615ull",
                                        "uint64_t", "0x7538", "uint64_t"};
constexpr ArithmeticTypeInfo doubleType{false,
                                        true,
                                        "-179769313486231570814527423731704356798070567525844996598917476803"
                                        "157260780028538760589558632766878171540458953514382464234321326889"
                                        "464182768467546703537516986049910576551282076245490090389328944075"
                                        "868508455133942304583236903222948165808559332123348274797826204144"
                                        "723168738177180919299881250404026184124858368.000000ll",
                                        "179769313486231570814527423731704356798070567525844996598917476803"
                                        "157260780028538760589558632766878171540458953514382464234321326889"
                                        "464182768467546703537516986049910576551282076245490090389328944075"
                                        "868508455133942304583236903222948165808559332123348274797826204144"
                                        "723168738177180919299881250404026184124858368.000000ll",
                                        "double",
                                        "0x6638",
                                        "double"};

// Vulkan format metadata keeps the format's own signedness and encoding.
constexpr FormatInfo boolFormat{ScalarType::Bool, 8, true, "bool"};
constexpr FormatInfo int8Format{ScalarType::Int8, 8, true, "int8_t"};
constexpr FormatInfo uint8Format{ScalarType::Uint8, 8, true, "uint8_t"};
constexpr FormatInfo int16Format{ScalarType::Int16, 16, true, "int16_t"};
constexpr FormatInfo uint16Format{ScalarType::Uint16, 16, true, "uint16_t"};
constexpr FormatInfo int32Format{ScalarType::Int32, 32, true, "int"};
constexpr FormatInfo uint32Format{ScalarType::Uint32, 32, true, "uint32_t"};
constexpr FormatInfo int64Format{ScalarType::Int64, 64, true, "int64_t"};
constexpr FormatInfo uint64Format{ScalarType::Uint64, 64, true, "uint64_t"};
constexpr FormatInfo float16Format{ScalarType::Float16, 16, false, "float16_t"};
constexpr FormatInfo bfloat16Format{ScalarType::BFloat16, 16, false, "bfloat16_t"};
constexpr FormatInfo float8e4m3Format{ScalarType::Float8E4M3, 8, false, "float8_e4m3_t"};
constexpr FormatInfo float8e5m2Format{ScalarType::Float8E5M2, 8, false, "float8_e5m2_t"};
constexpr FormatInfo float32Format{ScalarType::Float32, 32, false, "float"};
constexpr FormatInfo doubleFormat{ScalarType::Float64, 64, false, "double"};
} // namespace

float decodeReducedFloat(uint32_t rawValue, VkFormat format) {
    int exponentBits;
    int mantissaBits;
    switch (format) {
    case VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E4M3_ARM:
        exponentBits = 4;
        mantissaBits = 3;
        break;
    case VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E5M2_ARM:
        exponentBits = 5;
        mantissaBits = 2;
        break;
    case VK_FORMAT_R16_SFLOAT:
        exponentBits = 5;
        mantissaBits = 10;
        break;
    case VK_FORMAT_R16_SFLOAT_FPENCODING_BFLOAT16_ARM:
        exponentBits = 8;
        mantissaBits = 7;
        break;
    default:
        throw std::runtime_error("Expected an FP8, FP16, or BF16 format");
    }
    const uint32_t exponentMask = (1u << exponentBits) - 1u;
    const uint32_t mantissaMask = (1u << mantissaBits) - 1u;
    const uint32_t exponent = (rawValue >> mantissaBits) & exponentMask;
    const uint32_t mantissa = rawValue & mantissaMask;
    const int bias = (1 << (exponentBits - 1)) - 1;
    const bool e4m3 = format == VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E4M3_ARM;
    float value;
    if (e4m3 && exponent == exponentMask && mantissa == mantissaMask) {
        value = std::numeric_limits<float>::quiet_NaN();
    } else if (!e4m3 && exponent == exponentMask) {
        value = mantissa ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
    } else if (exponent == 0) {
        value = std::ldexp(float(mantissa), 1 - bias - mantissaBits);
    } else {
        value = std::ldexp(1.0f + (float(mantissa) / float(1u << mantissaBits)), int(exponent) - bias);
    }
    return (rawValue & (1u << (exponentBits + mantissaBits))) ? -value : value;
}

const FormatInfo *getFormatInfo(const VkFormat format) {
    switch (format) {
    case VK_FORMAT_R8_SINT:
        return &int8Format;
    case VK_FORMAT_R8_UINT:
    case VK_FORMAT_S8_UINT:
        return &uint8Format;
    case VK_FORMAT_R8_BOOL_ARM:
        return &boolFormat;
    case VK_FORMAT_R16_SINT:
        return &int16Format;
    case VK_FORMAT_R16_UINT:
        return &uint16Format;
    case VK_FORMAT_R16_SFLOAT:
        return &float16Format;
    case VK_FORMAT_R16_SFLOAT_FPENCODING_BFLOAT16_ARM:
        return &bfloat16Format;
    case VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E5M2_ARM:
        return &float8e5m2Format;
    case VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E4M3_ARM:
        return &float8e4m3Format;
    case VK_FORMAT_R32_SINT:
        return &int32Format;
    case VK_FORMAT_R32_UINT:
        return &uint32Format;
    case VK_FORMAT_R32_SFLOAT:
        return &float32Format;
    case VK_FORMAT_R64_SINT:
        return &int64Format;
    case VK_FORMAT_R64_UINT:
        return &uint64Format;
    case VK_FORMAT_R64_SFLOAT:
        return &doubleFormat;
    default:
        throw std::runtime_error("Unsupported tensor buffer format: " + std::to_string(format));
    }
}

const ArithmeticTypeInfo *getArithmeticTypeInfo(ScalarType type) {
    switch (type) {
    case ScalarType::Bool:
        return &boolType;
    case ScalarType::Int8:
        return &int8Type;
    case ScalarType::Uint8:
        return &uint8Type;
    case ScalarType::Int16:
        return &int16Type;
    case ScalarType::Uint16:
        return &uint16Type;
    case ScalarType::Int32:
        return &int32Type;
    case ScalarType::Uint32:
        return &uint32Type;
    case ScalarType::Int64:
        return &int64Type;
    case ScalarType::Uint64:
        return &uint64Type;
    case ScalarType::Float16:
        return &float16Type;
    case ScalarType::BFloat16:
        return &bfloat16Type;
    case ScalarType::Float8E4M3:
        return &float8e4m3Type;
    case ScalarType::Float8E5M2:
        return &float8e5m2Type;
    case ScalarType::Float32:
        return &float32Type;
    case ScalarType::Float64:
        return &doubleType;
    }
    throw std::runtime_error("Unsupported arithmetic type");
}

ScalarType getIntegerArithmeticType(uint32_t bitWidth, IntegerInterpretation interpretation) {
    switch (bitWidth) {
    case 8:
        return interpretation == IntegerInterpretation::Signed ? ScalarType::Int8 : ScalarType::Uint8;
    case 16:
        return interpretation == IntegerInterpretation::Signed ? ScalarType::Int16 : ScalarType::Uint16;
    case 32:
        return interpretation == IntegerInterpretation::Signed ? ScalarType::Int32 : ScalarType::Uint32;
    case 64:
        return interpretation == IntegerInterpretation::Signed ? ScalarType::Int64 : ScalarType::Uint64;
    default:
        throw std::runtime_error("Unsupported integer width: " + std::to_string(bitWidth));
    }
}

std::string_view getTensorInterfaceGlslType(VkFormat format) {
    const auto *storage = getFormatInfo(format);
    if (!storage->isInteger || storage->encoding == ScalarType::Bool) {
        return storage->glslType;
    }
    switch (storage->bitWidth) {
    case 8:
        return "uint8_t";
    case 16:
        return "uint16_t";
    case 32:
        return "uint32_t";
    case 64:
        return "uint64_t";
    default:
        throw std::runtime_error("Unsupported integer tensor width");
    }
}

void setDebugUtilsObjectName(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &loader,
                             VkDevice device, VkObjectType type, uint64_t handle, const std::string &name) {

    if (loader->vkSetDebugUtilsObjectNameEXT) {
        VkDebugUtilsObjectNameInfoEXT nameInfo{};
        nameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
        nameInfo.objectType = type;
        nameInfo.objectHandle = handle;
        nameInfo.pObjectName = name.c_str();

        loader->vkSetDebugUtilsObjectNameEXT(device, &nameInfo);
    }
}

} // namespace mlsdk::el::utils
