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

#include <functional>
#include <numeric>

#include <glslang/Include/glslang_c_interface.h>
#include <glslang/Public/resource_limits_c.h>

using namespace mlsdk::el::log;

namespace mlsdk::el::utils {

namespace {
Log layerLog("VMEL_COMMON_SEVERITY", "Layer");
} // namespace

size_t getElementCount(const std::vector<int64_t> &dimensions) {
    auto result = std::accumulate(dimensions.begin(), dimensions.end(), int64_t(1), std::multiplies<int64_t>());
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
constexpr FormatInfo int8Format{true, true, "-128", "127", "int8_t", "0x6931", "int8_t"};
constexpr FormatInfo uint8Format{true, false, "0u", "255u", "uint8_t", "0x7531", "uint8_t"};
constexpr FormatInfo boolFormat{true, false, "0", "1", "bool", "0x6231", "bool"};
constexpr FormatInfo int16Format{true, true, "-32768", "32767", "int16_t", "0x6932", "int16_t"};
constexpr FormatInfo uint16Format{true, false, "0u", "65535u", "uint16_t", "0x7532", "uint16_t"};
constexpr FormatInfo float16Format{false, true, "-65504.000000", "65504.000000", "float16_t", "0x6632", "float16_t"};
constexpr FormatInfo bfloat16Format{false,    true,   "-3.3895313892515355e+38", "3.3895313892515355e+38", "bfloat16_t",
                                    "0x6642", "float"};
constexpr FormatInfo float8e5m2Format{false, true, "-57344", "57344", "float8_e5m2_t", "0x664D", "float16_t"};
constexpr FormatInfo float8e4m3Format{false, true, "-448", "448", "float8_e4m3_t", "0x664E", "float16_t"};
constexpr FormatInfo int32Format{true, true, "-2147483648", "2147483647", "int", "0x6934", "int"};
constexpr FormatInfo uint32Format{true, false, "0u", "4294967295u", "uint32_t", "0x7534", "uint32_t"};
constexpr FormatInfo float32Format{false,
                                   true,
                                   "-340282346638528859811704183484516925440.000000",
                                   "340282346638528859811704183484516925440.000000",
                                   "float",
                                   "0x6634",
                                   "float"};
constexpr FormatInfo int64Format{true,     true,     "-9223372036854775808ll", "9223372036854775807ll", "int64_t",
                                 "0x6938", "int64_t"};
constexpr FormatInfo uint64Format{true, false, "0ull", "18446744073709551615ull", "uint64_t", "0x7538", "uint64_t"};
constexpr FormatInfo doubleFormat{false,
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
} // namespace

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

const FormatInfo *getFormatInfo(const VkFormat format, const bool isUnsigned) {
    if (isUnsigned) {
        switch (format) {
        case VK_FORMAT_R8_SINT:
            return getFormatInfo(VK_FORMAT_R8_UINT);
        case VK_FORMAT_R16_SINT:
            return getFormatInfo(VK_FORMAT_R16_UINT);
        case VK_FORMAT_R32_SINT:
            return getFormatInfo(VK_FORMAT_R32_UINT);
        case VK_FORMAT_R64_SINT:
            return getFormatInfo(VK_FORMAT_R64_UINT);
        default:
            return getFormatInfo(format);
        }
    } else {
        switch (format) {
        case VK_FORMAT_R8_UINT:
        case VK_FORMAT_S8_UINT:
            return getFormatInfo(VK_FORMAT_R8_SINT);
        case VK_FORMAT_R16_UINT:
            return getFormatInfo(VK_FORMAT_R16_SINT);
        case VK_FORMAT_R32_UINT:
            return getFormatInfo(VK_FORMAT_R32_SINT);
        case VK_FORMAT_R64_UINT:
            return getFormatInfo(VK_FORMAT_R64_SINT);
        default:
            return getFormatInfo(format);
        }
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
