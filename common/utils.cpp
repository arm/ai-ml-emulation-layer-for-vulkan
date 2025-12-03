/*
 * SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "mlel/utils.hpp"
#include "mlel/float.hpp"
#include "mlel/log.hpp"
#include <functional>
#include <glslang/Include/glslang_c_interface.h>
#include <glslang/Public/resource_limits_c.h>
#include <limits>

using namespace mlsdk::el::log;

namespace mlsdk::el::utils {

namespace {
Log layerLog("VMEL_COMMON_SEVERITY", "Layer");
} // namespace

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

#if defined(EXPERIMENTAL_MOLTEN_VK_SUPPORT) || defined(ANDROID)
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

template <typename T> class Format : public FormatBase {
  public:
    explicit Format(const std::string &_glslType, const std::string &_literalSuffix = "")
        : _glslType{_glslType}, charType{getCharType()}, literalSuffix{_literalSuffix} {}

    bool isInteger() const override { return std::numeric_limits<T>::is_integer; }
    bool isSigned() const override { return std::numeric_limits<T>::is_signed; }
    std::string lowest() const override { return std::to_string(std::numeric_limits<T>::lowest()) + literalSuffix; }
    std::string max() const override { return std::to_string(std::numeric_limits<T>::max()) + literalSuffix; }
    std::string glslType() const override { return _glslType; }
    std::string toInt() const override { return std::to_string(uint64_t(charType) << 8 | ('0' + sizeof(T))); }

  private:
    const std::string _glslType;
    const int charType;
    const std::string literalSuffix;

    static constexpr int getCharType() {
        if constexpr (std::numeric_limits<T>::is_integer) {
            if constexpr (std::numeric_limits<T>::digits == 1) {
                return 'b';
            } else if constexpr (std::numeric_limits<T>::is_signed) {
                return 'i';
            } else {
                return 'u';
            }
        } else {
            return 'f';
        }
    }
};

std::shared_ptr<FormatBase> makeFormat(const VkFormat format) {
    switch (format) {
    case VK_FORMAT_R8_SINT:
        return std::make_shared<Format<int8_t>>("int8_t");
    case VK_FORMAT_R8_UINT:
    case VK_FORMAT_S8_UINT:
        return std::make_shared<Format<uint8_t>>("uint8_t", "u");
    case VK_FORMAT_R8_BOOL_ARM:
        return std::make_shared<Format<bool>>("bool");
    case VK_FORMAT_R16_SINT:
        return std::make_shared<Format<int16_t>>("int16_t");
    case VK_FORMAT_R16_UINT:
        return std::make_shared<Format<uint16_t>>("uint16_t", "u");
    case VK_FORMAT_R16_SFLOAT:
        return std::make_shared<Format<float16>>("float16_t");
    case VK_FORMAT_R32_SINT:
        return std::make_shared<Format<int32_t>>("int");
    case VK_FORMAT_R32_UINT:
        return std::make_shared<Format<uint32_t>>("uint32_t", "u");
    case VK_FORMAT_R32_SFLOAT:
        return std::make_shared<Format<float>>("float");
    case VK_FORMAT_R64_SINT:
        return std::make_shared<Format<int64_t>>("int64_t", "ll");
    case VK_FORMAT_R64_UINT:
        return std::make_shared<Format<uint64_t>>("uint64_t", "ull");
    case VK_FORMAT_R64_SFLOAT:
        return std::make_shared<Format<double>>("double", "ll");
    default:
        throw std::runtime_error("Unsupported tensor buffer format: " + std::to_string(format));
    }
}

std::shared_ptr<FormatBase> makeFormat(const VkFormat format, const bool isUnsigned) {
    if (isUnsigned) {
        switch (format) {
        case VK_FORMAT_R8_SINT:
            return makeFormat(VK_FORMAT_R8_UINT);
        case VK_FORMAT_R16_SINT:
            return makeFormat(VK_FORMAT_R16_UINT);
        case VK_FORMAT_R32_SINT:
            return makeFormat(VK_FORMAT_R32_UINT);
        case VK_FORMAT_R64_SINT:
            return makeFormat(VK_FORMAT_R64_UINT);
        default:
            return makeFormat(format);
        }
    } else {
        switch (format) {
        case VK_FORMAT_R8_UINT:
        case VK_FORMAT_S8_UINT:
            return makeFormat(VK_FORMAT_R8_SINT);
        case VK_FORMAT_R16_UINT:
            return makeFormat(VK_FORMAT_R16_SINT);
        case VK_FORMAT_R32_UINT:
            return makeFormat(VK_FORMAT_R32_SINT);
        case VK_FORMAT_R64_UINT:
            return makeFormat(VK_FORMAT_R64_SINT);
        default:
            return makeFormat(format);
        }
    }
}

void setDebugUtilsObjectName(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &loader,
                             VkDevice device, VkObjectType type, uint64_t handle, const std::string &name) {

    if (loader->vkSetDebugUtilsObjectNameEXT) {
        VkDebugUtilsObjectNameInfoEXT nameInfo{};
        nameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
        nameInfo.objectType = static_cast<VkObjectType>(type);
        nameInfo.objectHandle = handle;
        nameInfo.pObjectName = name.c_str();

        loader->vkSetDebugUtilsObjectNameEXT(device, &nameInfo);
    }
}

} // namespace mlsdk::el::utils
