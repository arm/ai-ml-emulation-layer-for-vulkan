/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include "compute_graph_op.hpp"
#include "compute_optical_flow.hpp"
#include "pipeline_cache.hpp"
#include "shaders/precompiled_shaders.hpp"

#include <gtest/gtest.h>
#include <spirv-tools/libspirv.hpp>
#include <spirv/unified1/spirv.hpp11>

#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mlsdk::el::compute {
namespace {

struct PrecompiledShaderCase {
    std::string shader;
    std::vector<std::string> keys;
    std::string name;
};

std::vector<PrecompiledShaderCase> precompiledShaderCases() {
    return {
#include "shaders/tosa_shader_cases.hpp.inc"
    };
}

class PrecompiledShader : public testing::TestWithParam<PrecompiledShaderCase> {};

using Cache = std::shared_ptr<PipelineCache>;
using Formats = std::vector<VkFormat>;

struct ShaderSelector {
    // Key positions in the order accepted by the operator's public selector.
    std::vector<size_t> formatKeys;
    std::function<SpirvBinary(const Cache &, const PrecompiledShaderCase &, const Formats &)> invoke;
};

template <typename Function, size_t... I>
SpirvBinary invokeFormats(Function function, const Cache &cache, const Formats &formats, std::index_sequence<I...>) {
    return function(cache, formats.at(I)...);
}

template <typename... Args>
ShaderSelector formatSelector(SpirvBinary (*function)(const Cache &, Args...), std::vector<size_t> keys) {
    return {std::move(keys), [function](const Cache &cache, const PrecompiledShaderCase &, const Formats &formats) {
                return invokeFormats(function, cache, formats, std::index_sequence_for<Args...>{});
            }};
}

uint32_t accumulatorType(const std::string &type) {
    static const std::map<std::string, uint32_t> types{{"int", 1}, {"float16_t", 2}, {"float", 3}, {"int64_t", 4}};
    return types.at(type);
}

ShaderSelector convolutionSelector(SpirvBinary (*function)(const Cache &, VkFormat, VkFormat, VkFormat, uint32_t)) {
    return {{0, 2, 1}, [function](const Cache &cache, const PrecompiledShaderCase &test, const Formats &formats) {
                return function(cache, formats.at(0), formats.at(1), formats.at(2), accumulatorType(test.keys.at(3)));
            }};
}

ShaderSelector namedUnarySelector(SpirvBinary (*function)(const Cache &, VkFormat, const std::string &)) {
    return {{1}, [function](const Cache &cache, const PrecompiledShaderCase &test, const Formats &formats) {
                return function(cache, formats.at(0), test.keys.at(0));
            }};
}

const std::map<std::string, ShaderSelector> &operatorSelectors() {
    using namespace graph_op;
    static const std::map<std::string, ShaderSelector> selectors{
        {"argmax", formatSelector(Argmax::createSpirv, {0})},
        {"arithmetic_right_shift", formatSelector(ArithmeticRightShift::createSpirv, {0})},
        {"avgpool2d",
         {{0},
          [](const Cache &cache, const PrecompiledShaderCase &test, const Formats &formats) {
              return AvgPool2D::createSpirv(cache, formats.at(0), accumulatorType(test.keys.at(1)));
          }}},
        {"cast", formatSelector(Cast::createSpirv, {0, 1})},
        {"clamp", formatSelector(Clamp::createSpirv, {0})},
        {"concat", formatSelector(Concat::createSpirv, {0})},
        {"conv2d", convolutionSelector(Conv2D::createSpirv)},
        {"conv3d", convolutionSelector(Conv3D::createSpirv)},
        {"depthwise_conv2d", convolutionSelector(DepthwiseConv2D::createSpirv)},
        {"elementwise_binary",
         {{1, 2},
          [](const Cache &cache, const PrecompiledShaderCase &test, const Formats &formats) {
              return ElementwiseBinary::createSpirv(cache, formats.at(0), formats.at(1), test.keys.at(0));
          }}},
        {"elementwise_unary", namedUnarySelector(ElementwiseUnary::createSpirv)},
        {"fft2d", formatSelector(Fft2D::createSpirv, {})},
        {"gather", formatSelector(Gather::createSpirv, {1, 0})},
        {"matmul", formatSelector(Matmul::createSpirv, {0, 1})},
        {"maxpool2d", formatSelector(MaxPool2D::createSpirv, {0})},
        {"mul", formatSelector(Mul::createSpirv, {0, 1})},
        {"negate", formatSelector(Negate::createSpirv, {0})},
        {"pad", formatSelector(Pad::createSpirv, {0})},
        {"reduce", namedUnarySelector(Reduce::createSpirv)},
        {"rescale",
         {{0, 1, 2},
          [](const Cache &cache, const PrecompiledShaderCase &test, const Formats &formats) {
              return Rescale::createSpirv(cache, formats.at(0), formats.at(1), formats.at(2),
                                          test.keys.at(0).rfind("uint", 0) == 0, test.keys.at(1).rfind("uint", 0) == 0);
          }}},
        {"reshape", formatSelector(Reshape::createSpirv, {0})},
        {"resize", formatSelector(Resize::createSpirv, {0, 1})},
        {"reverse", formatSelector(Reverse::createSpirv, {0})},
        {"rfft2d", formatSelector(Rfft2D::createSpirv, {})},
        {"scatter", formatSelector(Scatter::createSpirv, {1, 0})},
        {"select", formatSelector(Select::createSpirv, {0})},
        {"slice", formatSelector(Slice::createSpirv, {0})},
        {"table", formatSelector(Table::createSpirv, {0, 1})},
        {"tile", formatSelector(Tile::createSpirv, {0})},
        {"transpose", formatSelector(Transpose::createSpirv, {0})},
        {"transpose_conv2d", convolutionSelector(TransposeConv2D::createSpirv)},
    };
    return selectors;
}

const Formats &storageFormats(std::string_view type) {
    static const std::map<std::string_view, Formats> formats{
        {"bool", {VK_FORMAT_R8_BOOL_ARM}},
        {"int8_t", {VK_FORMAT_R8_SINT, VK_FORMAT_R8_UINT}},
        {"uint8_t", {VK_FORMAT_R8_SINT, VK_FORMAT_R8_UINT}},
        {"int16_t", {VK_FORMAT_R16_SINT, VK_FORMAT_R16_UINT}},
        {"uint16_t", {VK_FORMAT_R16_SINT, VK_FORMAT_R16_UINT}},
        {"int", {VK_FORMAT_R32_SINT, VK_FORMAT_R32_UINT}},
        {"int64_t", {VK_FORMAT_R64_SINT, VK_FORMAT_R64_UINT}},
        {"float16_t", {VK_FORMAT_R16_SFLOAT}},
        {"float", {VK_FORMAT_R32_SFLOAT}},
        {"bfloat16_t", {VK_FORMAT_R16_SFLOAT_FPENCODING_BFLOAT16_ARM}},
        {"float8_e4m3_t", {VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E4M3_ARM}},
        {"float8_e5m2_t", {VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E5M2_ARM}},
    };
    return formats.at(type);
}

std::vector<Formats> storageCombinations(const PrecompiledShaderCase &test, const ShaderSelector &selector) {
    std::vector<Formats> combinations(1);
    for (const auto key : selector.formatKeys) {
        std::vector<Formats> expanded;
        for (const auto &combination : combinations) {
            for (const auto format : storageFormats(test.keys.at(key))) {
                auto next = combination;
                next.push_back(format);
                expanded.push_back(std::move(next));
            }
        }
        combinations = std::move(expanded);
    }
    return combinations;
}

void expectIntegerStorageCounterparts(SpirvBinary spirv) {
    ASSERT_GE(spirv.size(), 5u);
    // Every integer tensor operand must use the shared unsigned interface. Each
    // such operand can independently bind either its SINT or UINT storage format.
    const std::map<uint32_t, std::pair<VkFormat, VkFormat>> formats = {
        {8, {VK_FORMAT_R8_SINT, VK_FORMAT_R8_UINT}},
        {16, {VK_FORMAT_R16_SINT, VK_FORMAT_R16_UINT}},
        {32, {VK_FORMAT_R32_SINT, VK_FORMAT_R32_UINT}},
        {64, {VK_FORMAT_R64_SINT, VK_FORMAT_R64_UINT}},
    };
    std::map<uint32_t, std::pair<uint32_t, uint32_t>> integerTypes;
    size_t tensorTypes = 0;
    for (size_t offset = 5; offset < spirv.size();) {
        const auto *instruction = spirv.data() + offset;
        const auto count = instruction[0] >> 16;
        const auto opcode = static_cast<spv::Op>(instruction[0] & 0xffff);
        ASSERT_GT(count, 0u);
        ASSERT_LE(offset + count, spirv.size());
        if (opcode == spv::Op::OpTypeInt) {
            ASSERT_EQ(count, 4u);
            integerTypes.emplace(instruction[1], std::make_pair(instruction[2], instruction[3]));
        } else if (opcode == spv::Op::OpTypeTensorARM) {
            ++tensorTypes;
            ASSERT_GE(count, 3u);
            if (const auto type = integerTypes.find(instruction[2]); type != integerTypes.end()) {
                const auto [width, signedness] = type->second;
                EXPECT_EQ(signedness, 0u) << "Signed integer tensor interface";
                const auto format = formats.find(width);
                ASSERT_NE(format, formats.end());
                const auto &[signedFormat, unsignedFormat] = format->second;
                const auto expected = "uint" + std::to_string(width) + "_t";
                EXPECT_EQ(utils::getTensorInterfaceGlslType(signedFormat), expected);
                EXPECT_EQ(utils::getTensorInterfaceGlslType(unsignedFormat), expected);
            }
        }
        offset += count;
    }
    EXPECT_GT(tensorTypes, 0u);
}

TEST_P(PrecompiledShader, OperatorSelectsExpectedModule) {
    const auto cache = std::make_shared<PipelineCache>(nullptr, 0, VK_NULL_HANDLE);
    const auto &test = GetParam();
    auto key = test.shader;
    for (const auto &part : test.keys) {
        key += "_" + part;
    }
    const auto embedded = precompiledSpirvModules.find(key);
    ASSERT_NE(embedded, precompiledSpirvModules.end()) << key;
    const auto selector = operatorSelectors().find(test.shader);
    ASSERT_NE(selector, operatorSelectors().end()) << "No operator selector test for " << test.shader;
    const auto &[words, size] = embedded->second;
    expectIntegerStorageCounterparts({words, size});
    for (const auto &formats : storageCombinations(test, selector->second)) {
        SCOPED_TRACE(testing::PrintToString(formats));
        const auto selected = selector->second.invoke(cache, test, formats);
        EXPECT_EQ(selected.data(), words);
        EXPECT_EQ(selected.size(), size);
    }
}

TEST(PipelineCache, EveryTosaSelectorHasCases) {
    std::set<std::string> covered;
    for (const auto &test : precompiledShaderCases()) {
        covered.insert(test.shader);
    }
    for (const auto &[name, selector] : operatorSelectors()) {
        EXPECT_EQ(covered.count(name), 1u) << name;
    }
}

INSTANTIATE_TEST_SUITE_P(PipelineCache, PrecompiledShader, testing::ValuesIn(precompiledShaderCases()),
                         [](const auto &info) { return info.param.name; });

TEST(PipelineCache, EmbeddedModulesValidateForVulkan) {
    spvtools::SpirvTools tools{SPV_ENV_VULKAN_1_3};
    tools.SetMessageConsumer(
        [](spv_message_level_t, const char *, const spv_position_t &position, const char *message) {
            ADD_FAILURE() << "Word " << position.index << ": " << message;
        });
    ASSERT_FALSE(precompiledSpirvModules.empty());
    for (const auto &[name, module] : precompiledSpirvModules) {
        SCOPED_TRACE(name);
        const auto &[words, size] = module;
        EXPECT_TRUE(tools.Validate(words, size));
    }
}

TEST(PipelineCache, MotionShadersAvailableWithoutGlslCompilation) {
    const auto cache = std::make_shared<PipelineCache>(nullptr, 0, VK_NULL_HANDLE);
    // Keep expectations independent of the build's shader list so an omitted
    // module fails this test even when all remaining modules validate.
    for (const auto mode : {common::BlockMatchMode::MIN_SAD, common::BlockMatchMode::MIN_SAD_COST}) {
        const auto key = std::to_string(static_cast<uint32_t>(mode));
        SCOPED_TRACE("block_match_" + key);
        const auto expected = cache->lookup("block_match", {key});
        const auto costFormats =
            mode == common::BlockMatchMode::MIN_SAD ? Formats{VK_FORMAT_UNDEFINED} : storageFormats("int16_t");
        for (const auto format : costFormats) {
            SCOPED_TRACE(format);
            const auto selected = graph_op::BlockMatch::createSpirv(cache, mode, format);
            EXPECT_EQ(selected.data(), expected.data());
            EXPECT_EQ(selected.size(), expected.size());
        }
    }
    const auto rawSad = std::to_string(static_cast<uint32_t>(common::BlockMatchMode::RAW_SAD));
    for (const auto *costType : {"uint8_t", "uint16_t"}) {
        SCOPED_TRACE(costType);
        const auto expected = cache->lookup("block_match", {rawSad, costType});
        for (const auto format : storageFormats(costType)) {
            SCOPED_TRACE(format);
            const auto selected = graph_op::BlockMatch::createSpirv(cache, common::BlockMatchMode::RAW_SAD, format);
            EXPECT_EQ(selected.data(), expected.data());
            EXPECT_EQ(selected.size(), expected.size());
        }
    }
    const std::map<std::string, SpirvBinary> opticalShaders{
        {"rgb_to_y_img", optical_flow::RGBToY::createSpirv(cache, false, true)},
        {"rgb_to_y_full_buf", optical_flow::RGBToY::createSpirv(cache, true, false)},
        {"downsample_img", optical_flow::Downsample::createSpirv(cache)},
        {"mv_process_and_warp_buf", optical_flow::MVProcessAndWarp::createSpirv(cache)},
        {"dense_warp_img", optical_flow::DenseWarp::createSpirv(cache)},
        {"median_filter_img", optical_flow::MedianFilter::createSpirv(cache)},
        {"bilateral_filter_img", optical_flow::BilateralFilter::createSpirv(cache, true)},
        {"bilateral_filter_buf", optical_flow::BilateralFilter::createSpirv(cache, false)},
        {"subpixel_me_buf", optical_flow::SubpixelME::createSpirv(cache, false)},
        {"subpixel_me_acc_buf", optical_flow::SubpixelME::createSpirv(cache, true)},
        {"mv_replace_img", optical_flow::MVReplace::createSpirv(cache, false)},
        {"mv_replace_cost_img", optical_flow::MVReplace::createSpirv(cache, true)},
        {"block_match_of_flow", optical_flow::BlockMatch::createSpirv(cache, common::BlockMatchMode::MIN_SAD, false)},
        {"block_match_of_flow_cost_buf",
         optical_flow::BlockMatch::createSpirv(cache, common::BlockMatchMode::MIN_SAD_COST, false)},
        {"block_match_of_flow_cost_img",
         optical_flow::BlockMatch::createSpirv(cache, common::BlockMatchMode::MIN_SAD_COST, true)},
        {"block_match_of_cost_buf",
         optical_flow::BlockMatch::createSpirv(cache, common::BlockMatchMode::RAW_SAD, false)},
    };
    for (const auto &[expectedName, selected] : opticalShaders) {
        SCOPED_TRACE(expectedName);
        const auto expected = cache->lookup(expectedName, {});
        EXPECT_EQ(selected.data(), expected.data());
        EXPECT_EQ(selected.size(), expected.size());
    }
}

TEST(PipelineCache, ReportsMissingVariant) {
    PipelineCache cache{nullptr, 0, VK_NULL_HANDLE};
    try {
        // NEGATE exists, but TOSA 1.0 does not allow FP8 arithmetic for it.
        cache.lookup("negate", {"float8_e4m3_t", "float"});
        FAIL() << "Unsupported variant was accepted";
    } catch (const std::runtime_error &error) {
        EXPECT_STREQ(error.what(), "Missing precompiled shader: negate_float8_e4m3_t_float");
    }
}

} // namespace
} // namespace mlsdk::el::compute
