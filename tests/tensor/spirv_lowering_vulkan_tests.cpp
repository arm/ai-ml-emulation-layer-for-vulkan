/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include "mlel/tensor.hpp"
#include "mlel/utils.hpp"
#include "shaders/precompiled_shaders.hpp"
#include "tensor_test_utils.hpp"
#include "test_utils.hpp"
#include "vulkan_test_utils.hpp"

#include <array>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace mlsdk::el::tests {
namespace {

using utilities::Shape;
using utilities::Tensor;

std::vector<uint32_t> tensorShader() {
    TensorShaderSource source;
    source.tensors = {{"data", "uint", "1", 0, 0}};
    source.body = R"(
    tensorWriteARM(data, uint[](0), tensorSizeARM(data, 0));
)";
    return compileTensorShader(source);
}

// Deliberately use only the Tensor layer, to establish that direct application
// shaders reach it without graph processing. Create the device after each test
// chooses its lowering mode.
class TensorLoweringForVulkan : public testing::Test {
  protected:
    void ensureDevice() {
        if (device) {
            return;
        }
        const ScopedEnvironment implicitLayers("VK_INSTANCE_LAYERS", std::nullopt);
        auto context = std::make_shared<vk::raii::Context>();
        auto instance =
            std::make_shared<utilities::Instance>(context, std::vector<const char *>{"VK_LAYER_ML_Tensor_Emulation"});
        std::vector<const char *> extensions{VK_ARM_TENSORS_EXTENSION_NAME};
        auto physical = std::make_shared<utilities::PhysicalDevice>(instance, extensions);
        const auto &vkPhysical = &(*physical);
        vk::PhysicalDeviceMaintenance5FeaturesKHR maintenance5{};
        if (utils::hasExtension(vkPhysical.enumerateDeviceExtensionProperties(), VK_KHR_MAINTENANCE_5_EXTENSION_NAME)) {
            auto features =
                vkPhysical.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceMaintenance5FeaturesKHR>();
            inlineSupported = features.get<vk::PhysicalDeviceMaintenance5FeaturesKHR>().maintenance5;
            if (inlineSupported) {
                extensions.push_back(VK_KHR_MAINTENANCE_5_EXTENSION_NAME);
                maintenance5.maintenance5 = VK_TRUE;
            }
        }
        vk::PhysicalDeviceFeatures2 features{};
        features.features.shaderInt64 = VK_TRUE;
        vk::PhysicalDeviceTensorFeaturesARM tensors{};
        tensors.tensors = VK_TRUE;
        tensors.shaderTensorAccess = VK_TRUE;
        tensors.pNext = inlineSupported ? &maintenance5 : nullptr;
        features.pNext = &tensors;
        device = std::make_shared<utilities::Device>(physical, extensions, &features);
    }

    VkResult createModule(const std::vector<uint32_t> &code) {
        ensureDevice();
        const auto &vkDevice = &(*device);
        const VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
                                            code.size() * sizeof(uint32_t), code.data()};
        VkShaderModule module = VK_NULL_HANDLE;
        const auto result = vkDevice.getDispatcher()->vkCreateShaderModule(*vkDevice, &info, nullptr, &module);
        if (module != VK_NULL_HANDLE) {
            vkDevice.getDispatcher()->vkDestroyShaderModule(*vkDevice, module, nullptr);
        }
        return result;
    }

    VkResult createPipeline(const std::vector<uint32_t> &code, bool inlineModule = true) {
        ensureDevice();
        const auto &vkDevice = &(*device);
        const vk::DescriptorSetLayoutBinding binding{0, vk::DescriptorType::eTensorARM, 1,
                                                     vk::ShaderStageFlagBits::eCompute};
        const vk::raii::DescriptorSetLayout descriptorLayout{vkDevice,
                                                             vk::DescriptorSetLayoutCreateInfo{{}, 1, &binding}};
        const vk::DescriptorSetLayout rawDescriptorLayout = *descriptorLayout;
        const vk::raii::PipelineLayout layout{vkDevice, vk::PipelineLayoutCreateInfo{{}, 1, &rawDescriptorLayout}};
        const VkShaderModuleCreateInfo shader{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
                                              code.size() * sizeof(uint32_t), code.data()};
        VkComputePipelineCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        info.layout = *layout;
        info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        info.stage.pName = "main";
        vk::raii::ShaderModule module{nullptr};
        if (inlineModule) {
            info.stage.pNext = &shader;
        } else {
            module = vk::raii::ShaderModule(vkDevice, vk::ShaderModuleCreateInfo(shader));
            info.stage.module = *module;
        }
        VkPipeline pipeline = VK_NULL_HANDLE;
        const auto result =
            vkDevice.getDispatcher()->vkCreateComputePipelines(*vkDevice, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline);
        if (pipeline != VK_NULL_HANDLE) {
            vkDevice.getDispatcher()->vkDestroyPipeline(*vkDevice, pipeline, nullptr);
        }
        return result;
    }

    std::shared_ptr<utilities::Device> device;
    bool inlineSupported = false;
};

class SpecializedTensorPipeline : public utilities::PipelineBase {
  public:
    using PipelineBase::PipelineBase;

    void dispatch(const DescriptorMap &descriptors, const TensorSpecialization *specialization = nullptr,
                  const std::vector<uint32_t> &pushConstants = {}) {
        auto layout = pipelineLayout;
        if (!pushConstants.empty()) {
            std::vector<vk::DescriptorSetLayout> layouts;
            layouts.reserve(descriptorSetLayouts.size());
            for (const auto &setLayout : descriptorSetLayouts) {
                layouts.push_back(*setLayout);
            }
            const vk::PushConstantRange range{vk::ShaderStageFlagBits::eCompute, 0,
                                              uint32_t(pushConstants.size() * sizeof(uint32_t))};
            layout = std::make_shared<vk::raii::PipelineLayout>(
                &(*device), vk::PipelineLayoutCreateInfo{{}, uint32_t(layouts.size()), layouts.data(), 1, &range});
        }
        const auto info = specialization ? specialization->info() : vk::SpecializationInfo{};
        const vk::PipelineShaderStageCreateInfo stage{
            {}, vk::ShaderStageFlagBits::eCompute, *shaderModule, "main", specialization ? &info : nullptr};
        const vk::raii::Pipeline pipeline{&(*device), nullptr, vk::ComputePipelineCreateInfo{{}, stage, *layout}};
        auto bindings = createDescriptorSets(descriptors);
        std::vector<vk::DescriptorSet> sets;
        for (const auto &set : bindings.sets) {
            sets.push_back(*set);
        }
        auto commands = createCommandBuffer();
        commands.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        commands.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
        commands.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *layout, 0, sets, nullptr);
        if (!pushConstants.empty()) {
            commands.pushConstants<uint32_t>(*layout, vk::ShaderStageFlagBits::eCompute, 0, pushConstants);
        }
        commands.dispatch(1, 1, 1);
        commands.end();
        submitWork(commands);
    }
};

TEST_F(TensorLoweringForVulkan, DefaultCreatesTensorModule) {
    const ScopedEnvironment environment("VMEL_EXPERIMENTAL_TENSOR_LOWERING_MODE", std::nullopt);
    EXPECT_EQ(createModule(tensorShader()), VK_SUCCESS);
}

TEST_F(TensorLoweringForVulkan, ZeroCreatesTensorPipeline) {
    const ScopedEnvironment environment("VMEL_EXPERIMENTAL_TENSOR_LOWERING_MODE", "0");
    EXPECT_EQ(createPipeline(tensorShader(), false), VK_SUCCESS);
}

TEST_F(TensorLoweringForVulkan, DirectCreatesScalarTensorModule) {
    const ScopedEnvironment environment("VMEL_EXPERIMENTAL_TENSOR_LOWERING_MODE", "1");
    EXPECT_EQ(createModule(tensorShader()), VK_SUCCESS);
}

TEST_F(TensorLoweringForVulkan, DirectPreservesNonTensorModule) {
    const ScopedEnvironment environment("VMEL_EXPERIMENTAL_TENSOR_LOWERING_MODE", "1");
    const auto shader = compileGlsl("#version 460\nlayout(local_size_x = 1) in; void main() {}\n");
    EXPECT_EQ(createModule(shader), VK_SUCCESS);
}

TEST_F(TensorLoweringForVulkan, DirectRejectsMalformedModule) {
    const ScopedEnvironment environment("VMEL_EXPERIMENTAL_TENSOR_LOWERING_MODE", "1");
    EXPECT_EQ(createModule({0, 0, 0, 0, 0}), VK_ERROR_UNKNOWN);
    EXPECT_EQ(createModule({}), VK_ERROR_UNKNOWN);
}

TEST_F(TensorLoweringForVulkan, DirectCreatesScalarTensorPipeline) {
    const ScopedEnvironment environment("VMEL_EXPERIMENTAL_TENSOR_LOWERING_MODE", "1");
    EXPECT_EQ(createPipeline(tensorShader(), false), VK_SUCCESS);
}

TEST_F(TensorLoweringForVulkan, DirectRejectsUnsupportedInlineTensorPipeline) {
    const ScopedEnvironment environment("VMEL_EXPERIMENTAL_TENSOR_LOWERING_MODE", "1");
    ensureDevice();
    if (!inlineSupported) {
        GTEST_SKIP() << "Inline shader modules require maintenance5";
    }
    TensorShaderSource source;
    source.tensors = {{"data", "uint", "1", 0, 0}};
    source.declarations = "layout(constant_id = 0) const uint LENGTH = 2;";
    source.body = R"(
    uint values[LENGTH]; tensorReadARM(data, uint[](0), values);
)";
    const auto shader = compileTensorShader(source);
    EXPECT_EQ(createPipeline(shader), VK_ERROR_UNKNOWN);
}

TEST_F(TensorLoweringForVulkan, DirectPreservesNonTensorInlinePipeline) {
    const ScopedEnvironment environment("VMEL_EXPERIMENTAL_TENSOR_LOWERING_MODE", "1");
    ensureDevice();
    if (!inlineSupported) {
        GTEST_SKIP() << "Inline shader modules require maintenance5";
    }
    const auto shader = compileGlsl("#version 460\nlayout(local_size_x = 1) in; void main() {}\n");
    EXPECT_EQ(createPipeline(shader), VK_SUCCESS);
}

TEST_F(TensorLoweringForVulkan, DirectRejectsMalformedInlineModule) {
    const ScopedEnvironment environment("VMEL_EXPERIMENTAL_TENSOR_LOWERING_MODE", "1");
    ensureDevice();
    if (!inlineSupported) {
        GTEST_SKIP() << "Inline shader modules require maintenance5";
    }
    EXPECT_EQ(createPipeline({0, 0, 0, 0, 0}), VK_ERROR_UNKNOWN);
    EXPECT_EQ(createPipeline({}), VK_ERROR_UNKNOWN);
}

TEST_F(TensorLoweringForVulkan, DirectScalarBoundsAndRuntimeMetadata) {
    const ScopedEnvironment environment("VMEL_EXPERIMENTAL_TENSOR_LOWERING_MODE", "1");
    ensureDevice();
    TensorShaderSource source;
    source.tensors = {
        {"source", "uint", "2", 0, 0}, {"destination", "uint", "2", 0, 1}, {"results", "uint", "1", 0, 2}};
    source.body = R"(
    uint height = tensorSizeARM(source, 0);
    uint width = tensorSizeARM(source, 1);
    tensorWriteARM(results, uint[](0), height);
    tensorWriteARM(results, uint[](1), width);
    uint value;
    tensorReadARM(source, uint[](0, width), value);
    tensorWriteARM(results, uint[](2), value);
    tensorReadARM(source, uint[](0, uint(-1)), value);
    tensorWriteARM(results, uint[](3), value);
    tensorReadARM(source, uint[](height, 0), value);
    tensorWriteARM(results, uint[](4), value);
    tensorReadARM(source, uint[](0xffffffffu, 0xffffffffu), value);
    tensorWriteARM(results, uint[](5), value);
    tensorWriteARM(destination, uint[](0, width), 99u);
    tensorWriteARM(destination, uint[](uint(-1), 0), 99u);
    tensorWriteARM(destination, uint[](height, 0), 99u);
    for (uint y = 0; y < height; ++y) {
        for (uint x = 0; x < width; ++x) {
            uint coords[2] = uint[](y, x);
            if ((x & 1) == 0) {
                tensorReadARM(source, coords, value);
            } else {
                tensorReadARM(source, uint[](y, x), value);
            }
            tensorWriteARM(destination, coords, value);
        }
    }
)";
    const auto shader = compileTensorShader(source);
    for (const auto &[height, width, rowStride, elementStride] :
         std::vector<std::array<int64_t, 4>>{{2, 3, 32, 4}, {3, 2, 40, 8}, {2, 4, 16, 4}}) {
        SCOPED_TRACE(testing::PrintToString(std::array{height, width, rowStride, elementStride}));
        const Shape shape{vk::Format::eR32Uint, {height, width}, {rowStride, elementStride}};
        std::vector<uint32_t> values;
        for (int64_t y = 0; y < height; ++y) {
            for (int64_t x = 0; x < width; ++x) {
                values.push_back(static_cast<uint32_t>((100 * (y + 1)) + x));
            }
        }
        auto input = makeTestTensor(device, shape, values);
        auto output = makeTestTensor(device, shape);
        const Shape resultsShape{vk::Format::eR32Uint, {6}};
        auto results = makeTestTensor(device, resultsShape);
        const utilities::TensorComputePipeline::DescriptorMap descriptors{
            {{0, {input}}, {1, {output}}, {2, {results}}}};
        utilities::TensorComputePipeline pipeline(device, descriptors, shader);
        pipeline.dispatchSubmit(1, 1, 1);
        expectTensorValues<uint32_t>(*results, resultsShape,
                                     {static_cast<uint32_t>(height), static_cast<uint32_t>(width), 0, 0, 0, 0});
        expectTensorValues(*output, shape, values, 0xa5);
    }
}

class TensorRuntimeBoundsForVulkan : public TensorLoweringForVulkan, public testing::WithParamInterface<const char *> {
  protected:
    void checkReads(const Shape &shape, const std::vector<std::pair<std::vector<uint32_t>, uint32_t>> &cases,
                    bool specialised = false) {
        const ScopedEnvironment environment("VMEL_EXPERIMENTAL_TENSOR_LOWERING_MODE", GetParam());
        ensureDevice();
        const auto rank = shape.getDimensions().size();
        std::string readCoordinates;
        std::string coordinateValues;
        if (specialised) {
            readCoordinates = "uint coords[RANK];\n"
                              "for (uint axis = 0; axis < RANK; ++axis) {\n"
                              "    tensorReadARM(coordinates, uint[](axis), coords[axis]);\n"
                              "}\n";
            coordinateValues = "coords";
        } else {
            for (size_t axis = 0; axis < rank; ++axis) {
                const auto index = std::to_string(axis);
                const auto variable = "c" + index;
                readCoordinates.append("uint ")
                    .append(variable)
                    .append("; tensorReadARM(coordinates, uint[](")
                    .append(index)
                    .append("), ")
                    .append(variable)
                    .append(");\n");
                coordinateValues += (axis == 0 ? "" : ", ") + variable;
            }
            coordinateValues = "uint[](" + coordinateValues + ")";
        }
        TensorShaderSource source;
        source.tensors = {{"source", "uint", specialised ? "RANK" : std::to_string(rank), 0, 0},
                          {"coordinates", "uint", "1", 0, 1},
                          {"result", "uint", "1", 0, 2}};
        if (specialised) {
            source.declarations = "layout(constant_id = 0) const uint RANK = 1;\n";
        }
        source.body = readCoordinates + R"(
    uint value;
    tensorReadARM(source, COORDINATE_VALUES, value);
    tensorWriteARM(result, uint[](0), value);
)";
        utils::replaceAll(source.body, "COORDINATE_VALUES", coordinateValues);
        const auto shader = compileTensorShader(source);
        auto input = std::make_shared<Tensor>(device, shape);
        auto coordinates = std::make_shared<Tensor>(device, Shape{vk::Format::eR32Uint, {static_cast<int64_t>(rank)}});
        auto result = std::make_shared<Tensor>(device, Shape{vk::Format::eR32Uint, {1}});
        std::vector<uint32_t> inputData(input->size() / sizeof(uint32_t), 0xdeadbeef);
        for (size_t i = 0; i < shape.elementCount(); ++i) {
            inputData[shape.getElementOffset(i) / sizeof(uint32_t)] = 100 + static_cast<uint32_t>(i);
        }
        std::memcpy(input->data(), inputData.data(), input->size());
        const utilities::TensorComputePipeline::DescriptorMap descriptors{
            {{0, {input}}, {1, {coordinates}}, {2, {result}}}};
        // Preserve the fixed-rank test's pipeline reuse. The specialised path
        // overrides the same module's default rank when creating its pipeline.
        std::unique_ptr<utilities::TensorComputePipeline> fixedPipeline;
        std::unique_ptr<SpecializedTensorPipeline> specialisedPipeline;
        if (specialised) {
            specialisedPipeline = std::make_unique<SpecializedTensorPipeline>(device, descriptors, shader);
        } else {
            fixedPipeline = std::make_unique<utilities::TensorComputePipeline>(device, descriptors, shader);
        }
        const TensorSpecialization specialization{{0, static_cast<uint32_t>(rank)}};
        for (const auto &[coords, expected] : cases) {
            SCOPED_TRACE(testing::PrintToString(coords));
            ASSERT_EQ(coords.size(), rank);
            // Supply coordinates through tensor storage so invalid accesses cannot be folded away.
            std::memcpy(coordinates->data(), coords.data(), coordinates->size());
            std::memset(result->data(), 0xa5, result->size());
            if (specialised) {
                specialisedPipeline->dispatch(descriptors, &specialization);
            } else {
                fixedPipeline->dispatchSubmit(1, 1, 1);
            }
            uint32_t actual = 0;
            std::memcpy(&actual, result->data(), sizeof(actual));
            EXPECT_EQ(actual, expected);
        }
    }
};

TEST_P(TensorRuntimeBoundsForVulkan, RankOne) {
    checkReads(Shape{vk::Format::eR32Uint, {3}},
               {{{0}, 100}, {{2}, 102}, {{3}, 0}, {{std::numeric_limits<uint32_t>::max()}, 0}});
}

TEST_P(TensorRuntimeBoundsForVulkan, RankTwoWithPaddedRows) {
    const auto invalid = std::numeric_limits<uint32_t>::max();
    checkReads(Shape{vk::Format::eR32Uint, {2, 3}, {32, 4}}, {{{0, 0}, 100},
                                                              {{1, 2}, 105},
                                                              {{0, 3}, 0},
                                                              {{1, 3}, 0},
                                                              {{2, 0}, 0},
                                                              {{invalid, 0}, 0},
                                                              {{0, invalid}, 0},
                                                              {{1, invalid}, 0},
                                                              {{invalid, invalid}, 0}});
}

TEST_P(TensorRuntimeBoundsForVulkan, SpecializedRanks) {
    const auto invalid = std::numeric_limits<uint32_t>::max();
    for (uint32_t rank = 1; rank <= 6; ++rank) {
        SCOPED_TRACE(rank);
        const Shape shape{vk::Format::eR32Uint, std::vector<int64_t>(rank, 2)};
        const std::vector<uint32_t> last(rank, 1);
        std::vector<std::pair<std::vector<uint32_t>, uint32_t>> cases{{std::vector<uint32_t>(rank, 0), 100},
                                                                      {last, 99 + (1u << rank)}};
        for (uint32_t axis = 0; axis < rank; ++axis) {
            auto coords = last;
            coords[axis] = 2;
            cases.emplace_back(coords, 0);
            coords[axis] = invalid;
            cases.emplace_back(coords, 0);
        }
        cases.emplace_back(std::vector<uint32_t>(rank, invalid), 0);
        checkReads(shape, cases, true);
    }
}

INSTANTIATE_TEST_SUITE_P(LoweringModes, TensorRuntimeBoundsForVulkan, testing::Values("0", "1"),
                         [](const testing::TestParamInfo<const char *> &info) {
                             return std::string(info.param) == "0" ? "GlslToSpirv" : "DirectSpirv";
                         });

TEST_F(TensorLoweringForVulkan, DirectSpecializedRanksAndInt64Storage) {
    const ScopedEnvironment environment("VMEL_EXPERIMENTAL_TENSOR_LOWERING_MODE", "1");
    ensureDevice();
    TensorShaderSource source;
    source.tensors = {{"source", "uint64_t", "INPUT_RANK", 0, 0},
                      {"destination", "uint64_t", "OUTPUT_RANK", 0, 1},
                      {"results", "uint", "1", 0, 2}};
    source.declarations = R"(
layout(constant_id = 0) const uint INPUT_BASE = 1;
layout(constant_id = 1) const uint OUTPUT_BASE = 1;
const uint INPUT_RANK = INPUT_BASE + 1;
const uint OUTPUT_RANK = OUTPUT_BASE + 1;
)";
    source.extensions = "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n";
    source.body = R"(
    uint inputCoords[INPUT_RANK];
    uint outputCoords[OUTPUT_RANK];
    for (uint i = 0; i < INPUT_RANK; ++i) inputCoords[i] = 1;
    for (uint i = 0; i < OUTPUT_RANK; ++i) outputCoords[i] = 1;
    uint64_t value;
    tensorReadARM(source, inputCoords, value);
    tensorWriteARM(destination, outputCoords, uint64_t(-int64_t(value)));
    inputCoords[INPUT_RANK - 1] = 2;
    tensorReadARM(source, inputCoords, value);
    tensorWriteARM(results, uint[](0), uint(value));
    tensorWriteARM(results, uint[](1), tensorSizeARM(source, 0));
    outputCoords[OUTPUT_RANK - 1] = 2;
    tensorWriteARM(destination, outputCoords, uint64_t(0));
)";
    const auto shader = compileTensorShader(source);
    auto makeTensor = [&](uint32_t rank) {
        std::vector<int64_t> dimensions(rank, 2);
        std::vector<int64_t> strides(rank);
        int64_t stride = 16;
        for (size_t i = rank; i > 0; --i) {
            strides[i - 1] = stride;
            stride *= 3;
        }
        return std::make_shared<Tensor>(device, Shape{vk::Format::eR64Uint, dimensions, strides});
    };
    auto results = std::make_shared<Tensor>(device, Shape{vk::Format::eR32Uint, {2}});
    const utilities::TensorComputePipeline::DescriptorMap initial{
        {{0, {makeTensor(2)}}, {1, {makeTensor(2)}}, {2, {results}}}};
    // Recreating the module also exercises reuse of its cached lowered SPIR-V.
    for (uint32_t repeat = 0; repeat < 2; ++repeat) {
        SpecializedTensorPipeline pipeline(device, initial, shader);
        for (const auto &[inputRank, outputRank] :
             std::vector<std::pair<uint32_t, uint32_t>>{{2, 2}, {1, 6}, {6, 1}, {2, 3}, {3, 2}, {4, 5}, {5, 4}}) {
            SCOPED_TRACE(std::to_string(inputRank) + " -> " + std::to_string(outputRank));
            auto input = makeTensor(inputRank);
            auto output = makeTensor(outputRank);
            auto elementIndex = [](uint32_t rank) {
                size_t index = 0;
                size_t stride = 2;
                for (uint32_t i = 0; i < rank; ++i) {
                    index += stride;
                    stride *= 3;
                }
                return index;
            };
            const uint64_t expected = (uint64_t{1} << 40) + (uint64_t{inputRank} * 100) + outputRank;
            std::vector<uint64_t> inputData(input->size() / sizeof(uint64_t), 0xdeadbeefdeadbeef);
            std::vector<uint64_t> expectedOutput(output->size() / sizeof(uint64_t), 0xa5a5a5a5a5a5a5a5);
            inputData[elementIndex(inputRank)] = static_cast<uint64_t>(-static_cast<int64_t>(expected));
            expectedOutput[elementIndex(outputRank)] = expected;
            std::memcpy(input->data(), inputData.data(), input->size());
            std::memset(output->data(), 0xa5, output->size());
            std::memset(results->data(), 0xa5, results->size());
            const TensorSpecialization specialization{{0, inputRank - 1}, {1, outputRank - 1}};
            pipeline.dispatch({{{0, {input}}, {1, {output}}, {2, {results}}}},
                              inputRank == 2 && outputRank == 2 ? nullptr : &specialization);
            const std::array<uint32_t, 2> expectedResults{0, 2};
            EXPECT_EQ(std::memcmp(results->data(), expectedResults.data(), results->size()), 0);
            EXPECT_EQ(std::memcmp(output->data(), expectedOutput.data(), output->size()), 0);
        }
    }
}

TEST_F(TensorLoweringForVulkan, DirectBooleanStorageBoundsAndSpecializedRanks) {
    const ScopedEnvironment environment("VMEL_EXPERIMENTAL_TENSOR_LOWERING_MODE", "1");
    ensureDevice();
    TensorShaderSource source;
    source.tensors = {{"source", "bool", "INPUT_RANK", 0, 0},
                      {"destination", "bool", "OUTPUT_RANK", 0, 1},
                      {"results", "uint", "1", 0, 2}};
    source.declarations = R"(
layout(constant_id = 0) const uint INPUT_RANK = 2;
layout(constant_id = 1) const uint OUTPUT_RANK = 1;
)";
    source.body = R"(
    uint inputCoords[INPUT_RANK];
    uint outputCoords[OUTPUT_RANK];
    for (uint i = 0; i < INPUT_RANK; ++i) inputCoords[i] = 1;
    for (uint i = 0; i < OUTPUT_RANK; ++i) outputCoords[i] = 1;
    for (uint x = 0; x < 4; ++x) {
        inputCoords[INPUT_RANK - 1] = x;
        bool value;
        tensorReadARM(source, inputCoords, value);
        outputCoords[OUTPUT_RANK - 1] = x;
        tensorWriteARM(destination, outputCoords, value);
        outputCoords[OUTPUT_RANK - 1] = x + 4;
        tensorWriteARM(destination, outputCoords, !value);
    }
    tensorWriteARM(results, uint[](0), tensorSizeARM(source, 0));
    tensorWriteARM(results, uint[](1), tensorSizeARM(destination, 0));

    // Invalid tensor coordinates can still address padding inside the allocation.
    for (uint i = 0; i < INPUT_RANK; ++i) inputCoords[i] = 0;
    for (uint i = 0; i < OUTPUT_RANK; ++i) outputCoords[i] = 0;
    inputCoords[INPUT_RANK - 1] = 4;
    outputCoords[OUTPUT_RANK - 1] = 8;
    bool value;
    tensorReadARM(source, inputCoords, value);
    tensorWriteARM(results, uint[](2), uint(value));
    tensorWriteARM(destination, outputCoords, false);
    inputCoords[INPUT_RANK - 1] = 0xffffffffu;
    outputCoords[OUTPUT_RANK - 1] = 0xffffffffu;
    tensorReadARM(source, inputCoords, value);
    tensorWriteARM(results, uint[](3), uint(value));
    tensorWriteARM(destination, outputCoords, true);
)";
    const auto shader = compileTensorShader(source);
    auto makeShape = [](uint32_t rank, int64_t width, int64_t elementStride) {
        std::vector<int64_t> dimensions(rank);
        std::vector<int64_t> strides(rank);
        for (size_t i = rank; i > 0; --i) {
            dimensions[i - 1] = i == rank ? width : 2;
            strides[i - 1] = elementStride;
            elementStride *= dimensions[i - 1] + 1;
        }
        return Shape{vk::Format::eR8BoolARM, dimensions, strides};
    };
    auto results = std::make_shared<Tensor>(device, Shape{vk::Format::eR32Uint, {4}});
    const utilities::TensorComputePipeline::DescriptorMap initial{
        {{0, {std::make_shared<Tensor>(device, makeShape(2, 4, 3))}},
         {1, {std::make_shared<Tensor>(device, makeShape(1, 8, 2))}},
         {2, {results}}}};
    SpecializedTensorPipeline pipeline(device, initial, shader);
    for (const auto &[inputRank, outputRank] :
         std::vector<std::pair<uint32_t, uint32_t>>{{2, 1}, {1, 6}, {6, 3}, {4, 5}, {5, 2}, {3, 4}}) {
        SCOPED_TRACE(std::to_string(inputRank) + " -> " + std::to_string(outputRank));
        const auto inputShape = makeShape(inputRank, 4, 3);
        const auto outputShape = makeShape(outputRank, 8, 2);
        auto input = std::make_shared<Tensor>(device, inputShape);
        auto output = std::make_shared<Tensor>(device, outputShape);
        // The shader selects coordinate 1 on every axis except the innermost.
        auto rowOffset = [](const Shape &shape) {
            size_t offset = 0;
            const auto &strides = shape.getStrides();
            for (size_t i = 0; i + 1 < strides.size(); ++i) {
                offset += static_cast<size_t>(strides[i]);
            }
            return offset;
        };
        std::vector<uint8_t> inputData(input->size(), 0xa5);
        std::vector<uint8_t> expectedOutput(output->size(), 0xa5);
        const std::array<uint8_t, 4> values{0, 1, 0x80, 0xff};
        for (size_t x = 0; x < values.size(); ++x) {
            inputData[rowOffset(inputShape) + (x * 3)] = values[x];
            expectedOutput[rowOffset(outputShape) + (x * 2)] = values[x] != 0 ? 1 : 0;
            expectedOutput[rowOffset(outputShape) + ((x + 4) * 2)] = values[x] == 0 ? 1 : 0;
        }
        std::memcpy(input->data(), inputData.data(), input->size());
        std::memset(output->data(), 0xa5, output->size());
        std::memset(results->data(), 0xa5, results->size());
        const TensorSpecialization specialization{{0, inputRank}, {1, outputRank}};
        pipeline.dispatch({{{0, {input}}, {1, {output}}, {2, {results}}}},
                          inputRank == 2 && outputRank == 1 ? nullptr : &specialization);
        const std::array<uint32_t, 4> expectedResults{inputRank == 1 ? 4u : 2u, outputRank == 1 ? 8u : 2u, 0, 0};
        EXPECT_EQ(std::memcmp(results->data(), expectedResults.data(), results->size()), 0);
        EXPECT_EQ(std::memcmp(output->data(), expectedOutput.data(), output->size()), 0);
    }
}

TEST_F(TensorLoweringForVulkan, DirectSelectBroadcastsIndependentDescriptors) {
    const ScopedEnvironment environment("VMEL_EXPERIMENTAL_TENSOR_LOWERING_MODE", "1");
    ensureDevice();
    const auto &[words, size] = compute::precompiledSpirvModules.at("select_float");
    const std::vector<uint32_t> shader(words, words + size);
    const TensorSpecialization specialization{{0, 3}};
    const Shape outputShape{vk::Format::eR32Sfloat, {2, 3, 4}, {224, 64, 8}};
    // Predicate, true input, and false input broadcast along different axes.
    const std::vector<std::array<Shape, 3>> cases{
        {Shape{vk::Format::eR8Uint, {1, 3, 1}, {32, 5, 1}}, Shape{vk::Format::eR32Sfloat, {2, 1, 4}, {64, 40, 8}},
         Shape{vk::Format::eR32Sfloat, {1, 3, 1}, {80, 16, 4}}},
        {Shape{vk::Format::eR8Uint, {2, 1, 4}, {32, 16, 2}}, Shape{vk::Format::eR32Sfloat, {1, 3, 4}, {224, 64, 12}},
         Shape{vk::Format::eR32Sfloat, {2, 1, 1}, {28, 12, 4}}}};
    for (const auto &shapes : cases) {
        SCOPED_TRACE(testing::PrintToString(shapes[0].getDimensions()));
        std::vector<uint8_t> predicates(shapes[0].elementCount());
        std::vector<float> trueValues(shapes[1].elementCount());
        std::vector<float> falseValues(shapes[2].elementCount());
        for (size_t i = 0; i < predicates.size(); ++i) {
            predicates[i] = std::array<uint8_t, 3>{0, 1, 255}[i % 3];
        }
        for (size_t i = 0; i < trueValues.size(); ++i) {
            trueValues[i] = float(i + 1);
        }
        for (size_t i = 0; i < falseValues.size(); ++i) {
            falseValues[i] = -float(100 + i);
        }
        auto predicate = makeTestTensor(device, shapes[0], predicates);
        auto trueInput = makeTestTensor(device, shapes[1], trueValues);
        auto falseInput = makeTestTensor(device, shapes[2], falseValues);
        auto output = makeTestTensor(device, outputShape);
        const SpecializedTensorPipeline::DescriptorMap descriptors{
            {{0, {output}}}, {{0, {predicate}}}, {{0, {trueInput}}}, {{0, {falseInput}}}};
        SpecializedTensorPipeline pipeline(device, descriptors, shader);
        pipeline.dispatch(descriptors, &specialization);

        std::vector<float> expected(outputShape.elementCount());
        for (size_t n = 0; n < 2; ++n) {
            for (size_t y = 0; y < 3; ++y) {
                for (size_t x = 0; x < 4; ++x) {
                    auto inputIndex = [&](const Shape &shape) {
                        const auto &dims = shape.getDimensions();
                        const auto row = ((n % size_t(dims[0])) * size_t(dims[1])) + (y % size_t(dims[1]));
                        return (row * size_t(dims[2])) + (x % size_t(dims[2]));
                    };
                    const float value = predicates[inputIndex(shapes[0])] != 0 ? trueValues[inputIndex(shapes[1])]
                                                                               : falseValues[inputIndex(shapes[2])];
                    expected[(((n * 3) + y) * 4) + x] = value;
                }
            }
        }
        expectTensorValues(*output, outputShape, expected);
    }
}

TEST_F(TensorLoweringForVulkan, DirectBroadcastArithmeticAndRounding) {
    const ScopedEnvironment environment("VMEL_EXPERIMENTAL_TENSOR_LOWERING_MODE", "1");
    ensureDevice();
    auto left = makeTestTensor<int32_t>(device, Shape{vk::Format::eR32Uint, {2, 1}, {20, 4}}, {-7, 7});
    auto right = makeTestTensor<int32_t>(device, Shape{vk::Format::eR32Uint, {1, 3}, {40, 8}}, {0, 1, 2});
    auto left16 = makeTestTensor<int16_t>(device, Shape{vk::Format::eR16Uint, {2, 1}, {10, 2}}, {-7, 7});
    auto right16 = makeTestTensor<int16_t>(device, Shape{vk::Format::eR16Uint, {1, 3}, {20, 4}}, {0, 1, 2});
    const Shape outputShape{vk::Format::eR32Uint, {2, 3}, {36, 8}};
    const TensorSpecialization specialization{{0, 2}, {1, 2}};
    struct Case {
        std::string shader;
        uint32_t control;
        std::vector<int32_t> expected;
    };
    // Literal expected values distinguish signed arithmetic, both rounding modes,
    // and MUL's shift control. The final case reads 16-bit inputs and writes 32-bit results.
    const std::vector<Case> cases{{"elementwise_binary_ADD_int_int", 0, {-7, -6, -5, 7, 8, 9}},
                                  {"arithmetic_right_shift_int", 0, {-7, -4, -2, 7, 3, 1}},
                                  {"arithmetic_right_shift_int", 1, {-7, -3, -2, 7, 4, 2}},
                                  {"mul_int_int", 0, {0, -7, -14, 0, 7, 14}},
                                  {"mul_int_int", 1, {0, -3, -7, 0, 4, 7}},
                                  {"mul_int16_t_int", 1, {0, -3, -7, 0, 4, 7}}};
    for (const auto &test : cases) {
        SCOPED_TRACE(test.shader + " control=" + std::to_string(test.control));
        const auto &[words, size] = compute::precompiledSpirvModules.at(test.shader);
        auto output = makeTestTensor(device, outputShape);
        const bool narrowInputs = test.shader == "mul_int16_t_int";
        const SpecializedTensorPipeline::DescriptorMap descriptors{
            {{0, {output}}}, {{0, {narrowInputs ? left16 : left}}}, {{0, {narrowInputs ? right16 : right}}}};
        SpecializedTensorPipeline pipeline(device, descriptors, {words, words + size});
        pipeline.dispatch(descriptors, &specialization, {test.control});
        expectTensorValues(*output, outputShape, test.expected);
    }
}

TEST_F(TensorLoweringForVulkan, DirectRescaleScalarAndPerChannelParameters) {
    const ScopedEnvironment environment("VMEL_EXPERIMENTAL_TENSOR_LOWERING_MODE", "1");
    ensureDevice();
    auto input = makeTestTensor<int32_t>(device, Shape{vk::Format::eR32Uint, {2, 3}, {32, 8}}, {-2, 5, 4, 8, 1, 2});
    const Shape outputShape{vk::Format::eR8Uint, {2, 3}, {12, 2}};
    auto output = makeTestTensor(device, outputShape);
    auto scalarMultiplier = makeTestTensor<int32_t>(device, Shape{vk::Format::eR32Uint, {1}, {8}}, {1 << 30});
    auto scalarShift = makeTestTensor<uint8_t>(device, Shape{vk::Format::eR8Uint, {1}, {3}}, {32});
    auto channelMultipliers =
        makeTestTensor<int32_t>(device, Shape{vk::Format::eR32Uint, {3}, {8}}, {1 << 30, 1 << 29, 1 << 30});
    auto channelShifts = makeTestTensor<uint8_t>(device, Shape{vk::Format::eR8Uint, {3}, {3}}, {31, 31, 32});
    const SpecializedTensorPipeline::DescriptorMap initial{
        {{0, {output}}}, {{0, {input}}}, {{0, {scalarMultiplier}}}, {{0, {scalarShift}}}};
    const auto &[words, size] = compute::precompiledSpirvModules.at("rescale_int_int8_t_int");
    SpecializedTensorPipeline pipeline(device, initial, {words, words + size});
    struct Case {
        uint32_t doubleRound;
        uint32_t perChannel;
        std::vector<int8_t> expected;
    };
    // With input/output zero points 3/-2, these values distinguish both rounding
    // modes and reads of each channel's multiplier and shift from padded tensors.
    const std::vector<Case> cases{{0, 0, {-3, -1, -2, -1, -2, -2}},
                                  {1, 0, {-3, -1, -1, 0, -3, -2}},
                                  {0, 1, {-4, -1, -2, 1, -2, -2}},
                                  {1, 1, {-4, -1, -1, 1, -2, -2}}};
    for (const auto &test : cases) {
        SCOPED_TRACE(std::to_string(test.doubleRound) + "/" + std::to_string(test.perChannel));
        std::memset(output->data(), 0xa5, output->size());
        const TensorSpecialization specialization{{0, 2}, {1, 1}, {2, test.doubleRound}, {3, test.perChannel}};
        const SpecializedTensorPipeline::DescriptorMap descriptors{
            {{0, {output}}},
            {{0, {input}}},
            {{0, {test.perChannel ? channelMultipliers : scalarMultiplier}}},
            {{0, {test.perChannel ? channelShifts : scalarShift}}}};
        pipeline.dispatch(descriptors, &specialization, {3, static_cast<uint32_t>(-2)});
        expectTensorValues(*output, outputShape, test.expected);
    }
}

TEST_F(TensorLoweringForVulkan, DirectPadUsesFlatParametersOnMultipleAxes) {
    const ScopedEnvironment environment("VMEL_EXPERIMENTAL_TENSOR_LOWERING_MODE", "1");
    ensureDevice();
    auto input = makeTestTensor<uint32_t>(device, Shape{vk::Format::eR32Uint, {2, 3}, {40, 8}}, {1, 2, 3, 4, 5, 6});
    // Flat rank-1 tensor: before/after pairs for each input axis.
    auto padding = makeTestTensor<uint32_t>(device, Shape{vk::Format::eR32Uint, {4}, {8}}, {1, 1, 2, 1});
    const Shape outputShape{vk::Format::eR32Uint, {4, 6}, {64, 8}};
    auto output = makeTestTensor(device, outputShape);
    const SpecializedTensorPipeline::DescriptorMap descriptors{{{0, {output}}}, {{0, {input}}}, {{0, {padding}}}};
    const auto &[words, size] = compute::precompiledSpirvModules.at("pad_int");
    SpecializedTensorPipeline pipeline(device, descriptors, {words, words + size});
    const TensorSpecialization specialization{{0, 2}};
    // PAD's integer constant follows its real_t constant in the shader ABI.
    std::vector<uint32_t> constants((sizeof(real_t) / sizeof(uint32_t)) + 1, 0);
    constants.back() = 99;
    pipeline.dispatch(descriptors, &specialization, constants);
    expectTensorValues<uint32_t>(*output, outputShape, {99, 99, 99, 99, 99, 99, 99, 99, 1,  2,  3,  99,
                                                        99, 99, 4,  5,  6,  99, 99, 99, 99, 99, 99, 99});
}

TEST_F(TensorLoweringForVulkan, DirectTableReadsGlobalTensorFromHelper) {
    const ScopedEnvironment environment("VMEL_EXPERIMENTAL_TENSOR_LOWERING_MODE", "1");
    ensureDevice();
    auto input =
        makeTestTensor<int8_t>(device, Shape{vk::Format::eR8Uint, {2, 3}, {12, 2}}, {-128, -127, -1, 0, 126, 127});
    std::vector<int8_t> entries(256);
    for (size_t i = 0; i < entries.size(); ++i) {
        entries[i] = static_cast<int8_t>(127 - int(i));
    }
    auto table = makeTestTensor(device, Shape{vk::Format::eR8Uint, {256}, {2}}, entries);
    const Shape outputShape{vk::Format::eR8Uint, {2, 3}, {16, 3}};
    auto output = makeTestTensor(device, outputShape);
    const SpecializedTensorPipeline::DescriptorMap descriptors{{{0, {output}}}, {{0, {input}}}, {{0, {table}}}};
    const auto &[words, size] = compute::precompiledSpirvModules.at("table_int8_t_int8_t");
    SpecializedTensorPipeline pipeline(device, descriptors, {words, words + size});
    const TensorSpecialization specialization{{0, 2}};
    pipeline.dispatch(descriptors, &specialization);
    expectTensorValues<int8_t>(*output, outputShape, {127, 126, 0, -1, -127, -128});
}

TEST_F(TensorLoweringForVulkan, DirectReductionRestoresCoordinatesAfterLoopExits) {
    const ScopedEnvironment environment("VMEL_EXPERIMENTAL_TENSOR_LOWERING_MODE", "1");
    ensureDevice();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    auto input = makeTestTensor<float>(device, Shape{vk::Format::eR32Sfloat, {2, 3, 2}, {80, 24, 8}},
                                       {nan, 3, 2, nan, 1, 5, 7, 8, nan, -4, 6, nan});
    const Shape outputShape{vk::Format::eR32Sfloat, {2, 1, 2}, {40, 16, 8}};
    auto output = makeTestTensor(device, outputShape);
    const SpecializedTensorPipeline::DescriptorMap descriptors{{{0, {output}}}, {{0, {input}}}};
    const TensorSpecialization specialization{{0, 3}};
    for (const auto &[shader, expected] : std::vector<std::pair<std::string, std::vector<float>>>{
             {"reduce_REDUCE_MIN_float", {1, 3, 6, -4}}, {"reduce_REDUCE_MAX_float", {2, 5, 7, 8}}}) {
        const auto &[words, size] = compute::precompiledSpirvModules.at(shader);
        SpecializedTensorPipeline pipeline(device, descriptors, {words, words + size});
        // IGNORE (2) continues past NaNs; PROPAGATE (1) breaks on the first NaN.
        // Both must restore axis 1 before writing to the rank-3 output tensor.
        for (const uint32_t nanMode : {2u, 1u}) {
            SCOPED_TRACE(shader + " nanMode=" + std::to_string(nanMode));
            std::memset(output->data(), 0xa5, output->size());
            pipeline.dispatch(descriptors, &specialization, {1, nanMode});
            expectTensorValues(*output, outputShape, nanMode == 2 ? expected : std::vector<float>(4, nan));
        }
    }
}

TEST_F(TensorLoweringForVulkan, DirectArgmaxUsesDifferentInputAndOutputRanks) {
    const ScopedEnvironment environment("VMEL_EXPERIMENTAL_TENSOR_LOWERING_MODE", "1");
    ensureDevice();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    auto input = makeTestTensor<float>(device, Shape{vk::Format::eR32Sfloat, {2, 3, 2}, {80, 24, 8}},
                                       {nan, 3, 2, nan, 1, 5, 7, 8, nan, -4, 6, nan});
    const Shape outputShape{vk::Format::eR32Uint, {2, 2}, {32, 8}};
    auto output = makeTestTensor(device, outputShape);
    const SpecializedTensorPipeline::DescriptorMap descriptors{{{0, {output}}}, {{0, {input}}}};
    const auto &[words, size] = compute::precompiledSpirvModules.at("argmax_float");
    SpecializedTensorPipeline pipeline(device, descriptors, {words, words + size});
    const TensorSpecialization specialization{{0, 3}, {1, 2}};
    for (const auto &[nanMode, expected] :
         std::vector<std::pair<uint32_t, std::vector<uint32_t>>>{{2, {1, 2, 0, 0}}, {1, {0, 1, 1, 2}}}) {
        SCOPED_TRACE(nanMode);
        std::memset(output->data(), 0xa5, output->size());
        pipeline.dispatch(descriptors, &specialization, {1, nanMode});
        expectTensorValues(*output, outputShape, expected);
    }
}

TEST_F(TensorLoweringForVulkan, DirectArrayBoundsStridesAndSpecializedRanks) {
    const ScopedEnvironment environment("VMEL_EXPERIMENTAL_TENSOR_LOWERING_MODE", "1");
    ensureDevice();
    for (const uint32_t length : {1u, 2u, 4u}) {
        for (const bool specialized : {false, true}) {
            TensorShaderSource source;
            source.tensors = {{"source", "uint", "RANK", 0, 0},
                              {"destination", "uint", "RANK", 0, 1},
                              {"results", "uint", "1", 0, 2}};
            source.declarations =
                std::string(specialized ? "layout(constant_id = 0) const uint RANK = 2;" : "const uint RANK = 2;") +
                "\nconst uint LENGTH = " + std::to_string(length) + R"(;
layout(push_constant) uniform Parameters { uint start; uint outer; } params;
)";
            source.body = R"(
    uint coords[RANK];
    for (uint axis = 0; axis < RANK; ++axis) coords[axis] = 0;
    coords[0] = params.outer;
    coords[RANK - 1] = params.start;
    // Keep the accesses inside caller control flow as well as the helper guards.
    for (uint pass = 0; pass < 2; ++pass) {
        if (pass == 0) {
            uint values[LENGTH];
            tensorReadARM(source, coords, values);
            tensorWriteARM(results, uint[](0), values);
        } else {
            uint values[LENGTH];
            for (uint i = 0; i < LENGTH; ++i) values[i] = 101 + i;
            tensorWriteARM(destination, coords, values);
        }
    }
)";
            const auto shader = compileTensorShader(source);
            for (const uint32_t rank : specialized ? std::vector<uint32_t>{1, 2, 6} : std::vector<uint32_t>{2}) {
                SCOPED_TRACE(std::to_string(length) + " elements, rank " + std::to_string(rank) +
                             (specialized ? " specialized" : " literal"));
                std::vector<int64_t> dimensions(rank, 1);
                std::vector<int64_t> inputStrides(rank, 32);
                std::vector<int64_t> outputStrides(rank, 48);
                if (rank > 1) {
                    dimensions.front() = 2;
                    inputStrides.front() = 64;
                    outputStrides.front() = 96;
                }
                dimensions.back() = 3;
                inputStrides.back() = 8;
                outputStrides.back() = 12;
                const Shape inputShape{vk::Format::eR32Uint, dimensions, inputStrides};
                const Shape outputShape{vk::Format::eR32Uint, dimensions, outputStrides};
                const Shape resultShape{vk::Format::eR32Uint, {length}, {8}};
                const std::vector<uint32_t> inputValues =
                    rank == 1 ? std::vector<uint32_t>{11, 12, 13} : std::vector<uint32_t>{11, 12, 13, 21, 22, 23};
                auto input = makeTestTensor(device, inputShape, inputValues);
                auto output = makeTestTensor(device, outputShape);
                auto results = makeTestTensor(device, resultShape);
                const SpecializedTensorPipeline::DescriptorMap descriptors{
                    {{0, {input}}, {1, {output}}, {2, {results}}}};
                SpecializedTensorPipeline pipeline(device, descriptors, shader);
                const TensorSpecialization specialization{{0, rank}};
                for (const auto &[start, outer] : std::vector<std::pair<uint32_t, uint32_t>>{
                         {0, 0}, {1, 0}, {2, 0}, {3, 0}, {0xffffffffu, 0}, {1, 1}, {0, 2}}) {
                    if (rank == 1 && outer != 0) {
                        continue;
                    }
                    SCOPED_TRACE(std::to_string(outer) + "," + std::to_string(start));
                    std::memset(output->data(), 0xa5, output->size());
                    std::memset(results->data(), 0xa5, results->size());
                    pipeline.dispatch(descriptors, specialized ? &specialization : nullptr, {start, outer});
                    std::vector<uint32_t> expectedRead(length, 0);
                    std::vector<uint32_t> expectedWrite(inputValues.size(), 0xa5a5a5a5);
                    for (uint32_t i = 0; i < length; ++i) {
                        const uint64_t x = uint64_t(start) + i;
                        if (outer < 2 && x < 3) {
                            const auto index = (uint64_t(outer) * 3) + x;
                            expectedRead[i] = inputValues[index];
                            expectedWrite[index] = 101 + i;
                        }
                    }
                    expectTensorValues(*results, resultShape, expectedRead);
                    expectTensorValues(*output, outputShape, expectedWrite);
                }
                expectTensorValues(*input, inputShape, inputValues);
            }
        }
    }
}

TEST_F(TensorLoweringForVulkan, DirectMatmulArrayLhsScalarRhsAndTail) {
    const ScopedEnvironment environment("VMEL_EXPERIMENTAL_TENSOR_LOWERING_MODE", "1");
    ensureDevice();
    auto left = makeTestTensor<int8_t>(device, Shape{vk::Format::eR8Uint, {1, 2, 5}, {128, 40, 3}},
                                       {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto right = makeTestTensor<int8_t>(device, Shape{vk::Format::eR8Uint, {1, 5, 3}, {160, 24, 2}},
                                        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    const Shape outputShape{vk::Format::eR32Uint, {1, 2, 3}, {128, 40, 8}};
    auto output = makeTestTensor(device, outputShape);
    const SpecializedTensorPipeline::DescriptorMap descriptors{{{0, {output}}}, {{0, {left}}}, {{0, {right}}}};
    const auto &[words, size] = compute::precompiledSpirvModules.at("matmul_int8_t_int");
    SpecializedTensorPipeline pipeline(device, descriptors, {words, words + size});
    pipeline.dispatch(descriptors, nullptr, {1, 2});
    expectTensorValues<int32_t>(*output, outputShape, {80, 90, 100, 205, 240, 275});
}

TEST_F(TensorLoweringForVulkan, DirectConv2DArrayChannelTailsAndBiasForms) {
    const ScopedEnvironment environment("VMEL_EXPERIMENTAL_TENSOR_LOWERING_MODE", "1");
    ensureDevice();
    auto input =
        makeTestTensor<int8_t>(device, Shape{vk::Format::eR8Uint, {1, 1, 1, 5}, {64, 32, 16, 2}}, {1, 2, 3, 4, 5});
    auto weights = makeTestTensor<int8_t>(device, Shape{vk::Format::eR8Uint, {5, 1, 1, 5}, {64, 32, 32, 3}},
                                          {1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5});
    auto scalarBias = makeTestTensor<int32_t>(device, Shape{vk::Format::eR32Uint, {1}, {8}}, {7});
    auto channelBias =
        makeTestTensor<int32_t>(device, Shape{vk::Format::eR32Uint, {5}, {8}}, {100, 101, 102, 103, 104});
    const Shape outputShape{vk::Format::eR32Uint, {1, 1, 1, 5}, {96, 64, 48, 8}};
    auto output = makeTestTensor(device, outputShape);
    const auto &[words, size] = compute::precompiledSpirvModules.at("conv2d_int8_t_int8_t_int_int");
    for (const auto &[bias, expected] : std::vector<std::pair<std::shared_ptr<Tensor>, std::vector<int32_t>>>{
             {scalarBias, {27, 37, 47, 57, 67}}, {channelBias, {120, 131, 142, 153, 164}}}) {
        std::memset(output->data(), 0xa5, output->size());
        const SpecializedTensorPipeline::DescriptorMap descriptors{
            {{0, {output}}}, {{0, {input}}}, {{0, {weights}}}, {{0, {bias}}}};
        SpecializedTensorPipeline pipeline(device, descriptors, {words, words + size});
        // One channel group per dispatch, including the partial group at channel 4.
        // Push constants: zero points, padding, stride, dilation, and output-position offset.
        for (const uint32_t channel : {0u, 4u}) {
            pipeline.dispatch(descriptors, nullptr,
                              {1, static_cast<uint32_t>(-1), 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, channel});
        }
        expectTensorValues(*output, outputShape, expected);
    }
}

TEST_F(TensorLoweringForVulkan, DirectTableInterpolatesTwoElementReads) {
    const ScopedEnvironment environment("VMEL_EXPERIMENTAL_TENSOR_LOWERING_MODE", "1");
    ensureDevice();
    auto input = makeTestTensor<int16_t>(device, Shape{vk::Format::eR16Uint, {2, 3}, {32, 4}},
                                         {-32768, -32767, -1, 0, 32640, 32767});
    std::vector<int16_t> entries(513);
    for (size_t i = 0; i < entries.size(); ++i) {
        entries[i] = static_cast<int16_t>(int(i) - 256);
    }
    auto table = makeTestTensor(device, Shape{vk::Format::eR16Uint, {513}, {6}}, entries);
    const Shape outputShape{vk::Format::eR32Uint, {2, 3}, {40, 8}};
    auto output = makeTestTensor(device, outputShape);
    const SpecializedTensorPipeline::DescriptorMap descriptors{{{0, {output}}}, {{0, {input}}}, {{0, {table}}}};
    const auto &[words, size] = compute::precompiledSpirvModules.at("table_int16_t_int");
    SpecializedTensorPipeline pipeline(device, descriptors, {words, words + size});
    const TensorSpecialization specialization{{0, 2}};
    pipeline.dispatch(descriptors, &specialization);
    expectTensorValues<int32_t>(*output, outputShape, {-32768, -32767, -1, 0, 32640, 32767});
}

} // namespace
} // namespace mlsdk::el::tests
