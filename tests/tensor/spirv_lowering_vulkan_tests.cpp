/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include "mlel/tensor.hpp"
#include "mlel/utils.hpp"
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

std::vector<uint32_t> tensorShader() {
    return compileGlsl(R"(
#version 460
#extension GL_ARM_tensors : require
layout(local_size_x = 1) in;
layout(set = 0, binding = 0) uniform tensorARM<uint, 1> data;
void main() {
    tensorWriteARM(data, uint[](0), tensorSizeARM(data, 0));
}
)");
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
    const auto shader = compileGlsl(R"(
#version 460
#extension GL_ARM_tensors : require
layout(local_size_x = 1) in;
layout(constant_id = 0) const uint rank = 1;
layout(set = 0, binding = 0) uniform tensorARM<uint, rank> data;
void main() { uint value; tensorReadARM(data, uint[](0), value); }
)");
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
    const auto shader = compileGlsl(R"(
#version 460
#extension GL_ARM_tensors : require
layout(local_size_x = 1) in;
layout(set = 0, binding = 0) uniform tensorARM<uint, 2> source;
layout(set = 0, binding = 1) uniform tensorARM<uint, 2> destination;
layout(set = 0, binding = 2) uniform tensorARM<uint, 1> results;
void main() {
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
}
)");
    for (const auto &[height, width, rowStride, elementStride] :
         std::vector<std::array<int64_t, 4>>{{2, 3, 32, 4}, {3, 2, 40, 8}, {2, 4, 16, 4}}) {
        SCOPED_TRACE(testing::PrintToString(std::array{height, width, rowStride, elementStride}));
        const utilities::Shape shape{vk::Format::eR32Uint, {height, width}, {rowStride, elementStride}};
        auto input = std::make_shared<utilities::Tensor>(device, shape);
        auto output = std::make_shared<utilities::Tensor>(device, shape);
        auto results = std::make_shared<utilities::Tensor>(device, utilities::Shape{vk::Format::eR32Uint, {6}});
        std::vector<uint32_t> inputData(input->size() / sizeof(uint32_t), 0xdeadbeef);
        std::vector<uint32_t> expectedOutput(output->size() / sizeof(uint32_t), 0xa5a5a5a5);
        for (int64_t y = 0; y < height; ++y) {
            for (int64_t x = 0; x < width; ++x) {
                const auto index = static_cast<size_t>((y * rowStride) + (x * elementStride)) / sizeof(uint32_t);
                inputData[index] = static_cast<uint32_t>((100 * (y + 1)) + x);
                expectedOutput[index] = inputData[index];
            }
        }
        std::memcpy(input->data(), inputData.data(), input->size());
        std::memset(output->data(), 0xa5, output->size());
        std::memset(results->data(), 0xa5, results->size());
        const utilities::TensorComputePipeline::DescriptorMap descriptors{
            {{0, {input}}, {1, {output}}, {2, {results}}}};
        utilities::TensorComputePipeline pipeline(device, descriptors, shader);
        pipeline.dispatchSubmit(1, 1, 1);
        const std::array<uint32_t, 6> expectedResults{
            static_cast<uint32_t>(height), static_cast<uint32_t>(width), 0, 0, 0, 0};
        EXPECT_EQ(std::memcmp(results->data(), expectedResults.data(), results->size()), 0);
        EXPECT_EQ(std::memcmp(output->data(), expectedOutput.data(), output->size()), 0);
    }
}

class TensorRuntimeBoundsForVulkan : public TensorLoweringForVulkan, public testing::WithParamInterface<const char *> {
  protected:
    void checkReads(const utilities::Shape &shape,
                    const std::vector<std::pair<std::vector<uint32_t>, uint32_t>> &cases) {
        const ScopedEnvironment environment("VMEL_EXPERIMENTAL_TENSOR_LOWERING_MODE", GetParam());
        ensureDevice();
        const auto rank = shape.getDimensions().size();
        std::string readCoordinates;
        std::string coordinateValues;
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
        const auto shader =
            compileGlsl("#version 460\n#extension GL_ARM_tensors : require\n"
                        "layout(local_size_x = 1) in;\n"
                        "layout(set = 0, binding = 0) uniform tensorARM<uint, " +
                        std::to_string(rank) +
                        "> source;\n"
                        "layout(set = 0, binding = 1) uniform tensorARM<uint, 1> coordinates;\n"
                        "layout(set = 0, binding = 2) uniform tensorARM<uint, 1> result;\n"
                        "void main() {\n" +
                        readCoordinates + "uint value; tensorReadARM(source, uint[](" + coordinateValues +
                        "), value);\ntensorWriteARM(result, uint[](0), value);\n}\n");
        auto input = std::make_shared<utilities::Tensor>(device, shape);
        auto coordinates = std::make_shared<utilities::Tensor>(
            device, utilities::Shape{vk::Format::eR32Uint, {static_cast<int64_t>(rank)}});
        auto result = std::make_shared<utilities::Tensor>(device, utilities::Shape{vk::Format::eR32Uint, {1}});
        std::vector<uint32_t> inputData(input->size() / sizeof(uint32_t), 0xdeadbeef);
        for (size_t i = 0; i < shape.elementCount(); ++i) {
            inputData[shape.getElementOffset(i) / sizeof(uint32_t)] = 100 + static_cast<uint32_t>(i);
        }
        std::memcpy(input->data(), inputData.data(), input->size());
        const utilities::TensorComputePipeline::DescriptorMap descriptors{
            {{0, {input}}, {1, {coordinates}}, {2, {result}}}};
        utilities::TensorComputePipeline pipeline(device, descriptors, shader);
        for (const auto &[coords, expected] : cases) {
            SCOPED_TRACE(testing::PrintToString(coords));
            ASSERT_EQ(coords.size(), rank);
            // Supply coordinates after pipeline compilation so invalid accesses cannot be folded away.
            std::memcpy(coordinates->data(), coords.data(), coordinates->size());
            std::memset(result->data(), 0xa5, result->size());
            pipeline.dispatchSubmit(1, 1, 1);
            uint32_t actual = 0;
            std::memcpy(&actual, result->data(), sizeof(actual));
            EXPECT_EQ(actual, expected);
        }
    }
};

TEST_P(TensorRuntimeBoundsForVulkan, RankOne) {
    checkReads(utilities::Shape{vk::Format::eR32Uint, {3}},
               {{{0}, 100}, {{2}, 102}, {{3}, 0}, {{std::numeric_limits<uint32_t>::max()}, 0}});
}

TEST_P(TensorRuntimeBoundsForVulkan, RankTwoWithPaddedRows) {
    const auto invalid = std::numeric_limits<uint32_t>::max();
    checkReads(utilities::Shape{vk::Format::eR32Uint, {2, 3}, {32, 4}}, {{{0, 0}, 100},
                                                                         {{1, 2}, 105},
                                                                         {{0, 3}, 0},
                                                                         {{1, 3}, 0},
                                                                         {{2, 0}, 0},
                                                                         {{invalid, 0}, 0},
                                                                         {{0, invalid}, 0},
                                                                         {{1, invalid}, 0},
                                                                         {{invalid, invalid}, 0}});
}

INSTANTIATE_TEST_SUITE_P(LoweringModes, TensorRuntimeBoundsForVulkan, testing::Values("0", "1"),
                         [](const testing::TestParamInfo<const char *> &info) {
                             return std::string(info.param) == "0" ? "GlslToSpirv" : "DirectSpirv";
                         });

} // namespace
} // namespace mlsdk::el::tests
