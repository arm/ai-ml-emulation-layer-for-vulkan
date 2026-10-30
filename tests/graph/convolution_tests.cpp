/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include "compute_graph_op.hpp"
#include "mlel/pipeline.hpp"
#include "mlel/tensor.hpp"
#include "mlel/utils.hpp"
#include "vulkan_test_utils.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

namespace mlsdk::el::tests {

using namespace utilities;

namespace {

struct FloatConvCase {
    int64_t batches, inputChannels, outputChannels, kernel;
    std::array<int32_t, 4> pad;
    std::array<int32_t, 2> stride, dilation;
    bool scalarBias;
    bool cooperative;
    int64_t height = 7, width = 9;
};

// Use the real selector and shaders with small dispatch limits to cover every offset axis.
void dispatchFloatConv(const std::shared_ptr<Device> &device, const std::array<std::shared_ptr<Tensor>, 4> &tensors,
                       const FloatConvCase &test, bool fallback = false) {
    namespace graph = compute::graph_op;
    const auto &vkDevice = &(*device);
    const auto &physical = &(*device->getPhysicalDevice());
    const auto &instance = &(*device->getPhysicalDevice()->getInstance());
    auto loader =
        std::make_shared<vk::detail::DispatchLoaderDynamic>(*instance, instance.getDispatcher()->vkGetInstanceProcAddr,
                                                            *vkDevice, vkDevice.getDispatcher()->vkGetDeviceProcAddr);
    auto cache = std::make_shared<compute::PipelineCache>(nullptr, 0, VK_NULL_HANDLE);
    static thread_local const auto cooperativeCode = cache->lookup("conv2d_cooperative", {});
    static thread_local PFN_vkCreateShaderModule createShaderModule;
    static thread_local PFN_vkCmdDispatch cmdDispatch;
    static thread_local bool selected;
    static thread_local uint32_t dispatches;
    createShaderModule = loader->vkCreateShaderModule;
    cmdDispatch = loader->vkCmdDispatch;
    selected = false;
    dispatches = 0;
    loader->vkCreateShaderModule = [](VkDevice dev, const VkShaderModuleCreateInfo *info,
                                      const VkAllocationCallbacks *allocator, VkShaderModule *module) {
        selected = info->codeSize == cooperativeCode.size() * sizeof(uint32_t) &&
                   std::memcmp(info->pCode, cooperativeCode.data(), info->codeSize) == 0;
        return createShaderModule(dev, info, allocator, module);
    };
    loader->vkCmdDispatch = [](VkCommandBuffer command, uint32_t x, uint32_t y, uint32_t z) {
        ++dispatches;
        cmdDispatch(command, x, y, z);
    };
    loader->vkGetPhysicalDeviceProperties = [](VkPhysicalDevice, VkPhysicalDeviceProperties *properties) {
        *properties = {};
        std::fill_n(properties->limits.maxComputeWorkGroupCount, 3, 1u);
    };

    std::array<std::shared_ptr<compute::TensorDescriptor>, 4> descriptions;
    std::vector<vk::raii::DeviceMemory> memory;
    graph::TensorDescriptorMap bindings;
    for (size_t i = 0; i < tensors.size(); ++i) {
        descriptions[i] = std::make_shared<compute::TensorDescriptor>(loader, *physical, *vkDevice,
                                                                      tensors[i]->getTensorDescription());
        auto tensor = compute::TensorDescriptor::makeTensor(descriptions[i]);
        auto allocation = descriptions[i]->createInitializeDeviceMemory(tensors[i]->data());
        memory.emplace_back(vkDevice, allocation);
        tensor->bindTensorMemory(allocation, 0);
        bindings.emplace(descriptions[i], std::move(tensor));
    }
    const std::vector<int32_t> pad(test.pad.begin(), test.pad.end());
    const std::vector<int32_t> stride(test.stride.begin(), test.stride.end());
    const std::vector<int32_t> dilation(test.dilation.begin(), test.dilation.end());
    graph::GraphPipeline graphPipeline(loader, *physical, *vkDevice, cache);
    std::shared_ptr<graph::ComputePipelineBase> kernel;
    if (fallback) {
        kernel = std::make_shared<graph::Conv2D>(loader, *vkDevice, cache, descriptions[0], descriptions[3],
                                                 descriptions[1], descriptions[2], pad, stride, dilation, 0, 0, 3,
                                                 std::array<uint32_t, 3>{1, 1, 1}, false, "reference");
    } else {
        graphPipeline.makeConv2D(descriptions[0], descriptions[3], descriptions[1], descriptions[2], pad, stride,
                                 dilation, 0, 0, 3, "test");
        ASSERT_EQ(graphPipeline.getPipelines().size(), 1u);
        kernel = graphPipeline.getPipelines().front();
    }
    EXPECT_EQ(selected, !fallback && test.cooperative);
    graph::ComputeDescriptorSetMap sets;
    kernel->getComputePipelineLayout()->makeDescriptorSets(sets, bindings);
    for (const auto &[_, set] : sets) {
        set->updateDescriptorSet();
    }
    vk::raii::CommandPool pool(vkDevice, {{}, device->getPhysicalDevice()->getComputeFamilyIndex()});
    vk::raii::CommandBuffers commands(vkDevice, {*pool, vk::CommandBufferLevel::ePrimary, 1});
    auto &command = commands.front();
    command.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    kernel->cmdBindAndDispatch(*command, sets);
    const vk::MemoryBarrier hostRead{vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eHostRead};
    command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eHost, {}, hostRead,
                            {}, {});
    command.end();
    EXPECT_GT(dispatches, 1u);
    vk::raii::Queue queue(vkDevice, device->getPhysicalDevice()->getComputeFamilyIndex(), 0);
    const vk::CommandBuffer rawCommand = *command;
    queue.submit(vk::SubmitInfo{{}, {}, rawCommand}, nullptr);
    queue.waitIdle();
    const auto *result = memory.back().mapMemory(0, VK_WHOLE_SIZE);
    std::memcpy(tensors[3]->data(), result, tensors[3]->size());
    memory.back().unmapMemory();
}

void expectFloatConv(std::shared_ptr<Device> &device, const FloatConvCase &test) {
    const auto height = test.height, width = test.width;
    const auto outputHeight =
        (height + test.pad[0] + test.pad[1] - (test.kernel - 1) * test.dilation[0] - 1) / test.stride[0] + 1;
    const auto outputWidth =
        (width + test.pad[2] + test.pad[3] - (test.kernel - 1) * test.dilation[1] - 1) / test.stride[1] + 1;
    const auto input = std::make_shared<Tensor>(
        device, Shape{vk::Format::eR32Sfloat, {test.batches, height, width, test.inputChannels}});
    const auto weights = std::make_shared<Tensor>(
        device, Shape{vk::Format::eR32Sfloat, {test.outputChannels, test.kernel, test.kernel, test.inputChannels}});
    const auto bias =
        std::make_shared<Tensor>(device, Shape{vk::Format::eR32Sfloat, {test.scalarBias ? 1 : test.outputChannels}});
    const auto output = std::make_shared<Tensor>(
        device, Shape{vk::Format::eR32Sfloat, {test.batches, outputHeight, outputWidth, test.outputChannels}});
    auto fill = [](const auto &tensor, int multiplier, int modulus) {
        std::vector<float> values(tensor->size() / sizeof(float));
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = float((i * multiplier) % modulus) / 8.0f - 1.0f;
        }
        std::memcpy(tensor->data(), values.data(), tensor->size());
        return values;
    };
    const auto inputs = fill(input, 7, 17);
    const auto weightValues = fill(weights, 5, 19);
    const auto biases = fill(bias, 3, 11);
    std::memset(output->data(), 0xff, output->size());
    auto source = fileToString("conv2d_fp32.spvasm");
    const std::vector<std::pair<std::string, int32_t>> attributes{
        {"pad_top", test.pad[0]},         {"pad_bottom", test.pad[1]},     {"pad_left", test.pad[2]},
        {"pad_right", test.pad[3]},       {"stride_y", test.stride[0]},    {"stride_x", test.stride[1]},
        {"dilation_y", test.dilation[0]}, {"dilation_x", test.dilation[1]}};
    for (const auto &[name, value] : attributes) {
        const auto marker = "@" + name + "@";
        source.replace(source.find(marker), marker.size(), std::to_string(value));
    }
    const GraphPipeline::DescriptorMap descriptors{{{0, {input}}, {1, {weights}}, {2, {bias}}, {3, {output}}}};
    const GraphConstants constants;
    GraphPipeline pipeline(device, descriptors, constants, assembleSpirv(source));
    pipeline.dispatchSubmit();

    std::vector<float> expected;
    for (int64_t n = 0; n < test.batches; ++n) {
        for (int64_t oy = 0; oy < outputHeight; ++oy) {
            for (int64_t ox = 0; ox < outputWidth; ++ox) {
                for (int64_t oc = 0; oc < test.outputChannels; ++oc) {
                    float sum = 0;
                    for (int64_t ky = 0; ky < test.kernel; ++ky) {
                        for (int64_t kx = 0; kx < test.kernel; ++kx) {
                            const auto y = oy * test.stride[0] - test.pad[0] + ky * test.dilation[0];
                            const auto x = ox * test.stride[1] - test.pad[2] + kx * test.dilation[1];
                            if (y < 0 || y >= height || x < 0 || x >= width) {
                                continue;
                            }
                            for (int64_t ic = 0; ic < test.inputChannels; ++ic) {
                                const auto in = ((n * height + y) * width + x) * test.inputChannels + ic;
                                const auto w = ((oc * test.kernel + ky) * test.kernel + kx) * test.inputChannels + ic;
                                sum += inputs[in] * weightValues[w];
                            }
                        }
                    }
                    expected.push_back(sum + biases[test.scalarBias ? 0 : oc]);
                }
            }
        }
    }
    // Dyadic inputs keep every product and sum exact, including channel tails.
    ASSERT_EQ(std::memcmp(output->data(), expected.data(), output->size()), 0);
    std::memset(output->data(), 0xff, output->size());
    ASSERT_NO_FATAL_FAILURE(dispatchFloatConv(device, {input, weights, bias, output}, test));
    ASSERT_EQ(std::memcmp(output->data(), expected.data(), output->size()), 0);
}

TEST_F(MLEmulationLayerGraphForVulkan, Conv2DPointwiseSharedWeights) {
    for (const auto &test :
         std::vector<FloatConvCase>{{1, 16, 16, 1, {0, 0, 0, 0}, {1, 1}, {1, 1}, false, true},
                                    {2, 17, 19, 1, {1, 2, 2, 1}, {2, 1}, {1, 1}, true, false},
                                    {1, 33, 35, 1, {0, 0, 0, 0}, {1, 1}, {2, 3}, false, true},
                                    {1, 3, 5, 1, {0, 0, 0, 0}, {1, 1}, {1, 1}, true, false},
                                    {1, 17, 19, 2, {0, 0, 0, 0}, {1, 1}, {1, 1}, false, false},
                                    {1, 256, 16, 1, {64, 64, 64, 64}, {1, 1}, {1, 1}, false, false, 1, 1}}) {
        SCOPED_TRACE(testing::Message() << test.inputChannels << "x" << test.outputChannels << " k=" << test.kernel);
        ASSERT_NO_FATAL_FAILURE(expectFloatConv(device, test));
    }
}

TEST_F(MLEmulationLayerGraphForVulkan, Conv2DThreeByThreeSharedWeights) {
    for (const auto &test :
         std::vector<FloatConvCase>{{1, 16, 16, 3, {1, 1, 1, 1}, {1, 1}, {1, 1}, false, true},
                                    {2, 17, 19, 3, {2, 4, 1, 3}, {2, 1}, {2, 2}, true, true},
                                    {1, 33, 35, 3, {0, 0, 0, 0}, {1, 1}, {1, 1}, false, true},
                                    {1, 17, 19, 3, {5, 0, 4, 1}, {1, 1}, {1, 1}, false, false},
                                    {1, 3, 5, 3, {0, 0, 0, 0}, {1, 1}, {1, 1}, true, false},
                                    {1, 17, 19, 5, {2, 2, 2, 2}, {1, 1}, {1, 1}, false, false},
                                    {1, 256, 16, 3, {64, 64, 64, 64}, {1, 1}, {1, 1}, false, false, 1, 1}}) {
        SCOPED_TRACE(testing::Message() << test.inputChannels << "x" << test.outputChannels << " k=" << test.kernel);
        ASSERT_NO_FATAL_FAILURE(expectFloatConv(device, test));
    }
}

void expectConvRounding(const std::shared_ptr<Device> &device, int64_t kernel) {
    const FloatConvCase test{1, 17, 19, kernel, {0, 0, 0, 0}, {1, 1}, {1, 1}, false, true};
    auto input = std::make_shared<Tensor>(device, Shape{vk::Format::eR32Sfloat, {1, 7, 9, 17}});
    auto weights = std::make_shared<Tensor>(device, Shape{vk::Format::eR32Sfloat, {19, kernel, kernel, 17}});
    auto bias = std::make_shared<Tensor>(device, Shape{vk::Format::eR32Sfloat, {19}});
    auto output = std::make_shared<Tensor>(device, Shape{vk::Format::eR32Sfloat, {1, 8 - kernel, 10 - kernel, 19}});
    auto *values = reinterpret_cast<float *>(input->data());
    std::fill_n(values, input->size() / sizeof(float), 0.0f);
    for (size_t i = 0; i < input->size() / sizeof(float); i += 17) {
        // (2^24 + 1) - 2^24 is zero; regrouping to 2^24 + (1 - 2^24) gives one.
        values[i] = 16777216.0f;
        values[i + 1] = 1.0f;
        values[i + 2] = -16777216.0f;
    }
    std::fill_n(reinterpret_cast<float *>(weights->data()), weights->size() / sizeof(float), 1.0f);
    std::memset(bias->data(), 0, bias->size());
    std::memset(output->data(), 0xff, output->size());
    ASSERT_NO_FATAL_FAILURE(dispatchFloatConv(device, {input, weights, bias, output}, test, true));
    std::vector<float> reference(output->size() / sizeof(float));
    ASSERT_EQ(std::memcmp(output->data(), reference.data(), output->size()), 0);
    std::memset(output->data(), 0xff, output->size());
    ASSERT_NO_FATAL_FAILURE(dispatchFloatConv(device, {input, weights, bias, output}, test));
    EXPECT_EQ(std::memcmp(output->data(), reference.data(), output->size()), 0);
}

TEST_F(MLEmulationLayerGraphForVulkan, Conv2DPointwisePreservesRoundingOrder) { expectConvRounding(device, 1); }

TEST_F(MLEmulationLayerGraphForVulkan, Conv2DThreeByThreePreservesRoundingOrder) { expectConvRounding(device, 3); }

void expectConv2DStorageFormats(std::shared_ptr<Device> &device, vk::Format inputFormat, vk::Format weightFormat,
                                vk::Format outputFormat, bool negateWeights = false) {

    auto inputTensor = std::make_shared<Tensor>(device, Shape{inputFormat, std::vector<int64_t>{1, 8, 8, 3}});
    std::iota(inputTensor->data(), inputTensor->data() + inputTensor->size(), uint8_t{});

    auto weightTensor = std::make_shared<Tensor>(device, Shape{weightFormat, std::vector<int64_t>{3, 2, 2, 3}});
    std::fill(weightTensor->data(), weightTensor->data() + weightTensor->size(),
              negateWeights ? uint8_t{255} : uint8_t{1});

    auto biasTensor = std::make_shared<Tensor>(device, Shape{outputFormat, std::vector<int64_t>{3}});
    for (size_t i = 0; i < biasTensor->size(); i++) {
        if ((i % 4) == 0) {
            *(biasTensor->data() + i) = uint8_t(i / 4);
        } else {
            *(biasTensor->data() + i) = 0;
        }
    }

    auto outputTensor = std::make_shared<Tensor>(device, Shape{outputFormat, std::vector<int64_t>{1, 4, 4, 3}});
    const GraphPipeline::DescriptorMap descriptorMap = {
        {
            // set 0
            {
                0,             // binding
                {inputTensor}, // tensor
            },
            {
                1,              // binding
                {weightTensor}, // tensor
            },
            {
                2,            // binding
                {biasTensor}, // tensor
            },
            {
                3,              // binding
                {outputTensor}, // tensor
            },
        },
    };

    GraphConstants graphConstants;

    const auto spirv = assembleSpirv(fileToString("conv2d.spvasm"));
    auto graphPipeline = std::make_shared<GraphPipeline>(device, descriptorMap, graphConstants, spirv);

    graphPipeline->dispatchSubmit();

    std::cout << "INPUT" << std::endl;
    inputTensor->print();

    std::cout << "OUTPUT" << std::endl;
    outputTensor->print();

    const uint32_t ref[1][4][4][3] = {
        // batch
        {
            // y=0
            {
                //      c=0         c=1         c=2        b  y  x  c
                {0x000000ae, 0x000000af, 0x000000b0}, // (0, 0, 0, c)
                {0x000000f6, 0x000000f7, 0x000000f8}, // (0, 0, 1, c)
                {0x0000013e, 0x0000013f, 0x00000140}, // (0, 0, 2, c)
                {0x00000186, 0x00000187, 0x00000188}, // (0, 0, 3, c)
            },
            // y=1
            {
                {0x000002ee, 0x000002ef, 0x000002f0}, // (0, 1, 0, c)
                {0x00000336, 0x00000337, 0x00000338}, // (0, 1, 1, c)
                {0x0000037e, 0x0000037f, 0x00000380}, // (0, 1, 2, c)
                {0x000003c6, 0x000003c7, 0x000003c8}, // (0, 1, 3, c)
            },
            // y=2
            {
                {0x0000052e, 0x0000052f, 0x00000530}, // (0, 2, 0, c)
                {0x00000176, 0x00000177, 0x00000178}, // (0, 2, 1, c)
                {0xffffffbe, 0xffffffbf, 0xffffffc0}, // (0, 2, 2, c)
                {0x00000006, 0x00000007, 0x00000008}, // (0, 2, 3, c)
            },
            // y=3
            {
                {0xfffffb6e, 0xfffffb6f, 0xfffffb70}, // (0, 3, 0, c)
                {0xfffffbb6, 0xfffffbb7, 0xfffffbb8}, // (0, 3, 1, c)
                {0xfffffbfe, 0xfffffbff, 0xfffffc00}, // (0, 3, 2, c)
                {0xfffffc46, 0xfffffc47, 0xfffffc48}, // (0, 3, 3, c)
            },
        },
    };

    std::array<int32_t, 48> expected{};
    std::memcpy(expected.data(), &ref, sizeof(ref));
    if (negateWeights) {
        for (size_t i = 0; i < expected.size(); ++i) {
            expected[i] = -expected[i] + 2 * static_cast<int32_t>(i % 3);
        }
    }
    // Compare the encoded result so negative TOSA values also match R32_UINT storage.
    ASSERT_EQ(std::memcmp(outputTensor->data(), expected.data(), sizeof(expected)), 0) << "Output mismatch";
}

TEST_F(MLEmulationLayerGraphForVulkan, Conv2D) {
    expectConv2DStorageFormats(device, vk::Format::eR8Sint, vk::Format::eR8Sint, vk::Format::eR32Sint);
}

using Conv2DStorageFormats = GraphTestWithParam<std::tuple<vk::Format, vk::Format, vk::Format, bool>>;

TEST_P(Conv2DStorageFormats, PreservesSignedTosaValues) {
    const auto [inputFormat, weightFormat, outputFormat, negateWeights] = GetParam();
    expectConv2DStorageFormats(device, inputFormat, weightFormat, outputFormat, negateWeights);
}

INSTANTIATE_TEST_SUITE_P(ConvolutionRegression, Conv2DStorageFormats,
                         testing::Combine(testing::Values(vk::Format::eR8Sint, vk::Format::eR8Uint),
                                          testing::Values(vk::Format::eR8Sint, vk::Format::eR8Uint),
                                          testing::Values(vk::Format::eR32Sint, vk::Format::eR32Uint), testing::Bool()),
                         [](const auto &info) {
                             const auto &params = info.param;
                             return "Input" + vk::to_string(std::get<0>(params)) + "_Weight" +
                                    vk::to_string(std::get<1>(params)) + "_Output" +
                                    vk::to_string(std::get<2>(params)) +
                                    (std::get<3>(params) ? "_NegativeWeights" : "_PositiveWeights");
                         });

TEST_F(MLEmulationLayerGraphForVulkan, Conv2DDispatchesBeyondZWorkgroupLimit) {
    constexpr int64_t outputChannels = 262144;

    auto inputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, {1, 1, 1, 4}});
    std::fill(inputTensor->data(), inputTensor->data() + inputTensor->size(), 1);

    auto weightTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, {outputChannels, 1, 1, 4}});
    std::fill(weightTensor->data(), weightTensor->data() + weightTensor->size(), 1);

    auto biasTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR32Sint, {outputChannels}});
    std::fill(biasTensor->data(), biasTensor->data() + biasTensor->size(), 0);

    auto outputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR32Sint, {1, 1, 1, outputChannels}});
    const GraphPipeline::DescriptorMap descriptorMap = {{
        {0, {inputTensor}},
        {1, {weightTensor}},
        {2, {biasTensor}},
        {3, {outputTensor}},
    }};

    const auto spirv = assembleSpirv(fileToString("conv2d_large_output_channels.spvasm"));
    auto graphPipeline = std::make_shared<GraphPipeline>(device, descriptorMap, GraphConstants{}, spirv);
    graphPipeline->dispatchSubmit();

    const std::vector<int32_t> expected(outputChannels, 4);
    ASSERT_TRUE(outputTensor->compare(expected.data(), expected.size() * sizeof(expected[0]))) << "Output mismatch";
}

class Conv2DLargeSpatialDimension : public testing::TestWithParam<std::tuple<std::string, int64_t, int64_t>> {};

TEST_P(Conv2DLargeSpatialDimension, DispatchesBeyondWorkgroupLimit) {
    constexpr int64_t channels = 4;
    const auto &[shaderFile, height, width] = GetParam();

    auto device = createDevice();

    auto inputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, {1, height, width, channels}});
    std::fill(inputTensor->data(), inputTensor->data() + inputTensor->size(), 1);

    auto weightTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, {channels, 1, 1, channels}});
    std::fill(weightTensor->data(), weightTensor->data() + weightTensor->size(), 1);

    auto biasTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR32Sint, {channels}});
    std::fill(biasTensor->data(), biasTensor->data() + biasTensor->size(), 0);

    auto outputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR32Sint, {1, height, width, channels}});
    const GraphPipeline::DescriptorMap descriptorMap = {{
        {0, {inputTensor}},
        {1, {weightTensor}},
        {2, {biasTensor}},
        {3, {outputTensor}},
    }};

    const auto spirv = assembleSpirv(fileToString(shaderFile));
    auto graphPipeline = std::make_shared<GraphPipeline>(device, descriptorMap, GraphConstants{}, spirv);
    graphPipeline->dispatchSubmit();

    const std::vector<int32_t> expected(static_cast<size_t>(height * width * channels), channels);
    ASSERT_TRUE(outputTensor->compare(expected.data(), expected.size() * sizeof(expected[0]))) << "Output mismatch";
}

INSTANTIATE_TEST_SUITE_P(MLEmulationLayerForVulkan, Conv2DLargeSpatialDimension,
                         testing::Values(std::make_tuple("conv2d_large_width.spvasm", 1, 524281),
                                         std::make_tuple("conv2d_large_height.spvasm", 524281, 1)),
                         [](const auto &info) {
                             return "Height" + std::to_string(std::get<1>(info.param)) + "_Width" +
                                    std::to_string(std::get<2>(info.param));
                         });

TEST_F(MLEmulationLayerGraphForVulkan, Conv2DAccumulatorInt64) {

    auto inputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR16Sint, {1, 2, 2, 4}});
    auto weightTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR8Sint, {4, 1, 1, 4}});
    auto biasTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR64Sint, {4}});
    auto outputTensor = std::make_shared<Tensor>(device, Shape{vk::Format::eR64Sint, {1, 2, 2, 4}});

    const std::array<int16_t, 16> inputValues = {
        1,  2,  3,  4,  //
        5,  6,  7,  8,  //
        9,  10, 11, 12, //
        13, 14, 15, 16,
    };
    const std::array<int8_t, 16> weightValues = {
        1,  2,  3, 4, //
        -1, 0,  1, 2, //
        2,  -3, 4, -5, -2, -1, 0, 1,
    };
    const std::array<int64_t, 4> biasValues = {10, -20, 30, -40};
    const std::array<int64_t, 16> expectedValues = {
        40,  -10, 18, -40, //
        80,  -2,  10, -48, //
        120, 6,   2,  -56, //
        160, 14,  -6, -64,
    };

    ASSERT_EQ(inputTensor->size(), inputValues.size() * sizeof(inputValues[0]));
    ASSERT_EQ(weightTensor->size(), weightValues.size() * sizeof(weightValues[0]));
    ASSERT_EQ(biasTensor->size(), biasValues.size() * sizeof(biasValues[0]));
    ASSERT_EQ(outputTensor->size(), expectedValues.size() * sizeof(expectedValues[0]));

    std::memcpy(inputTensor->data(), inputValues.data(), inputTensor->size());
    std::memcpy(weightTensor->data(), weightValues.data(), weightTensor->size());
    std::memcpy(biasTensor->data(), biasValues.data(), biasTensor->size());

    const GraphPipeline::DescriptorMap descriptorMap = {{
        {0, {inputTensor}},
        {1, {weightTensor}},
        {2, {biasTensor}},
        {3, {outputTensor}},
    }};

    GraphConstants graphConstants;

    const auto spirv = assembleSpirv(fileToString("conv2_acc_int64.spvasm"));
    auto graphPipeline = std::make_shared<GraphPipeline>(device, descriptorMap, graphConstants, spirv);
    graphPipeline->dispatchSubmit();

    ASSERT_TRUE(outputTensor->compare(expectedValues.data(), expectedValues.size() * sizeof(expectedValues[0])))
        << "Output mismatch";
}

TEST_F(MLEmulationLayerGraphForVulkan, Conv3DLargeStridePadRegression) {
    constexpr int64_t kBatch = 1;
    constexpr int64_t kInputDepth = 3;
    constexpr int64_t kInputHeight = 2;
    constexpr int64_t kInputWidth = 2;
    constexpr int64_t kInputChannels = 4;
    constexpr int64_t kOutputChannels = 3;
    constexpr int64_t kKernelDepth = 2;
    constexpr int64_t kKernelHeight = 2;
    constexpr int64_t kKernelWidth = 8192;
    constexpr int64_t kOutputDepth = 3;
    constexpr int64_t kOutputHeight = 2;
    constexpr int64_t kOutputWidth = 2;

    auto inputTensor = std::make_shared<Tensor>(
        device, Shape{vk::Format::eR8Sint,
                      std::vector<int64_t>{kBatch, kInputDepth, kInputHeight, kInputWidth, kInputChannels}});
    auto outputTensor = std::make_shared<Tensor>(
        device, Shape{vk::Format::eR32Sint,
                      std::vector<int64_t>{kBatch, kOutputDepth, kOutputHeight, kOutputWidth, kOutputChannels}});

    const GraphPipeline::DescriptorMap descriptorMap = {{
        {0, {inputTensor}},
        {1, {outputTensor}},
    }};

    const std::array<int8_t, 48> inputValues = {
        -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
        -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
        -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
    };
    const std::array<int32_t, 36> expectedValues = {
        0,    -548, 137, 0, 0, 0, 548,   -685, -411, 0, 0, 0, -1233, 274,  685, 0, 0, 0,
        -822, -411, 274, 0, 0, 0, -1233, 274,  685,  0, 0, 0, -822,  -411, 274, 0, 0, 0,
    };
    const std::array<int8_t, 96> relevantWeights = {
        1, -1, 2,  -1, 1,  0,  -1, 0,  1, 2, 2, -1, 2, 2, 0, 1,  1,  -2, -1, -2, 0,  2,  -2, 0,
        1, 1,  -2, 1,  0,  -2, 2,  -1, 1, 0, 0, 2,  2, 0, 0, -1, 0,  0,  -1, -1, -1, 0,  -1, -2,
        1, 1,  -2, 2,  -1, 1,  1,  -2, 0, 0, 1, 2,  1, 2, 0, -2, 0,  2,  1,  -2, 1,  -1, -1, -1,
        0, -2, 2,  -2, 0,  -1, 0,  -1, 0, 2, 0, -2, 2, 1, 1, 0,  -1, 0,  1,  -2, -2, 2,  1,  0,
    };
    std::vector<int8_t> weightValues(
        static_cast<size_t>(kOutputChannels * kKernelDepth * kKernelHeight * kKernelWidth * kInputChannels), 0);
    GraphConstants graphConstants;

    ASSERT_EQ(inputValues.size(), inputTensor->size());
    ASSERT_EQ(expectedValues.size(), outputTensor->size() / sizeof(expectedValues[0]));

    std::memcpy(inputTensor->data(), inputValues.data(), inputValues.size() * sizeof(inputValues[0]));

    const auto setWeight = [&](int64_t oc, int64_t kd, int64_t kh, int64_t kx, int64_t ic, int8_t value) {
        const auto flatIndex = static_cast<size_t>(
            (((((oc * kKernelDepth) + kd) * kKernelHeight + kh) * kKernelWidth + kx) * kInputChannels) + ic);
        weightValues[flatIndex] = value;
    };

    for (size_t oc = 0; oc < static_cast<size_t>(kOutputChannels); ++oc) {
        for (size_t kd = 0; kd < static_cast<size_t>(kKernelDepth); ++kd) {
            for (size_t kh = 0; kh < static_cast<size_t>(kKernelHeight); ++kh) {
                const size_t blockBase =
                    ((oc * static_cast<size_t>(kKernelDepth) + kd) * static_cast<size_t>(kKernelHeight) + kh) * 8;
                for (size_t ic = 0; ic < static_cast<size_t>(kInputChannels); ++ic) {
                    setWeight(static_cast<int64_t>(oc), static_cast<int64_t>(kd), static_cast<int64_t>(kh), 8190,
                              static_cast<int64_t>(ic), relevantWeights[blockBase + ic]);
                    setWeight(static_cast<int64_t>(oc), static_cast<int64_t>(kd), static_cast<int64_t>(kh), 8191,
                              static_cast<int64_t>(ic), relevantWeights[blockBase + 4 + ic]);
                }
            }
        }
    }

    graphConstants.makeGraphPipelineConstantTensor(
        0, Shape{vk::Format::eR8Sint, {kOutputChannels, kKernelDepth, kKernelHeight, kKernelWidth, kInputChannels}},
        weightValues);

    const auto spirv = assembleSpirv(fileToString("conv3d_large_stride.spvasm"));
    auto graphPipeline = std::make_shared<GraphPipeline>(device, descriptorMap, graphConstants, spirv);
    graphPipeline->dispatchSubmit();

    ASSERT_TRUE(outputTensor->compare(expectedValues.data(), expectedValues.size() * sizeof(expectedValues[0])))
        << "Output mismatch";
}

void expectConv3DInlineEncodedConstantPipelineCreates(std::shared_ptr<Device> &device, const std::string &shaderFile,
                                                      vk::Format format, vk::Format outputFormat) {

    auto inputTensor = std::make_shared<Tensor>(device, Shape{format, std::vector<int64_t>{1, 1, 1, 1, 1}});
    auto outputTensor = std::make_shared<Tensor>(device, Shape{outputFormat, std::vector<int64_t>{1, 1, 1, 1, 1}});

    if (format == vk::Format::eR16SfloatFpencodingBfloat16ARM) {
        const uint16_t inputValue = 0x3f80;
        std::memcpy(inputTensor->data(), &inputValue, sizeof(inputValue));
    } else {
        *inputTensor->data() = format == vk::Format::eR8SfloatFpencodingFloat8E4M3ARM ? 0x38 : 0x3c;
    }

    const GraphPipeline::DescriptorMap descriptorMap = {{
        {0, {inputTensor}},
        {1, {outputTensor}},
    }};

    const auto spirv = assembleSpirv(fileToString(shaderFile));
    auto graphPipeline = std::make_shared<GraphPipeline>(device, descriptorMap, GraphConstants{}, spirv);
    ASSERT_NE(graphPipeline, nullptr);
    ASSERT_NO_THROW(graphPipeline->dispatchSubmit());
    uint16_t actual;
    std::memcpy(&actual, outputTensor->data(), sizeof(actual));
    const uint16_t expected = 0x4000;
    EXPECT_EQ(actual, expected); // 1 * 1 + 1 = 2, exact in FP16 and BF16.
}

TEST_F(MLEmulationLayerGraphForVulkan, Conv3DInlineBFloat16ConstantRegression) {
    expectConv3DInlineEncodedConstantPipelineCreates(device, "conv3d_inline_bf16_constant.spvasm",
                                                     vk::Format::eR16SfloatFpencodingBfloat16ARM,
                                                     vk::Format::eR16SfloatFpencodingBfloat16ARM);
}

TEST_F(MLEmulationLayerGraphForVulkan, Conv3DInlineFloat8E5M2ConstantRegression) {
    expectConv3DInlineEncodedConstantPipelineCreates(device, "conv3d_inline_fp8e5m2_constant.spvasm",
                                                     vk::Format::eR8SfloatFpencodingFloat8E5M2ARM,
                                                     vk::Format::eR16Sfloat);
}

TEST_F(MLEmulationLayerGraphForVulkan, Conv3DInlineFloat8E4M3ConstantRegression) {
    expectConv3DInlineEncodedConstantPipelineCreates(device, "conv3d_inline_fp8e4m3_constant.spvasm",
                                                     vk::Format::eR8SfloatFpencodingFloat8E4M3ARM,
                                                     vk::Format::eR16Sfloat);
}

} // namespace
} // namespace mlsdk::el::tests
