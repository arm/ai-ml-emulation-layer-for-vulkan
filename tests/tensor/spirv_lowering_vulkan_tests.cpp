/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include "mlel/utils.hpp"
#include "test_utils.hpp"
#include "vulkan_test_utils.hpp"

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

TEST_F(TensorLoweringForVulkan, DirectRejectsTensorModule) {
    const ScopedEnvironment environment("VMEL_EXPERIMENTAL_TENSOR_LOWERING_MODE", "1");
    EXPECT_EQ(createModule(tensorShader()), VK_ERROR_UNKNOWN);
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

TEST_F(TensorLoweringForVulkan, DirectRejectsInlineTensorPipeline) {
    const ScopedEnvironment environment("VMEL_EXPERIMENTAL_TENSOR_LOWERING_MODE", "1");
    ensureDevice();
    if (!inlineSupported) {
        GTEST_SKIP() << "Inline shader modules require maintenance5";
    }
    EXPECT_EQ(createPipeline(tensorShader()), VK_ERROR_UNKNOWN);
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

} // namespace
} // namespace mlsdk::el::tests
