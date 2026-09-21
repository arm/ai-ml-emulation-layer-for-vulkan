/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

#include "mlel/device.hpp"
#include "mlel/pipeline.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mlsdk::el::tests {

using utilities::Device;
using utilities::GraphPipeline;

std::string fileToString(const std::string &filename);
// Test modules must pass SPIR-V validation for the Vulkan target environment.
std::vector<uint32_t> assembleSpirv(const std::string &text);
std::vector<uint32_t> compileGlsl(const std::string &text);
std::shared_ptr<Device> createDevice();

struct ProfilingGraph {
    GraphPipeline::DescriptorMap descriptorMap;
    std::shared_ptr<GraphPipeline> pipeline;
};

ProfilingGraph makeMaxPoolProfilingGraph(std::shared_ptr<Device> &device);
void submitGraphWithoutFence(const std::shared_ptr<Device> &device, const ProfilingGraph &graph, bool waitDevice);

// Keep the fixture type shared across translation units so each GoogleTest
// suite has one device even when its test cases live in different source files.
template <typename Base> class GraphTestWithDevice : public Base {
  protected:
    static void SetUpTestSuite() { device = createDevice(); }
    static void TearDownTestSuite() { device.reset(); }
    inline static std::shared_ptr<Device> device;
};

using MLEmulationLayerGraphForVulkan = GraphTestWithDevice<testing::Test>;
template <typename Param> using GraphTestWithParam = GraphTestWithDevice<testing::TestWithParam<Param>>;

} // namespace mlsdk::el::tests
