/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

#include "mlel/tensor.hpp"
#include "vulkan_test_utils.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace mlsdk::el::tests {

struct TensorShaderBinding {
    std::string name;
    std::string elementType;
    // GLSL expression: either a literal rank or a specialization constant expression.
    std::string rank;
    uint32_t set;
    uint32_t binding;
};

struct TensorShaderSource {
    std::vector<TensorShaderBinding> tensors;
    std::string declarations;
    std::string body;
    std::string extensions;
};

// Extra declarations precede the bindings so they can define rank constants.
// The shader body stays explicit in each test; only boilerplate is generated.
inline std::vector<uint32_t> compileTensorShader(const TensorShaderSource &shader) {
    std::string source = "#version 460\n#extension GL_ARM_tensors : require\n" + shader.extensions +
                         "\nlayout(local_size_x = 1) in;\n" + shader.declarations + "\n";
    for (const auto &tensor : shader.tensors) {
        source += "layout(set = " + std::to_string(tensor.set) + ", binding = " + std::to_string(tensor.binding) +
                  ") uniform tensorARM<" + tensor.elementType + ", " + tensor.rank + "> " + tensor.name + ";\n";
    }
    return compileGlsl(source + "\nvoid main() {\n" + shader.body + "\n}\n");
}

// Own both the 32-bit values and their ID-to-offset mapping. Construct the Vulkan
// view on demand, so copying or moving this helper cannot leave stale pointers.
class TensorSpecialization {
  public:
    TensorSpecialization(std::initializer_list<std::pair<uint32_t, uint32_t>> constants) {
        for (const auto &[id, value] : constants) {
            entries.emplace_back(id, uint32_t(values.size() * sizeof(uint32_t)), sizeof(uint32_t));
            values.push_back(value);
        }
    }

    vk::SpecializationInfo info() const {
        return {uint32_t(entries.size()), entries.data(), values.size() * sizeof(uint32_t), values.data()};
    }

  private:
    std::vector<vk::SpecializationMapEntry> entries;
    std::vector<uint32_t> values;
};

inline std::shared_ptr<utilities::Tensor> makeTestTensor(const std::shared_ptr<utilities::Device> &device,
                                                         const utilities::Shape &shape, uint8_t fill = 0xa5) {
    auto tensor = std::make_shared<utilities::Tensor>(device, shape);
    std::memset(tensor->data(), fill, tensor->size());
    return tensor;
}

template <typename T>
std::shared_ptr<utilities::Tensor> makeTestTensor(const std::shared_ptr<utilities::Device> &device,
                                                  const utilities::Shape &shape, const std::vector<T> &values) {
    if (shape.elementCount() != values.size() || shape.getFormatSize() != sizeof(T)) {
        throw std::runtime_error("Tensor fixture size mismatch");
    }
    auto tensor = makeTestTensor(device, shape);
    for (size_t i = 0; i < values.size(); ++i) {
        std::memcpy(tensor->data() + shape.getElementOffset(i), &values[i], sizeof(T));
    }
    return tensor;
}

// Padding checks are opt-in: tests of ignored out-of-bounds writes need them,
// while tests concerned only with logical tensor values can omit them.
template <typename T>
void expectTensorValues(const utilities::Tensor &tensor, const utilities::Shape &shape, const std::vector<T> &values,
                        std::optional<uint8_t> padding = std::nullopt) {
    ASSERT_EQ(shape.elementCount(), values.size());
    ASSERT_EQ(shape.getFormatSize(), sizeof(T));
    ASSERT_EQ(tensor.size(), shape.getSize());
    std::vector<uint8_t> bytes(tensor.data(), tensor.data() + tensor.size());
    for (size_t i = 0; i < values.size(); ++i) {
        SCOPED_TRACE(i);
        const auto offset = shape.getElementOffset(i);
        T value;
        std::memcpy(&value, bytes.data() + offset, sizeof(T));
        if constexpr (std::is_floating_point_v<T>) {
            if (std::isnan(values[i])) {
                EXPECT_TRUE(std::isnan(value));
            } else {
                EXPECT_EQ(value, values[i]);
            }
        } else {
            EXPECT_EQ(value, values[i]);
        }
        if (padding) {
            std::memset(bytes.data() + offset, *padding, sizeof(T));
        }
    }
    if (padding) {
        EXPECT_TRUE(std::all_of(bytes.begin(), bytes.end(), [&](uint8_t byte) { return byte == *padding; }));
    }
}

} // namespace mlsdk::el::tests
