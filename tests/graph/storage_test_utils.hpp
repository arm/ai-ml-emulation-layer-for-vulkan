/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

#include "mlel/tensor.hpp"
#include "vulkan_test_utils.hpp"

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlsdk::el::tests {

using utilities::Shape;
using utilities::Tensor;

std::vector<uint32_t> makeStorageConversionGraph(const std::string &inputType, const std::string &outputType,
                                                 uint32_t inputRank, uint32_t outputRank, bool binary,
                                                 const std::string &operation, const std::string &constants = "");

template <typename T>
std::shared_ptr<Tensor> storageTestTensor(std::shared_ptr<Device> &device, vk::Format format,
                                          const std::vector<int64_t> &shape, const std::vector<T> &values) {
    auto tensor = std::make_shared<Tensor>(device, Shape{format, shape});
    if (tensor->size() != values.size() * sizeof(T)) {
        throw std::runtime_error("Storage test tensor size mismatch");
    }
    std::memcpy(tensor->data(), values.data(), tensor->size());
    return tensor;
}

} // namespace mlsdk::el::tests
