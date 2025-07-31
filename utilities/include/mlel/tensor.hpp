/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "mlel/device.hpp"
#include "mlel/float.hpp"

#include <vulkan/vulkan.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

namespace mlsdk::el::utilities {

/*******************************************************************************
 * Shape
 *******************************************************************************/

class Shape {
  public:
    Shape(const vk::Format _format, const std::vector<int64_t> &_dimensions, const std::vector<int64_t> &_strides = {});

    const std::vector<int64_t> &getDimensions() const;
    const std::vector<int64_t> &getStrides() const;
    size_t getSize() const;
    size_t getFormatSize() const;
    vk::Format getFormat() const;
    size_t elementCount() const;
    size_t getElementOffset(size_t index) const;

  private:
    std::vector<int64_t> createStrides(const std::vector<int64_t> &_strides) const;
    vk::Format format;
    std::vector<int64_t> dimensions;
    std::vector<int64_t> strides;
};

/*******************************************************************************
 * Tensor
 *******************************************************************************/

class Tensor {
  public:
    // Set useForCopy to make this tensor having the buffer bit set in BufferInfo struct
    Tensor(const std::shared_ptr<Device> &_device, const Shape &shape, const std::vector<uint8_t> &data = {},
           bool useForCopy = false);
    Tensor(const std::shared_ptr<Device> &_device, const Shape &shape, const uint8_t *pointer, const size_t size,
           bool useForCopy = false);

    vk::TensorARM const &operator&() const { return *tensor; }

    template <typename T> bool compare(const T *ref, const size_t size) const {
        switch (shape.getFormat()) {
        case vk::Format::eR8Snorm:
        case vk::Format::eR8Sscaled:
        case vk::Format::eR8Sint:
        case vk::Format::eR8Srgb:
            return compare(reinterpret_cast<const int8_t *>(pointer), ref, size);
        case vk::Format::eR8Unorm:
        case vk::Format::eR8Uscaled:
        case vk::Format::eR8Uint:
        case vk::Format::eR8BoolARM:
            return compare(reinterpret_cast<const uint8_t *>(pointer), ref, size);
        case vk::Format::eR16Snorm:
        case vk::Format::eR16Sscaled:
        case vk::Format::eR16Sint:
            return compare(reinterpret_cast<const int16_t *>(pointer), ref, size);
        case vk::Format::eR16Unorm:
        case vk::Format::eR16Uscaled:
        case vk::Format::eR16Uint:
            return compare(reinterpret_cast<const uint16_t *>(pointer), ref, size);
        case vk::Format::eR16Sfloat:
            return compare<float16>(reinterpret_cast<const float16 *>(pointer), ref, size, 1.0e-03);
        case vk::Format::eR32Sint:
            return compare(reinterpret_cast<const int32_t *>(pointer), ref, size);
        case vk::Format::eR32Uint:
            return compare(reinterpret_cast<const uint32_t *>(pointer), ref, size);
        case vk::Format::eR32Sfloat:
            return compare(reinterpret_cast<const float *>(pointer), ref, size, 1.0e-06);
        case vk::Format::eR64Sint:
            return compare(reinterpret_cast<const int64_t *>(pointer), ref, size);
        case vk::Format::eR64Uint:
            return compare(reinterpret_cast<const uint64_t *>(pointer), ref, size);
        case vk::Format::eR64Sfloat:
            return compare(reinterpret_cast<const double *>(pointer), ref, size, 1.0e-06);
        default:
            throw std::runtime_error(std::string("Unsupported format for compare: ") +
                                     vk::to_string(shape.getFormat()));
        }
    }

    template <typename T, typename U> bool stridedCompare(const Tensor &ref, const double tolerance = 0) const {
        for (size_t idx = 0; idx < shape.elementCount(); ++idx) {
            auto srcOffsetBytes = getElementOffset(idx);
            auto refOffsetBytes = ref.getElementOffset(idx);
            auto srcPtr = reinterpret_cast<T *>(data() + srcOffsetBytes);
            auto refPtr = reinterpret_cast<U *>(ref.data() + refOffsetBytes);
            if (!isClose(*srcPtr, *refPtr, tolerance)) {
                std::cout << "Output mismatch at position " << std::dec << idx << std::endl;
                std::cout << "src=" << +(*srcPtr) << ", ref=" << +(*refPtr) << ", diff=" << fabs(*srcPtr - *refPtr)
                          << std::endl;
                return false;
            }
        }
        return true;
    }
    size_t size() const;
    size_t getElementOffset(size_t index) const;
    uint8_t *data() const;
    void setData(const uint8_t *pointer, const size_t size);
    void clear();
    vk::Format getFormat() const;
    const std::vector<int64_t> &getDimensions() const;
    const vk::TensorDescriptionARM &getTensorDescription() const;
    void updateDescriptorSet(const vk::raii::DescriptorSet &descriptorSet, uint32_t binding, uint32_t arrayIndex) const;
    void print() const;

  private:
    static bool isClose(const double a, const double b, const double tolerance) {
        if (std::isnan(a) && std::isnan(b)) {
            return true;
        }

        if (std::isinf(a) && std::isinf(b)) {
            return true;
        }

        return fabs(a - b) <= tolerance * fmax(1.0, fmax(fabs(a), fabs(b)));
    }

    template <typename T, typename U>
    bool compare(const T *src, const U *ref, const size_t size, const double tolerance = 0) const {
        for (size_t i = 0; i < (size / sizeof(U)); i++) {
            if (!isClose(double(src[i]), double(ref[i]), tolerance)) {
                std::cout << "Output mismatch at position " << std::dec << i << ", byte offset " << i * sizeof(T)
                          << std::endl;
                std::cout << "src=" << +src[i] << ", ref=" << +ref[i]
                          << ", diff=" << fabs(static_cast<double>(src[i]) - static_cast<double>(ref[i])) << std::endl;
                return false;
            }
        }

        return true;
    }

    vk::raii::TensorARM createTensor(bool useForCopy) const;
    std::shared_ptr<vk::raii::DeviceMemory> allocateTensorMemory() const;
    vk::MemoryRequirements getTensorMemoryRequirements() const;
    void *bindAndMapTensor() const;
    vk::raii::TensorViewARM createTensorView() const;
    vk::TensorDescriptionARM createTensorDescription() const;

    const Shape shape;
    std::shared_ptr<PhysicalDevice> physicalDevice;
    std::shared_ptr<Device> device;
    vk::raii::TensorARM tensor;
    vk::MemoryRequirements memoryRequirements;
    std::shared_ptr<vk::raii::DeviceMemory> deviceMemory;
    void *pointer;
    vk::raii::TensorViewARM tensorView;
    vk::TensorDescriptionARM tensorDescription;
};

} // namespace mlsdk::el::utilities
