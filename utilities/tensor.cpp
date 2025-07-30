/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "mlel/tensor.hpp"

#include "mlel/exception.hpp"

#include <iostream>
#include <numeric>
#include <vulkan/vulkan_format_traits.hpp>

namespace mlsdk::el::utilities {

/*******************************************************************************
 * Shape
 *******************************************************************************/

Shape::Shape(const vk::Format _format, const std::vector<int64_t> &_dimensions, const std::vector<int64_t> &_strides)
    : format{_format}, dimensions{_dimensions}, strides{createStrides(_strides)} {}

const std::vector<int64_t> &Shape::getDimensions() const { return dimensions; }

const std::vector<int64_t> &Shape::getStrides() const { return strides; };

size_t Shape::getSize() const { return dimensions[0] * strides[0]; }

size_t Shape::elementCount() const {
    return static_cast<size_t>(
        std::abs(std::accumulate(dimensions.begin(), dimensions.end(), int64_t(1), std::multiplies<int64_t>())));
}

size_t Shape::getElementOffset(size_t index) const {
    std::vector<size_t> coordinates(dimensions.size());
    size_t byteOffset = 0;

    for (int i = dimensions.size() - 1; i >= 0; i--) {
        coordinates[i] = index % dimensions[i];
        index /= dimensions[i];
        byteOffset += coordinates[i] * strides[i];
    }

    return byteOffset;
}

std::vector<int64_t> Shape::createStrides(const std::vector<int64_t> &_strides) const {
    if (!_strides.empty()) {
        if (_strides.size() == dimensions.size()) {
            return _strides;
        } else {
            throw std::runtime_error("Stride size should be equal to dimension size.");
        }
    }
    std::vector<int64_t> strides;
    strides.resize(dimensions.size());

    if (strides.size() > 0) {
        strides.back() = getFormatSize();
        for (size_t i = strides.size() - 1; i > 0; i--) {
            strides[i - 1] = strides[i] * dimensions[i];
        }
    }

    return strides;
}

size_t Shape::getFormatSize() const { return vk::blockSize(format); }

vk::Format Shape::getFormat() const { return format; }

/*******************************************************************************
 * Tensor
 *******************************************************************************/

Tensor::Tensor(const std::shared_ptr<Device> &_device, const Shape &_shape, const std::vector<uint8_t> &_data)
    : Tensor(_device, _shape, _data.data(), _data.size()) {}

Tensor::Tensor(const std::shared_ptr<Device> &_device, const Shape &_shape, const uint8_t *_pointer, const size_t _size)
    : shape{_shape}, physicalDevice{_device->getPhysicalDevice()}, device{_device}, tensor{createTensor()},
      memoryRequirements{getTensorMemoryRequirements()}, deviceMemory{allocateTensorMemory()},
      pointer{bindAndMapTensor()}, tensorView{createTensorView()}, tensorDescription{createTensorDescription()} {
    setData(_pointer, _size);
}

size_t Tensor::size() const { return shape.getSize(); }

size_t Tensor::getElementOffset(size_t index) const { return shape.getElementOffset(index); }

uint8_t *Tensor::data() const { return reinterpret_cast<uint8_t *>(pointer); }

vk::Format Tensor::getFormat() const { return shape.getFormat(); }

const std::vector<int64_t> &Tensor::getDimensions() const { return shape.getDimensions(); }

void Tensor::setData(const uint8_t *_pointer, const size_t _size) {
    VK_ASSERT_GE(size(), _size, "Tensor data is larger than tensor size");

    std::copy(_pointer, _pointer + _size, data());
    std::fill(data() + _size, data() + size(), 0);
}

void Tensor::clear() { memset(data(), 0, size()); }

const vk::TensorDescriptionARM &Tensor::getTensorDescription() const { return tensorDescription; }

void Tensor::updateDescriptorSet(const vk::raii::DescriptorSet &descriptorSet, uint32_t binding,
                                 uint32_t arrayIndex) const {
    const vk::WriteDescriptorSetTensorARM writeDescriptorSetTensor{
        1,              // tensor view count
        &(*tensorView), // tensor views
    };

    const vk::WriteDescriptorSet writeDescriptorSet{
        *descriptorSet,                 // descriptor set
        binding,                        // binding
        arrayIndex,                     // dst array element
        1,                              // descriptor count
        vk::DescriptorType::eTensorARM, // descriptor type
        nullptr,                        // image info
        nullptr,                        // buffer info
        nullptr,                        // texel buffer view
        &writeDescriptorSetTensor       // next, tensor info
    };
    (&(*device)).updateDescriptorSets({1, &writeDescriptorSet}, {0, nullptr});
}

void Tensor::print() const {
    const auto *p = data();

    if (std::all_of(p, p + size(), [](const auto value) { return value == uint8_t(0); })) {
        std::cout << "All zeros, size: " << size() << std::endl;
        return;
    }

    std::ios_base::fmtflags coutFlags(std::cout.flags());

    std::cout << std::hex << std::setfill('0');

    for (uint32_t i = 0; i < size(); i++) {
        if ((i % 16) == 0) {
            std::cout << std::endl << std::setw(8) << i << ": ";
        }

        std::cout << std::setw(2) << static_cast<unsigned>(p[i]) << " ";
    }

    std::cout << std::endl;
    std::cout.flags(coutFlags);
}

vk::raii::TensorARM Tensor::createTensor() const {
    const vk::TensorDescriptionARM tensorDescription{
        vk::TensorTilingARM::eLinear,           // tiling
        shape.getFormat(),                      // format
        uint32_t(shape.getDimensions().size()), // dimensions count
        shape.getDimensions().data(),           // dimensions
        shape.getStrides().data(),              // strides
        {},                                     // usage flags
    };

    const vk::TensorCreateInfoARM tensorCreateInfo{
        {},                          // flags
        &tensorDescription,          // tensor description
        vk::SharingMode::eExclusive, // sharing mode
        0,                           // queue family index count
    };

    return {&(*device), tensorCreateInfo};
}

std::shared_ptr<vk::raii::DeviceMemory> Tensor::allocateTensorMemory() const {
    // TODO Pass requirements.memoryRequirements.memoryTypeBits
    return device->allocateDeviceMemory(memoryRequirements.size, vk::MemoryPropertyFlagBits::eHostVisible);
}

vk::MemoryRequirements Tensor::getTensorMemoryRequirements() const {
    const vk::TensorMemoryRequirementsInfoARM requirementsInfo{
        *tensor,
    };
    vk::MemoryRequirements2 memoryRequirements = (&(*device)).getTensorMemoryRequirementsARM(requirementsInfo);

    return memoryRequirements.memoryRequirements;
}

void *Tensor::bindAndMapTensor() const {
    const vk::BindTensorMemoryInfoARM bindInfo{
        *tensor,        // tensor
        **deviceMemory, // device memory
        {}              // memory offset
    };
    (&(*device)).bindTensorMemoryARM(bindInfo);
    void *pointer = deviceMemory->mapMemory({}, VK_WHOLE_SIZE, {});

    return pointer;
}

vk::raii::TensorViewARM Tensor::createTensorView() const {
    const vk::TensorViewCreateInfoARM tensorViewCreateInfo{
        {},                // flags
        *tensor,           // tensor
        shape.getFormat(), // format
    };
    vk::raii::TensorViewARM tensorView{&(*device), tensorViewCreateInfo};

    return tensorView;
}

vk::TensorDescriptionARM Tensor::createTensorDescription() const {
    vk::TensorDescriptionARM tensorDescription{
        vk::TensorTilingARM::eLinear,                        // tiling
        shape.getFormat(),                                   // format
        static_cast<uint32_t>(shape.getDimensions().size()), // dimension count
        shape.getDimensions().data(),                        // dimensions
        nullptr,                                             // strides
        vk::TensorUsageFlagBitsARM::eDataGraph               // usage
    };

    return tensorDescription;
}

} // namespace mlsdk::el::utilities
