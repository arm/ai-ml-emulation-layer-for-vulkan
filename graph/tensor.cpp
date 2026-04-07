/*
 * SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "tensor.hpp"

#include "graph_log.hpp"

#include <exception>
#include <numeric>
#include <vulkan/vulkan_format_traits.hpp>

using namespace mlsdk::el::log;

namespace mlsdk::el::compute {

/*******************************************************************************
 * Tensor
 *******************************************************************************/

Tensor::Tensor(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader, VkDevice _device,
               std::shared_ptr<TensorDescriptor> _tensorDescriptor, VkTensorARM _tensorARM,
               VkTensorViewARM _tensorViewARM)
    : loader{_loader}, device{_device}, tensorDescriptor{std::move(_tensorDescriptor)}, tensorARM{_tensorARM},
      tensorViewARM{_tensorViewARM} {}

Tensor::~Tensor() {
    if (tensorViewARM != nullptr) {
        loader->vkDestroyTensorViewARM(device, tensorViewARM, nullptr);
    }

    if (tensorARM != nullptr) {
        loader->vkDestroyTensorARM(device, tensorARM, nullptr);
    }
}

std::shared_ptr<TensorDescriptor> Tensor::getTensorDescriptor() const { return tensorDescriptor; }

VkTensorARM Tensor::getVkTensorARM() const { return tensorARM; }

VkTensorViewARM Tensor::getVkTensorViewARM() const { return tensorViewARM; }

VkDeviceSize Tensor::bindTensorMemory(VkDeviceMemory deviceMemory, VkDeviceSize offset) const {
    const VkBindTensorMemoryInfoARM bindInfo = {
        VK_STRUCTURE_TYPE_BIND_TENSOR_MEMORY_INFO_ARM, // type
        nullptr,                                       // next
        tensorARM,                                     // tensor
        deviceMemory,                                  // device memory
        offset,                                        // memory offset
    };

    if (loader->vkBindTensorMemoryARM(device, 1, &bindInfo) != VK_SUCCESS) {
        throw std::runtime_error("Failed to bind tensor to memory");
    }

    return tensorDescriptor->getMemoryRequirementsSize();
}

/*******************************************************************************
 * TensorDescriptor
 *******************************************************************************/

TensorDescriptor::TensorDescriptor(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader,
                                   VkPhysicalDevice _physicalDevice, VkDevice _device, const VkFormat _format,
                                   const std::vector<int64_t> &_dimensions, const std::vector<int64_t> &_strides)
    : loader{_loader}, physicalDevice{_physicalDevice}, device{_device}, format{_format},
      dimensions{_dimensions}, strides{_strides} {}

TensorDescriptor::TensorDescriptor(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader,
                                   VkPhysicalDevice _physicalDevice, VkDevice _device,
                                   const VkTensorDescriptionARM &_tensorDescription)
    : loader{_loader}, physicalDevice{_physicalDevice}, device{_device}, format{_tensorDescription.format},
      dimensions{_tensorDescription.pDimensions, _tensorDescription.pDimensions + _tensorDescription.dimensionCount},
      strides{createStrides(_tensorDescription)} {}

std::shared_ptr<Tensor> TensorDescriptor::makeTensor(const std::shared_ptr<TensorDescriptor> &_this) {
    const auto tensorDescription = _this->getTensorDescription();
    auto *tensorARM = _this->createTensorARM(tensorDescription);
    auto *tensorViewARM = _this->createTensorViewARM(tensorARM, _this->format);

    auto tensor = std::make_shared<Tensor>(_this->loader, _this->device, _this, tensorARM, tensorViewARM);

    graphLog(Severity::Debug) << "Create tensor. tensor=" << tensor << " " << *tensor << std::endl;

    return tensor;
}

VkDeviceMemory TensorDescriptor::createInitializeDeviceMemory(const void *data) {
    if (data == nullptr) {
        return VK_NULL_HANDLE;
    }

    const auto requirements = getMemoryRequirements();
    auto *deviceMemory = allocateDeviceMemory(requirements.size, requirements.memoryTypeBits);

    void *dst;
    auto ret = loader->vkMapMemory(device, deviceMemory, 0, VK_WHOLE_SIZE, {}, &dst);
    if (ret != VK_SUCCESS) {
        throw std::runtime_error("Failed to memory map memory for constant tensor");
    }

    std::copy(static_cast<const char *>(data), static_cast<const char *>(data) + getSize(), static_cast<char *>(dst));

    loader->vkUnmapMemory(device, deviceMemory);

    return deviceMemory;
}

VkDeviceMemory TensorDescriptor::allocateDeviceMemory(const size_t size, const uint32_t memoryTypeBits) const {
    const auto memoryTypeIndices = getMemoryTypeIndices(
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memoryTypeBits);
    for (const auto memoryTypeIndex : memoryTypeIndices) {
        const VkMemoryAllocateInfo memoryAllocateInfo{
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, // type
            nullptr,                                // next
            size,                                   // size
            memoryTypeIndex,                        // memory type index
        };

        VkDeviceMemory deviceMemory;
        auto ret = loader->vkAllocateMemory(device, &memoryAllocateInfo, nullptr, &deviceMemory);
        if (ret == VK_SUCCESS) {
            return deviceMemory;
        }
    }

    throw std::runtime_error("Failed to allocate memory for constant tensor");
}

std::vector<uint32_t> TensorDescriptor::getMemoryTypeIndices(const VkMemoryPropertyFlags memoryPropertyFlags,
                                                             const uint32_t memoryTypeBits) const {
    VkPhysicalDeviceMemoryProperties memoryProperties;
    loader->vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

    std::vector<uint32_t> memoryTypeIndices;

    // Compile a list of memory allocation infos
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
        // Exclude memory types that are not part of mask
        if (((memoryTypeBits >> i) & 1) == 0) {
            continue;
        }

        const auto &memoryType = memoryProperties.memoryTypes[i];

        // Check that all required memory properties are supported
        if ((memoryType.propertyFlags & memoryPropertyFlags) != memoryPropertyFlags) {
            continue;
        }

        // Add memory type
        memoryTypeIndices.emplace_back(i);
    }

    // Sort infos in priority order
    std::sort(memoryTypeIndices.begin(), memoryTypeIndices.end(),
              [&memoryProperties](const auto &leftIndex, const auto &rightIndex) {
                  const auto &leftMemoryType = memoryProperties.memoryTypes[leftIndex];
                  const auto &rightMemoryType = memoryProperties.memoryTypes[rightIndex];

                  const auto &leftHeap = memoryProperties.memoryHeaps[leftMemoryType.heapIndex];
                  const auto &rightHeap = memoryProperties.memoryHeaps[rightMemoryType.heapIndex];

                  // Prioritize device local memory, it is likely faster
                  const auto leftDeviceLocal = leftMemoryType.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                  const auto rightDeviceLocal = rightMemoryType.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                  if (leftDeviceLocal != rightDeviceLocal) {
                      return leftDeviceLocal > rightDeviceLocal;
                  }

                  // Select the larger heap
                  return leftHeap.size > rightHeap.size;
              });

    return memoryTypeIndices;
}

VkFormat TensorDescriptor::getFormat() const { return format; }

const std::vector<int64_t> &TensorDescriptor::getDimensions() const { return dimensions; }

uint32_t TensorDescriptor::getRank() const { return static_cast<uint32_t>(dimensions.size()); }

size_t TensorDescriptor::getShapeSize() const {
    return static_cast<size_t>(
        std::abs(std::accumulate(dimensions.begin(), dimensions.end(), int64_t(1), std::multiplies<int64_t>())));
}

size_t TensorDescriptor::getSize() const {
    if (strides.data() != nullptr && strides[0] < 0) {
        throw std::runtime_error("Strides must be greater than zero.");
    }
    return strides.data() != nullptr ? static_cast<size_t>(dimensions[0U] * strides[0])
                                     : getShapeSize() * vk::blockSize(vk::Format(format));
}

uint64_t TensorDescriptor::getReferenceCounter() const { return referenceCounter; }

void TensorDescriptor::incrementReferenceCounter() { referenceCounter++; }

ComputePipelineBase *TensorDescriptor::getPipeline() const { return pipeline; }

void TensorDescriptor::setPipeline(ComputePipelineBase *_pipeline) { pipeline = _pipeline; }

VkMemoryRequirements TensorDescriptor::getMemoryRequirements() {
    if (vkMemoryRequirements) {
        return vkMemoryRequirements.value();
    }

    const auto tensorDescription = getTensorDescription();

    const VkTensorCreateInfoARM tensorCreateInfo = {
        VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_ARM, // type
        nullptr,                                  // next
        0,                                        // flags
        &tensorDescription,                       // tensor description
        VK_SHARING_MODE_EXCLUSIVE,                // sharing mode
        0,                                        // queue family index count
        nullptr,                                  // queue family indices
    };

    const VkDeviceTensorMemoryRequirementsARM requirementsInfo = {
        VK_STRUCTURE_TYPE_DEVICE_TENSOR_MEMORY_REQUIREMENTS_ARM, // type
        nullptr,                                                 // next
        &tensorCreateInfo,                                       // tensor create info
    };

    VkMemoryRequirements2 memoryRequirements = {};
    memoryRequirements.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
    loader->vkGetDeviceTensorMemoryRequirementsARM(device, &requirementsInfo, &memoryRequirements);

    // Cache result
    vkMemoryRequirements = memoryRequirements.memoryRequirements;

    return memoryRequirements.memoryRequirements;
}

VkDeviceSize TensorDescriptor::getMemoryRequirementsSize() { return getMemoryRequirements().size; }

VkTensorDescriptionARM TensorDescriptor::getTensorDescription() const {
    return VkTensorDescriptionARM{
        VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM, // type
        nullptr,                                  // next
        VK_TENSOR_TILING_LINEAR_ARM,              // tiling
        format,                                   // format
        static_cast<uint32_t>(dimensions.size()), // dimensions count
        dimensions.data(),                        // dimensions
        strides.data(),                           // strides
        VK_TENSOR_USAGE_SHADER_BIT_ARM,           // usage flags
    };
}

std::vector<int64_t> TensorDescriptor::createStrides(const VkTensorDescriptionARM &tensorDescription) const {
    if (tensorDescription.pStrides != nullptr) {
        return std::vector<int64_t>{tensorDescription.pStrides,
                                    tensorDescription.pStrides + tensorDescription.dimensionCount};
    }

    return {};
}

std::vector<VkQueueFamilyProperties> TensorDescriptor::enumerateQueueFamilyProperties() const {
    uint32_t queueFamilyPropertiesCount;
    loader->vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertiesCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyPropertiesCount);
    loader->vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertiesCount,
                                                     queueFamilyProperties.data());

    return queueFamilyProperties;
}

uint32_t TensorDescriptor::getComputeFamilyIndex() const {
    auto queueFamilyProperties = enumerateQueueFamilyProperties();

    for (uint32_t i = 0; i < queueFamilyProperties.size(); i++) {
        auto &property = queueFamilyProperties[i];

        if (property.queueFlags & VK_QUEUE_COMPUTE_BIT) {
            return i;
        }
    }

    throw std::runtime_error("Failed to find queue family index");
}

VkTensorARM TensorDescriptor::createTensorARM(const VkTensorDescriptionARM &tensorDescription) const {
    auto queueFamilyIndex = getComputeFamilyIndex();

    const VkTensorCreateInfoARM tensorCreateInfo = {
        VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_ARM, // type
        nullptr,                                  // next
        0,                                        // flags
        &tensorDescription,                       // tensor description
        VK_SHARING_MODE_EXCLUSIVE,                // sharing mode
        1,                                        // queue family index count
        &queueFamilyIndex,                        // queue family indices
    };

    VkTensorARM tensor;
    if (loader->vkCreateTensorARM(device, &tensorCreateInfo, nullptr, &tensor) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create tensor.");
    }

    return tensor;
}

VkTensorViewARM TensorDescriptor::createTensorViewARM(const VkTensorARM tensor, const VkFormat _format) const {
    const VkTensorViewCreateInfoARM tensorViewCreateInfo = {
        VK_STRUCTURE_TYPE_TENSOR_VIEW_CREATE_INFO_ARM, // type
        nullptr,                                       // next
        0,                                             // flags
        tensor,                                        // tensor
        _format,                                       // format
    };

    VkTensorViewARM tensorView;
    if (loader->vkCreateTensorViewARM(device, &tensorViewCreateInfo, nullptr, &tensorView) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create tensor view.");
    }

    return tensorView;
}

Log &operator<<(Log &os, const Tensor &tensor) {
    return os << "tensorARM=" << tensor.getVkTensorARM() << " " << *tensor.getTensorDescriptor();
}

Log &operator<<(Log &os, const TensorDescriptor &tensor) {
    return os << "format=" << vk::to_string(vk::Format(tensor.getFormat())) << ", shape=" << tensor.getDimensions();
}

/*******************************************************************************
 * VirtualTensor
 *******************************************************************************/

VirtualTensor::VirtualTensor(const std::shared_ptr<TensorDescriptor> &_tensor, ComputePipelineBase *_parent,
                             ComputePipelineBase *_descendant)
    : tensor{_tensor}, parent{_parent}, descendant{_descendant} {
    tensor->incrementReferenceCounter();
}

bool VirtualTensor::getVisited() const { return visited; }

void VirtualTensor::setVisited(const bool _visited) { visited = _visited; }

std::shared_ptr<TensorDescriptor> VirtualTensor::getTensor() const { return tensor; }

ComputePipelineBase *VirtualTensor::getParentPipeline() const { return parent; }

ComputePipelineBase *VirtualTensor::getDescendantPipeline() const { return descendant; }

} // namespace mlsdk::el::compute
