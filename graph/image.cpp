/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "image.hpp"

#include "graph_log.hpp"
#include "mlel/utils.hpp"

#include <exception>
#include <numeric>
#include <vulkan/vulkan_format_traits.hpp>

using namespace mlsdk::el::log;
using namespace mlsdk::el::utils;

namespace mlsdk::el::compute {

/*******************************************************************************
 * Image
 *******************************************************************************/
Image::Image(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &loader,
             VkPhysicalDevice physicalDevice, VkDevice device, Usage usage, VkExtent3D dim, VkFormat format,
             VkImageTiling tiling, bool isCached, const std::string &debugName)
    : loader_{loader}, physicalDevice_{physicalDevice}, device_{device}, usage_{usage}, dim_{dim}, format_{format},
      tiling_{tiling}, imageType_{Type::Internal}, isCached_{isCached}, debugName_{debugName} {
    componentCount_ = vk::componentCount(vk::Format(format_));
    blockSize_ = vk::blockSize(vk::Format(format_));
    if (isImageSample() || isImageStore()) {
        makeImage();
    }
    if (usage == Usage::BufferStoreLoad) {
        makeBuffer();
    } else if (usage == Usage::BufferStoreImageSample || usage == Usage::ImageStoreBufferLoad) {
        makeBufferAlias();
    }
}

Image::Image(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &loader, Usage usage,
             VkExtent3D dim, VkFormat format, VkImageLayout layout)
    : loader_{loader}, usage_(usage), dim_(dim), format_(format), imageLayout_(layout), imageType_(Type::Placeholder) {
    componentCount_ = vk::componentCount(vk::Format(format_));
    blockSize_ = vk::blockSize(vk::Format(format_));
}

Image::~Image() {
    if (!isInternal()) {
        return;
    }

    if (imageView_) {
        loader_->vkDestroyImageView(device_, imageView_, nullptr);
    }
    if (image_) {
        loader_->vkDestroyImage(device_, image_, nullptr);
    }
    if (buffer_) {
        loader_->vkDestroyBuffer(device_, buffer_, nullptr);
    }
}

void Image::makeBuffer() {
    sizeInBytes_ = dim_.height * dim_.width * blockSize_;
    rowPitch_ = dim_.width * blockSize_;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.flags = 0;
    bufferInfo.size = sizeInBytes_;
    bufferInfo.usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult res = loader_->vkCreateBuffer(device_, &bufferInfo, VK_NULL_HANDLE, &buffer_);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("Failed to create buffer");
    }

    VkMemoryRequirements2 mrqs{};
    mrqs.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
    const VkBufferMemoryRequirementsInfo2 memInfo = {
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2,
        nullptr,
        buffer_,
    };
    loader_->vkGetBufferMemoryRequirements2(device_, &memInfo, &mrqs);
    memoryRequirements_ = mrqs.memoryRequirements;

    if (!debugName_.empty()) {
        setDebugUtilsObjectName(loader_, device_, VK_OBJECT_TYPE_BUFFER, reinterpret_cast<uint64_t>(buffer_),
                                debugName_ + "_buf");
    }
}

/* Build a VkImage + VkImageView and wrap them */
void Image::makeImage() {
    VkImageUsageFlags imageUsage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (isImageSample()) {
        imageUsage |= VK_IMAGE_USAGE_SAMPLED_BIT;
    }
    if (isImageStore()) {
        imageUsage |= VK_IMAGE_USAGE_STORAGE_BIT;
    }

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent = {dim_.width, dim_.height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format_;
    imageInfo.tiling = tiling_;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = imageUsage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult res = loader_->vkCreateImage(device_, &imageInfo, VK_NULL_HANDLE, &image_);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("Failed to create image");
    }

    VkMemoryRequirements2 mrqs{};
    mrqs.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
    const VkImageMemoryRequirementsInfo2 memInfo = {
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
        nullptr,
        image_,
    };
    loader_->vkGetImageMemoryRequirements2(device_, &memInfo, &mrqs);
    memoryRequirements_ = mrqs.memoryRequirements;

    if (tiling_ == VK_IMAGE_TILING_LINEAR) // This case includes Buffer alias
    {
        const VkImageSubresource subResource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0};
        VkSubresourceLayout subResourceLayout;
        loader_->vkGetImageSubresourceLayout(device_, image_, &subResource, &subResourceLayout);
        sizeInBytes_ = subResourceLayout.size;
        rowPitch_ = subResourceLayout.rowPitch;
    }

    if (!debugName_.empty()) {
        setDebugUtilsObjectName(loader_, device_, VK_OBJECT_TYPE_IMAGE, reinterpret_cast<uint64_t>(image_),
                                debugName_ + "_img");
    }
}

void Image::makeBufferAlias() {
    assert(sizeInBytes_ > 0);
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.flags = 0;
    bufferInfo.size = sizeInBytes_;
    bufferInfo.usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult res = loader_->vkCreateBuffer(device_, &bufferInfo, VK_NULL_HANDLE, &buffer_);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("Failed to create buffer alias");
    }

    if (!debugName_.empty()) {
        setDebugUtilsObjectName(loader_, device_, VK_OBJECT_TYPE_BUFFER, reinterpret_cast<uint64_t>(buffer_),
                                debugName_ + "_buf");
    }
}

void Image::bindToMemory(VkDeviceMemory memory, VkDeviceSize baseOffset) {
    assert(isInternal());
    if (hasBuffer()) {
        auto res = loader_->vkBindBufferMemory(device_, buffer_, memory, baseOffset + memoryOffset_);
        if (res != VK_SUCCESS) {
            throw std::runtime_error("Failed to bind buffer alias");
        }
    }
    if (hasImage()) {
        auto res = loader_->vkBindImageMemory(device_, image_, memory, baseOffset + memoryOffset_);
        if (res != VK_SUCCESS) {
            throw std::runtime_error("Failed to bind image memory");
        }

        VkImageViewCreateInfo imageViewInfo{};
        imageViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        imageViewInfo.image = image_;
        imageViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        imageViewInfo.format = format_;
        imageViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageViewInfo.subresourceRange.baseMipLevel = 0;
        imageViewInfo.subresourceRange.levelCount = 1;
        imageViewInfo.subresourceRange.baseArrayLayer = 0;
        imageViewInfo.subresourceRange.layerCount = 1;

        res = loader_->vkCreateImageView(device_, &imageViewInfo, VK_NULL_HANDLE, &imageView_);
        if (res != VK_SUCCESS) {
            throw std::runtime_error("Failed to create image view");
        }
    }
}

void Image::setExternalDescriptor(VkDescriptorSet descriptorSet, uint32_t binding, VkImageView imageView) {
    assert(!isInternal());
    imageType_ = Type::External;
    externalDescriptorSet_ = descriptorSet;
    externalBinding_ = binding;
    externalImageView_ = imageView;
}

std::vector<uint32_t> Image::candidateMemoryTypes(uint32_t bits, VkMemoryPropertyFlags wanted) const {
    VkPhysicalDeviceMemoryProperties memProps{};
    loader_->vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memProps);

    std::vector<uint32_t> out;
    out.reserve(memProps.memoryTypeCount);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) == 0) {
            continue;
        }
        if ((memProps.memoryTypes[i].propertyFlags & wanted) != wanted) {
            continue;
        }
        out.emplace_back(i);
    }
    std::sort(out.begin(), out.end(), [&](uint32_t a, uint32_t b) {
        auto ha = memProps.memoryHeaps[memProps.memoryTypes[a].heapIndex];
        auto hb = memProps.memoryHeaps[memProps.memoryTypes[b].heapIndex];
        bool aDev = memProps.memoryTypes[a].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        bool bDev = memProps.memoryTypes[b].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        if (aDev != bDev) {
            return aDev > bDev;
        }
        return ha.size > hb.size;
    });
    return out;
}

VkDeviceMemory Image::allocateDeviceMemory(VkMemoryRequirements mrq, VkMemoryPropertyFlags wanted) {
    for (auto idx : candidateMemoryTypes(mrq.memoryTypeBits, wanted)) {
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, mrq.size, idx};
        VkDeviceMemory mem{};
        if (loader_->vkAllocateMemory(device_, &ai, nullptr, &mem) == VK_SUCCESS) {
            return mem;
        }
    }
    throw std::runtime_error("Failed to allocate memory");
}

std::tuple<VkPipelineStageFlags2, VkAccessFlags2, VkImageLayout> Image::barrierProps(BarrierState state) {
    switch (state) {
    case BarrierState::Undefined:
        return {VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE, VK_IMAGE_LAYOUT_UNDEFINED};
    case BarrierState::TransferSrc:
        return {VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL};
    case BarrierState::TransferDst:
        return {VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL};
    case BarrierState::ShaderRead:
        return {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    case BarrierState::ShaderWrite:
        return {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL};
    case BarrierState::HostRead:
        return {VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT, VK_IMAGE_LAYOUT_GENERAL};
    case BarrierState::HostWrite:
        return {VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL};

    case BarrierState::GraphRead:
        return {VK_PIPELINE_STAGE_2_DATA_GRAPH_BIT_ARM, VK_ACCESS_2_DATA_GRAPH_READ_BIT_ARM,
                VK_IMAGE_LAYOUT_TENSOR_ALIASING_ARM};
    case BarrierState::GraphWrite:
        return {VK_PIPELINE_STAGE_2_DATA_GRAPH_BIT_ARM, VK_ACCESS_2_DATA_GRAPH_WRITE_BIT_ARM,
                VK_IMAGE_LAYOUT_TENSOR_ALIASING_ARM};

    default:
        assert(false);
    }
    return {VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE, VK_IMAGE_LAYOUT_UNDEFINED};
}

void Image::makeBarrier(VkCommandBuffer cmdBuf, BarrierState newState) {
    if (isExternal()) {
        // application-owned images should be externally synchronised
        return;
    }

    VkImageMemoryBarrier2 imageBarrier;
    VkBufferMemoryBarrier2 bufferBarrier;
    VkDependencyInfo dependencyInfo{};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;

    const auto [srcStageMask, srcAccessMask, srcImageLayout] = barrierProps(state_);
    const auto [dstStageMask, dstAccessMask, dstImageLayout] = barrierProps(newState);

    if (hasBuffer()) {
        bufferBarrier = {
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
            nullptr,
            srcStageMask,
            srcAccessMask,
            dstStageMask,
            dstAccessMask,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            buffer_,
            0,
            sizeInBytes_,
        };
        dependencyInfo.bufferMemoryBarrierCount = 1;
        dependencyInfo.pBufferMemoryBarriers = &bufferBarrier;
    }

    if (hasImage()) {
        imageBarrier = {
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            nullptr,
            srcStageMask,
            srcAccessMask,
            dstStageMask,
            dstAccessMask,
            srcImageLayout,
            dstImageLayout,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            image_,
            VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
        };
        dependencyInfo.imageMemoryBarrierCount = 1;
        dependencyInfo.pImageMemoryBarriers = &imageBarrier;
    }

    loader_->vkCmdPipelineBarrier2(cmdBuf, &dependencyInfo);

    state_ = newState;
    imageLayout_ = dstImageLayout;
}

void Image::swapHandles(Image &other) {
    assert(usage_ == other.usage_);
    assert(dim_.height == other.dim_.height);
    assert(dim_.width == other.dim_.width);
    assert(dim_.depth == other.dim_.depth);
    assert(format_ == other.format_);
    assert(tiling_ == other.tiling_);
    std::swap(buffer_, other.buffer_);
    std::swap(image_, other.image_);
    std::swap(imageView_, other.imageView_);
    std::swap(imageLayout_, other.imageLayout_);
    std::swap(state_, other.state_);
}

bool Image::isCompatible(const std::shared_ptr<Image> &a, const std::shared_ptr<Image> &b) {
    if (!a || !b) {
        return false;
    }
    return (a->usage() == b->usage()) && (a->width() == b->width()) && (a->height() == b->height()) &&
           (a->format() == b->format());
}

} // namespace mlsdk::el::compute
