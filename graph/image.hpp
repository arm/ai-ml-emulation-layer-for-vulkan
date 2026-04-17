/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include <vulkan/vulkan.hpp>

#include <memory>
#include <optional>

namespace mlsdk::el::compute {

/*******************************************************************************
 * Image
 *******************************************************************************/

class Image {
  public:
    enum class Usage {
        BufferStoreLoad,        // Producer stores to Buffer, consumer loads from Buffer
        ImageStoreSample,       // Producer stores to Image, consumer samples from Image
        BufferStoreImageSample, // Producer stores to Buffer, consumer samples from Image
        ImageStoreBufferLoad,   // Producer stores to Image, consumer loads from Buffer
        NoStoreImageSample,     // Read-only. Consumer samples from Image
        HostBuffer,             // Host visible and coherent buffer. Mainly for staging buffer
    };

    enum class BarrierState {
        Undefined,
        TransferSrc,
        TransferDst,
        ShaderRead,
        ShaderWrite,
        HostRead,
        HostWrite,
        GraphRead,
        GraphWrite,
    };

    enum class Type {
        Internal,    // owned by the OpticalFlow instance
        External,    // owned by the application
        Placeholder, // placeholder to ensure compatability later
    };

    // Factories
    static std::shared_ptr<Image>
    makeInternal(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &loader,
                 VkPhysicalDevice physicalDevice, VkDevice device, Usage usage, VkExtent3D dim, VkFormat format,
                 VkImageTiling tiling, bool isCached = false, const std::string &debugName = "") {
        return std::make_shared<Image>(loader, physicalDevice, device, usage, dim, format, tiling, isCached, debugName);
    }

    static std::shared_ptr<Image>
    makePlaceholder(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &loader, Usage usage,
                    VkExtent3D dim, VkFormat format, VkImageLayout layout) {
        return std::make_shared<Image>(loader, usage, dim, format, layout);
    }

    // Creates an internal image
    Image(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &loader,
          VkPhysicalDevice physicalDevice, VkDevice device, Usage usage, VkExtent3D dim, VkFormat format,
          VkImageTiling tiling, bool isCached, const std::string &debugName);
    // Creates a placeholder image
    Image(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &loader, Usage usage,
          VkExtent3D dim, VkFormat format, VkImageLayout layout);

    ~Image();

    // Internal resource creation
    void makeImage();
    void makeBuffer();
    void makeBufferAlias();
    // External resource connection
    void setExternalDescriptor(VkDescriptorSet descriptorSet, uint32_t binding, VkImageView imageView);

    // Memory helpers
    VkMemoryRequirements getMemoryRequirements() const;
    void setMemoryOffset(VkDeviceSize offset);
    void bindToMemory(VkDeviceMemory memory, VkDeviceSize baseOffset);
    VkDeviceMemory allocateDeviceMemory(VkMemoryRequirements mrq, VkMemoryPropertyFlags wantedProps);

    void makeBarrier(VkCommandBuffer cmdBuf, BarrierState newState);

    // Utils
    void swapHandles(Image &other);

    bool isBufferStore() const;
    bool isBufferLoad() const;
    bool isImageStore() const;
    bool isImageSample() const;

    bool isLinearTiling() const;

    bool isPlaceholder() const;
    bool isExternal() const;
    bool isInternal() const;
    bool isCached() const;

    bool hasImage() const;
    bool hasBuffer() const;

    uint32_t height() const;
    uint32_t width() const;
    uint32_t stride() const;
    uint32_t componentCount() const;
    Usage usage() const;
    VkFormat format() const;

    VkImageView getImageView() const;
    VkImageLayout getImageLayout() const;
    VkBuffer getBuffer() const;

    // checks if two images are compatible for binding
    static bool isCompatible(const std::shared_ptr<Image> &a, const std::shared_ptr<Image> &b);

  private:
    std::vector<uint32_t> candidateMemoryTypes(uint32_t memoryTypeBits, VkMemoryPropertyFlags wanted) const;
    std::tuple<VkPipelineStageFlags2, VkAccessFlags2, VkImageLayout> barrierProps(BarrierState state);

    std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> loader_;
    VkPhysicalDevice physicalDevice_;
    VkDevice device_;

    Usage usage_ = Usage::BufferStoreLoad;
    VkExtent3D dim_;
    BarrierState state_ = BarrierState::Undefined;
    VkDeviceSize sizeInBytes_ = 0;
    VkDeviceSize blockSize_ = 1;
    VkDeviceSize rowPitch_ = 1;
    uint32_t componentCount_ = 1;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    VkImageTiling tiling_ = VK_IMAGE_TILING_LINEAR;
    VkImageLayout imageLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    Type imageType_;
    bool isCached_ = false;
    std::string debugName_;

    VkMemoryRequirements memoryRequirements_;
    VkDeviceSize memoryOffset_;

    // Internal resource handles
    VkImageView imageView_ = VK_NULL_HANDLE;
    VkImage image_ = VK_NULL_HANDLE;
    VkBuffer buffer_ = VK_NULL_HANDLE;

    // External images only
    VkDescriptorSet externalDescriptorSet_ = VK_NULL_HANDLE;
    uint32_t externalBinding_ = 0;
    VkImageView externalImageView_ = VK_NULL_HANDLE;
};

inline bool Image::isBufferStore() const {
    return usage_ == Usage::BufferStoreLoad || usage_ == Usage::BufferStoreImageSample;
}
inline bool Image::isBufferLoad() const {
    return usage_ == Usage::BufferStoreLoad || usage_ == Usage::ImageStoreBufferLoad;
}
inline bool Image::isImageStore() const {
    return usage_ == Usage::ImageStoreSample || usage_ == Usage::ImageStoreBufferLoad;
}
inline bool Image::isImageSample() const {
    return usage_ == Usage::ImageStoreSample || usage_ == Usage::BufferStoreImageSample ||
           usage_ == Usage::NoStoreImageSample;
}

inline bool Image::isLinearTiling() const { return tiling_ == VK_IMAGE_TILING_LINEAR; }

inline bool Image::isPlaceholder() const { return imageType_ == Type::Placeholder; }
inline bool Image::isExternal() const { return imageType_ == Type::External; }
inline bool Image::isInternal() const { return imageType_ == Type::Internal; }
inline bool Image::isCached() const { return isCached_; }

inline bool Image::hasBuffer() const { return buffer_ != VK_NULL_HANDLE; }
inline bool Image::hasImage() const { return image_ != VK_NULL_HANDLE; }

inline VkImageView Image::getImageView() const { return isInternal() ? imageView_ : externalImageView_; }
inline VkImageLayout Image::getImageLayout() const { return imageLayout_; }
inline VkBuffer Image::getBuffer() const { return buffer_; }

inline VkMemoryRequirements Image::getMemoryRequirements() const { return memoryRequirements_; }
inline void Image::setMemoryOffset(VkDeviceSize offset) { memoryOffset_ = offset; }

inline Image::Usage Image::usage() const { return usage_; }
inline VkFormat Image::format() const { return format_; }
inline uint32_t Image::height() const { return dim_.height; }
inline uint32_t Image::width() const { return dim_.width; }
inline uint32_t Image::stride() const {
    if (isLinearTiling()) {
        return static_cast<uint32_t>(rowPitch_ / blockSize_);
    }
    // This should not be used in the shader
    return 0;
}
inline uint32_t Image::componentCount() const { return componentCount_; }

} // namespace mlsdk::el::compute
