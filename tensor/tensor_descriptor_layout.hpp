/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace mlsdk::el::layer {

inline constexpr std::size_t maxTensorRank = 6;

// Shared memory layout for host-written tensor metadata and shader accesses.
struct TensorDescriptorLayout {
    template <typename T, std::size_t ALIGN> struct alignas(ALIGN) AlignAs {
        T v;

        template <typename U> AlignAs &operator=(const U &val) {
            v = val;
            return *this;
        }
    };

    uint64_t address;
    uint32_t narrowOffsets;
    AlignAs<int64_t, 16> dimensions[maxTensorRank];
    AlignAs<int64_t, 16> strides[maxTensorRank];
};

inline bool supports32BitTensorOffsets(const std::vector<int64_t> &dimensions, const std::vector<int64_t> &strides) {
    if (dimensions.empty() || dimensions.size() > maxTensorRank || dimensions.size() != strides.size()) {
        return false;
    }
    // The narrow shader handles at most four elements per access; reserve three for the tail.
    constexpr int64_t limit = int64_t(std::numeric_limits<uint32_t>::max()) - 3;
    int64_t offset = 0;
    for (size_t axis = 0; axis < dimensions.size(); ++axis) {
        const auto dimension = dimensions[axis];
        const auto stride = strides[axis];
        if (dimension <= 0 || dimension > limit || stride < 0 || stride > limit ||
            (stride != 0 && dimension - 1 > (limit - offset) / stride)) {
            return false;
        }
        offset += (dimension - 1) * stride;
    }
    return true;
}

} // namespace mlsdk::el::layer
