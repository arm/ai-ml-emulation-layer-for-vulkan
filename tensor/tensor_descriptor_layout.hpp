/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

#include <cstddef>
#include <cstdint>

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
    AlignAs<int64_t, 16> dimensions[maxTensorRank];
    AlignAs<int64_t, 16> strides[maxTensorRank];
};

} // namespace mlsdk::el::layer
