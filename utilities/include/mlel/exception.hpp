/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include <exception>
#include <stdexcept>
#include <string>

namespace mlsdk::el::utilities {

/*******************************************************************************
 * Defines
 *******************************************************************************/

inline void VK_ASSERT(const bool result, const std::string &message = "") {
    if (!result) {
        throw std::runtime_error(message);
    }
}

template <typename T, typename U> void VK_ASSERT_EQ(const T &left, const U &right, const std::string &message = "") {
    VK_ASSERT(left == right, message);
}

template <typename T, typename U> void VK_ASSERT_NE(const T &left, const U &right, const std::string &message = "") {
    VK_ASSERT(left != right, message);
}

template <typename T, typename U> void VK_ASSERT_GE(const T &left, const U &right, const std::string &message = "") {
    VK_ASSERT(left >= right, message);
}

template <typename T, typename U> void VK_ASSERT_GT(const T &left, const U &right, const std::string &message = "") {
    VK_ASSERT(left > right, message);
}

} // namespace mlsdk::el::utilities
