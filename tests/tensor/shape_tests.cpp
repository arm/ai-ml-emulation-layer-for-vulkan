/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include "mlel/tensor.hpp"

#include <gtest/gtest.h>

using mlsdk::el::utilities::Shape;

TEST(TensorShape, GetElementOffsetNonPacked) {
    const Shape shape{vk::Format::eR64Sint, {2, 3, 4}, {160, 40, 8}};

    EXPECT_EQ(shape.getElementOffset(0), 0u);
    EXPECT_EQ(shape.getElementOffset(1), 8u);
    EXPECT_EQ(shape.getElementOffset(3), 24u);
    EXPECT_EQ(shape.getElementOffset(4), 40u);
    EXPECT_EQ(shape.getElementOffset(11), 104u);
    EXPECT_EQ(shape.getElementOffset(12), 160u);
    EXPECT_EQ(shape.getElementOffset(23), 264u);
}
