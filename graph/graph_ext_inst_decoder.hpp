/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

namespace spvtools::opt {

class GraphExtInstContext;
class Instruction;

class GraphExtInstDecoder {
  public:
    virtual ~GraphExtInstDecoder() = default;
    virtual void handleOp(const Instruction *opExtInst) const = 0;
};

} // namespace spvtools::opt
