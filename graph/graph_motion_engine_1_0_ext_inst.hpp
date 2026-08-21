/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/*******************************************************************************
 * Includes
 *******************************************************************************/

#pragma once

#include "graph_ext_inst_decoder.hpp"

#include <string>
#include <string_view>

/*******************************************************************************
 * GraphPass extended instruction sets
 *******************************************************************************/

namespace spvtools::opt {

inline constexpr std::string_view motionEngine100 = "Arm.MotionEngine.100";

class GraphMotionEngine10ExtInst final : public GraphExtInstDecoder {
  public:
    explicit GraphMotionEngine10ExtInst(GraphExtInstContext &context) : context{context} {}
    void handleOp(const Instruction *opExtInst) const override;

  private:
    void handleMinSad(const Instruction *opExtInst, const std::string &debugName) const;
    void handleMinSadCost(const Instruction *opExtInst, const std::string &debugName) const;
    void handleRawSad(const Instruction *opExtInst, const std::string &debugName) const;

    GraphExtInstContext &context;
};

} // namespace spvtools::opt
