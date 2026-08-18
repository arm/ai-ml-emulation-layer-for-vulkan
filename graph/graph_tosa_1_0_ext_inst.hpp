/*
 * SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/*******************************************************************************
 * Includes
 *******************************************************************************/

#pragma once

#include "compute_graph_op.hpp"
#include "graph_ext_inst_decoder.hpp"

#include <functional>
#include <string>
#include <string_view>

/*******************************************************************************
 * GraphPass extended instruction sets
 *******************************************************************************/

namespace spvtools::opt {

inline constexpr std::string_view tosaSpv100 = "TOSA.001000.1";
using mlsdk::el::compute::graph_op::GraphPipeline;

class GraphTosa10ExtInst final : public GraphExtInstDecoder {
  public:
    explicit GraphTosa10ExtInst(GraphExtInstContext &context) : context{context} {}
    void handleOp(const Instruction *opExtInst) const override;

  private:
    void handleArgmax(const Instruction *opExtInst, const std::string &debugName) const;
    void handleArithmeticRightShift(const Instruction *opExtInst, const std::string &debugName) const;
    void handleAvgPool2D(const Instruction *opExtInst, const std::string &debugName) const;
    void handleCast(const Instruction *opExtInst, const std::string &debugName) const;
    void handleClamp(const Instruction *opExtInst, const std::string &debugName) const;
    void handleConcat(const Instruction *opExtInst, const std::string &debugName) const;
    void handleConv2D(const Instruction *opExtInst, const std::string &debugName) const;
    void handleConv3D(const Instruction *opExtInst, const std::string &debugName) const;
    void handleDepthwiseConv2D(const Instruction *opExtInst, const std::string &debugName) const;
    void handleElementwiseBinary(
        const Instruction *opExtInst, const std::string &debugName,
        const std::function<void(GraphPipeline *, const std::shared_ptr<mlsdk::el::compute::TensorDescriptor> &,
                                 const std::shared_ptr<mlsdk::el::compute::TensorDescriptor> &,
                                 const std::shared_ptr<mlsdk::el::compute::TensorDescriptor> &, const std::string &)>
            &function) const;
    void handleElementwiseUnary(
        const Instruction *opExtInst, const std::string &debugName,
        const std::function<void(GraphPipeline *, const std::shared_ptr<mlsdk::el::compute::TensorDescriptor> &,
                                 const std::shared_ptr<mlsdk::el::compute::TensorDescriptor> &, const std::string &)>
            &function) const;
    void handleFft2D(const Instruction *opExtInst, const std::string &debugName) const;
    void handleGather(const Instruction *opExtInst, const std::string &debugName) const;
    void handleMatmul(const Instruction *opExtInst, const std::string &debugName) const;
    void handleMaximum(const Instruction *opExtInst, const std::string &debugName) const;
    void handleMaxPool2D(const Instruction *opExtInst, const std::string &debugName) const;
    void handleMinimum(const Instruction *opExtInst, const std::string &debugName) const;
    void handleMul(const Instruction *opExtInst, const std::string &debugName) const;
    void handleNegate(const Instruction *opExtInst, const std::string &debugName) const;
    void handleReduce(
        const Instruction *opExtInst, const std::string &debugName,
        const std::function<void(GraphPipeline *, const std::shared_ptr<mlsdk::el::compute::TensorDescriptor> &,
                                 const std::shared_ptr<mlsdk::el::compute::TensorDescriptor> &, const uint32_t,
                                 const std::string &)> &function) const;
    void handleReduceMax(const Instruction *opExtInst, const std::string &debugName) const;
    void handleReduceMin(const Instruction *opExtInst, const std::string &debugName) const;
    void handlePad(const Instruction *opExtInst, const std::string &debugName) const;
    void handleRescale(const Instruction *opExtInst, const std::string &debugName) const;
    void handleReshape(const Instruction *opExtInst, const std::string &debugName) const;
    void handleResize(const Instruction *opExtInst, const std::string &debugName) const;
    void handleReverse(const Instruction *opExtInst, const std::string &debugName) const;
    void handleRfft2D(const Instruction *opExtInst, const std::string &debugName) const;
    void handleScatter(const Instruction *opExtInst, const std::string &debugName) const;
    void handleSelect(const Instruction *opExtInst, const std::string &debugName) const;
    void handleSlice(const Instruction *opExtInst, const std::string &debugName) const;
    void handleTable(const Instruction *opExtInst, const std::string &debugName) const;
    void handleTile(const Instruction *opExtInst, const std::string &debugName) const;
    void handleTranspose(const Instruction *opExtInst, const std::string &debugName) const;
    void handleTransposeConv2D(const Instruction *opExtInst, const std::string &debugName) const;

    GraphExtInstContext &context;
};

} // namespace spvtools::opt
