/*
 * SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/*******************************************************************************
 * Includes
 *******************************************************************************/

#pragma once

#include "compute.hpp"
#include "spirv_pass.hpp"

static constexpr std::string_view tosaSpv100 = "TOSA.001000.1";
static constexpr std::string_view motionEngine100 = "Arm.MotionEngine.100";

/*******************************************************************************
 * GraphPass TosaSpv100
 *******************************************************************************/

namespace spvtools::opt {

class GraphPassTosaSpv100 final : public GraphPassBase {
  public:
    explicit GraphPassTosaSpv100(mlsdk::el::compute::GraphPipeline &_pipelines) : GraphPassBase(_pipelines) {}
    const char *name() const override { return "graph-pass-tosaspv-v100"; }

  private:
    void handleGraph(const Graph *graph) override;
    void handleTosaInst(const Instruction *opExtInst);
    void handleMotionEngineInst(const Instruction *opExtInst);

    void handleArgmax(const Instruction *opExtInst, const std::string &debugName);
    void handleArithmeticRightShift(const Instruction *opExtInst, const std::string &debugName);
    void handleAvgPool2D(const Instruction *opExtInst, const std::string &debugName);
    void handleCast(const Instruction *opExtInst, const std::string &debugName);
    void handleClamp(const Instruction *opExtInst, const std::string &debugName);
    void handleConcat(const Instruction *opExtInst, const std::string &debugName);
    void handleConv2D(const Instruction *opExtInst, const std::string &debugName);
    void handleConv3D(const Instruction *opExtInst, const std::string &debugName);
    void handleDepthwiseConv2D(const Instruction *opExtInst, const std::string &debugName);
    void handleElementwiseBinary(
        const Instruction *opExtInst, const std::string &debugName,
        std::function<void(mlsdk::el::compute::GraphPipeline *,
                           const std::shared_ptr<mlsdk::el::compute::TensorDescriptor> &,
                           const std::shared_ptr<mlsdk::el::compute::TensorDescriptor> &,
                           const std::shared_ptr<mlsdk::el::compute::TensorDescriptor> &, const std::string &)>
            function);
    void handleElementwiseUnary(
        const Instruction *opExtInst, const std::string &debugName,
        std::function<void(mlsdk::el::compute::GraphPipeline *,
                           const std::shared_ptr<mlsdk::el::compute::TensorDescriptor> &,
                           const std::shared_ptr<mlsdk::el::compute::TensorDescriptor> &, const std::string &)>
            function);
    void handleFft2D(const Instruction *opExtInst, const std::string &debugName);
    void handleGather(const Instruction *opExtInst, const std::string &debugName);
    void handleMatmul(const Instruction *opExtInst, const std::string &debugName);
    void handleMaximum(const Instruction *opExtInst, const std::string &debugName);
    void handleMaxPool2D(const Instruction *opExtInst, const std::string &debugName);
    void handleMinimum(const Instruction *opExtInst, const std::string &debugName);
    void handleMul(const Instruction *opExtInst, const std::string &debugName);
    void handleNegate(const Instruction *opExtInst, const std::string &debugName);
    void handleReduce(
        const Instruction *opExtInst, const std::string &debugName,
        std::function<
            void(mlsdk::el::compute::GraphPipeline *, const std::shared_ptr<mlsdk::el::compute::TensorDescriptor> &,
                 const std::shared_ptr<mlsdk::el::compute::TensorDescriptor> &, const uint32_t, const std::string &)>
            function);
    void handleReduceMax(const Instruction *opExtInst, const std::string &debugName);
    void handleReduceMin(const Instruction *opExtInst, const std::string &debugName);
    void handlePad(const Instruction *opExtInst, const std::string &debugName);
    void handleRescale(const Instruction *opExtInst, const std::string &debugName);
    void handleReshape(const Instruction *opExtInst, const std::string &debugName);
    void handleResize(const Instruction *opExtInst, const std::string &debugName);
    void handleReverse(const Instruction *opExtInst, const std::string &debugName);
    void handleRfft2D(const Instruction *opExtInst, const std::string &debugName);
    void handleScatter(const Instruction *opExtInst, const std::string &debugName);
    void handleSelect(const Instruction *opExtInst, const std::string &debugName);
    void handleSlice(const Instruction *opExtInst, const std::string &debugName);
    void handleTable(const Instruction *opExtInst, const std::string &debugName);
    void handleTile(const Instruction *opExtInst, const std::string &debugName);
    void handleTranspose(const Instruction *opExtInst, const std::string &debugName);
    void handleTransposeConv2D(const Instruction *opExtInst, const std::string &debugName);

    void handleMinSad(const Instruction *opExtInst, const std::string &debugName);
    void handleMinSadCost(const Instruction *opExtInst, const std::string &debugName);
    void handleRawSad(const Instruction *opExtInst, const std::string &debugName);
};

} // namespace spvtools::opt
