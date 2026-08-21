/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

#include "graph_ext_inst_context.hpp"
#include "source/opt/pass.h"

#include <spirv-tools/optimizer.hpp>

namespace spvtools::opt {

class GraphPassExtInst final : public Pass {
  public:
    explicit GraphPassExtInst(GraphPipeline &_graphPipeline) : graphPipeline{_graphPipeline} {}
    ~GraphPassExtInst() override = default;
    const char *name() const override { return "graph-pass-extinst"; }

  private:
    Status Process() override;
    void handleGraphs(GraphExtInstContext &loweringContext);
    void handleGraph(const Graph *graph, GraphExtInstContext &loweringContext);

    GraphPipeline &graphPipeline;
};

} // namespace spvtools::opt

namespace spvtools {

inline Optimizer::PassToken createGraphPass(opt::GraphPipeline &graphPipeline) {
    return Optimizer::PassToken{MakeUnique<opt::GraphPassExtInst>(graphPipeline)};
}

} // namespace spvtools
