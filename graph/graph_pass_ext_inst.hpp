/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

#include "graph_ext_inst_context.hpp"
#include "graph_ext_inst_registry.hpp"

#include "source/opt/pass.h"
#include <utility>

#include <spirv-tools/optimizer.hpp>

namespace spvtools::opt {

class GraphPassExtInst final : public Pass {
  public:
    GraphPassExtInst(GraphPipeline &_graphPipeline, GraphExtInstImports imports)
        : graphPipeline{_graphPipeline}, imports{std::move(imports)} {}
    ~GraphPassExtInst() override = default;
    const char *name() const override { return "graph-pass-extinst"; }

  private:
    using Decoders = std::unordered_map<uint32_t, GraphExtInstDecoder *>;

    Status Process() override;
    void handleGraphs(GraphExtInstContext &loweringContext, const Decoders &decoders);
    void handleGraph(const Graph *graph, const Decoders &decoders);

    GraphPipeline &graphPipeline;
    GraphExtInstImports imports;
};

} // namespace spvtools::opt

namespace spvtools {

inline Optimizer::PassToken createGraphPass(opt::GraphPipeline &graphPipeline, opt::GraphExtInstImports imports) {
    return Optimizer::PassToken{MakeUnique<opt::GraphPassExtInst>(graphPipeline, std::move(imports))};
}

} // namespace spvtools
