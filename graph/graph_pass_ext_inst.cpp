/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "graph_pass_ext_inst.hpp"
#include "graph_ext_inst_registry.hpp"
#include "graph_log.hpp"

/*******************************************************************************
 * GraphPass extended instruction sets
 *******************************************************************************/
using namespace mlsdk::el::log;

namespace spvtools::opt {

Pass::Status GraphPassExtInst::Process() {
    GraphExtInstContext loweringContext{*context(), graphPipeline};
    std::unordered_map<const GraphExtInstSetDescriptor *, std::unique_ptr<GraphExtInstDecoder>> decoderCache;
    Decoders decoders;
    for (const auto &[id, descriptor] : imports) {
        auto &decoder = decoderCache[descriptor];
        if (!decoder) {
            decoder = descriptor->createDecoder(loweringContext);
        }
        decoders.emplace(id, decoder.get());
    }
    handleGraphs(loweringContext, decoders);
    return Status::SuccessWithChange;
}

void GraphPassExtInst::handleGraphs(GraphExtInstContext &loweringContext, const Decoders &decoders) {
    const auto &module = *get_module();

    for (const auto &graphEntry : module.graph_entry_points()) {
        graphLog(Severity::Info) << graphEntry << std::endl;

        const auto *graph = loweringContext.getGraphById(graphEntry.GetOperand(0));
        assert(graph != nullptr);

        loweringContext.handleGraphConstants();
        loweringContext.handleInputsAndOutputs(graphEntry);
        handleGraph(graph, decoders);
    }
}

void GraphPassExtInst::handleGraph(const Graph *graph, const Decoders &decoders) {
    // Iterate over instructions in the graph
    for (const auto &opExtInst : graph->instructions()) {

        // OpExtInst <OpExtInstImport id> <extended instruction opcode> [arguments]
        switch (opExtInst->opcode()) {
        case spv::Op::OpExtInst:
            break;
        case spv::Op::OpCompositeExtract:
            continue;
        default:
            throw std::runtime_error(std::string("Unsupported graph instruction ") +
                                     std::to_string(static_cast<unsigned>(opExtInst->opcode())));
        }

        const auto importId = opExtInst->GetInOperand(0).AsId();
        const auto decoder = decoders.find(importId);
        if (decoder == decoders.end()) {
            throw std::runtime_error("Unregistered extended instruction-set import id " + std::to_string(importId));
        }
        try {
            decoder->second->handleOp(opExtInst.get());
        } catch (const std::bad_alloc &) {
            throw;
        } catch (const std::exception &error) {
            throw std::runtime_error(imports.at(importId)->importName + " import %" + std::to_string(importId) +
                                     " opcode " + std::to_string(opExtInst->GetSingleWordInOperand(1)) + " result %" +
                                     std::to_string(opExtInst->result_id()) + ": " + error.what());
        }
    }
}

} // namespace spvtools::opt
