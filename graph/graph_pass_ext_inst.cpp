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

#include <algorithm>
#include <unordered_map>

/*******************************************************************************
 * GraphPass extended instruction sets
 *******************************************************************************/
using namespace mlsdk::el::log;

namespace spvtools::opt {

Pass::Status GraphPassExtInst::Process() {
    GraphExtInstContext loweringContext{*context(), graphPipeline};
    handleGraphs(loweringContext);
    return Status::SuccessWithChange;
}

void GraphPassExtInst::handleGraphs(GraphExtInstContext &loweringContext) {
    const auto &module = *get_module();

    for (const auto &graphEntry : module.graph_entry_points()) {
        graphLog(Severity::Info) << graphEntry << std::endl;

        const auto *graph = loweringContext.getGraphById(graphEntry.GetOperand(0));
        assert(graph != nullptr);

        loweringContext.handleGraphConstants();
        loweringContext.handleInputsAndOutputs(graphEntry);
        handleGraph(graph, loweringContext);
    }
}

void GraphPassExtInst::handleGraph(const Graph *graph, GraphExtInstContext &loweringContext) {
    const auto instructionSets = makeGraphExtInstRegistry(loweringContext);
    std::unordered_map<uint32_t, const GraphExtInstDecoder *> decodersByImportId;

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
        auto decoder = decodersByImportId.find(importId);
        if (decoder == decodersByImportId.end()) {
            const auto *importInstruction = get_def_use_mgr()->GetDef(importId);
            if (importInstruction == nullptr || importInstruction->opcode() != spv::Op::OpExtInstImport) {
                throw std::runtime_error("Invalid extended instruction-set import id " + std::to_string(importId));
            }
            const auto &importName = importInstruction->GetInOperand(0).AsString();
            const auto instructionSet =
                std::find_if(instructionSets.begin(), instructionSets.end(),
                             [&importName](const auto &entry) { return entry.importName == importName; });
            if (instructionSet == instructionSets.end()) {
                throw std::runtime_error(std::string("Unsupported extension ") + importName);
            }
            decoder = decodersByImportId.emplace(importId, instructionSet->decoder.get()).first;
        }
        decoder->second->handleOp(opExtInst.get());
    }
}

} // namespace spvtools::opt
