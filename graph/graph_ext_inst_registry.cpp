/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "graph_ext_inst_registry.hpp"

#include "graph_motion_engine_1_0_ext_inst.hpp"
#include "graph_tosa_ext_inst.hpp"
#include "source/opt/build_module.h"

#include <spirv-tools/libspirv.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

/*******************************************************************************
 * GraphPass extended instruction sets
 *******************************************************************************/
namespace spvtools::opt {

namespace {

template <typename Decoder> std::unique_ptr<GraphExtInstDecoder> makeDecoder(GraphExtInstContext &context) {
    return std::make_unique<Decoder>(context);
}

const std::vector<GraphExtInstSetDescriptor> &registeredGraphExtInstSets() {
    static const std::vector<GraphExtInstSetDescriptor> instructionSets{
        {std::string(tosaSpv100), &makeDecoder<GraphTosaExtInst>},
#if VMEL_ENABLE_TOSA_1X
        {std::string(tosaSpv1x), &makeDecoder<GraphTosaExtInst>},
#endif
        {std::string(motionEngine100), &makeDecoder<GraphMotionEngine10ExtInst>},
    };
    return instructionSets;
}

} // namespace

const GraphExtInstSetDescriptor *findGraphExtInstSet(const std::string_view importName) {
    const auto &instructionSets = registeredGraphExtInstSets();
    const auto found = std::find_if(instructionSets.begin(), instructionSets.end(),
                                    [&importName](const auto &entry) { return entry.importName == importName; });
    return found == instructionSets.end() ? nullptr : &*found;
}

GraphModuleInfo analyzeGraphModule(const uint32_t *code, size_t wordCount) {
    std::string diagnostic;
    auto ir = BuildModule(
        SPV_ENV_UNIVERSAL_1_6,
        [&diagnostic](spv_message_level_t, const char *, const spv_position_t &position, const char *message) {
            diagnostic = std::string(message) + " at word " + std::to_string(position.index);
        },
        code, wordCount);
    if (!ir || !ir->module()) {
        throw std::runtime_error("Cannot decode graph module: " + diagnostic);
    }
    auto &module = *ir->module();
    if (module.graphs().empty()) {
        throw std::runtime_error("Module contains no graph");
    }
    // Tools cannot analyze scaled tensor types yet, including unused ones.
    for (const auto &inst : module.types_values()) {
        if (std::string_view(spvOpcodeString(static_cast<uint32_t>(inst.opcode()))) == "TypeTensorScaledBlockARM") {
            throw std::runtime_error(
                "OpTypeTensorScaledBlockARM is unsupported; rejected before optimizer type analysis");
        }
    }
    GraphModuleInfo info;
    for (const auto &inst : module.ext_inst_imports()) {
        const auto name = inst.GetInOperand(0).AsString();
        const auto *descriptor = findGraphExtInstSet(name);
        if (!descriptor) {
            if (name.rfind("TOSA.", 0) == 0 || name.rfind("Arm.MotionEngine.", 0) == 0) {
                throw std::runtime_error("Unsupported graph extended instruction set: " + name);
            }
            continue;
        }
        info.extInstImports.emplace(inst.result_id(), descriptor);
        info.tosa |= name == tosaSpv100 || name == tosaSpv1x;
        info.motionEngine |= name == motionEngine100;
    }
    return info;
}

} // namespace spvtools::opt
