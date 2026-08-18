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
#include "graph_tosa_1_0_ext_inst.hpp"

#include <algorithm>

/*******************************************************************************
 * GraphPass extended instruction sets
 *******************************************************************************/
namespace spvtools::opt {

namespace {

using GraphExtInstDecoderFactory = std::unique_ptr<GraphExtInstDecoder> (*)(GraphExtInstContext &);

struct GraphExtInstSetDescriptor {
    std::string importName;
    GraphExtInstDecoderFactory createDecoder;
};

template <typename Decoder> std::unique_ptr<GraphExtInstDecoder> makeDecoder(GraphExtInstContext &context) {
    return std::make_unique<Decoder>(context);
}

const std::vector<GraphExtInstSetDescriptor> &registeredGraphExtInstSets() {
    static const std::vector<GraphExtInstSetDescriptor> instructionSets{
        {std::string(tosaSpv100), &makeDecoder<GraphTosa10ExtInst>},
        {std::string(motionEngine100), &makeDecoder<GraphMotionEngine10ExtInst>},
    };
    return instructionSets;
}

} // namespace

bool isRegisteredGraphExtInstImport(const std::string_view importName) {
    const auto &instructionSets = registeredGraphExtInstSets();
    return std::find_if(instructionSets.begin(), instructionSets.end(), [&importName](const auto &entry) {
               return entry.importName == importName;
           }) != instructionSets.end();
}

std::vector<GraphExtInstSetEntry> makeGraphExtInstRegistry(GraphExtInstContext &context) {
    std::vector<GraphExtInstSetEntry> registry;
    for (const auto &descriptor : registeredGraphExtInstSets()) {
        registry.push_back({descriptor.importName, descriptor.createDecoder(context)});
    }
    return registry;
}

} // namespace spvtools::opt
