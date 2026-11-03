/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

#include "graph_ext_inst_decoder.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>

namespace spvtools::opt {

class GraphExtInstContext;

using GraphExtInstDecoderFactory = std::unique_ptr<GraphExtInstDecoder> (*)(GraphExtInstContext &);

struct GraphExtInstSetDescriptor {
    std::string importName;
    GraphExtInstDecoderFactory createDecoder;
};

// Descriptors belong to the static registry and outlive module analysis and passes.
using GraphExtInstImports = std::unordered_map<uint32_t, const GraphExtInstSetDescriptor *>;

struct GraphModuleInfo {
    GraphExtInstImports extInstImports;
    bool tosa = false;
    bool motionEngine = false;
};

const GraphExtInstSetDescriptor *findGraphExtInstSet(std::string_view importName);

// Resolve imports and reject scaled declarations before optimizer type analysis.
// Module validity is a caller precondition.
GraphModuleInfo analyzeGraphModule(const uint32_t *code, size_t wordCount);

} // namespace spvtools::opt
