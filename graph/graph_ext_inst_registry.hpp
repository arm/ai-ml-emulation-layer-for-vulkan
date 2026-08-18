/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

#include "graph_ext_inst_decoder.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace spvtools::opt {

class GraphExtInstContext;

struct GraphExtInstSetEntry {
    std::string importName;
    std::unique_ptr<GraphExtInstDecoder> decoder;
};

bool isRegisteredGraphExtInstImport(std::string_view importName);
std::vector<GraphExtInstSetEntry> makeGraphExtInstRegistry(GraphExtInstContext &context);

} // namespace spvtools::opt
