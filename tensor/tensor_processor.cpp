/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include "tensor_processor.hpp"

#include "descriptor_binding.hpp"
#include "mlel/utils.hpp"
#include "tensor_log.hpp"
#include <map>
#include <set>
#include <spirv_cross_error_handling.hpp>
#include <sstream>

using namespace mlsdk::el::utils;
using namespace mlsdk::el::log;

namespace mlsdk::el::layer {

TensorProcessor::TensorProcessor(std::vector<uint32_t> spirv_) : m_spirv{std::move(spirv_)} {
    try {
        m_spirvCompiler = std::make_unique<CompilerTensorAsBuffer>(m_spirv);
        bool hasTensor = m_spirvCompiler->hasTensors();
        bool isCompute = m_spirvCompiler->isCompute();
        m_isTensorCompute = hasTensor && isCompute;
        if (hasTensor && !isCompute) {
            tensorLog(Severity::Error)
                << "SPIR-V at: " << m_spirv.data()
                << " uses OpTypeTensorARM but is not a compute shader. This is unsupported by the Tensor Layer."
                << std::endl;
            m_isValid = false;
        } else {
            m_isValid = true;
        }
    } catch (spirv_cross::CompilerError const &) {
        // bypass SPIRV that can't be compiled with SPIRV-Cross, and log a warning
        tensorLog(Severity::Error) << "SPIR-V at:" << m_spirv.data() << " could not be parsed." << std::endl;
        m_isValid = false;
    }
}

bool TensorProcessor::isTensorComputeShader() const { return m_isTensorCompute; }

bool TensorProcessor::isValidShader() const { return m_isValid; }

std::vector<uint32_t> TensorProcessor::getNewSpirv() const {
    if (!isTensorComputeShader()) {
        return m_spirv;
    }

    std::string glslSource = m_spirvCompiler->compile();

    tensorLog(Severity::Debug) << glslSource;

    return glslToSpirv(glslSource);
}
} // namespace mlsdk::el::layer
