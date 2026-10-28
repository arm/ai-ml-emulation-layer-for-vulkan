/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include "spirv_tensor_lowering.hpp"

#include "source/opt/pass.h"

#include <spirv-tools/optimizer.hpp>

#include <memory>

namespace mlsdk::el::layer {
namespace {

class TensorLoweringPass final : public spvtools::opt::Pass {
  public:
    const char *name() const override { return "tensor-lowering"; }

  private:
    Status Process() override;
};

spvtools::opt::Pass::Status TensorLoweringPass::Process() {
    const auto *features = context()->get_feature_mgr();
    if (!features->HasCapability(spv::Capability::TensorsARM) &&
        !features->HasExtension(spvtools::Extension::kSPV_ARM_tensors)) {
        return Status::SuccessWithoutChange;
    }

    // TODO: Transform tensor types and interfaces.
    // TODO: Generate read/write helpers and size-query replacements.
    // TODO: Remove replaced TensorARM declarations.
    if (consumer()) {
        consumer()(SPV_MSG_ERROR, nullptr, {}, "Unsupported TensorARM module in direct SPIR-V lowering");
    }
    return Status::Failure;
}

} // namespace

bool lowerTensorSpirv(std::vector<uint32_t> &spirv, bool &loweredTensorCompute,
                      const spvtools::MessageConsumer &messageConsumer) {
    loweredTensorCompute = false;
    spvtools::Optimizer optimizer{SPV_ENV_VULKAN_1_3};
    optimizer.SetMessageConsumer(messageConsumer);
    optimizer.RegisterPass(spvtools::Optimizer::PassToken{std::make_unique<TensorLoweringPass>()});

    spvtools::OptimizerOptions options;
    options.set_run_validator(false);
    std::vector<uint32_t> output;
    if (!optimizer.Run(spirv.data(), spirv.size(), &output, options)) {
        spirv.clear();
        return false;
    }
    // Optimizer serialization can omit OpNop; preserve unchanged modules exactly.
    return true;
}

} // namespace mlsdk::el::layer
