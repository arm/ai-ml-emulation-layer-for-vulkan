/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/*******************************************************************************
 * Includes
 *******************************************************************************/

#pragma once

#include "compute_graph_op.hpp"
#include "mlel/float.hpp"
#include "source/opt/pass.h"

#include <numeric>
#include <spirv-tools/optimizer.hpp>
#include <type_traits>

/*******************************************************************************
 * Base Graph Pass
 *******************************************************************************/

using namespace mlsdk::el::compute::graph_op;

namespace spvtools {

namespace opt {

enum RoundingMode {
    SingleRound = 1,
    InexactRound = 2,
    DoubleRound = 3,
};

class GraphPassBase : public Pass {
  public:
    GraphPassBase(GraphPipeline &_graphPipeline) : graphPipeline{_graphPipeline} {}

    ~GraphPassBase() override = default;

  protected:
    Status Process() override;
    virtual void handleGraph(const Graph *graph) = 0;

    void handleGraphConstants();
    void handleGraphs();
    void handleInputsAndOutputs(const Instruction &opGraphEntryPoint);
    const Graph *getGraphById(const Operand &operand);

    const analysis::TensorARM *getTensorType(const Operand &operand) const;
    const analysis::TensorARM *getTensorType(uint32_t id) const;
    std::tuple<uint64_t, uint64_t> getDescriptorSetAndBinding(const Operand &operand);
    std::tuple<uint64_t, uint64_t, std::shared_ptr<mlsdk::el::compute::TensorDescriptor>>
    getTensorByDecoration(const Operand &operand, uint32_t arrayIndex);
    std::shared_ptr<mlsdk::el::compute::TensorDescriptor> getTensor(const Instruction &instruction,
                                                                    uint32_t arrayIndex = 0);
    std::shared_ptr<mlsdk::el::compute::TensorDescriptor> getTensor(const Operand &operand, uint32_t arrayIndex = 0);
    std::shared_ptr<mlsdk::el::compute::TensorDescriptor> makeTensor(const analysis::TensorARM *tensor) const;
    std::shared_ptr<mlsdk::el::compute::TensorDescriptor> getOrMakeCompositeTensor(uint32_t id);
    std::shared_ptr<mlsdk::el::compute::TensorDescriptor> makeCompositeTensor(uint32_t id) const;
    VkFormat getVkFormat(const analysis::Type *type) const;
    bool getBoolConstant(const Operand &operand);

    // Temp implementation until debug info is truly available
    std::string extractDebugInfoFromSPV(const Instruction *, const std::string &defaultname) { return defaultname; }

    template <typename T = uint32_t> std::vector<T> getConstVector(const Operand &operand) const {
        return getConstVector<T>(operand.AsId());
    }

    template <typename T = uint32_t>
    void getFlattenedCompositeConstant(const spvtools::opt::analysis::CompositeConstant *composite,
                                       std::vector<T> &kernel) const {
        const auto &components = composite->GetComponents();
        kernel.reserve(kernel.size() + components.size());
        for (const auto *component : components) {
            if (const auto *innerComposite = component->AsCompositeConstant()) {
                getFlattenedCompositeConstant(innerComposite, kernel);
            } else {
                kernel.push_back(getConstScalar<T>(component));
            }
        }
    }

    bool shouldDecodeTensorConstantFromInstruction(const Instruction *instruction,
                                                   const analysis::Constant *constant) const {
        if (!instruction) {
            return false;
        }

        const auto *type = context()->get_type_mgr()->GetType(instruction->type_id());
        if (!type || !type->AsTensorARM()) {
            return false;
        }

        switch (instruction->opcode()) {
        case spv::Op::OpConstantComposite:
        case spv::Op::OpSpecConstantComposite:
        case spv::Op::OpConstantCompositeReplicateEXT:
        case spv::Op::OpSpecConstantCompositeReplicateEXT:
            return true;
        case spv::Op::OpConstantNull:
            return constant == nullptr;
        default:
            return false;
        }
    }

    template <typename T = uint32_t>
    void appendConstantVectorFromInstruction(uint32_t id, std::vector<T> &kernel) const {
        const auto *instruction = context()->get_def_use_mgr()->GetDef(id);
        if (!instruction) {
            throw std::runtime_error("Missing definition for constant id: " + std::to_string(id));
        }

        switch (instruction->opcode()) {
        case spv::Op::OpConstantComposite:
        case spv::Op::OpSpecConstantComposite:
        case spv::Op::OpConstantCompositeReplicateEXT:
        case spv::Op::OpSpecConstantCompositeReplicateEXT:
            for (uint32_t i = 0; i < instruction->NumInOperands(); ++i) {
                appendConstantVectorFromInstruction<T>(instruction->GetInOperand(i).AsId(), kernel);
            }
            return;
        case spv::Op::OpConstantNull: {
            const auto *type = context()->get_type_mgr()->GetType(instruction->type_id());
            if (const auto *tensorType = type ? type->AsTensorARM() : nullptr) {
                kernel.resize(kernel.size() + getElementCount(tensorType->shape_id()), 0);
            } else {
                kernel.push_back(0);
            }
            return;
        }
        default:
            if (const auto *constant = context()->get_constant_mgr()->FindDeclaredConstant(id)) {
                kernel.push_back(getConstScalar<T>(constant));
                return;
            }
            throw std::runtime_error("Unsupported constant definition opcode for id " + std::to_string(id) + ": " +
                                     std::to_string(static_cast<uint32_t>(instruction->opcode())));
        }
    }

    template <typename T = uint32_t> std::vector<T> getTensorConstantVectorFromInstruction(uint32_t id) const {
        const auto *instruction = context()->get_def_use_mgr()->GetDef(id);
        if (!instruction) {
            throw std::runtime_error("Missing definition for constant id: " + std::to_string(id));
        }

        std::vector<T> kernel;
        const bool isSplat = isCompositeReplicateConstantOpcode(instruction->opcode());
        appendConstantVectorFromInstruction<T>(id, kernel);

        if (isSplat) {
            assert(kernel.size() == 1);
            const auto *tensorType = getTensorType(id);
            const auto elemCount = getElementCount(tensorType->shape_id());
            kernel.resize(elemCount, kernel.front());
        }

        return kernel;
    }

    template <typename T = uint32_t> std::vector<T> getConstVector(const uint32_t id) const {
        const auto *constant = context()->get_constant_mgr()->FindDeclaredConstant(id);
        std::vector<T> kernel;

        {
            // TODO: Remove this fallback once our SPIRV-Tools dependency includes a
            // ConstantManager fix for TensorARM OpConstantCompositeReplicateEXT and
            // OpSpecConstantCompositeReplicateEXT constants.
            const auto *instruction = context()->get_def_use_mgr()->GetDef(id);
            if (shouldDecodeTensorConstantFromInstruction(instruction, constant)) {
                return getTensorConstantVectorFromInstruction<T>(id);
            }
            if (!constant) {
                throw std::runtime_error("Missing declared constant for id: " + std::to_string(id));
            }
        }

        if (const auto *composite = constant->AsCompositeConstant()) {
            const auto *instruction = context()->get_def_use_mgr()->GetDef(id);
            const bool isSplat = isCompositeReplicateConstantOpcode(instruction->opcode());
            getFlattenedCompositeConstant(composite, kernel);

            if (isSplat) {
                assert(kernel.size() == 1);
                const auto *tensorType = getTensorType(id);
                const auto elemCount = getElementCount(tensorType->shape_id());
                kernel.resize(elemCount, kernel.front());
            }
        } else if (const auto *null = constant->AsNullConstant(); null != nullptr) {
            if (const auto *tensor = constant->type()->AsTensorARM()) {
                // TensorARM: zero-initialize a composite tensor with the total element count
                const auto elemCount = getElementCount(tensor->shape_id());
                kernel.resize(elemCount, 0);
            } else {
                assert(false);
            }
        } else {
            assert(false);
        }

        return kernel;
    }

    template <typename T = int64_t> T getConstScalar(const Operand &operand) const {
        return getConstScalar<T>(context()->get_constant_mgr()->FindDeclaredConstant(operand.AsId()));
    }

    template <typename T = int64_t> T getConstScalar(const analysis::Constant *constant) const {
        const auto *intConstant = constant->AsIntConstant();
        if (intConstant) {
            const auto *type = intConstant->type()->AsInteger();

            if (type->IsSigned()) {
                return static_cast<T>(constant->GetSignExtendedValue());
            }

            switch (type->width()) {
            case 8:
                return T(int8_t(constant->GetZeroExtendedValue()));
            case 16:
                return T(int16_t(constant->GetZeroExtendedValue()));
            case 32:
                return T(int32_t(constant->GetZeroExtendedValue()));
            case 64:
                return T(int64_t(constant->GetZeroExtendedValue()));
            default:
                throw std::runtime_error(std::string("Unsupported integer constant width: ") +
                                         std::to_string(type->width()));
            }
        }

        const auto *floatConstant = constant->AsFloatConstant();
        if (floatConstant) {
            const auto *type = floatConstant->type()->AsFloat();

            switch (type->width()) {
            case 8: {
                if (type->encoding() == spv::FPEncoding::Float8E5M2EXT) {
                    const auto value = uint8_t(floatConstant->words()[0]);
                    const auto &fp = reinterpret_cast<const float8_e5m2 &>(value);
                    return T(fp);
                }
                if (type->encoding() == spv::FPEncoding::Float8E4M3EXT) {
                    const auto value = uint8_t(floatConstant->words()[0]);
                    const auto &fp = reinterpret_cast<const float8_e4m3 &>(value);
                    return T(fp);
                }
                throw std::runtime_error(std::string("Unsupported 8-bit float encoding: ") +
                                         std::to_string(static_cast<uint32_t>(type->encoding())));
            }
            case 16: {
                const auto value = uint16_t(floatConstant->words()[0]);
                const auto &fp = reinterpret_cast<const float16 &>(value);
                return T(fp);
            }
            case 32:
                return T(floatConstant->GetFloatValue());
            case 64:
                return T(floatConstant->GetDoubleValue());
            default:
                throw std::runtime_error(std::string("Unsupported constant float width: ") +
                                         std::to_string(type->width()));
            }
        }

        const auto *boolConstant = constant->AsBoolConstant();
        if (boolConstant) {
            return T(boolConstant->value() ? 1 : 0);
        }

        throw std::runtime_error(std::string("Unsupported constant type: ") + std::to_string(constant->type()->kind()) +
                                 " for requested return type: " + typeid(T).name());
    }

    static bool isBFloat16(const spvtools::opt::analysis::Float *f);
    static bool isFloat8E5M2(const spvtools::opt::analysis::Float *f);
    static bool isFloat8E4M3(const spvtools::opt::analysis::Float *f);

    GraphPipeline &graphPipeline;

  private:
    // Local cache from SPIR-V result id to the tensor descriptors used while lowering a graph.
    // Slot 1 is for multi-result logical values (for example FFT-style ops), not descriptor array elements.
    std::map<uint32_t, std::array<std::shared_ptr<mlsdk::el::compute::TensorDescriptor>, 2>> tensorMap;

    size_t getElementCount(uint32_t id) const;

    static bool isCompositeReplicateConstantOpcode(spv::Op opcode);

    template <typename T, spv::FPEncoding fpEncoding>
    std::vector<T> getConstVector(const spvtools::opt::Instruction *instruction,
                                  const std::vector<int64_t> &dimensions) const;
};

} // namespace opt

/*******************************************************************************
 * Create pass
 *******************************************************************************/

template <typename T, std::enable_if_t<std::is_base_of_v<opt::GraphPassBase, T>, bool> = true>
Optimizer::PassToken CreateGraphPass(GraphPipeline &graphPipeline) {
    return Optimizer::PassToken{MakeUnique<T>(graphPipeline)};
}

} // namespace spvtools
