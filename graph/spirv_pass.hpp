/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/*******************************************************************************
 * Includes
 *******************************************************************************/

#pragma once

#include "compute.hpp"
#include "mlel/float.hpp"
#include "source/opt/pass.h"

#include <numeric>
#include <spirv-tools/optimizer.hpp>
#include <type_traits>

/*******************************************************************************
 * Base Graph Pass
 *******************************************************************************/

namespace spvtools {

namespace opt {

enum RoundingMode {
    SingleRound = 1,
    InexactRound = 2,
    DoubleRound = 3,
};

class GraphPassBase : public Pass {
  public:
    GraphPassBase(mlsdk::el::compute::GraphPipeline &_graphPipeline) : graphPipeline{_graphPipeline} {}

    ~GraphPassBase() override = default;

  protected:
    Status Process() override;
    virtual void handleGraph(const Graph *graph) = 0;

    void handleGraphConstants();
    void handleGraphs();
    void handleInputsAndOutputs(const Instruction &opGraphEntryPoint);
    const Graph *getGraphById(const Operand &operand);
    std::tuple<std::vector<analysis::TensorARM *>, std::vector<analysis::TensorARM *>>
    getGraphType(const Operand &operand);
    analysis::TensorARM *getTensorType(const Operand &operand, const uint32_t index = 0) const;
    analysis::TensorARM *getTensorType(uint32_t id, const uint32_t index = 0) const;
    std::tuple<uint64_t, uint64_t> getDescriptorSetAndBinding(const Operand &operand);
    std::tuple<uint64_t, uint64_t, std::shared_ptr<mlsdk::el::compute::TensorDescriptor>>
    getTensorByDecoration(const Operand &operand, const uint32_t arrayIndex);
    void mapTensorByDecoration(uint32_t resultId, const Operand &operand, const uint32_t arrayIndex = 0);
    std::shared_ptr<mlsdk::el::compute::TensorDescriptor> getTensor(const Instruction &instruction,
                                                                    const uint32_t arrayIndex = 0);
    std::shared_ptr<mlsdk::el::compute::TensorDescriptor> getTensor(const Operand &operand,
                                                                    const uint32_t arrayIndex = 0);
    std::shared_ptr<mlsdk::el::compute::TensorDescriptor> makeTensor(const analysis::TensorARM *tensor) const;
    std::shared_ptr<mlsdk::el::compute::TensorDescriptor> getOrMakeCompositeTensor(const uint32_t id) const;
    std::shared_ptr<mlsdk::el::compute::TensorDescriptor> makeCompositeTensor(const uint32_t id) const;
    VkFormat getVkFormat(const analysis::Type *type) const;
    bool getBoolConstant(const Operand &operand);
    std::string extractDebugInfoFromSPV(const Instruction *opExtInst, const std::string &defaultname);
    template <typename T = uint32_t> std::vector<T> getConstVector(const Operand &operand) const {
        return getConstVector<T>(operand.AsId());
    }

    template <typename T = uint32_t>
    void getFlattenedCompositeConstant(const spvtools::opt::analysis::CompositeConstant *composite,
                                       std::vector<T> &kernel) const {
        for (const auto &component : composite->GetComponents()) {
            if (const auto &innerComposite = component->AsCompositeConstant()) {
                getFlattenedCompositeConstant(innerComposite, kernel);
            } else {
                kernel.push_back(getConstScalar<T>(component));
            }
        }
    }

    template <typename T = uint32_t> std::vector<T> getConstVector(const uint32_t id) const {
        const auto &constant = context()->get_constant_mgr()->FindDeclaredConstant(id);
        std::vector<T> kernel;

        if (const auto &composite = constant->AsCompositeConstant()) {
            const auto &instruction = context()->get_def_use_mgr()->GetDef(id);
            bool isSplat = instruction->opcode() == spv::Op::OpConstantCompositeReplicateEXT ||
                           instruction->opcode() == spv::Op::OpSpecConstantCompositeReplicateEXT;
            getFlattenedCompositeConstant(composite, kernel);

            if (isSplat) {
                assert(kernel.size() == 1);
                const auto tensorType = getTensorType(id);
                const auto &dimensions = getConstVector<int64_t>(tensorType->shape_id());
                size_t compositeCount =
                    std::accumulate(dimensions.begin(), dimensions.end(), size_t{1},
                                    [](size_t acc, int64_t dim) { return acc * static_cast<size_t>(dim); });
                kernel.resize(compositeCount, kernel.front());
            }
        } else if (const auto &null = constant->AsNullConstant()) {
            if (const auto &tensor = constant->type()->AsTensorARM()) {
                // TensorARM rank=1, shape=[array size]
                // Tensor must be rank 1, and first element of the shape is the number of vector elements
                const auto &shape = getConstVector<T>(tensor->shape_id());
                assert(shape.size() == 1);
                kernel.resize(static_cast<size_t>(shape[0]));
            } else {
                assert(false);
            }
        } else {
            assert(false);
        }

        return kernel;
    }

    template <typename T = int64_t> T getConstScalar(const Operand &operand, const bool isUnsigned = false) const {
        return getConstScalar<T>(context()->get_constant_mgr()->FindDeclaredConstant(operand.AsId()), isUnsigned);
    }

    template <typename T = int64_t>
    T getConstScalar(const analysis::Constant *constant, const bool isUnsigned = false) const {
        const auto &intConstant = constant->AsIntConstant();
        if (intConstant) {
            const auto &type = intConstant->type()->AsInteger();

            if (type->IsSigned()) {
                return static_cast<T>(constant->GetSignExtendedValue());
            } else {
                switch (type->width()) {
                case 8:
                    return isUnsigned ? T(constant->GetZeroExtendedValue())
                                      : T(int8_t(constant->GetZeroExtendedValue()));
                case 16:
                    return isUnsigned ? T(constant->GetZeroExtendedValue())
                                      : T(int16_t(constant->GetZeroExtendedValue()));
                case 32:
                    return isUnsigned ? T(constant->GetZeroExtendedValue())
                                      : T(int32_t(constant->GetZeroExtendedValue()));
                case 64:
                    return isUnsigned ? T(constant->GetZeroExtendedValue())
                                      : T(int64_t(constant->GetZeroExtendedValue()));
                default:
                    throw std::runtime_error(std::string("Unsupported integer constant width: ") +
                                             std::to_string(type->width()));
                }
            }
        }

        const auto &floatConstant = constant->AsFloatConstant();
        if (floatConstant) {
            const auto &type = floatConstant->type()->AsFloat();

            switch (type->width()) {
            case 16: {
                const uint16_t value = uint16_t(floatConstant->words()[0]);
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

        const auto &boolConstant = constant->AsBoolConstant();
        if (boolConstant) {
            return T(boolConstant->value() ? 1 : 0);
        }

        throw std::runtime_error(std::string("Unsupported constant type: ") + std::to_string(constant->type()->kind()));
    }

    mlsdk::el::compute::GraphPipeline &graphPipeline;
    VkDevice device = {};
    std::map<uint32_t, std::array<std::shared_ptr<mlsdk::el::compute::TensorDescriptor>, 2>> tensorMap;
};

} // namespace opt

/*******************************************************************************
 * Create pass
 *******************************************************************************/

template <typename T, std::enable_if_t<std::is_base_of_v<opt::GraphPassBase, T>, bool> = true>
Optimizer::PassToken CreateGraphPass(mlsdk::el::compute::GraphPipeline &graphPipeline) {
    return Optimizer::PassToken{MakeUnique<T>(graphPipeline)};
}

} // namespace spvtools
