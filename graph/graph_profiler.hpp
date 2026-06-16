/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "compute_graph_op.hpp"
#include "compute_optical_flow.hpp"

#include <nlohmann/json.hpp>
#include <vulkan/vulkan.hpp>

#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace mlsdk::el::layer {

/*******************************************************************************
 * GraphProfiler
 *******************************************************************************/

enum class ProfilingPipelineKind {
    GRAPH_OP,
    TOSA,
    MOTION_ENGINE,
};

class GraphProfiler {
  public:
    static bool isEnabled();

    GraphProfiler(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader,
                  VkPhysicalDevice _physicalDevice, VkDevice _device);
    ~GraphProfiler();

    GraphProfiler(const GraphProfiler &) = delete;
    GraphProfiler &operator=(const GraphProfiler &) = delete;
    GraphProfiler(GraphProfiler &&) = delete;
    GraphProfiler &operator=(GraphProfiler &&) = delete;

    mlsdk::el::compute::graph_op::ComputePipelineDispatchDecorator
    makeDispatchDecorator(VkPipeline dataGraphPipeline, VkCommandBuffer commandBuffer, uint32_t queueFamilyIndex,
                          uint32_t pipelineCount, ProfilingPipelineKind pipelineKind = ProfilingPipelineKind::TOSA);
    mlsdk::el::compute::optical_flow::ComputePipelineDispatchDecorator
    makeOpticalFlowDispatchDecorator(VkPipeline dataGraphPipeline, VkCommandBuffer commandBuffer,
                                     uint32_t queueFamilyIndex, uint32_t pipelineCount);
    bool hasProfiledCommandBuffers(const std::vector<VkCommandBuffer> &commandBuffers) const;
    void prepareCommandBuffersForSubmit(const std::vector<VkCommandBuffer> &commandBuffers);
    void registerSubmit(VkQueue queue, const std::vector<VkCommandBuffer> &commandBuffers, VkFence fence);
    void registerExecuteCommands(VkCommandBuffer commandBuffer, uint32_t commandBufferCount,
                                 const VkCommandBuffer *pCommandBuffers);
    void collectFence(VkFence fence);
    void collectQueue(VkQueue queue);
    void collectDevice();
    void clearCommandBuffer(VkCommandBuffer commandBuffer);
    std::string getPipelineJson(VkPipeline dataGraphPipeline);

  private:
    struct QueryPoolRecord;
    struct Sample;
    struct Aggregate;
    struct SubmitRecord;
    using Submissions = std::vector<std::shared_ptr<SubmitRecord>>;
    using CompletedSubmissions = std::vector<std::pair<std::shared_ptr<SubmitRecord>, std::vector<Sample>>>;

    class LockedState {
      public:
        uint64_t nextGraphDispatchIndex();
        void addCommandBufferRecord(VkCommandBuffer commandBuffer, const std::shared_ptr<QueryPoolRecord> &record);
        bool hasProfiledCommandBuffers(const std::vector<VkCommandBuffer> &commandBuffers) const;
        void registerSubmit(VkQueue queue, const std::vector<VkCommandBuffer> &commandBuffers, VkFence fence);
        void registerExecuteCommands(VkCommandBuffer commandBuffer, uint32_t commandBufferCount,
                                     const VkCommandBuffer *pCommandBuffers);
        Submissions getSubmissionsForFence(VkFence fence) const;
        Submissions getSubmissionsForQueue(VkQueue queue) const;
        Submissions getSubmissionsForDevice() const;
        Submissions getSubmissionsForCommandBuffers(const std::vector<VkCommandBuffer> &commandBuffers) const;
        Submissions getSubmissionsForCommandBuffer(VkCommandBuffer commandBuffer) const;
        void completeSubmissions(const CompletedSubmissions &completedSubmissions);
        void clearCommandBuffer(VkCommandBuffer commandBuffer);
        void clearAllCommandBuffers();
        std::vector<Sample> getSamples() const;
        std::vector<Sample> getSamples(VkPipeline dataGraphPipeline) const;

      private:
        std::vector<std::shared_ptr<QueryPoolRecord>>
        getRecordsForCommandBuffersLocked(const std::vector<VkCommandBuffer> &commandBuffers) const;
        void removeSubmissionsForCommandBufferLocked(VkCommandBuffer commandBuffer);

        mutable std::mutex mutex;
        uint64_t graphDispatchCounter{};
        uint64_t submissionCounter{};
        std::map<VkCommandBuffer, std::vector<std::shared_ptr<QueryPoolRecord>>> commandBufferRecords;
        Submissions submissions;
        std::vector<Sample> samples;
    };

    VkQueryPool getQueryPool(uint32_t queueFamilyIndex, uint32_t pipelineCount) const;

    std::shared_ptr<QueryPoolRecord> makeRecord(VkQueryPool queryPool, VkCommandBuffer commandBuffer,
                                                VkPipeline dataGraphPipeline);
    bool collectSubmission(const std::shared_ptr<SubmitRecord> &submission, std::vector<Sample> &newSamples);
    void collectSubmissions(const Submissions &submitRecords);
    void clearAllCommandBuffers();
    std::string makeJson() const;
    std::string makeJson(VkPipeline dataGraphPipeline) const;
    std::string makeJson(const std::vector<Sample> &profileSamples) const;
    static nlohmann::ordered_json toJson(const Sample &sample);
    static nlohmann::ordered_json toJson(const Aggregate &aggregate, const std::string &pipelineKind,
                                         const std::string &operatorName);
    bool supportsTimestampQueries(uint32_t queueFamilyIndex) const;

    std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> loader;
    VkPhysicalDevice physicalDevice{};
    VkDevice device{};
    float timestampPeriod{};
    std::vector<bool> queueFamilyTimestampSupport;
    LockedState state;
};

} // namespace mlsdk::el::layer
