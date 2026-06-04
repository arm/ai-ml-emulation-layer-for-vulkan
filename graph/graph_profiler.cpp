/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "graph_profiler.hpp"

#include "graph_log.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iterator>

using namespace mlsdk::el::compute::graph_op;
using namespace mlsdk::el::log;

namespace mlsdk::el::layer {
namespace {

bool isTruthyEnvironmentValue(const char *value) {
    if (value == nullptr || value[0] == '\0') {
        return false;
    }

    std::string str{value};
    std::transform(str.begin(), str.end(), str.begin(),
                   [](const unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return str != "0" && str != "false" && str != "off" && str != "no";
}

std::string normalizeOperatorName(const std::string &operatorName) {
    if (operatorName.empty()) {
        return "UNKNOWN";
    }
    return operatorName;
}

std::string profilingPipelineKindToString(ProfilingPipelineKind pipelineKind) {
    switch (pipelineKind) {
    case ProfilingPipelineKind::GRAPH_OP:
        return "graph_op";
    case ProfilingPipelineKind::TOSA:
        return "tosa";
    case ProfilingPipelineKind::MOTION_ENGINE:
        return "motion_engine";
    }

    return "unknown";
}

} // namespace

bool GraphProfiler::isEnabled() { return isTruthyEnvironmentValue(std::getenv("VMEL_GRAPH_PROFILING")); }

GraphProfiler::GraphProfiler(const std::shared_ptr<VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic> &_loader,
                             VkPhysicalDevice _physicalDevice, VkDevice _device)
    : loader{_loader}, physicalDevice{_physicalDevice}, device{_device} {
    VkPhysicalDeviceProperties properties{};
    loader->vkGetPhysicalDeviceProperties(physicalDevice, &properties);
    timestampPeriod = properties.limits.timestampPeriod;

    uint32_t queueFamilyCount = 0;
    loader->vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyCount);
    if (queueFamilyCount != 0) {
        loader->vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount,
                                                         queueFamilyProperties.data());
    }
    queueFamilyTimestampSupport.reserve(queueFamilyProperties.size());
    for (const auto &property : queueFamilyProperties) {
        queueFamilyTimestampSupport.push_back(property.timestampValidBits != 0);
    }
}

GraphProfiler::~GraphProfiler() { clearAllCommandBuffers(); }

ComputePipelineDispatchDecorator GraphProfiler::makeDispatchDecorator(VkPipeline dataGraphPipeline,
                                                                      VkCommandBuffer commandBuffer,
                                                                      uint32_t queueFamilyIndex, uint32_t pipelineCount,
                                                                      ProfilingPipelineKind pipelineKind) {
    auto pipelineKindString = profilingPipelineKindToString(pipelineKind);
    if (pipelineCount == 0 || !supportsTimestampQueries(queueFamilyIndex)) {
        return {};
    }

    const auto queryCount = pipelineCount * 2;
    VkQueryPoolCreateInfo queryPoolCreateInfo{
        VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, // sType
        nullptr,                                  // pNext
        0,                                        // flags
        VK_QUERY_TYPE_TIMESTAMP,                  // queryType
        queryCount,                               // queryCount
        0,                                        // pipelineStatistics
    };

    VkQueryPool queryPool = VK_NULL_HANDLE;
    const VkResult res = loader->vkCreateQueryPool(device, &queryPoolCreateInfo, nullptr, &queryPool);
    if (res != VK_SUCCESS) {
        graphLog(Severity::Error) << "Failed to create graph profiling query pool" << std::endl;
        return {};
    }

    auto record = makeRecord(queryPool, commandBuffer, dataGraphPipeline);
    loader->vkCmdResetQueryPool(commandBuffer, queryPool, 0, queryCount);
    state.addCommandBufferRecord(commandBuffer, record);

    return [this, record, queryPool, pipelineCount, pipelineKind = std::move(pipelineKindString)](
               VkCommandBuffer cmdBuffer, ComputePipelineBase &pipeline,
               const ComputeDescriptorSetMap &descriptorSetMap, uint32_t pipelineIndex) {
        const auto sampleIndex = static_cast<uint32_t>(record->samples.size());
        if (sampleIndex >= pipelineCount) {
            graphLog(Severity::Error) << "Graph profiling query pool is too small for recorded dispatches" << std::endl;
            pipeline.cmdBindAndDispatch(cmdBuffer, descriptorSetMap);
            return;
        }

        const uint32_t beforeQuery = sampleIndex * 2;
        const uint32_t afterQuery = beforeQuery + 1;
        auto operatorName = normalizeOperatorName(pipeline.getDebugName());
        record->samples.push_back({pipelineIndex, beforeQuery, afterQuery, pipelineKind, std::move(operatorName)});

        loader->vkCmdWriteTimestamp2(cmdBuffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, queryPool, beforeQuery);
        pipeline.cmdBindAndDispatch(cmdBuffer, descriptorSetMap);
        loader->vkCmdWriteTimestamp2(cmdBuffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, queryPool, afterQuery);
    };
}

mlsdk::el::compute::optical_flow::ComputePipelineDispatchDecorator
GraphProfiler::makeOpticalFlowDispatchDecorator(VkPipeline dataGraphPipeline, VkCommandBuffer commandBuffer,
                                                uint32_t queueFamilyIndex, uint32_t pipelineCount) {
    if (pipelineCount == 0 || !supportsTimestampQueries(queueFamilyIndex)) {
        return {};
    }

    const auto queryCount = pipelineCount * 2;
    VkQueryPoolCreateInfo queryPoolCreateInfo{
        VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, // sType
        nullptr,                                  // pNext
        0,                                        // flags
        VK_QUERY_TYPE_TIMESTAMP,                  // queryType
        queryCount,                               // queryCount
        0,                                        // pipelineStatistics
    };

    VkQueryPool queryPool = VK_NULL_HANDLE;
    const VkResult res = loader->vkCreateQueryPool(device, &queryPoolCreateInfo, nullptr, &queryPool);
    if (res != VK_SUCCESS) {
        graphLog(Severity::Error) << "Failed to create optical-flow profiling query pool" << std::endl;
        return {};
    }

    auto record = makeRecord(queryPool, commandBuffer, dataGraphPipeline);
    loader->vkCmdResetQueryPool(commandBuffer, queryPool, 0, queryCount);
    state.addCommandBufferRecord(commandBuffer, record);

    return [this, record, queryPool, pipelineCount](VkCommandBuffer cmdBuffer,
                                                    mlsdk::el::compute::optical_flow::ComputePipeline &pipeline,
                                                    uint32_t pipelineIndex) {
        const auto sampleIndex = static_cast<uint32_t>(record->samples.size());
        if (sampleIndex >= pipelineCount) {
            graphLog(Severity::Error) << "Optical-flow profiling query pool is too small for recorded dispatches"
                                      << std::endl;
            pipeline.bindAndDispatch(cmdBuffer);
            return;
        }

        const uint32_t beforeQuery = sampleIndex * 2;
        const uint32_t afterQuery = beforeQuery + 1;
        auto operatorName = normalizeOperatorName(pipeline.getDebugName());
        record->samples.push_back({pipelineIndex, beforeQuery, afterQuery, "optical_flow", std::move(operatorName)});

        loader->vkCmdWriteTimestamp2(cmdBuffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, queryPool, beforeQuery);
        pipeline.bindAndDispatch(cmdBuffer);
        loader->vkCmdWriteTimestamp2(cmdBuffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, queryPool, afterQuery);
    };
}

bool GraphProfiler::hasProfiledCommandBuffers(const std::vector<VkCommandBuffer> &commandBuffers) const {
    return state.hasProfiledCommandBuffers(commandBuffers);
}

void GraphProfiler::registerSubmit(VkQueue queue, const std::vector<VkCommandBuffer> &commandBuffers, VkFence fence) {
    state.registerSubmit(queue, commandBuffers, fence);
}

void GraphProfiler::registerExecuteCommands(VkCommandBuffer commandBuffer, uint32_t commandBufferCount,
                                            const VkCommandBuffer *pCommandBuffers) {
    state.registerExecuteCommands(commandBuffer, commandBufferCount, pCommandBuffers);
}

void GraphProfiler::collectFence(VkFence fence) {
    if (fence == VK_NULL_HANDLE) {
        return;
    }

    collectSubmissions(state.getSubmissionsForFence(fence));
}

void GraphProfiler::collectQueue(VkQueue queue) { collectSubmissions(state.getSubmissionsForQueue(queue)); }

void GraphProfiler::collectDevice() { collectSubmissions(state.getSubmissionsForDevice()); }

void GraphProfiler::clearCommandBuffer(VkCommandBuffer commandBuffer) {
    collectSubmissions(state.getSubmissionsForCommandBuffer(commandBuffer));
    state.clearCommandBuffer(commandBuffer);
}

bool GraphProfiler::collectSubmission(const std::shared_ptr<SubmitRecord> &submission,
                                      std::vector<Sample> &newSamples) {
    for (const auto &record : submission->queryRecords) {
        std::vector<uint64_t> timestamps(record->samples.size() * 2);
        if (timestamps.empty()) {
            continue;
        }

        const VkResult res = loader->vkGetQueryPoolResults(
            device, record->queryPool, 0, static_cast<uint32_t>(timestamps.size()),
            timestamps.size() * sizeof(uint64_t), timestamps.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
        if (res == VK_NOT_READY) {
            return false;
        }
        if (res != VK_SUCCESS) {
            graphLog(Severity::Error) << "Failed to read graph profiling timestamp results" << std::endl;
            return false;
        }

        for (const auto &sampleInfo : record->samples) {
            const auto before = timestamps[sampleInfo.beforeQuery];
            const auto after = timestamps[sampleInfo.afterQuery];
            const auto delta = after >= before ? after - before : 0;
            newSamples.push_back({submission->submissionIndex, record->graphDispatchIndex, record->commandBuffer,
                                  record->dataGraphPipeline, sampleInfo.pipelineIndex, sampleInfo.pipelineKind,
                                  sampleInfo.operatorName, before, after, delta,
                                  static_cast<double>(delta) * static_cast<double>(timestampPeriod) / 1000000.0});
        }
    }
    return true;
}

void GraphProfiler::collectSubmissions(const Submissions &submitRecords) {
    CompletedSubmissions completedSubmissions;
    for (const auto &submission : submitRecords) {
        std::vector<Sample> newSamples;
        if (collectSubmission(submission, newSamples)) {
            completedSubmissions.emplace_back(submission, std::move(newSamples));
        }
    }

    if (completedSubmissions.empty()) {
        return;
    }

    state.completeSubmissions(completedSubmissions);
}

std::string GraphProfiler::getPipelineJson(VkPipeline dataGraphPipeline) {
    const VkResult waitResult = loader->vkDeviceWaitIdle(device);
    if (waitResult == VK_SUCCESS) {
        collectDevice();
    } else {
        graphLog(Severity::Error) << "Failed waiting for graph profiling device idle before query" << std::endl;
    }

    return makeJson(dataGraphPipeline);
}

std::shared_ptr<GraphProfiler::QueryPoolRecord>
GraphProfiler::makeRecord(VkQueryPool queryPool, VkCommandBuffer commandBuffer, VkPipeline dataGraphPipeline) {
    auto deleter = [capturedLoader = loader, capturedDevice = device](QueryPoolRecord *record) {
        if (record->queryPool != VK_NULL_HANDLE) {
            capturedLoader->vkDestroyQueryPool(capturedDevice, record->queryPool, nullptr);
        }
        delete record;
    };

    auto record = std::shared_ptr<QueryPoolRecord>(new QueryPoolRecord{}, std::move(deleter));
    record->queryPool = queryPool;
    record->commandBuffer = commandBuffer;
    record->dataGraphPipeline = dataGraphPipeline;
    record->graphDispatchIndex = state.nextGraphDispatchIndex();
    return record;
}

uint64_t GraphProfiler::LockedState::nextGraphDispatchIndex() {
    std::lock_guard lock(mutex);
    return graphDispatchCounter++;
}

void GraphProfiler::LockedState::addCommandBufferRecord(VkCommandBuffer commandBuffer,
                                                        const std::shared_ptr<QueryPoolRecord> &record) {
    std::lock_guard lock(mutex);
    commandBufferRecords[commandBuffer].push_back(record);
}

bool GraphProfiler::LockedState::hasProfiledCommandBuffers(const std::vector<VkCommandBuffer> &commandBuffers) const {
    std::lock_guard lock(mutex);
    return !getRecordsForCommandBuffersLocked(commandBuffers).empty();
}

void GraphProfiler::LockedState::registerSubmit(VkQueue queue, const std::vector<VkCommandBuffer> &commandBuffers,
                                                VkFence fence) {
    std::lock_guard lock(mutex);
    auto records = getRecordsForCommandBuffersLocked(commandBuffers);
    if (records.empty()) {
        return;
    }

    auto submission = std::make_shared<SubmitRecord>();
    submission->submissionIndex = submissionCounter++;
    submission->queue = queue;
    submission->fence = fence;
    submission->commandBuffers = commandBuffers;
    submission->queryRecords = std::move(records);
    submissions.push_back(std::move(submission));
}

void GraphProfiler::LockedState::registerExecuteCommands(VkCommandBuffer commandBuffer, uint32_t commandBufferCount,
                                                         const VkCommandBuffer *pCommandBuffers) {
    if (commandBufferCount == 0 || pCommandBuffers == nullptr) {
        return;
    }

    std::lock_guard lock(mutex);
    auto &records = commandBufferRecords[commandBuffer];
    for (uint32_t i = 0; i < commandBufferCount; ++i) {
        const auto it = commandBufferRecords.find(pCommandBuffers[i]);
        if (it == commandBufferRecords.end()) {
            continue;
        }
        for (const auto &record : it->second) {
            if (std::find(records.begin(), records.end(), record) == records.end()) {
                records.push_back(record);
            }
        }
    }
}

std::vector<std::shared_ptr<GraphProfiler::QueryPoolRecord>>
GraphProfiler::LockedState::getRecordsForCommandBuffersLocked(
    const std::vector<VkCommandBuffer> &commandBuffers) const {
    std::vector<std::shared_ptr<QueryPoolRecord>> records;
    for (auto *const commandBuffer : commandBuffers) {
        const auto it = commandBufferRecords.find(commandBuffer);
        if (it == commandBufferRecords.end()) {
            continue;
        }

        for (const auto &record : it->second) {
            if (std::find(records.begin(), records.end(), record) == records.end()) {
                records.push_back(record);
            }
        }
    }
    return records;
}

GraphProfiler::Submissions GraphProfiler::LockedState::getSubmissionsForFence(VkFence fence) const {
    std::lock_guard lock(mutex);
    Submissions submitRecords;
    for (const auto &submission : submissions) {
        if (submission->fence == fence) {
            submitRecords.push_back(submission);
        }
    }
    return submitRecords;
}

GraphProfiler::Submissions GraphProfiler::LockedState::getSubmissionsForQueue(VkQueue queue) const {
    std::lock_guard lock(mutex);
    Submissions submitRecords;
    for (const auto &submission : submissions) {
        if (submission->queue == queue) {
            submitRecords.push_back(submission);
        }
    }
    return submitRecords;
}

GraphProfiler::Submissions GraphProfiler::LockedState::getSubmissionsForDevice() const { return submissions; }

GraphProfiler::Submissions
GraphProfiler::LockedState::getSubmissionsForCommandBuffer(VkCommandBuffer commandBuffer) const {
    std::lock_guard lock(mutex);
    Submissions submitRecords;
    for (const auto &submission : submissions) {
        const auto hasCommandBuffer = std::find(submission->commandBuffers.begin(), submission->commandBuffers.end(),
                                                commandBuffer) != submission->commandBuffers.end();
        const auto hasQueryRecord =
            std::any_of(submission->queryRecords.begin(), submission->queryRecords.end(),
                        [commandBuffer](const auto &record) { return record->commandBuffer == commandBuffer; });
        if (hasCommandBuffer || hasQueryRecord) {
            submitRecords.push_back(submission);
        }
    }
    return submitRecords;
}

void GraphProfiler::LockedState::completeSubmissions(const CompletedSubmissions &completedSubmissions) {
    std::lock_guard lock(mutex);
    for (const auto &[submission, newSamples] : completedSubmissions) {
        const auto it = std::find(submissions.begin(), submissions.end(), submission);
        if (it == submissions.end()) {
            continue;
        }
        samples.insert(samples.end(), newSamples.begin(), newSamples.end());
        submissions.erase(it);
    }
}

void GraphProfiler::LockedState::clearCommandBuffer(VkCommandBuffer commandBuffer) {
    std::lock_guard lock(mutex);
    commandBufferRecords.erase(commandBuffer);
    for (auto &[_, records] : commandBufferRecords) {
        records.erase(
            std::remove_if(records.begin(), records.end(),
                           [commandBuffer](const auto &record) { return record->commandBuffer == commandBuffer; }),
            records.end());
    }
    removeSubmissionsForCommandBufferLocked(commandBuffer);
}

void GraphProfiler::LockedState::clearAllCommandBuffers() {
    std::lock_guard lock(mutex);
    commandBufferRecords.clear();
    submissions.clear();
}

void GraphProfiler::LockedState::removeSubmissionsForCommandBufferLocked(VkCommandBuffer commandBuffer) {
    submissions.erase(
        std::remove_if(submissions.begin(), submissions.end(),
                       [commandBuffer](const auto &submission) {
                           const auto hasCommandBuffer =
                               std::find(submission->commandBuffers.begin(), submission->commandBuffers.end(),
                                         commandBuffer) != submission->commandBuffers.end();
                           const auto hasQueryRecord = std::any_of(
                               submission->queryRecords.begin(), submission->queryRecords.end(),
                               [commandBuffer](const auto &record) { return record->commandBuffer == commandBuffer; });
                           return hasCommandBuffer || hasQueryRecord;
                       }),
        submissions.end());
}

std::vector<GraphProfiler::Sample> GraphProfiler::LockedState::getSamples() const {
    std::lock_guard lock(mutex);
    return samples;
}

std::vector<GraphProfiler::Sample> GraphProfiler::LockedState::getSamples(VkPipeline dataGraphPipeline) const {
    std::lock_guard lock(mutex);
    std::vector<Sample> pipelineSamples;
    std::copy_if(samples.begin(), samples.end(), std::back_inserter(pipelineSamples),
                 [dataGraphPipeline](const auto &sample) { return sample.dataGraphPipeline == dataGraphPipeline; });
    return pipelineSamples;
}

void GraphProfiler::clearAllCommandBuffers() { state.clearAllCommandBuffers(); }

nlohmann::ordered_json GraphProfiler::toJson(const Sample &sample) {
    return {{"submission", sample.submissionIndex},   {"graph_dispatch", sample.graphDispatchIndex},
            {"pipeline_index", sample.pipelineIndex}, {"pipeline_kind", sample.pipelineKind},
            {"operator_name", sample.operatorName},   {"cycle_count_before", sample.before},
            {"cycle_count_after", sample.after},      {"cycle_count_delta", sample.delta},
            {"time_ms", sample.milliseconds}};
}

nlohmann::ordered_json GraphProfiler::toJson(const Aggregate &aggregate, const std::string &pipelineKind,
                                             const std::string &operatorName) {
    const auto average = aggregate.count == 0 ? 0.0 : aggregate.totalMilliseconds / aggregate.count;
    return {{"pipeline_kind", pipelineKind},
            {"operator_name", operatorName},
            {"dispatch_count", aggregate.count},
            {"total_time_ms", aggregate.totalMilliseconds},
            {"average_time_ms", average},
            {"min_time_ms", aggregate.count == 0 ? 0.0 : aggregate.minMilliseconds},
            {"max_time_ms", aggregate.maxMilliseconds}};
}

std::string GraphProfiler::makeJson() const { return makeJson(state.getSamples()); }

std::string GraphProfiler::makeJson(VkPipeline dataGraphPipeline) const {
    return makeJson(state.getSamples(dataGraphPipeline));
}

std::string GraphProfiler::makeJson(const std::vector<Sample> &profileSamples) const {
    using AggregateKey = std::pair<std::string, std::string>;
    std::map<AggregateKey, Aggregate> aggregates;
    for (const auto &sample : profileSamples) {
        auto &aggregate = aggregates[{sample.pipelineKind, sample.operatorName}];
        aggregate.count++;
        aggregate.totalMilliseconds += sample.milliseconds;
        aggregate.minMilliseconds = std::min(aggregate.minMilliseconds, sample.milliseconds);
        aggregate.maxMilliseconds = std::max(aggregate.maxMilliseconds, sample.milliseconds);
    }

    using Json = nlohmann::ordered_json;

    Json sampleJson = Json::array();
    for (const auto &sample : profileSamples) {
        sampleJson.push_back(toJson(sample));
    }

    Json aggregateJson = Json::array();
    for (const auto &[key, aggregate] : aggregates) {
        const auto &[pipelineKind, operatorName] = key;
        aggregateJson.push_back(toJson(aggregate, pipelineKind, operatorName));
    }

    Json output;
    output["timestamp_period_ns"] = timestampPeriod;
    output["samples"] = std::move(sampleJson);
    output["by_operator"] = std::move(aggregateJson);
    return output.dump(2);
}

bool GraphProfiler::supportsTimestampQueries(uint32_t queueFamilyIndex) const {
    return queueFamilyIndex < queueFamilyTimestampSupport.size() && queueFamilyTimestampSupport[queueFamilyIndex];
}

} // namespace mlsdk::el::layer
