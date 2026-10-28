/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

#include "mlel/device.hpp"
#include <optional>
#include <string>
#include <string_view>

namespace mlsdk::el::tests {

class ScopedEnvironment {
  public:
    ScopedEnvironment(std::string_view _name, std::optional<std::string_view> _value);
    ~ScopedEnvironment();

    ScopedEnvironment(const ScopedEnvironment &) = delete;
    ScopedEnvironment &operator=(const ScopedEnvironment &) = delete;
    ScopedEnvironment(ScopedEnvironment &&) = delete;
    ScopedEnvironment &operator=(ScopedEnvironment &&) = delete;

  private:
    void set(const std::string &value) const;
    void unset() const;

    std::string name;
    std::optional<std::string> oldValue;
};

inline bool computeQueueSupportsTimestampQueries(const std::shared_ptr<mlsdk::el::utilities::Device> &device) {
    const auto &physicalDevice = device->getPhysicalDevice();
    const auto queueFamilyIndex = physicalDevice->getComputeFamilyIndex();
    const auto queueFamilyProperties = (&(*physicalDevice)).getQueueFamilyProperties();
    return queueFamilyIndex < queueFamilyProperties.size() &&
           queueFamilyProperties[queueFamilyIndex].timestampValidBits != 0;
}

} // namespace mlsdk::el::tests
