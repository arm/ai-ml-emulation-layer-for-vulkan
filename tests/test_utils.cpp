/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include "test_utils.hpp"

#include <cstdlib>

namespace mlsdk::el::tests {

ScopedEnvironment::ScopedEnvironment(std::string_view _name, std::optional<std::string_view> _value) : name{_name} {
    const auto *const currentValue = std::getenv(name.c_str());
    if (currentValue != nullptr) {
        oldValue = currentValue;
    }
    if (_value.has_value()) {
        set(std::string{*_value});
    } else {
        unset();
    }
}

ScopedEnvironment::~ScopedEnvironment() {
    if (oldValue.has_value()) {
        set(oldValue.value());
    } else {
        unset();
    }
}

void ScopedEnvironment::set(const std::string &value) const {
#if defined(_WIN32)
    _putenv_s(name.c_str(), value.c_str());
#else
    setenv(name.c_str(), value.c_str(), 1);
#endif
}

void ScopedEnvironment::unset() const {
#if defined(_WIN32)
    _putenv_s(name.c_str(), "");
#else
    unsetenv(name.c_str());
#endif
}

} // namespace mlsdk::el::tests
