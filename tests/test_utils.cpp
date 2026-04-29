/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include "test_utils.hpp"

#include <cstdlib>

namespace mlsdk::el::tests {

ScopedEnvironment::ScopedEnvironment(const char *_name, const std::string &_value) : name{_name} {
    const auto *const currentValue = std::getenv(name.c_str());
    if (currentValue != nullptr) {
        oldValue = currentValue;
    }
    set(_value);
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
