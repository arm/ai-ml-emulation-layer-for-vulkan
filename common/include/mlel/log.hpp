/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include <algorithm>
#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace mlsdk::el::log {

/*******************************************************************************
 * Severity
 *******************************************************************************/

enum class Severity { Error, Warning, Info, Debug };

/*******************************************************************************
 * Log
 *******************************************************************************/

class Log {
  public:
    Log(const std::string &_environmentVariable, const std::string &_loggerName,
        const Severity _defaultLogLevel = Severity::Error);

    /**
     * Output log entry if log level is higher or equal to the severity.
     */
    template <typename T> Log &operator<<(const T &output) {
        if (enabled(severity)) {
            *os << output;
        }
        return *this;
    }

    std::ostream *getStreamMutable() { return os; }

    /**
     * Handle std::functions.
     */
    Log &operator<<(std::ostream &(*f)(std::ostream &));

    /**
     * Set log level, output log header and return reference to logger.
     */
    Log &operator()(const Severity _severity);

    /**
     * Return true if log is enable for severity.
     */
    bool enabled(const Severity _severity) const;

  private:
    static Severity getLogLevel(const std::string &environmentVariable, const Severity defaultLogLevel);
    std::string severityToString() const;

    static inline const std::map<std::string, Severity> stringToSeverityMap = {
        {std::string("error"), Severity::Error},
        {std::string("warning"), Severity::Warning},
        {std::string("info"), Severity::Info},
        {std::string("debug"), Severity::Debug}};

    static inline const std::array<std::string, 4> severityStringsArr = {std::string("Error"), std::string("Warning"),
                                                                         std::string("Info"), std::string("Debug")};

    Severity logLevel;
    std::string loggerName;
    Severity severity{Severity::Info};
    std::ostream *os;
};

template <typename T> auto getValue(T value) {
    // Specialization for int8_t and uint8_t to print as integers, not characters
    if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
        return int{value};
    } else {
        return value;
    }
};

template <typename T> Log &operator<<(Log &os, const std::vector<T> &v) {
    os << std::dec << "[";
    auto it = v.begin();
    if (it != v.end()) {
        os << getValue(*it);
        ++it;
    }
    while (it != v.end()) {
        os << ", " << getValue(*it);
        ++it;
    }
    os << "]";
    return os;
}

struct StringLineNumber {
    explicit StringLineNumber(const std::string &_str) : str{_str} {}
    const std::string &str;
};

Log &operator<<(Log &os, const StringLineNumber &s);

struct HexDump {
    HexDump(const uint8_t *_pointer, const size_t _size, const size_t _width = 16)
        : pointer{_pointer}, size{_size}, width{_width} {}
    const uint8_t *pointer;
    const size_t size;
    const size_t width;
};

Log &operator<<(Log &os, const HexDump &dump);

} // namespace mlsdk::el::log
