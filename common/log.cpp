/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "mlel/log.hpp"

#include <algorithm>
#include <array>

/*******************************************************************************
 * Log
 *******************************************************************************/

namespace mlsdk::el::log {
namespace {
constexpr std::array<std::pair<std::string_view, Severity>, 4> stringToSeverity = {
    {{"Error", Severity::Error}, {"Warning", Severity::Warning}, {"Info", Severity::Info}, {"Debug", Severity::Debug}}};

bool equalsIgnoreCase(std::string_view a, std::string_view b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), [](unsigned char c1, unsigned char c2) {
               return std::tolower(c1) == std::tolower(c2);
           });
}

Severity getLogLevel(const std::string &environmentVariable, const Severity defaultLogLevel) {
    const char *logLevelCharArr = std::getenv(environmentVariable.c_str());
    if (logLevelCharArr == nullptr) {
        return defaultLogLevel;
    }

    const std::string logLevelStr(logLevelCharArr);
    for (const auto &[name, severity] : stringToSeverity) {
        if (equalsIgnoreCase(name, logLevelStr)) {
            return severity;
        }
    }
    return defaultLogLevel;
}
} // namespace

Log::Log(const std::string &_environmentVariable, const std::string &_loggerName, const Severity _defaultLogLevel)
    : logLevel{getLogLevel(_environmentVariable, _defaultLogLevel)}, loggerName(_loggerName), os(&std::cout) {}

Log &Log::operator<<(std::ostream &(*f)(std::ostream &)) {
    if (enabled(severity)) {
        *os << f;
    }
    return *this;
}

Log &Log::operator()(const Severity _severity) {
    severity = _severity;
    if (enabled(severity)) {
        *os << '[' << loggerName << "][" << severityToString() << "] ";
    }
    return *this;
}

bool Log::enabled(const Severity _severity) const { return logLevel >= _severity; }

std::string_view Log::severityToString() const {
    const auto index = size_t(severity);
    return index >= stringToSeverity.size() ? "Unknown" : stringToSeverity[index].first;
}

Log &operator<<(Log &os, const StringLineNumber &s) {
    std::string::size_type pastPos{};
    unsigned line{1};
    os << std::resetiosflags(std::ios_base::dec) << '\n';
    for (auto curPos = s.str.find('\n', pastPos); curPos != std::string::npos; curPos = s.str.find('\n', pastPos)) {
        os << std::setw(3) << line++ << ": " << s.str.substr(pastPos, curPos - pastPos + 1);
        pastPos = curPos + 1;
    }
    os << std::setw(3) << line++ << ": " << s.str.substr(pastPos);
    return os;
}

Log &operator<<(Log &os, const HexDump &dump) {
    std::ios osStateOrig(nullptr);
    osStateOrig.copyfmt(*(os.getStreamMutable()));

    os << std::resetiosflags(std::ios_base::hex);
    os.getStreamMutable()->fill('0');

    for (size_t i = 0; i < dump.size; i++) {
        if ((i % dump.width) == 0) {
            os << std::endl << std::setw(8) << i << ": ";
        }
        os << std::setw(2) << static_cast<unsigned>(dump.pointer[i]) << ' ';
    }

    os.getStreamMutable()->copyfmt(osStateOrig);
    os << std::endl;

    return os;
}

} // namespace mlsdk::el::log
