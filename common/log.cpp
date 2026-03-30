/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "mlel/log.hpp"

/*******************************************************************************
 * Log
 *******************************************************************************/

namespace mlsdk::el::log {

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
        *os << "[" << loggerName << "][" << severityToString() << "] ";
    }
    return *this;
}

bool Log::enabled(const Severity _severity) const { return logLevel >= _severity; }

Severity Log::getLogLevel(const std::string &environmentVariable, const Severity defaultLogLevel) {
    char const *logLevelCharArr = std::getenv(environmentVariable.c_str());
    if (logLevelCharArr == nullptr) {
        return defaultLogLevel;
    }

    std::string logLevelStr(logLevelCharArr);
    std::transform(logLevelStr.begin(), logLevelStr.end(), logLevelStr.begin(), ::tolower);
    auto it = stringToSeverityMap.find(logLevelStr);
    if (it == stringToSeverityMap.end()) {
        return defaultLogLevel;
    }
    return it->second;
}

std::string Log::severityToString() const {
    return size_t(severity) >= severityStringsArr.size() ? "Unknown" : severityStringsArr[uint32_t(severity)];
}

Log &operator<<(Log &os, const StringLineNumber &s) {
    std::string::size_type pastPos{};
    unsigned line{1};
    os << std::resetiosflags(std::ios_base::dec) << "\n";
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
        os << std::setw(2) << static_cast<unsigned>(dump.pointer[i]) << " ";
    }

    os.getStreamMutable()->copyfmt(osStateOrig);
    os << std::endl;

    return os;
}

} // namespace mlsdk::el::log
