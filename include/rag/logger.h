#pragma once

// Minimal logging wrapper. Uses spdlog if available, otherwise prints to stderr.

#ifdef RAG_HAS_SPDLOG

#include <spdlog/spdlog.h>

#else

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace rag {
namespace log {

enum level { info, warn, error };

inline std::string _fmt(const std::string& tmpl,
                         const std::vector<std::string>& args) {
    std::string result;
    size_t pos = 0;
    for (auto& arg : args) {
        size_t p = tmpl.find("{}", pos);
        if (p == std::string::npos) break;
        result += tmpl.substr(pos, p - pos);
        result += arg;
        pos = p + 2;
    }
    result += tmpl.substr(pos);
    return result;
}

template<typename... Args>
inline void _log(level lv, const std::string& fmt, Args&&... args) {
    const char* tag = "";
    switch (lv) {
        case info:  tag = "[INFO] ";  break;
        case warn:  tag = "[WARN] ";  break;
        case error: tag = "[ERROR] "; break;
    }

    std::vector<std::string> arg_strs;
    (arg_strs.push_back((std::ostringstream{} << args).str()), ...);

    std::cerr << tag << _fmt(fmt, arg_strs) << std::endl;
}

} // namespace log
} // namespace rag

#define SPDLOG_info(...)  ::rag::log::_log(::rag::log::info, __VA_ARGS__)
#define SPDLOG_warn(...)  ::rag::log::_log(::rag::log::warn, __VA_ARGS__)
#define SPDLOG_error(...) ::rag::log::_log(::rag::log::error, __VA_ARGS__)

#endif // RAG_HAS_SPDLOG
