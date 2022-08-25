#ifndef LOG_UTILS_H
#define LOG_UTILS_H

/**
 * @file LogUtils.h
 * @brief Utilities for structured logging
 * */

#include <string>

using std::string;

namespace log_utils {

enum class LOG_LEVEL { TRACE, DEBUG, INFO, WARN, ERROR, FATAL, OFF };
enum class EVENT_STATE { START, STOP };

auto initialise_global_logging(
  LOG_LEVEL default_log_level = LOG_LEVEL::WARN
) -> void;
auto log(LOG_LEVEL level, string s) -> void;
auto log_timed_event(
  string component,
  EVENT_STATE start_stop,
  string message = "",
  LOG_LEVEL level = LOG_LEVEL::TRACE
) -> void;

}  // namespace log_utils
#endif
