#ifndef LOGGER_H
#define LOGGER_H

/**
 * @file Logger.h
 * @brief Utilities for structured logging
 */

#include <string>

namespace log_utils {

using std::string;

class Logger {
private:
  Logger() = default;

public:
  enum class LOG_LEVEL { TRACE, DEBUG, INFO, WARN, ERROR, FATAL, OFF };
  enum class EVENT_STATE { START, STOP };

  static auto
  initialise_global_logging(LOG_LEVEL default_log_level = LOG_LEVEL::WARN)
    -> void;
  static auto log(LOG_LEVEL level, const string &message) -> void;
  static auto log_timed_event(
    const string &component,
    EVENT_STATE start_stop,
    const string &message = "",
    LOG_LEVEL level = LOG_LEVEL::DEBUG
  ) -> void;
};

}  // namespace log_utils
#endif
