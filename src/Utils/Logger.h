#ifndef LOGGER_H
#define LOGGER_H

/**
 * @file Logger.h
 * @brief Utilities for structured logging
 * */

#include <string>

using std::string;

namespace log_utils {

class Logger {
  private:
    static bool json_pattern;
    Logger(){};

  public:
    enum class LOG_LEVEL { TRACE, DEBUG, INFO, WARN, ERROR, FATAL, OFF };
    enum class EVENT_STATE { START, STOP };

    static auto initialise_global_logging(
      LOG_LEVEL default_log_level = LOG_LEVEL::WARN,
      bool set_json_pattern = true
    ) -> void;
    static auto log(LOG_LEVEL level, const string &s) -> void;
    static auto log_timed_event(
      string component,
      EVENT_STATE start_stop,
      const string &message = "",
      LOG_LEVEL level = LOG_LEVEL::TRACE
    ) -> void;
};

}  // namespace log_utils
#endif
