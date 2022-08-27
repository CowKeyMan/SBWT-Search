// json format idea from: https://github.com/gabime/spdlog/issues/1797

#include <string>

#include <spdlog/common.h>
#include <unordered_map>

#include "Utils/Logger.h"
#include "spdlog/cfg/env.h"
#include "spdlog/spdlog.h"

using fmt::format;
using spdlog::level::level_enum;
using std::string;
using std::unordered_map;

namespace log_utils {

bool Logger::json_pattern = false;

const unordered_map<Logger::LOG_LEVEL, level_enum> level_to_spd{
  { Logger::LOG_LEVEL::TRACE, level_enum::trace },
  { Logger::LOG_LEVEL::DEBUG, level_enum::debug },
  { Logger::LOG_LEVEL::INFO, level_enum::info },
  { Logger::LOG_LEVEL::WARN, level_enum::warn },
  { Logger::LOG_LEVEL::ERROR, level_enum::err },
  { Logger::LOG_LEVEL::FATAL, level_enum::critical },
  { Logger::LOG_LEVEL::OFF, level_enum::off },
};

auto Logger::initialise_global_logging(
  Logger::LOG_LEVEL default_log_level, bool set_json_pattern
) -> void {
  json_pattern = set_json_pattern;
  spdlog::set_level(level_to_spd.at(default_log_level));
  spdlog::cfg::load_env_levels();
  if (json_pattern) {
    string pattern
      = R"({"time": "%Y-%m-%dT%H:%M:%S.%f%z", "level": "%^%l%$", "process": %P, "thread": %t, "log": %v})";
    spdlog::set_pattern(pattern);
  }
}

auto Logger::log(LOG_LEVEL level, const string &s) -> void {
  string message;
  if (json_pattern) {
    message = format(R"({{"type": "message", "message": "{}"}})", s);
  } else {
    message = s;
  }
  spdlog::log(level_to_spd.at(level), message);
}

auto Logger::log_timed_event(
  const string &component,
  EVENT_STATE start_stop,
  const string &message,
  LOG_LEVEL level
) -> void {
  string state = start_stop == EVENT_STATE::START ? "start" : "stop";
  string json_message = format(
    R"({{"type": "timed_event", "state": "{}", "component": "{}", "message": "{}"}})",
    state,
    component,
    message
  );
  spdlog::log(level_to_spd.at(level), json_message);
}

}  // namespace log_utils
