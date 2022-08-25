// json format idea from: https://github.com/gabime/spdlog/issues/1797

#include <string>

#include <spdlog/common.h>
#include <unordered_map>

#include "Utils/LogUtils.h"
#include "spdlog/cfg/env.h"
#include "spdlog/spdlog.h"

using fmt::format;
using spdlog::level::level_enum;
using std::string;
using std::unordered_map;

namespace log_utils {

const unordered_map<LOG_LEVEL, level_enum> level_to_spd{
  { LOG_LEVEL::TRACE, level_enum::trace },
  { LOG_LEVEL::DEBUG, level_enum::debug },
  { LOG_LEVEL::INFO, level_enum::info },
  { LOG_LEVEL::WARN, level_enum::warn },
  { LOG_LEVEL::ERROR, level_enum::err },
  { LOG_LEVEL::FATAL, level_enum::critical },
  { LOG_LEVEL::OFF, level_enum::off },
};

auto initialise_global_logging(LOG_LEVEL default_log_level) -> void {
  spdlog::set_level(level_to_spd.at(default_log_level));
  spdlog::cfg::load_env_levels();
  string pattern
    = R"({"time": "%Y-%m-%dT%H:%M:%S.%f%z", "level": "%^%l%$", "process": %P, "thread": %t, "log": %v})";
  spdlog::set_pattern(pattern);
}

auto log(LOG_LEVEL level, string s) -> void {
  string message = format(R"({{"type": "message", "message": "{}"}})", s);
  spdlog::log(level_to_spd.at(level), message);
}

auto log_timed_event(
  string component, EVENT_STATE start_stop, string message, LOG_LEVEL level
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
