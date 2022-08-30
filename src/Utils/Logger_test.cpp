// spdlog to stream code from:
// https://stackoverflow.com/questions/66473052/how-can-i-read-spdlog-output-in-a-google-test

#include <cstdlib>
#include <string>

#include <gtest/gtest.h>

#include "Utils/Logger.h"
#include "spdlog/sinks/ostream_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

using std::basic_streambuf;
using std::char_traits;
using std::cout;
using std::make_shared;
using std::ostream;
using std::string;
using std::stringstream;

namespace log_utils {
class LogUtilsTest: public ::testing::Test {
  protected:
    basic_streambuf<char, char_traits<char>> *old_buf;
    stringstream stream;
    int i;
    char buffer[500];

    auto SetUp(bool set_json_pattern) -> void {
      auto ostream_sink = make_shared<spdlog::sinks::ostream_sink_st>(stream);
      auto ostream_logger = std::make_shared<spdlog::logger>("", ostream_sink);
      ostream_logger->set_level(spdlog::level::debug);
      spdlog::set_default_logger(ostream_logger);
      Logger::initialise_global_logging(
        Logger::LOG_LEVEL::TRACE, set_json_pattern
      );
    }

    auto TearDown() -> void override {
      spdlog::drop_all();
      spdlog::set_default_logger(spdlog::stdout_color_mt(""));
    }
};

TEST_F(LogUtilsTest, NormalLog) {
  SetUp(false);
  Logger::log(Logger::LOG_LEVEL::TRACE, "hello");
  auto s = stream.str();
  sscanf(
    s.c_str(),
    R"([%d-%d-%d %d:%d:%d.%d] %s %s)",
    &i,
    &i,
    &i,
    &i,
    &i,
    &i,
    &i,
    buffer,
    &buffer[250]
  );
  ASSERT_EQ("[trace]", string(buffer));
  ASSERT_EQ("hello", string(&buffer[250]));
}

TEST_F(LogUtilsTest, NormalLogJson) {
  SetUp(true);
  Logger::log(Logger::LOG_LEVEL::WARN, "hello");
  auto s = stream.str();
  sscanf(
    s.c_str(),
    R"({"time": "%d-%d-%dT%d:%d:%d.%d+%d:%d", "level": "warning", "process": %d, "thread": %d, "log": {"type": "message", "message": %s)",
    &i,
    &i,
    &i,
    &i,
    &i,
    &i,
    &i,
    &i,
    &i,
    &i,
    &i,
    buffer
  );
  ASSERT_EQ(R"("hello"}})", string(buffer));
}

TEST_F(LogUtilsTest, TimeEvent) {
  SetUp(true);
  Logger::log_timed_event("test", Logger::EVENT_STATE::START, "hello");
  auto s = stream.str();
  sscanf(
    s.c_str(),
    R"({"time": "%d-%d-%dT%d:%d:%d.%d+%d:%d", "level": "trace", "process": %d, "thread": %d, "log": {"type": %s "state": "start", "component": "test", "message": %s)",
    &i,
    &i,
    &i,
    &i,
    &i,
    &i,
    &i,
    &i,
    &i,
    &i,
    &i,
    buffer,
    &(buffer[250])
  );
  ASSERT_EQ(R"("timed_event",)", string(buffer));
  ASSERT_EQ(R"("hello"}})", string(&buffer[250]));
}

}  // namespace log_utils
