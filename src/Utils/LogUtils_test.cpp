// spdlog to stream code from:
// https://stackoverflow.com/questions/66473052/how-can-i-read-spdlog-output-in-a-google-test

#include <cstdlib>
#include <iostream>

#include <gtest/gtest.h>

#include "Utils/LogUtils.h"
#include "spdlog/sinks/ostream_sink.h"
#include "spdlog/spdlog.h"

using std::basic_streambuf;
using std::char_traits;
using std::cout;
using std::ostream;
using std::stringstream;

namespace log_utils {

class LogUtilsTest: public ::testing::Test {
  protected:
    basic_streambuf<char, char_traits<char>> *old_buf;
    stringstream stream;
    int i;
    char buffer[500];

    auto SetUp() -> void override {
      auto ostream_logger = spdlog::get("gtest_logger");
      if (!ostream_logger) {
        auto ostream_sink
          = std::make_shared<spdlog::sinks::ostream_sink_st>(stream);
        ostream_logger
          = std::make_shared<spdlog::logger>("gtest_logger", ostream_sink);
        ostream_logger->set_pattern(">%v<");
        ostream_logger->set_level(spdlog::level::debug);
      }
      spdlog::set_default_logger(ostream_logger);
      initialise_global_logging(LOG_LEVEL::TRACE);
    }

    auto TearDown() -> void override { spdlog::drop_all(); }
};

TEST_F(LogUtilsTest, NormalLogJson) {
  log(LOG_LEVEL::WARN, "hello");
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

using namespace std;

TEST_F(LogUtilsTest, TimeEvent) {
  log_timed_event("test", EVENT_STATE::START, "hello");
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
