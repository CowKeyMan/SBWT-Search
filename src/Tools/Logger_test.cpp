// spdlog to stream code from:
// https://stackoverflow.com/questions/66473052/how-can-i-read-spdlog-output-in-a-google-test

#include <array>
#include <cstdlib>
#include <string>

#include <gtest/gtest.h>

#include "Tools/Logger.h"
#include "spdlog/sinks/ostream_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

namespace log_utils {

using std::make_shared;
using std::ostream;
using std::string;
using std::stringstream;

class LogUtilsTest: public ::testing::Test {
private:
  stringstream ss;

protected:
  auto get_stream_string() -> string { return ss.str(); };

  auto SetUp() -> void override {
    auto ostream_sink = make_shared<spdlog::sinks::ostream_sink_st>(ss);
    auto ostream_logger = std::make_shared<spdlog::logger>("", ostream_sink);
    ostream_logger->set_level(spdlog::level::debug);
    spdlog::set_default_logger(ostream_logger);
    Logger::initialise_global_logging(Logger::LOG_LEVEL::TRACE);
  }

  auto TearDown() -> void override {
    spdlog::drop_all();
    spdlog::set_default_logger(spdlog::stdout_color_mt(""));
    Logger::initialise_global_logging(Logger::LOG_LEVEL::OFF);
  }
};

TEST_F(LogUtilsTest, NormalLogJson) {
  const auto buffer_size = 500;
  string buffer("\0", buffer_size);
  int i = -1;
  Logger::log(Logger::LOG_LEVEL::WARN, "hello");
  auto s = get_stream_string();
  auto value = sscanf(  // NOLINT(cppcoreguidelines-pro-type-vararg,
                        // hicpp-vararg, cert-err34-c)
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
    buffer.data()
  );
  ASSERT_EQ(12, value);
  // the conversion of string is made to cut off all the trailing '\0'
  ASSERT_EQ(R"("hello"}})", buffer.substr(0, buffer.find_first_of('\0')));
}

TEST_F(LogUtilsTest, TimeEvent) {
  const auto buffer_size = 500;
  string buffer("\0", buffer_size);
  string buffer2("\0", buffer_size);
  int i = -1;
  Logger::log_timed_event("test", Logger::EVENT_STATE::START, "hello");
  auto s = get_stream_string();
  auto value = sscanf(  // NOLINT(cppcoreguidelines-pro-type-vararg,
                        // hicpp-vararg, cert-err34-c)
    s.c_str(),
    R"({"time": "%d-%d-%dT%d:%d:%d.%d+%d:%d", "level": "debug", "process": %d, "thread": %d, "log": {"type": %s "state": "start", "component": "test", "message": %s)",
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
    buffer.data(),
    buffer2.data()
  );
  ASSERT_EQ(13, value);
  // the conversion of string is made to cut off all the trailing '\0'
  ASSERT_EQ(R"("timed_event",)", buffer.substr(0, buffer.find_first_of('\0')));
  ASSERT_EQ(R"("hello"}})", buffer2.substr(0, buffer2.find_first_of('\0')));
}

}  // namespace log_utils
