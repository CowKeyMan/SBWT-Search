#include <gtest/gtest.h>

#include "Tools/Logger.h"

using log_utils::Logger;

auto main(int argc, char **argv) -> int{
  Logger::initialise_global_logging(Logger::LOG_LEVEL::OFF);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
