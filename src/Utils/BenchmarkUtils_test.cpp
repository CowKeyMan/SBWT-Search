#include <thread>

#include <gtest/gtest.h>

#include "BenchmarkUtils.hpp"

TEST(BenchmarkUtils, TimeIt) {
  ASSERT_EQ(0, TIME_IT_TOTAL);
  TIME_IT(std::this_thread::sleep_for(std::chrono::milliseconds(2)));
  ASSERT_LE(2, TIME_IT_TOTAL);
}
