#include <gtest/gtest.h>

#include "Utils/CircularBuffer.hpp"

namespace utils {

TEST(CircularBufferTest, NormalUsage) {
  CircularBuffer<int> buffer(3, 9);
  ASSERT_EQ(3, buffer.size());
  ASSERT_EQ(9, buffer.current());
  buffer.current() = 1;
  ASSERT_EQ(1, buffer.current());
  buffer.step_forward();
  ASSERT_EQ(9, buffer.current());
  buffer.current() = 2;
  ASSERT_EQ(2, buffer.current());
  buffer.step_forward();
  buffer.step_forward();
  ASSERT_EQ(1, buffer.current());
}

}
