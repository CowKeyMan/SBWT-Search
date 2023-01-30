#include "gtest/gtest.h"

#include "Tools/CircularBuffer.hpp"

namespace structure_utils {

TEST(CircularBufferTest, NormalUsage) {
  CircularBuffer<int> buffer(2, 9);
  ASSERT_EQ(0, buffer.size());
  ASSERT_EQ(2, buffer.capacity());
  ASSERT_EQ(9, buffer.current_write());
  buffer.current_write() = 1;
  buffer.step_write();
  ASSERT_EQ(1, buffer.current_read());
  buffer.set(0, 5);
  ASSERT_EQ(5, buffer.current_read());
  buffer.current_write() = 2;
  buffer.step_write();
  buffer.step_read();
  ASSERT_EQ(2, buffer.current_read());
  buffer.step_read();
  ASSERT_EQ(5, buffer.current_read());
}

}  // namespace structure_utils
