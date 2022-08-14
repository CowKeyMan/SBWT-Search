#include <gtest/gtest.h>

#include "Utils/CircularQueue.hpp"

namespace utils {

TEST(CircularQueueTest, NormalUsage) {
  CircularQueue<int> q(3);
  ASSERT_EQ(3, q.capacity());
  ASSERT_TRUE(q.empty());
  q.push(1);
  q.push(2);
  ASSERT_EQ(1, q.front());
  q.push(3);
  ASSERT_TRUE(q.full());
  q.pop();
  ASSERT_FALSE(q.full());
  ASSERT_EQ(2, q.size());
  q.push(4);
  ASSERT_EQ(2, q.front());
  ASSERT_EQ(3, q.size());
}

}
