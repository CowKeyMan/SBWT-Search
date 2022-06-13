#include <gtest/gtest.h>

#include "Functions.h"

namespace functions {

TEST(FunctionTest, Add) {
  ASSERT_EQ(3, add(1, 2));
  ASSERT_EQ(1, add(-1, 2));
}

TEST(FunctionTest, Mul) {
  ASSERT_EQ(2, mul(1, 2));
  ASSERT_EQ(-2, mul(-1, 2));
  ASSERT_EQ(6, mul(3, 2));
}

}
