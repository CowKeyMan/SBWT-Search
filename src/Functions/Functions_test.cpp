#include <gtest/gtest.h>

#include "Functions.h"

namespace functions {

TEST(BinaryFunctionTest, Add) {
  ASSERT_EQ(3, add(1, 2));
  ASSERT_EQ(1, add(-1, 2));
}

TEST(BinaryFunctionTest, Mul) {
  ASSERT_EQ(2, mul(1, 2));
  ASSERT_EQ(-2, mul(-1, 2));
  ASSERT_EQ(6, mul(3, 2));
}

TEST(BinaryFunctionTest, Sub) {
  ASSERT_EQ(-1, sub(1, 2));
  ASSERT_EQ(-3, sub(-1, 2));
  ASSERT_EQ(1, sub(3, 2));
}

TEST(BinaryFunctionTest, Div) {
  ASSERT_EQ(-1, div(2, -2));
  ASSERT_EQ(3, div(9, 3));
  ASSERT_EQ(3, div(7, 2));
}

}
