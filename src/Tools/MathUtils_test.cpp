#include <memory>
#include <stdint.h>

#include "gtest/gtest.h"

#include "Tools/MathUtils.hpp"

using u64 = uint64_t;

namespace math_utils {

TEST(MathUtils, RoundUp) {
  ASSERT_EQ(64, round_up<u64>(45, 64));
  ASSERT_EQ(128, round_up<u64>(65, 64));
  ASSERT_EQ(64, round_up<u64>(64, 64));
}

TEST(MathUtils, RoundDown) {
  ASSERT_EQ(0, round_down<u64>(45, 64));
  ASSERT_EQ(64, round_down<u64>(65, 64));
  ASSERT_EQ(64, round_down<u64>(64, 64));
}

TEST(MathUtils, DivideAndRound) {
  ASSERT_EQ(2ULL, divide_and_round<u64>(3ULL, 2ULL));
}

}  // namespace math_utils
