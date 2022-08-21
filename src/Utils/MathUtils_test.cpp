#include <memory>
#include <stdint.h>

#include "gtest/gtest.h"

#include "Utils/MathUtils.hpp"

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

TEST(MathUtils, DivisibleByPowerOfTwo) {
  ASSERT_EQ(divisible_by_power_of_two<u64>(32, 4), true);
  ASSERT_EQ(divisible_by_power_of_two<u64>(1024, 64), true);
  ASSERT_EQ(divisible_by_power_of_two<u64>(64, 1024), false);
}

}  // namespace math_utils
