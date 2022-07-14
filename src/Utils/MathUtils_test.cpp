#include <gtest/gtest.h>

#include "Utils/MathUtils.hpp"
#include "Utils/TypeDefinitionUtils.h"

using sbwt_search::u64;

TEST(MathUtils, RoundUp) {
  ASSERT_EQ(64, round_up<u64>(45, 64));
  ASSERT_EQ(128, round_up<u64>(65, 64));
  ASSERT_EQ(64, round_up<u64>(64, 64));
}

TEST(MathUtils, DivisibleByPowerOfTwo) {
  ASSERT_EQ(divisible_by_power_of_two<u64>(32, 4), true);
  ASSERT_EQ(divisible_by_power_of_two<u64>(1024, 64), true);
  ASSERT_EQ(divisible_by_power_of_two<u64>(64, 1024), false);
}
