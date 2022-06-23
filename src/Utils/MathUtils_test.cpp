#include <gtest/gtest.h>

#include "GlobalDefinitions.h"
#include "MathUtils.hpp"

using sbwt_search::u64;

TEST(MathUtils, RoundUp) {
  ASSERT_EQ(64, round_up<u64>(45, 64));
  ASSERT_EQ(128, round_up<u64>(65, 64));
  ASSERT_EQ(64, round_up<u64>(64, 64));
}
