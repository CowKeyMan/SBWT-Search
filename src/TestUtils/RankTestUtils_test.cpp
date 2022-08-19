#include <stdexcept>
#include <string>
#include <vector>

#include "gtest/gtest_pred_impl.h"
#include <gtest/gtest-message.h>
#include <gtest/gtest-test-part.h>

#include "TestUtils/BitVectorTestUtils.hpp"
#include "TestUtils/RankTestUtils.hpp"

using std::runtime_error;
using std::string;

namespace sbwt_search {

TEST(RankTestUtilsTest, TestRank) {
  ASSERT_EQ(0, dummy_cpu_rank<false>(&bit_vector[0], 2));
  ASSERT_EQ(1, dummy_cpu_rank<false>(&bit_vector[0], 63));
  ASSERT_EQ(2, dummy_cpu_rank<false>(&bit_vector[0], 64));
  ASSERT_EQ(20, dummy_cpu_rank<false>(&bit_vector[0], 64 * 4 + 64 - 5));
  ASSERT_EQ(38, dummy_cpu_rank<false>(&bit_vector[0], 64 * 9 + 64 - 2));
}

TEST(RankTestUtilsTest, TestRankReversed) {
  ASSERT_EQ(1, dummy_cpu_rank<true>(&bit_vector[0], 1));
  ASSERT_EQ(20, dummy_cpu_rank<true>(&bit_vector[0], 64 * 4 + 5));
}

}  // namespace sbwt_search
