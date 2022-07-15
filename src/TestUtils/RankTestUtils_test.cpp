#include <gtest/gtest.h>

#include "TestUtils/BitVectorTestUtils.hpp"
#include "TestUtils/RankTestUtils.hpp"

using std::runtime_error;
using std::string;

namespace sbwt_search {

TEST(RankTestUtilsTest, TestRank) {
  ASSERT_EQ(0, dummy_cpu_rank<false>(&bit_vector[0], 2));
  ASSERT_EQ(2, dummy_cpu_rank<false>(&bit_vector[0], 63));
  ASSERT_EQ(20, dummy_cpu_rank<false>(&bit_vector[0], 64 * 4 + 63 - 5));
  ASSERT_EQ(38, dummy_cpu_rank<false>(&bit_vector[0], 64 * 9 + 63 - 2));
  ASSERT_EQ(21, dummy_cpu_rank<true>(&bit_vector[0], 64 * 4 + 6 - 1));
}

}
