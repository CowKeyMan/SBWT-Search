#include <stdexcept>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "TestUtils/BitVectorTestUtils.hpp"
#include "TestUtils/RankTestUtils.hpp"

using std::runtime_error;
using std::string;

namespace sbwt_search {

TEST(RankTestUtilsTest, TestRank) {
  ASSERT_EQ(1, dummy_cpu_rank(&bit_vector[0], 1));
  ASSERT_EQ(20, dummy_cpu_rank(&bit_vector[0], 64 * 4 + 5));
}

}  // namespace sbwt_search
