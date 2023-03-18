#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include "Tools/StdUtils.hpp"

namespace std_utils {

using std::vector;

TEST(StdUtils, SplitVector) {
  const u64 vector_size = 20;
  const u64 split_size = 16;
  vector<int> v(vector_size, 1);
  auto split_v = split_vector(v, split_size);
  ASSERT_EQ(split_v.size(), split_size);
  u64 count = 0;
  for (auto &v : split_v) { count += std::accumulate(v.begin(), v.end(), 0); }
  ASSERT_EQ(count, vector_size);
}

}  // namespace std_utils
