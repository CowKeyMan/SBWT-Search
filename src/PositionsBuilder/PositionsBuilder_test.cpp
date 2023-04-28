#include <algorithm>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "PositionsBuilder/PositionsBuilder.h"

namespace sbwt_search {

using std::numeric_limits;
using std::string;
using std::vector;

/*
simulating the following sequences:
  "ACgT",  // size: 4, starting_index: 0
  "gA",  // size: 2, starting_index: 4
  "GAT",  // size: 3, starting_index: 6
  "GtCa",  // size: 4, starting_index: 9
  "AAAAaA",  // size: 6, starting_index: 13
  "GCAG"  // size: 4, starting_index: 19
*/

const auto kmer_size = 3;
const auto seq_size = 23;

TEST(PositionsBuilderTest, WithLastPosition) {
  const vector<u64> chars_before_newline
    = {4, 6, 9, 13, 19, 23, numeric_limits<u64>::max()};
  const vector<u64> expected = {0, 1, 6, 9, 10, 13, 14, 15, 16, 19, 20};
  auto host = PositionsBuilder(kmer_size);
  PinnedVector<u64> positions(9999);
  host.build_positions(chars_before_newline, seq_size, positions);
  ASSERT_EQ(positions.to_vector(), expected);
}

TEST(PositionsBuilderTest, WithNoLastPosition) {
  const vector<u64> chars_before_newline
    = {4, 6, 9, 13, 19, numeric_limits<u64>::max()};
  const vector<u64> expected = {0, 1, 6, 9, 10, 13, 14, 15, 16, 19, 20};
  auto host = PositionsBuilder(kmer_size);
  PinnedVector<u64> positions(9999);
  host.build_positions(chars_before_newline, seq_size, positions);
  ASSERT_EQ(positions.to_vector(), expected);
}

}  // namespace sbwt_search
