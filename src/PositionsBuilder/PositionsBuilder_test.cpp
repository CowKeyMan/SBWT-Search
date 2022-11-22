#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "PositionsBuilder/PositionsBuilder.h"

using std::accumulate;
using std::string;
using std::transform;
using std::vector;

namespace sbwt_search {

const vector<string> seqs = {
  "ACgT",  // size: 4, starting_index: 0
  "gA",  // size: 2, starting_index: 4
  "GAT",  // size: 3, starting_index: 6
  "GtCa",  // size: 4, starting_index: 9
  "AAAAaA",  // size: 6, starting_index: 13
  "GCAG"  // size: 4, starting_index: 19
};
auto kmer_size = 3;

const vector<size_t> expected = { 0, 1, 6, 9, 10, 13, 14, 15, 16, 19, 20 };
vector<size_t> string_breaks = { 4, 6, 9, 13, 19, 23 };
const auto seq_size = 23;

TEST(PositionsBuilderTest, WithNoLastPosition) {
  auto host = PositionsBuilder(kmer_size);
  vector<size_t> positions;
  host.build_positions(string_breaks, seq_size, positions);
  ASSERT_EQ(positions, expected);
}

TEST(PositionsBuilderTest, WithLastPosition) {
  string_breaks.pop_back();
  auto host = PositionsBuilder(kmer_size);
  vector<size_t> positions;
  host.build_positions(string_breaks, seq_size, positions);
  ASSERT_EQ(positions, expected);
}

}  // namespace sbwt_search
