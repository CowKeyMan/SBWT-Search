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
vector<size_t> chars_before_newline = { 4, 6, 9, 13, 19, 23, size_t(-1) };
const auto seq_size = 23;

TEST(PositionsBuilderTest, WithLastPosition) {
  auto host = PositionsBuilder(kmer_size);
  vector<size_t> positions;
  host.build_positions(chars_before_newline, seq_size, positions);
  ASSERT_EQ(positions, expected);
}

TEST(PositionsBuilderTest, WithNoLastPosition) {
  chars_before_newline.pop_back();
  chars_before_newline.pop_back();
  chars_before_newline.push_back(size_t(-1));
  auto host = PositionsBuilder(kmer_size);
  vector<size_t> positions;
  host.build_positions(chars_before_newline, seq_size, positions);
  ASSERT_EQ(positions, expected);
}

}  // namespace sbwt_search
