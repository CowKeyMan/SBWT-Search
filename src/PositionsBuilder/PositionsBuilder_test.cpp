#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "PositionsBuilder/PositionsBuilder.h"
#include "TestUtils/GeneralTestUtils.hpp"
#include "Utils/TypeDefinitions.h"

using std::partial_sum;
using std::string;
using std::transform;
using std::vector;

namespace sbwt_search {

const vector<string> string_seqs = {
  "ACgT",  // 00011011
  "gA",  // 1000
  "GAT",  // 100011
  "GtCa",  // 10110100
  "AAAAaAAaAAAAAAAaAAAAAAAAAAAAAAAA",  // 32 As = 64 0s
  "GC"  // 1001
};
auto kmer_size = 3;

auto build_expected() -> vector<u64> {
  auto expected = vector<u64>();
  auto cumsum_string_length = 0;
  for (auto &s: string_seqs) {
    for (int i = 0; i < s.size() - kmer_size + 1; ++i) {
      expected.push_back(i + cumsum_string_length);
    }
    cumsum_string_length += s.size();
  }
  return expected;
}

TEST(PositionsBuilder, Build) {
  auto string_lengths = vector<u64>(string_seqs.size());
  transform(
    string_seqs.begin(),
    string_seqs.end(),
    string_lengths.begin(),
    [](string s) -> u64 { return s.size(); }
  );
  auto cumsum_string_lengths = vector<u64>(string_seqs.size() + 1);
  partial_sum(
    string_lengths.begin(), string_lengths.end(), cumsum_string_lengths.begin() + 1
  );
  auto positions_per_string = vector<u64>(string_seqs.size());
  transform(
    string_lengths.begin(),
    string_lengths.end(),
    positions_per_string.begin(),
    [](u64 x) -> u64 { return (x - 2) > 0 ? x - 2 : 0; }
  );
  auto cumsum_positions_per_string = vector<u64>(string_seqs.size() + 1);
  partial_sum(
    positions_per_string.begin(),
    positions_per_string.end(),
    cumsum_positions_per_string.begin() + 1
  );

  auto positions
    = PositionsBuilder(cumsum_positions_per_string.back())
        .get_positions(
          cumsum_positions_per_string, cumsum_string_lengths, kmer_size
        );
  auto expected = build_expected();
  assert_vectors_equal<u64>(*positions, expected);
}

};
