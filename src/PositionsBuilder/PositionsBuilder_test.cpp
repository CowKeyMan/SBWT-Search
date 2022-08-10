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
  "ACgT",  // size: 4, starting_index: 0
  "gA",  // size: 2, starting_index: 4
  "GAT",  // size: 3, starting_index: 6
  "GtCa",  // size: 4, starting_index: 9
  "AAAAaA",  // size: 6, starting_index: 13
  "GC"  // size: 2, starting_index: 19
};
auto kmer_size = 3;

const vector<u64> expected = { 0, 1, 6, 9, 10, 13, 14, 15, 16 };

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
    string_lengths.begin(),
    string_lengths.end(),
    cumsum_string_lengths.begin() + 1
  );
  auto positions_per_string = vector<u64>(string_seqs.size());
  transform(
    string_lengths.begin(),
    string_lengths.end(),
    positions_per_string.begin(),
    [](u64 x) -> u64 { return (x - (kmer_size - 1)) > 0 ? x - (kmer_size - 1) : 0; }
  );
  auto cumsum_positions_per_string = vector<u64>(string_seqs.size() + 1);
  partial_sum(
    positions_per_string.begin(),
    positions_per_string.end(),
    cumsum_positions_per_string.begin() + 1
  );
  auto positions = PositionsBuilder().get_positions(
    cumsum_positions_per_string, cumsum_string_lengths, kmer_size
  );
  assert_vectors_equal<u64>(*positions, expected);
}

};
