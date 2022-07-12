#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "RankIndexBuilder/CpuRankIndexBuilder.hpp"
#include "Utils/TestUtils.hpp"
#include "Utils/TypeDefinitionUtils.h"

using std::vector;

namespace sbwt_search {

constexpr u64 get_ones(const int ones) {
  u64 result = 0;
  for (int i = 1; i <= ones; ++i) { result |= (1ULL << i); }
  return result;
}

const vector<u64> bit_vector = {
  // # Hyper Block 1 - total: 130
  // ## Super Block 1 - total: 30
  // ### Basic Block 01 - total: 5
  get_ones(2),
  get_ones(3),
  // ### Basic Block 02 - total: 10
  get_ones(1),
  get_ones(9),
  // ### Basic Block 03 - total: 10
  get_ones(10),
  get_ones(0),
  // ### Basic Block 04 - total: 5
  get_ones(1),
  get_ones(4),

  // ## Super Block 2 - total: 100
  // ### Basic Block 05 - total: 10
  get_ones(5),
  get_ones(5),
  // ### Basic Block 06 - total: 65
  get_ones(60),
  get_ones(5),
  // ### Basic Block 07 - total: 20
  get_ones(13),
  get_ones(7),
  // ### Basic Block 08 - total: 5
  get_ones(0),
  get_ones(5),

  // # Hyper Block 2 - total: 2
  // ## Super Block 3 - total: 2
  // ### Basic Block 09 - total: 2
  get_ones(2),
  get_ones(0),  // PADDING TO NEXT SUPER BLOCK

  get_ones(0),  // PADDING TO NEXT SUPER BLOCK
  get_ones(0),  // PADDING TO NEXT SUPER BLOCK

  get_ones(0),  // PADDING TO NEXT SUPER BLOCK
  get_ones(0),  // PADDING TO NEXT SUPER BLOCK

  get_ones(0),  // PADDING TO NEXT SUPER BLOCK
  get_ones(0)  // PADDING TO NEXT SUPER BLOCK
};

vector<u64> layer_0 = { 0, 130 };
vector<u64> layer_1_2 = {
  (0ULL << 32) | (5ULL << 20) | (10ULL << 10) | (10ULL << 0),
  (30ULL << 32) | (10ULL << 20) | (65ULL << 10) | (20ULL << 0),
  (0ULL << 32) | (2ULL << 20) | (0ULL << 10) | (0ULL << 0),
};
vector<u64> c_map = { 1, 132 + 1, 132 * 2 + 1, 132 * 3 + 1, 132 * 4 + 1 };

BitVectorSbwtContainer build_container() {
  auto four_vectors
    = vector<vector<u64>>{ bit_vector, bit_vector, bit_vector, bit_vector };
  return BitVectorSbwtContainer(
    move(four_vectors[0]),
    move(four_vectors[1]),
    move(four_vectors[2]),
    move(four_vectors[3]),
    64 * 17  // 17 u64s total
  );
}

TEST(RankIndexBuilderTest, BuildIndex) {
  auto container = build_container();
  CpuRankIndexBuilder<
    BitVectorSbwtContainer,
    64 * 8,  // 8 u64s = 1 super block
    64 * 8 * 2  // 2 super blocks + 1 hyper block
    >
    host(container);
  host.build_index();
  assert_vectors_equal<u64>(
    layer_0, container.get_layer_0(static_cast<ACGT>(0))
  );
  assert_vectors_equal<u64>(
    layer_1_2, container.get_layer_1_2(static_cast<ACGT>(0))
  );
  assert_vectors_equal<u64>(c_map, container.get_c_map());
}

}