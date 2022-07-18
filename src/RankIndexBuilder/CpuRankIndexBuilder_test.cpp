#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "RankIndexBuilder/RankIndexBuilder.hpp"
#include "SbwtContainer/CpuSbwtContainer.hpp"
#include "TestUtils/BitVectorTestUtils.hpp"
#include "TestUtils/GeneralTestUtils.hpp"
#include "Utils/TypeDefinitions.h"

using std::vector;

namespace sbwt_search {

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
    expected_layer_0, container.get_layer_0(static_cast<ACGT>(0))
  );
  assert_vectors_equal<u64>(
    expected_layer_1_2, container.get_layer_1_2(static_cast<ACGT>(0))
  );
  assert_vectors_equal<u64>(c_map, container.get_c_map());
}

}
