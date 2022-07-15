#include <algorithm>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "Rank/Rank_test.hpp"
#include "RankIndexBuilder/RankIndexBuilder.hpp"
#include "TestUtils/BitVectorTestUtils.hpp"
#include "TestUtils/RankTestUtils.hpp"
#include "Utils/TypeDefinitionUtils.h"

using std::generate;
using std::srand;
using std::vector;

namespace sbwt_search {

TEST(RankTest, RankTest) {
  std::srand(42);
  auto bit_vector = vector<u64>(1000);
  std::generate(bit_vector.begin(), bit_vector.end(), std::rand);
  // execute and test, but first make rank work with template parameters
  auto builder
    = SingleIndexBuilder<1024, 1ULL << 32>(1000 * 64, &bit_vector[0]);
  builder.build();
  auto layer_0 = builder.get_layer_0();
  auto layer_1_2 = builder.get_layer_1_2();
  auto test_indexes
    = vector<u64>({ 60, 120, 180, 240, 320 - 1, 320, 1000, 3540, 5000, 6001 });
  auto actual = get_rank_output(bit_vector, layer_0, layer_1_2, test_indexes);
  auto expected = vector<u64>(test_indexes.size());
  for (auto i = 0; i < test_indexes.size(); ++i) {
    expected[i] = dummy_cpu_rank<false>(&bit_vector[0], test_indexes[i]);
  }
  ASSERT_EQ(expected, actual);
}

}
