#include <algorithm>
#include <cstdlib>
#include <vector>

#include "gtest/gtest.h"
#include <pstl/glue_algorithm_defs.h>

#include "Rank/Rank_test.hpp"
#include "RankIndexBuilder/RankIndexBuilder.hpp"
#include "TestUtils/RankTestUtils.hpp"
#include "Utils/TypeDefinitions.h"

using std::generate;
using std::srand;
using std::vector;

namespace sbwt_search {

class RankTestClass {
  public:
    vector<u64> layer_0, layer_1_2, expected, bit_vector;
    const vector<u64> test_indexes{ 60,  120,  180,  240,  320 - 1,
                                    320, 1000, 3540, 5000, 6001 };

    RankTestClass() {
      std::srand(42);
      bit_vector = vector<u64>(1000);
      std::generate(bit_vector.begin(), bit_vector.end(), std::rand);
      // execute and test, but first make rank work with template parameters
      auto builder
        = SingleIndexBuilder<1024, 1ULL << 32>(1000 * 64, &bit_vector[0]);
      builder.build();
      layer_0 = builder.get_layer_0();
      layer_1_2 = builder.get_layer_1_2();
      expected = vector<u64>(test_indexes.size());
      for (auto i = 0; i < test_indexes.size(); ++i) {
        expected[i]
          = dummy_cpu_rank(&bit_vector[0], test_indexes[i]);
      }
    }
};

TEST(RankTest, RankTest) {
  auto host = RankTestClass();
  auto actual = get_rank_output(
    host.bit_vector, host.layer_0, host.layer_1_2, host.test_indexes
  );
  ASSERT_EQ(host.expected, actual);
}

}  // namespace sbwt_search
