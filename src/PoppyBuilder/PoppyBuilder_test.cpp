#include <deque>
#include <memory>

#include "gtest/gtest.h"
#include <gtest/gtest.h>
#include <sdsl/int_vector.hpp>

#include "PoppyBuilder/PoppyBuilder.h"
#include "TestUtils/BitVectorTestUtils.hpp"
#include "TestUtils/GeneralTestUtils.hpp"
#include "Utils/TypeDefinitions.h"

using std::make_shared;
using std::move;
using std::shared_ptr;

namespace sbwt_search {

TEST(PoppyBuilderTest, BuildIndex) {
  auto v = vector_to_sdsl(bit_array, 64 * bit_array.size());
  PoppyBuilder host(v.size(), v.data(), 64 * 8 * 2, 64 * 8);
  // 8 u64s = 1 super block, 2 super blocks = 1 hyper block
  host.build();
  assert_vectors_equal<u64>(
    expected_layer_1_2, host.get_layer_1_2(), __FILE__, __LINE__
  );
  assert_vectors_equal<u64>(
    expected_layer_0, host.get_layer_0(), __FILE__, __LINE__
  );
  ASSERT_EQ(total_1s, host.get_total_count());
}

}  // namespace sbwt_search
