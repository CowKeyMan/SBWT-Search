#include <limits>

#include <gtest/gtest.h>
#include <sdsl/rank_support_v5.hpp>
#include <sdsl/util.hpp>

#include "PoppyBuilder/PoppyBuilder.h"
#include "Tools/RNGUtils.hpp"
#include "Tools/TypeDefinitions.h"
#include "UtilityKernels/Rank_test.h"
#include "sdsl/int_vector.hpp"

namespace sbwt_search {

using gpu_utils::GpuPointer;
using rng_utils::get_uniform_generator;

TEST(RankTest, TestAll) {
  const u64 num_bits = 1000;
  sdsl::bit_vector v;
  v.bit_resize(num_bits);
  const auto num_elements = v.capacity() / u64_bits;
  auto rng = get_uniform_generator<u64>(0, std::numeric_limits<u64>::max());
  for (u64 i = 0; i < num_elements; ++i) { v.set_int(i * u64_bits, rng()); }
  sdsl::rank_support_v5 rank_support;
  rank_support.set_vector(&v);
  sdsl::util::init_support(rank_support, &v);
  auto poppy = PoppyBuilder({v.data(), v.size()}, num_bits).get_poppy();
  auto d_v = GpuPointer<u64>(v.data(), num_elements);
  auto d_poppy_layer_0 = GpuPointer<u64>(poppy.layer_0);
  auto d_poppy_layer_1_2 = GpuPointer<u64>(poppy.layer_1_2);
  for (u64 i = 0; i < num_bits; ++i) {
    ASSERT_EQ(
      get_rank(d_v, d_poppy_layer_0, d_poppy_layer_1_2, i), rank_support.rank(i)
    ) << "Unequal at index "  // LCOV_EXCL_LINE
      << i;
  }
}

}  // namespace sbwt_search
