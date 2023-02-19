#include <limits>

#include <gtest/gtest.h>
#include <sdsl/util.hpp>

#include "Tools/RNGUtils.hpp"
#include "Tools/TypeDefinitions.h"
#include "UtilityKernels/GetBoolFromBitVector_test.h"
#include "sdsl/int_vector.hpp"

namespace sbwt_search {

using gpu_utils::GpuPointer;
using rng_utils::get_uniform_generator;

TEST(GetBoolFromBitVectorTest, TestAll) {
  const u64 num_bits = 1000;
  sdsl::bit_vector v;
  v.bit_resize(num_bits);
  const auto num_elements = v.capacity() / u64_bits;
  auto rng = get_uniform_generator<u64>(0, std::numeric_limits<u64>::max());
  for (u64 i = 0; i < num_elements; ++i) { v.set_int(i * u64_bits, rng()); }
  auto d_v = GpuPointer<u64>(v.data(), num_elements);
  for (u64 i = 0; i < num_bits; ++i) {
    ASSERT_EQ(v[i], get_bool_from_bit_vector(d_v, i))
      << "Unequal at index " << i;
  }
}

}  // namespace sbwt_search
