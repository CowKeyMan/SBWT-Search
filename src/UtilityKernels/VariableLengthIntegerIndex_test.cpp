#include <limits>

#include <gtest/gtest.h>
#include <sdsl/util.hpp>

#include "Tools/BitDefinitions.h"
#include "Tools/RNGUtils.hpp"
#include "Tools/TypeDefinitions.h"
#include "UtilityKernels/VariableLengthIntegerIndex_test.h"
#include "sdsl/int_vector.hpp"

namespace sbwt_search {

using bit_utils::set_bits;
using gpu_utils::GpuPointer;
using rng_utils::get_uniform_generator;

TEST(GetVariableLengthIntegerIndexTest, TestAll) {
  const u64 num_bits = 10000;
  for (u64 width : {1, 10, 20, 30, 40, 50, 60, 64}) {
    sdsl::int_vector v;
    v.width(width);
    v.bit_resize(num_bits);
    const auto num_elements = v.capacity() / u64_bits;
    auto rng = get_uniform_generator<u64>(0, std::numeric_limits<u64>::max());
    for (u64 i = 0; i < num_elements; ++i) { v.set_int(i * u64_bits, rng()); }
    // the +1 is added for convenience since the function always references the
    // next u64 bits as well
    auto d_v = GpuPointer<u64>(v.data(), num_elements + 1);
    for (u64 i = 0; i < v.size(); ++i) {
      ASSERT_EQ(
        v[i], get_variable_length_int_index(d_v, width, set_bits.at(width), i)
      ) << "Unequal at index "
        << i << " with width " << width;
    }
  }
}

}  // namespace sbwt_search
