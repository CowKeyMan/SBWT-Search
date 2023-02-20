#ifndef VARIABLE_LENGTH_INTEGER_INDEX_CUH
#define VARIABLE_LENGTH_INTEGER_INDEX_CUH

/**
 * @file VariableLengthIntegerIndex.cuh
 * @brief Given a u64 integer vector whose size is not a regular number such as
 * 32 or 64 bits, we can index it using this function. So for example, given a
 * vector of u64s which contain the bits for a vector of integers of 12 bits
 * each, we can get the 1st, 2nd,...,nth element using this function. Note: only
 * works for variables with number of bits less than or equal to the bits of the
 * container (so if we have a list of u64 elements, the bit size of the
 * containing elements must be less than 64. This kernel is usually used for
 * indexing sdsl int_vectors of irregular size. The same implementation is done
 * by sdsl:
 * https://github.com/simongog/sdsl-lite/blob/master/include/sdsl/bits.hpp#L501
 */

#include "Tools/TypeDefinitions.h"
#include "hip/hip_runtime.h"

namespace sbwt_search {

inline __device__ auto d_variable_length_int_index(
  const u64 *container, u32 width, u64 width_set_bits, u64 index
) -> u64 {
  u64 index_1 = (index * width) / u64_bits;
  u64 offset = (index * width) % u64_bits;
  u64 elem_1 = container[index_1];
  u64 result = elem_1 >> offset;
  if (offset + width > u64_bits) {
    u64 elem_2 = container[index_1 + 1];
    u64 part_2 = (elem_2 << (u64_bits - offset));
    result = result | part_2;
  }
  return result & width_set_bits;
}

}  // namespace sbwt_search

#endif
