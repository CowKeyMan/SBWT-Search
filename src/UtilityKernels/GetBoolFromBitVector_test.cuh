#ifndef RANK_TEST_CUH
#define RANK_TEST_CUH

/**
 * @file Rank_test.cuh
 * @brief A simple kernel which performs the GetBoolFromBitVector function for a
 * single item in the gpu. Used only for testing
 */

#include "UtilityKernels/GetBoolFromBitVector.cuh"
#include "hip/hip_runtime.h"

namespace sbwt_search {

__global__ void d_global_get_bool_from_bit_vector(
  const u64 *bit_vector, const u64 index, u64 *result
) {
  result[0] = d_get_bool_from_bit_vector(bit_vector, index);
}

}  // namespace sbwt_search

#endif
