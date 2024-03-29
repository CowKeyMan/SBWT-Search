#ifndef GET_BOOL_FROM_BIT_VECTOR_TEST_CUH
#define GET_BOOL_FROM_BIT_VECTOR_TEST_CUH

/**
 * @file GetBoolFromBitVector_test.cuh
 * @brief A simple kernel which performs the GetBoolFromBitVector function for a
 * single item in the gpu. Used only for testing
 */

#include "UtilityKernels/GetBoolFromBitVector.cuh"
#include "hip/hip_runtime.h"

namespace sbwt_search {

__global__ auto d_global_get_bool_from_bit_vector(
  const u64 *bit_vector, const u64 index, u64 *result
) -> void {
  result[0] = d_get_bool_from_bit_vector(bit_vector, index);
}

}  // namespace sbwt_search

#endif
