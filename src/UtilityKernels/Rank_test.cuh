#ifndef RANK_TEST_CUH
#define RANK_TEST_CUH

/**
 * @file Rank_test.cuh
 * @brief A simple kernel which performs rank function for a single item in the
 * gpu. Used only for testing
 */

#include "UtilityKernels/Rank.cuh"
#include "hip/hip_runtime.h"

namespace sbwt_search {

__global__ void d_global_rank(
  const u64 *bit_vector,
  const u64 *layer_0,
  const u64 *layer_1_2,
  const u64 index,
  u64 *result
) {
  result[0] = d_rank(bit_vector, layer_0, layer_1_2, index);
}

}  // namespace sbwt_search

#endif
