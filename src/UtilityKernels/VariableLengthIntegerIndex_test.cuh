#ifndef VARIABLE_LENGTH_INTEGER_INDEX_TEST_CUH
#define VARIABLE_LENGTH_INTEGER_INDEX_TEST_CUH

/**
 * @file VariableLengthIntegerIndex_test.cuh
 * @brief Test kernel for VariableLengthIntegerIndex
 */

#include "Tools/TypeDefinitions.h"
#include "UtilityKernels/VariableLengthIntegerIndex.cuh"
#include "hip/hip_runtime.h"

namespace sbwt_search {

__global__ auto d_global_variable_length_int_index(
  const u64 *container, u32 width, u64 width_set_bits, u64 index, u64 *result
) -> void {
  result[0]
    = d_variable_length_int_index(container, width, width_set_bits, index);
}

}  // namespace sbwt_search

#endif
