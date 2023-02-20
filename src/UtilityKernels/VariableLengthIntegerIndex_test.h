#ifndef VARIABLE_LENGTH_INTEGER_INDEX_TEST_H
#define VARIABLE_LENGTH_INTEGER_INDEX_TEST_H

/**
 * @file VariableLengthIntegerIndex_test.h
 * @brief Header for functions used in testing the d_variable_length_int_index
 */

#include "Tools/GpuPointer.h"
#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using gpu_utils::GpuPointer;

auto get_variable_length_int_index(
  const GpuPointer<u64> &container, u64 width, u64 width_set_bits, u64 index
) -> u64;

}  // namespace sbwt_search

#endif
