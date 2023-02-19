#ifndef GET_BOOL_FROM_BIT_VECTOR_TEST_H
#define GET_BOOL_FROM_BIT_VECTOR_TEST_H

/**
 * @file GetBoolFromBitVector_test.h
 * @brief Header for function used in testing the GetBoolFromBitVector kernel
 */

#include "Tools/GpuPointer.h"
#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using gpu_utils::GpuPointer;

auto get_bool_from_bit_vector(const GpuPointer<u64> &bit_vector, u64 index)
  -> bool;

}  // namespace sbwt_search

#endif
