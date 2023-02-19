#ifndef GET_BOOL_FROM_BIT_VECTOR_CUH
#define GET_BOOL_FROM_BIT_VECTOR_CUH

/**
 * @file GetBoolFromBitVector.cuh
 * @brief Given a bit vector (which is an array of u64s, where each bit in each
 * element should be considered individually), this function will get the index
 * of a certain bit within the whole bit vector and return wether it is False
 * (0) or True (1). For example, given the index 78, this will be found on the
 * 2nd element of the u64 array, and we must check the (78 - 64 =) 14th bit.
 * This is the same as the sdsl-lite library's bit vector.
 */

#include "Tools/TypeDefinitions.h"
#include "hip/hip_runtime.h"

inline __device__ auto d_get_bool_from_bit_vector(u64 *bitmap, u64 index)
  -> bool {
  auto elem_idx = index / u64_bits;
  auto elem_offset = index / u64_bits;
  return (bitmap[elem_idx] & (1ULL << elem_offset)) > 0;
}

#endif
