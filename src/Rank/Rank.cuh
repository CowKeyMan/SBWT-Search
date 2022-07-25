#ifndef RANK_CUH
#define RANK_CUH

/**
 * @file Rank.cuh
 * @brief Rank implementation on GPU
 * */

#include "Utils/BitDefinitions.h"
#include "Utils/BitVectorUtils.h"
#include "Utils/TypeDefinitions.h"

namespace sbwt_search {

inline constexpr __host__ __device__ size_t log_base_2(const u64 n) {
  return ((n < 2) ? 0 : 1 + log_base_2(n / 2));
}

template <u64 superblock_bits, u64 hyperblock_bits, bool reversed_bits>
__device__ u64 d_rank(
  const u64 *bit_vector,
  const u64 *layer_0,
  const u64 *layer_1_2,
  const u64 index
) {
  const u64 basicblocks_in_superblock = 4;
  const u64 basicblock_bits = superblock_bits / basicblocks_in_superblock;
  u64 entry_basicblock = 0;
  const ulonglong2 *bit_vector_128b
    = reinterpret_cast<const ulonglong2 *>(bit_vector);
  const u64 target_shift_right = 64U - (index % 64U);
  const u64 ints_in_basicblock = basicblock_bits / 64;
  const u64 in_basicblock_index = (index / 64) % ints_in_basicblock;
#pragma unroll  // calculating entry_basicblock 2 ints at a time
  for (u64 i = 0; i < ints_in_basicblock; i += 2) {
    ulonglong2 data_128b = bit_vector_128b
      [(index / 128) - ((index / 128) % (ints_in_basicblock / 2)) + i / 2];
    if (reversed_bits) {
      entry_basicblock += __popcll(
                            data_128b.x
                            << (((i + 0) == in_basicblock_index)
                                * target_shift_right)
                          )
                        * ((i + 0) <= in_basicblock_index);
      entry_basicblock += __popcll(
                            data_128b.y
                            << (((i + 1) == in_basicblock_index)
                                * target_shift_right)
                          )
                        * ((i + 1) <= in_basicblock_index);
    } else {
      entry_basicblock
        += __popcll(
             data_128b.x
             >> (((i + 0) == in_basicblock_index) * target_shift_right)
           )
         * ((i + 0) <= in_basicblock_index);
      entry_basicblock
        += __popcll(
             data_128b.y
             >> (((i + 1) == in_basicblock_index) * target_shift_right)
           )
         * ((i + 1) <= in_basicblock_index);
    }
  }
  const u64 entry_layer_1_2 = layer_1_2[index / superblock_bits];
  u64 entry_layer_2_joined
    = (entry_layer_1_2 & thirty_1s)
   >> (10 * (3U - ((index % superblock_bits) / basicblock_bits)));
  const u64 entry_layer_2 = ((entry_layer_2_joined >> 20))
                          + ((entry_layer_2_joined >> 10) & ten_1s)
                          + ((entry_layer_2_joined >> 00) & ten_1s);
  const u64 entry_layer_1 = entry_layer_1_2 >> 32;
  const u64 entry_layer_0 = layer_0[index / hyperblock_bits];
  return entry_basicblock + entry_layer_2 + entry_layer_1 + entry_layer_0;
}

}

#endif
