#ifndef RANK_CUH
#define RANK_CUH

/**
 * @file Rank.cuh
 * @brief Rank implementation on GPU
 */

#include <memory>

#include "Global/GlobalDefinitions.h"
#include "Tools/BitDefinitions.h"
#include "Tools/GpuIntrinsics.cuh"
#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

inline __device__ auto d_rank(
  const u64 *bit_vector,
  const u64 *layer_0,
  const u64 *layer_1_2,
  const u64 index
) -> u64 {
  const u64 basicblocks_in_superblock = 4;
  const u64 basicblock_bits = superblock_bits / basicblocks_in_superblock;
  u64 entry_basicblock = 0;
  const auto *bit_vector_128b
    = reinterpret_cast<const ulonglong2 *>(bit_vector);
  const u64 target_shift = 64U - (index % 64U);
  const u64 ints_in_basicblock = basicblock_bits / 64;
  const u64 in_basicblock_index = (index / 64) % ints_in_basicblock;
#pragma unroll  // calculating entry_basicblock 2 ints at a time
  for (u64 i = 0; i < ints_in_basicblock; i += 2) {
    ulonglong2 data_128b = bit_vector_128b
      [(index / 128) - ((index / 128) % (ints_in_basicblock / 2)) + i / 2];
    // WARNING: UNDEFINED BEHAVIOUR HERE WHEN target_shift = 64
    // (expected to equal 0 - works well on CUDA)
    entry_basicblock
      += __popcll(
           data_128b.x << (((i + 0) == in_basicblock_index) * target_shift)
         )
        * ((i + 0) <= in_basicblock_index)
      + __popcll(
          data_128b.y << (((i + 1) == in_basicblock_index) * target_shift)
        )
        * ((i + 1) <= in_basicblock_index);
  }
  const u64 entry_layer_1_2 = layer_1_2[index / superblock_bits];
  u64 entry_layer_2_joined = (entry_layer_1_2 & thirty_1s)
    >> (10 * (3U - ((index % superblock_bits) / basicblock_bits)));
  const u64 entry_layer_2 = ((entry_layer_2_joined >> 20))
    + ((entry_layer_2_joined >> 10) & ten_1s)
    + ((entry_layer_2_joined >> 00) & ten_1s);
  const u64 entry_layer_1 = entry_layer_1_2 >> 32;
  const u64 entry_layer_0 = layer_0[index / hyperblock_bits];
  return entry_basicblock + entry_layer_2 + entry_layer_1 + entry_layer_0;
}

}  // namespace sbwt_search

#endif
