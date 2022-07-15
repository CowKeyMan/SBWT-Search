#ifndef RANK_CUH
#define RANK_CUH

/**
 * @file Rank.cuh
 * @brief Rank implementation on GPU
 * */

#include "Utils/TypeDefinitionUtils.h"

namespace sbwt_search {

inline constexpr __host__ __device__ size_t log_base_2(const u64 n) {
  return ((n < 2) ? 0 : 1 + log_base_2(n / 2));
}

const auto two_1s = 3ULL;
const auto ten_1s = 0x3ffULL;

template <u64 superblock_bits, u64 hyperblock_bits>
__device__ u64 d_rank(
  const u64 *const data,
  const u64 *const layer_0,
  const u64 *const layer_1_2,
  const u64 index
) {
  constexpr const u64 basicblock_bits = superblock_bits / 4;
  constexpr const u64 superblock_bits_log_2 = log_base_2(superblock_bits);
  constexpr const u64 hyperblock_bits_log_2 = log_base_2(hyperblock_bits);
  constexpr const u64 ints_in_basicblock = basicblock_bits / 64;
  const u64 entry_layer_0 = layer_0[index >> hyperblock_bits_log_2];
  const u64 entry_layer_1_2 = layer_1_2[index >> superblock_bits_log_2];
  const u64 entry_layer_1 = entry_layer_1_2 >> hyperblock_bits_log_2;
  const u64 entry_layer_2
    = (((entry_layer_1_2 >> 20) & ten_1s)
       * (0 < ((index % superblock_bits) / basicblock_bits)))
    + (((entry_layer_1_2 >> 10) & ten_1s)
       * (1 < ((index % superblock_bits) / basicblock_bits)))
    + (((entry_layer_1_2 >> 00) & ten_1s)
       * (2 < ((index % superblock_bits) / basicblock_bits)));
  u64 entry_basic_block = 0;
  const u64 in_basicblock_index = (index / 64) % ints_in_basicblock;
  const u64 initial_basicblock_index
    = (index / 64) - (index / 64) % ints_in_basicblock;
  const u64 target_shift_right = 63U - (index % 64);
#pragma unroll 4
  for (u64 i = 0; i < ints_in_basicblock; ++i) {
    entry_basic_block += __popcll(
      (data[initial_basicblock_index + i] * (i <= in_basicblock_index))
      >> ((i == in_basicblock_index) * target_shift_right)
    );
  }
  auto result
    = entry_layer_0 + entry_layer_1 + entry_layer_2 + entry_basic_block;
  return result;
}

}

#endif
