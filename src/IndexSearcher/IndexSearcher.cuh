#ifndef INDEX_SEARCHER_CUH
#define INDEX_SEARCHER_CUH

/**
 * @file IndexSearcher.cuh
 * @brief Search implementation
 */

#include "Tools/BitDefinitions.h"
#include "Tools/KernelUtils.cuh"
#include "Tools/TypeDefinitions.h"
#include "UtilityKernels/GetBoolFromBitVector.cuh"
#include "UtilityKernels/Rank.cuh"
#include "hip/hip_runtime.h"

namespace sbwt_search {

using bit_utils::two_1s;
using gpu_utils::get_idx;

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
template <bool move_to_key_kmer>
__global__ void d_search(
  const u32 kmer_size,
  const u64 *const c_map,
  const u64 *const *const acgt,
  const u64 *const *const layer_0,
  const u64 *const *const layer_1_2,
  const u64 *const presearch_left,
  const u64 *const presearch_right,
  const u64 *const kmer_positions,
  const u64 *const bit_seqs,
  const u64 *const key_kmer_marks,
  u64 *out
) {
  const u32 idx = get_idx();
  const u64 kmer_index = kmer_positions[idx] * 2;
  const u64 first_part = (bit_seqs[kmer_index / 64] << (kmer_index % 64));
  const u64 second_part
    = (bit_seqs[kmer_index / 64 + 1] >> (64 - (kmer_index % 64)))
    & static_cast<u64>(-static_cast<u64>((kmer_index % 64) != 0));
  const u64 kmer = first_part | second_part;
  constexpr const u64 presearch_mask = (2ULL << (presearch_letters * 2)) - 1;
  const u32 presearched
    = (kmer >> (64 - presearch_letters * 2)) & presearch_mask;
  u64 node_left = presearch_left[presearched];
  u64 node_right = presearch_right[presearched];
  for (u64 i = kmer_positions[idx] + presearch_letters;
       i < kmer_positions[idx] + kmer_size;
       ++i) {
    const u32 c = (bit_seqs[i / 32] >> (62 - (i % 32) * 2)) & two_1s;
    node_left = c_map[c] + d_rank(acgt[c], layer_0[c], layer_1_2[c], node_left);
    node_right = c_map[c]
      + d_rank(acgt[c], layer_0[c], layer_1_2[c], node_right + 1) - 1;
  }
  if (node_left > node_right) {
    out[idx] = -1ULL;
    return;
  }
  if (move_to_key_kmer) {
    while (!d_get_bool_from_bit_vector(key_kmer_marks, node_left)) {
      for (u32 i = 0; i < 4; ++i) {
        if (d_get_bool_from_bit_vector(acgt[i], node_left)) {
          node_left
            = c_map[i] + d_rank(acgt[i], layer_0[i], layer_1_2[i], node_left);
          break;
        }
      }
    }
  }
  out[idx] = node_left;
}
// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

}  // namespace sbwt_search

#endif
