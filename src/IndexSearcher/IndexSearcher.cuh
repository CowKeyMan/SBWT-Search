#ifndef INDEX_SEARCHER_CUH
#define INDEX_SEARCHER_CUH

/**
 * @file IndexSearcher.cuh
 * @brief Search implementation
 */

#include "Tools/BitDefinitions.h"
#include "Tools/KernelUtils.cuh"
#include "Tools/TypeDefinitions.h"
#include "UtilityKernels/Rank.cuh"
#include "hip/hip_runtime.h"

namespace sbwt_search {

using bit_utils::two_1s;
using gpu_utils::get_idx;

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
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
  u64 *out
) {
  const auto rank = d_rank;
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
  for (u32 i = presearch_letters * 2; i < kmer_size * 2; i += 2) {
    const u32 c = (kmer >> (62 - i)) & two_1s;
    node_left = c_map[c] + rank(acgt[c], layer_0[c], layer_1_2[c], node_left);
    node_right
      = c_map[c] + rank(acgt[c], layer_0[c], layer_1_2[c], node_right + 1) - 1;
  }
  if (node_left > node_right) { node_left = -1ULL; }
  out[idx] = node_left;
}
// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

}  // namespace sbwt_search

#endif
