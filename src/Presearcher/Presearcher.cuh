#ifndef PRESEARCHER_CUH
#define PRESEARCHER_CUH

/**
 * @file Presearcher.cuh
 * @brief Device function for presearching
 */

#include "Rank/Rank.cuh"
#include "Tools/KernelUtils.cuh"

namespace sbwt_search {

using gpu_utils::get_idx;

__global__ void d_presearch(
  const u64 *const c_map,
  const u64 *const *const acgt,
  const u64 *const *const layer_0,
  const u64 *const *const layer_1_2,
  u64 *presearch_left,
  u64 *presearch_right
) {
  const auto rank = d_rank;
  const u32 kmer = get_idx();
  u32 c = (kmer >> (presearch_letters * 2 - 2)) & two_1s;
  u64 node_left = c_map[c];
  u64 node_right = c_map[c + 1] - 1;
#pragma unroll
  for (u32 i = presearch_letters * 2 - 4;; i -= 2) {
    c = (kmer >> i) & two_1s;
    node_left = c_map[c] + rank(acgt[c], layer_0[c], layer_1_2[c], node_left);
    node_right
      = c_map[c] + rank(acgt[c], layer_0[c], layer_1_2[c], node_right + 1) - 1;
    if (i == 0) { break; }
  }
  presearch_left[kmer] = node_left;
  presearch_right[kmer] = node_right;
}

}  // namespace sbwt_search

#endif
