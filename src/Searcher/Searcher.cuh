#ifndef SEARCHER_CUH
#define SEARCHER_CUH

/**
 * @file Searcher.cuh
 * @brief Search implementation
 * */

#include <memory>
#include <utility>
#include <vector>

#include "Rank/Rank.cuh"
#include "SbwtContainer/GpuSbwtContainer.cuh"
#include "Utils/CudaKernelUtils.cuh"
#include "Utils/CudaUtils.cuh"
#include "Utils/MathUtils.hpp"
#include "Utils/TypeDefinitions.h"
#include "Utils/GlobalDefinitions.h"

using gpu_utils::get_idx;
using math_utils::round_up;
using std::shared_ptr;
using std::vector;

namespace sbwt_search {

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
    & (-((kmer_index % 64) != 0));
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
  if (node_left > node_right) node_left = -1ULL;
  out[idx] = node_left;
}

class SearcherGpu {
  private:
    shared_ptr<GpuSbwtContainer> container;

  public:
    SearcherGpu(shared_ptr<GpuSbwtContainer> container): container(container) {}

    auto search(
      const vector<u64> &bit_seqs,
      vector<u64> &kmer_positions,
      vector<u64> &results
    ) -> void {
      u32 blocks_per_grid
        = round_up<u64>(kmer_positions.size(), threads_per_block)
        / threads_per_block;
      auto d_bit_seqs = GpuPointer<u64>(bit_seqs);
      auto memory_reserved
        = round_up<u64>(kmer_positions.size(), superblock_bits);
      auto d_kmer_positions = GpuPointer<u64>(memory_reserved);
      d_kmer_positions.set(kmer_positions, kmer_positions.size());
      d_kmer_positions.memset(
        kmer_positions.size(), memory_reserved - kmer_positions.size(), 0
      );
      results.resize(kmer_positions.size());
      if (kmer_positions.size() > 0) {
        d_search<<<blocks_per_grid, threads_per_block>>>(
          container->get_kmer_size(),
          container->get_c_map().get(),
          container->get_acgt_pointers().get(),
          container->get_layer_0_pointers().get(),
          container->get_layer_1_2_pointers().get(),
          container->get_presearch_left().get(),
          container->get_presearch_right().get(),
          d_kmer_positions.get(),
          d_bit_seqs.get(),
          d_kmer_positions.get()
        );
        GPU_CHECK(cudaPeekAtLastError());
        GPU_CHECK(cudaDeviceSynchronize());
        d_kmer_positions.copy_to(results, kmer_positions.size());
      }
    }
};

}  // namespace sbwt_search

#endif
