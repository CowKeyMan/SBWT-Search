#ifndef SEARCHER_CUH
#define SEARCHER_CUH

/**
 * @file Searcher.cuh
 * @brief Search implementation
 * */

#include <utility>
#include <vector>

#include "Rank/Rank.cuh"
#include "SbwtContainer/GpuSbwtContainer.cuh"
#include "Utils/CudaUtilFunctions.cuh"
#include "Utils/TypeDefinitions.h"

using std::vector;

namespace sbwt_search {

template <
  u64 superblock_bits,
  u64 hyperblock_bits,
  u64 presearch_letters,
  u32 kmer_size>
__global__ void d_search(
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
  const auto rank = d_rank<superblock_bits, hyperblock_bits>;
  const u32 idx = get_idx();
  const u64 kmer_index = kmer_positions[idx] * 2;
  const u64 first_part = (bit_seqs[kmer_index / 64] << (kmer_index % 64));
  const u64 second_part
    = (bit_seqs[(kmer_index / 64) + 1] >> (64 - (kmer_index % 64)));
  const u64 kmer = first_part | second_part;
  constexpr const u32 presearch_mask = (2ULL << (presearch_letters * 2)) - 1;
  const u32 presearched
    = (kmer >> 64 - (presearch_letters * 2)) & presearch_mask;
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

class Searcher {
  private:
    GpuSbwtContainer *const container;

  public:
    Searcher(GpuSbwtContainer *const container): container(container) {}
    template <
      u32 threads_per_block,
      u64 superblock_bits,
      u64 hyperblock_bits,
      u32 presearch_letters,
      u32 kmer_size>
    vector<u64>
    search(const vector<u64> &kmer_positions, const vector<u64> &bit_seqs) {
      auto blocks_per_grid = kmer_positions.size() / threads_per_block;
      auto d_bit_seqs = CudaPointer<u64>(bit_seqs);
      auto d_kmer_positions = CudaPointer<u64>(kmer_positions);
      d_search<superblock_bits, hyperblock_bits, presearch_letters, kmer_size>
        <<<blocks_per_grid, threads_per_block>>>(
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
      CUDA_CHECK(cudaPeekAtLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      auto result = vector<u64>(kmer_positions.size());
      d_kmer_positions.copy_to(result);
      return result;
    }
};

}

#endif
