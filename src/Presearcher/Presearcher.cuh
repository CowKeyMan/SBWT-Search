#ifndef PRESEARCHER_CUH
#define PRESEARCHER_CUH

/**
 * @file Presearcher.cuh
 * @brief Does the presearch for the sbwt index searching
 * */

#include "Rank/Rank.cuh"
#include "SbwtContainer/GpuSbwtContainer.cuh"
#include "Utils/CudaUtilFunctions.cuh"
#include "Utils/MathUtils.hpp"
#include "Utils/TypeDefinitions.h"

using std::move;
using std::shared_ptr;
using std::make_unique;

namespace sbwt_search {

template <
  u64 superblock_bits,
  u64 hyperblock_bits,
  u32 presearch_letters,
  bool reversed_bits>
__global__ void d_presearch(
  const u64 *const c_map,
  const u64 *const *const acgt,
  const u64 *const *const layer_0,
  const u64 *const *const layer_1_2,
  u64 *presearch_left,
  u64 *presearch_right
) {
  const auto rank = d_rank<superblock_bits, hyperblock_bits, reversed_bits>;
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
    if (i == 0) break;
  }
  presearch_left[kmer] = node_left;
  presearch_right[kmer] = node_right;
}

class Presearcher {
  private:
    shared_ptr<GpuSbwtContainer> container;

  public:
    Presearcher(shared_ptr<GpuSbwtContainer> container): container(container) {}
    template <
      u32 threads_per_block,
      u64 superblock_bits,
      u64 hyperblock_bits,
      u32 presearch_letters,
      bool reversed_bits>
    void presearch() {
      constexpr const auto presearch_times
        = round_up<size_t>(1ULL << (presearch_letters * 2), threads_per_block);
      auto blocks_per_grid = presearch_times / threads_per_block;
      auto presearch_left = make_unique<CudaPointer<u64>>(presearch_times);
      auto presearch_right = make_unique<CudaPointer<u64>>(presearch_times);
      d_presearch<
        superblock_bits,
        hyperblock_bits,
        presearch_letters,
        reversed_bits><<<blocks_per_grid, threads_per_block>>>(
        container->get_c_map().get(),
        container->get_acgt_pointers().get(),
        container->get_layer_0_pointers().get(),
        container->get_layer_1_2_pointers().get(),
        presearch_left->get(),
        presearch_right->get()
      );
      CUDA_CHECK(cudaPeekAtLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      container->set_presearch(move(presearch_left), move(presearch_right));
    }
};

}

#endif
