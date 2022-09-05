#ifndef PRESEARCHER_CUH
#define PRESEARCHER_CUH

/**
 * @file Presearcher.cuh
 * @brief Does the presearch for the sbwt index searching
 * */

#include "Rank/Rank.cuh"
#include "SbwtContainer/GpuSbwtContainer.cuh"
#include "Utils/CudaKernelUtils.cuh"
#include "Utils/CudaUtils.cuh"
#include "Utils/GlobalDefinitions.h"
#include "Utils/MathUtils.hpp"
#include "Utils/TypeDefinitions.h"
#include "Utils/Logger.h"

using gpu_utils::get_idx;
using math_utils::round_up;
using std::make_unique;
using std::move;
using std::shared_ptr;
using log_utils::Logger;

namespace sbwt_search {

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
    void presearch() {
      constexpr const auto presearch_times
        = round_up<size_t>(1ULL << (presearch_letters * 2), threads_per_block);
      auto blocks_per_grid = presearch_times / threads_per_block;
      auto presearch_left = make_unique<GpuPointer<u64>>(presearch_times);
      auto presearch_right = make_unique<GpuPointer<u64>>(presearch_times);
      Logger::log_timed_event("PresearchFunction", Logger::EVENT_STATE::START);
      d_presearch<<<blocks_per_grid, threads_per_block>>>(
        container->get_c_map().get(),
        container->get_acgt_pointers().get(),
        container->get_layer_0_pointers().get(),
        container->get_layer_1_2_pointers().get(),
        presearch_left->get(),
        presearch_right->get()
      );
      GPU_CHECK(cudaPeekAtLastError());
      GPU_CHECK(cudaDeviceSynchronize());
      Logger::log_timed_event("PresearchFunction", Logger::EVENT_STATE::STOP);
      container->set_presearch(move(presearch_left), move(presearch_right));
    }
};

}  // namespace sbwt_search

#endif
