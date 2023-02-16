#include "Global/GlobalDefinitions.h"
#include "Rank/Rank.cuh"
#include "Searcher/Searcher.h"
#include "Tools/GpuUtils.h"
#include "Tools/KernelUtils.cuh"
#include "Tools/Logger.h"
#include "Tools/MathUtils.hpp"
#include "fmt/core.h"

namespace sbwt_search {

using fmt::format;
using gpu_utils::get_idx;
using log_utils::Logger;
using math_utils::round_up;

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

auto Searcher::launch_search_kernel(u64 queries, u64 batch_id) -> void {
  cudaEvent_t search_start;  // NOLINT(cppcoreguidelines-init-variables)
  cudaEvent_t search_stop;   // NOLINT(cppcoreguidelines-init-variables)
  GPU_CHECK(cudaEventCreate(&search_start));
  GPU_CHECK(cudaEventCreate(&search_stop));
  u32 blocks_per_grid
    = round_up<u64>(queries, threads_per_block) / threads_per_block;
  cudaEventRecord(search_start);
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
  cudaEventRecord(search_stop);
  GPU_CHECK(cudaPeekAtLastError());
  GPU_CHECK(cudaDeviceSynchronize());
  float milliseconds = -1;
  GPU_CHECK(cudaEventElapsedTime(&milliseconds, search_start, search_stop));
  Logger::log(
    Logger::LOG_LEVEL::DEBUG,
    format("Batch {} took {} ms to search in the GPU", batch_id, milliseconds)
  );
}

}  // namespace sbwt_search
