#include "Global/GlobalDefinitions.h"
#include "Searcher/Searcher.cuh"
#include "Searcher/Searcher.h"
#include "Tools/GpuUtils.h"
#include "Tools/KernelUtils.cuh"
#include "Tools/Logger.h"
#include "Tools/MathUtils.hpp"
#include "UtilityKernels/Rank.cuh"
#include "fmt/core.h"

namespace sbwt_search {

using fmt::format;
using log_utils::Logger;
using math_utils::round_up;

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
