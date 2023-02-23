#include "Global/GlobalDefinitions.h"
#include "Searcher/Searcher.cuh"
#include "Searcher/Searcher.h"
#include "Tools/GpuUtils.h"
#include "Tools/KernelUtils.cuh"
#include "Tools/Logger.h"
#include "Tools/MathUtils.hpp"
#include "UtilityKernels/Rank.cuh"
#include "fmt/core.h"
#include "hip/hip_runtime.h"

namespace sbwt_search {

using fmt::format;
using log_utils::Logger;
using math_utils::round_up;

auto Searcher::launch_search_kernel(u64 num_queries, u64 batch_id) -> void {
  hipEvent_t search_start;  // NOLINT(cppcoreguidelines-init-variables)
  hipEvent_t search_stop;   // NOLINT(cppcoreguidelines-init-variables)
  GPU_CHECK(hipEventCreate(&search_start));
  GPU_CHECK(hipEventCreate(&search_stop));
  u32 blocks_per_grid
    = round_up<u64>(num_queries, threads_per_block) / threads_per_block;
  GPU_CHECK(hipEventRecord(search_start));
  hipLaunchKernelGGL(
    d_search,
    blocks_per_grid,
    threads_per_block,
    0,
    nullptr,
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
  GPU_CHECK(hipEventRecord(search_stop));
  GPU_CHECK(hipPeekAtLastError());
  GPU_CHECK(hipDeviceSynchronize());
  float milliseconds = -1;
  GPU_CHECK(hipEventElapsedTime(&milliseconds, search_start, search_stop));
  Logger::log(
    Logger::LOG_LEVEL::DEBUG,
    format("Batch {} took {} ms to search in the GPU", batch_id, milliseconds)
  );
}

}  // namespace sbwt_search
