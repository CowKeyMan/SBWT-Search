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
  u32 blocks_per_grid
    = round_up<u64>(num_queries, threads_per_block) / threads_per_block;
  hipEvent_t &start_timer_ = *reinterpret_cast<hipEvent_t *>(start_timer.get());
  hipEvent_t &end_timer_ = *reinterpret_cast<hipEvent_t *>(end_timer.get());
  GPU_CHECK(hipEventRecord(start_timer_););
  hipLaunchKernelGGL(
    d_search,
    blocks_per_grid,
    threads_per_block,
    0,
    *reinterpret_cast<hipStream_t *>(gpu_stream.get()),
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
  GPU_CHECK(hipEventRecord(end_timer_));
  GPU_CHECK(hipPeekAtLastError());
  GPU_CHECK(hipDeviceSynchronize());
  float milliseconds = -1;
  GPU_CHECK(hipEventElapsedTime(&milliseconds, start_timer_, end_timer_));
  Logger::log(
    Logger::LOG_LEVEL::DEBUG,
    format("Batch {} took {} ms to search in the GPU", batch_id, milliseconds)
  );
}

}  // namespace sbwt_search
