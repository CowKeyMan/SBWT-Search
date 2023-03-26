#include "Global/GlobalDefinitions.h"
#include "IndexSearcher/IndexSearcher.cuh"
#include "IndexSearcher/IndexSearcher.h"
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

auto IndexSearcher::launch_search_kernel(u64 num_queries, u64 batch_id)
  -> void {
  u32 blocks_per_grid
    = round_up<u64>(num_queries, threads_per_block) / threads_per_block;
  start_timer.record(&gpu_stream);
  hipLaunchKernelGGL(
    d_search,
    blocks_per_grid,
    threads_per_block,
    0,
    *static_cast<hipStream_t *>(gpu_stream.get()),
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
  end_timer.record(&gpu_stream);
  GPU_CHECK(hipPeekAtLastError());
  GPU_CHECK(hipStreamSynchronize(*static_cast<hipStream_t *>(gpu_stream.get()))
  );
  float millis = start_timer.time_elapsed_ms(end_timer);
  Logger::log(
    Logger::LOG_LEVEL::DEBUG,
    format("Batch {} took {} ms to search in the GPU", batch_id, millis)
  );
}

}  // namespace sbwt_search
