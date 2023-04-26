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
  if (move_to_key_kmer) {
    hipLaunchKernelGGL(
      d_search<true>,
      blocks_per_grid,
      threads_per_block,
      0,
      *static_cast<hipStream_t *>(gpu_stream.data()),
      container->get_kmer_size(),
      container->get_c_map().data(),
      container->get_acgt_pointers().data(),
      container->get_layer_0_pointers().data(),
      container->get_layer_1_2_pointers().data(),
      container->get_presearch_left().data(),
      container->get_presearch_right().data(),
      d_kmer_positions.data(),
      d_bit_seqs.data(),
      container->get_key_kmer_marks().data(),
      d_kmer_positions.data()
    );
  } else {
    hipLaunchKernelGGL(
      d_search<false>,
      blocks_per_grid,
      threads_per_block,
      0,
      *static_cast<hipStream_t *>(gpu_stream.data()),
      container->get_kmer_size(),
      container->get_c_map().data(),
      container->get_acgt_pointers().data(),
      container->get_layer_0_pointers().data(),
      container->get_layer_1_2_pointers().data(),
      container->get_presearch_left().data(),
      container->get_presearch_right().data(),
      d_kmer_positions.data(),
      d_bit_seqs.data(),
      nullptr,
      d_kmer_positions.data()
    );
  }
  end_timer.record(&gpu_stream);
  GPU_CHECK(hipPeekAtLastError());
  GPU_CHECK(hipStreamSynchronize(*static_cast<hipStream_t *>(gpu_stream.data()))
  );
  float millis = start_timer.time_elapsed_ms(end_timer);
  Logger::log(
    Logger::LOG_LEVEL::DEBUG,
    format("Batch {} took {} ms to search in the GPU", batch_id, millis)
  );
}

}  // namespace sbwt_search
