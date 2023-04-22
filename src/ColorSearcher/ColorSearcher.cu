#include "ColorSearcher/ColorPostProcessor.cuh"
#include "ColorSearcher/ColorSearcher.cuh"
#include "ColorSearcher/ColorSearcher.h"
#include "Tools/BitDefinitions.h"
#include "Tools/GpuUtils.h"
#include "Tools/Logger.h"
#include "Tools/MathUtils.hpp"
#include "fmt/core.h"
#include "hip/hip_runtime.h"

namespace sbwt_search {

using bit_utils::set_bits;
using fmt::format;
using log_utils::Logger;
using math_utils::divide_and_ceil;

auto ColorSearcher::launch_search_kernel(u64 num_queries, u64 batch_id)
  -> void {
  Logger::log_timed_event(
    format("SearcherSearch_{}", stream_id),
    Logger::EVENT_STATE::START,
    format("batch {}", batch_id)
  );
  u64 blocks_per_grid = divide_and_ceil<u64>(num_queries, threads_per_block);
  start_timer.record(&gpu_stream);
  hipLaunchKernelGGL(
    d_color_search,
    blocks_per_grid,
    threads_per_block,
    0,
    *static_cast<hipStream_t *>(gpu_stream.get()),
    d_sbwt_index_idxs.get(),
    container->key_kmer_marks.get(),
    container->key_kmer_marks_poppy_layer_0.get(),
    container->key_kmer_marks_poppy_layer_1_2.get(),
    container->color_set_idxs.get(),
    container->color_set_idxs_width,
    set_bits.at(container->color_set_idxs_width),
    container->is_dense_marks.get(),
    container->is_dense_marks_poppy_layer_0.get(),
    container->is_dense_marks_poppy_layer_1_2.get(),
    container->dense_arrays.get(),
    container->dense_arrays_intervals.get(),
    container->dense_arrays_intervals_width,
    set_bits.at(container->dense_arrays_intervals_width),
    container->sparse_arrays.get(),
    container->sparse_arrays_width,
    set_bits.at(container->sparse_arrays_width),
    container->sparse_arrays_intervals.get(),
    container->sparse_arrays_intervals_width,
    set_bits.at(container->sparse_arrays_intervals_width),
    container->num_colors,
    d_fat_results.get()
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
  Logger::log_timed_event(
    format("SearcherSearch_{}", stream_id),
    Logger::EVENT_STATE::STOP,
    format("batch {}", batch_id)
  );
}

auto ColorSearcher::launch_combine_kernel(
  u64 num_warps, u64 num_reads, u64 num_colors, u64 batch_id
) -> void {
  Logger::log_timed_event(
    format("SearcherPostProcess_{}", stream_id),
    Logger::EVENT_STATE::START,
    format("batch {}", batch_id)
  );
  u64 blocks_per_grid
    = divide_and_ceil<u64>(num_reads * num_colors, threads_per_block);
  auto &d_warps_before_new_read = d_sbwt_index_idxs;
  hipLaunchKernelGGL(
    d_post_process,
    blocks_per_grid,
    threads_per_block,
    0,
    *static_cast<hipStream_t *>(gpu_stream.get()),
    d_fat_results.get(),
    d_warps_before_new_read.get(),
    num_warps,
    num_reads,
    num_colors,
    d_results.get()
  );
  Logger::log_timed_event(
    format("SearcherPostProcess_{}", stream_id),
    Logger::EVENT_STATE::STOP,
    format("batch {}", batch_id)
  );
}

}  // namespace sbwt_search
