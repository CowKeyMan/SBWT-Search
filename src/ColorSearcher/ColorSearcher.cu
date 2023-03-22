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
using math_utils::round_up;

auto ColorSearcher::launch_search_kernel(u64 num_queries, u64 batch_id)
  -> void {
  u32 blocks_per_grid
    = round_up<u64>(num_queries, threads_per_block) / threads_per_block;
  start_timer.record(&gpu_stream);
  hipLaunchKernelGGL(
    d_color_search,
    blocks_per_grid,
    threads_per_block,
    0,
    *static_cast<hipStream_t *>(gpu_stream.get()),
    d_sbwt_index_idxs.get(),
    container->core_kmer_marks.get(),
    container->core_kmer_marks_poppy_layer_0.get(),
    container->core_kmer_marks_poppy_layer_1_2.get(),
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
    d_results.get()
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
