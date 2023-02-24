#include <algorithm>
#include <iostream>
#include <limits>
#include <span>
#include <vector>
using std::cerr;
using std::cout;
using std::endl;
using std::vector;
template <class T>
auto print_vec(
  const vector<T> &v, uint64_t limit = std::numeric_limits<uint64_t>::max()
) {
  cout << "---------------------" << endl;
  for (int i = 0; i < std::min(limit, v.size()); ++i) { cout << v[i] << " "; }
  cout << endl << "---------------------" << endl;
}

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
  hipEvent_t search_start;  // NOLINT(cppcoreguidelines-init-variables)
  hipEvent_t search_stop;   // NOLINT(cppcoreguidelines-init-variables)
  GPU_CHECK(hipEventCreate(&search_start));
  GPU_CHECK(hipEventCreate(&search_stop));
  u32 blocks_per_grid
    = round_up<u64>(num_queries, threads_per_block) / threads_per_block;
  cout << blocks_per_grid << endl;
  GPU_CHECK(hipEventRecord(search_start));
  hipLaunchKernelGGL(
    d_color_search,
    blocks_per_grid,
    threads_per_block,
    0,
    nullptr,
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
