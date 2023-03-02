#ifndef COLOR_SEARCHER_CUH
#define COLOR_SEARCHER_CUH

/**
 * @file ColorSearcher.cuh
 * @brief Search kernel for searching for the colors and merging them. The
 * '_set_bits' variables take a u64 with the least significant <width> bits set
 * to 1, while the rest are 0s. The <width> is the width of the respective
 * variable length vector that they correspond to.
 */

#include <limits>

#include "Global/GlobalDefinitions.h"
#include "Tools/KernelUtils.cuh"
#include "Tools/TypeDefinitions.h"
#include "UtilityKernels/GetBoolFromBitVector.cuh"
#include "UtilityKernels/Rank.cuh"
#include "UtilityKernels/VariableLengthIntegerIndex.cuh"
#include "hip/hip_runtime.h"

namespace sbwt_search {

using gpu_utils::get_idx;
using std::numeric_limits;

__device__ void d_dense_get_arrays_start_end(
  u64 color_set_idx,
  u64 *is_dense_marks,
  u64 *is_dense_marks_poppy_layer_0,
  u64 *is_dense_marks_poppy_layer_1_2,
  u64 *dense_arrays_intervals,
  u64 dense_arrays_intervals_width,
  u64 dense_arrays_intervals_width_set_bits,
  u64 &arrays_start,
  u64 &arrays_end
);
__device__ bool d_dense_get_next_color_present(
  const u64 color_idx, u64 &array_idx, u64 *dense_arrays
);

__device__ void d_sparse_get_arrays_start_end(
  u64 color_set_idx,
  u64 *is_dense_marks,
  u64 *is_dense_marks_poppy_layer_0,
  u64 *is_dense_marks_poppy_layer_1_2,
  u64 *sparse_arrays_intervals,
  u64 sparse_arrays_intervals_width,
  u64 sparse_arrays_intervals_width_set_bits,
  u64 &arrays_start,
  u64 &arrays_end
);
__device__ bool d_sparse_get_next_color_present(
  const u64 color_idx,
  u64 &array_idx,
  u64 *sparse_arrays,
  u64 sparse_arrays_width,
  u64 sparse_arrays_width_set_bits
);

__global__ void d_color_search(
  u64 *sbwt_index_idxs,
  u64 *core_kmer_marks,
  u64 *core_kmer_marks_poppy_layer_0,
  u64 *core_kmer_marks_poppy_layer_1_2,
  u64 *color_set_idxs,
  u32 color_set_idxs_width,
  u64 color_set_idxs_width_set_bits,
  u64 *is_dense_marks,
  u64 *is_dense_marks_poppy_layer_0,
  u64 *is_dense_marks_poppy_layer_1_2,
  u64 *dense_arrays,
  u64 *dense_arrays_intervals,
  u32 dense_arrays_intervals_width,
  u64 dense_arrays_intervals_width_set_bits,
  u64 *sparse_arrays,
  u32 sparse_arrays_width,
  u64 sparse_arrays_width_set_bits,
  u64 *sparse_arrays_intervals,
  u32 sparse_arrays_intervals_width,
  u64 sparse_arrays_intervals_width_set_bits,
  u64 num_colors,
  u64 *results
) {
  u64 thread_idx = get_idx();
  u64 sbwt_index_idx = sbwt_index_idxs[thread_idx];
  if (sbwt_index_idx == static_cast<u64>(-1)) { return; }
  u64 color_set_idxs_idx = d_rank(
    core_kmer_marks,
    core_kmer_marks_poppy_layer_0,
    core_kmer_marks_poppy_layer_1_2,
    sbwt_index_idx
  );
  u64 color_set_idx = d_variable_length_int_index(
    color_set_idxs,
    color_set_idxs_width,
    color_set_idxs_width_set_bits,
    color_set_idxs_idx
  );
  bool is_dense = d_get_bool_from_bit_vector(is_dense_marks, color_set_idx);
  u64 arrays_start = 0;
  u64 arrays_end = 0;
  if (is_dense) {
    d_dense_get_arrays_start_end(
      color_set_idx,
      is_dense_marks,
      is_dense_marks_poppy_layer_0,
      is_dense_marks_poppy_layer_1_2,
      dense_arrays_intervals,
      dense_arrays_intervals_width,
      dense_arrays_intervals_width_set_bits,
      arrays_start,
      arrays_end
    );
  } else {
    d_sparse_get_arrays_start_end(
      color_set_idx,
      is_dense_marks,
      is_dense_marks_poppy_layer_0,
      is_dense_marks_poppy_layer_1_2,
      sparse_arrays_intervals,
      sparse_arrays_intervals_width,
      sparse_arrays_intervals_width_set_bits,
      arrays_start,
      arrays_end
    );
  }
  u64 array_idx = arrays_start;
  // TODO: shfl to find minimum color_idx and maximum color_idx
  // TODO: get next color
  for (u64 color_idx = 0; color_idx < num_colors; ++color_idx) {
    // TODO: get minimum of each thread and only consider that color
    // Use find first set bit (__ffsll)
    bool color_present = false;
    if (array_idx < arrays_end) {
      if (is_dense) {
        color_present
          = d_dense_get_next_color_present(color_idx, array_idx, dense_arrays);
      } else {
        color_present = d_sparse_get_next_color_present(
          color_idx,
          array_idx,
          sparse_arrays,
          sparse_arrays_width,
          sparse_arrays_width_set_bits
        );
      }
    }
#if (defined(__HIP_CPU_RT__) || defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__))
    u64 ballot_result = __ballot(color_present);
    if (thread_idx % gpu_warp_size == 0) {
      results[num_colors * thread_idx / gpu_warp_size + color_idx]
        = __popcll(ballot_result);
    }
#elif (defined(__HIP_PLATFORM_NVCC__) || defined(__HIP_PLATFORM_NVIDIA__))
    const unsigned full_mask = 0xFFFFFFFF;
    int ballot_result = __ballot_sync(full_mask, color_present);
    if (thread_idx % gpu_warp_size == 0) {
      results[num_colors * thread_idx / gpu_warp_size + color_idx]
        = __popc(ballot_result);
    }
#else
#error("No runtime defined");
#endif
  }
}

__device__ void d_dense_get_arrays_start_end(
  u64 color_set_idx,
  u64 *is_dense_marks,
  u64 *is_dense_marks_poppy_layer_0,
  u64 *is_dense_marks_poppy_layer_1_2,
  u64 *dense_arrays_intervals,
  u64 dense_arrays_intervals_width,
  u64 dense_arrays_intervals_width_set_bits,
  u64 &arrays_start,
  u64 &arrays_end
) {
  u64 starts_index = d_rank(
    is_dense_marks,
    is_dense_marks_poppy_layer_0,
    is_dense_marks_poppy_layer_1_2,
    color_set_idx
  );
  arrays_start = d_variable_length_int_index(
    dense_arrays_intervals,
    dense_arrays_intervals_width,
    dense_arrays_intervals_width_set_bits,
    starts_index
  );
  arrays_end = d_variable_length_int_index(
    dense_arrays_intervals,
    dense_arrays_intervals_width,
    dense_arrays_intervals_width_set_bits,
    starts_index + 1
  );
}

__device__ bool d_dense_get_next_color_present(
  const u64 color_idx, u64 &array_idx, u64 *dense_arrays
) {
  ++array_idx;
  return d_get_bool_from_bit_vector(dense_arrays, array_idx);
}

__device__ void d_sparse_get_arrays_start_end(
  u64 color_set_idx,
  u64 *is_dense_marks,
  u64 *is_dense_marks_poppy_layer_0,
  u64 *is_dense_marks_poppy_layer_1_2,
  u64 *sparse_arrays_intervals,
  u64 sparse_arrays_intervals_width,
  u64 sparse_arrays_intervals_width_set_bits,
  u64 &arrays_start,
  u64 &arrays_end
) {
  u64 starts_index = color_set_idx
    - d_rank(is_dense_marks,
             is_dense_marks_poppy_layer_0,
             is_dense_marks_poppy_layer_1_2,
             color_set_idx);
  arrays_start = d_variable_length_int_index(
    sparse_arrays_intervals,
    sparse_arrays_intervals_width,
    sparse_arrays_intervals_width_set_bits,
    starts_index
  );
  arrays_end = d_variable_length_int_index(
    sparse_arrays_intervals,
    sparse_arrays_intervals_width,
    sparse_arrays_intervals_width_set_bits,
    starts_index + 1
  );
}

__device__ bool d_sparse_get_next_color_present(
  const u64 color_idx,
  u64 &array_idx,
  u64 *sparse_arrays,
  u64 sparse_arrays_width,
  u64 sparse_arrays_width_set_bits
) {
  u64 next_color = d_variable_length_int_index(
    sparse_arrays, sparse_arrays_width, sparse_arrays_width_set_bits, array_idx
  );
  if (color_idx == next_color) {
    ++array_idx;
    return true;
  }
  return false;
}

}  // namespace sbwt_search

#endif
