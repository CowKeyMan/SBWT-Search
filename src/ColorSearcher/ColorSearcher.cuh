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
const unsigned full_mask = 0xFFFFFFFF;

__device__ auto d_dense_get_arrays_start_end(
  const u64 color_set_idx,
  const u64 *is_dense_marks,
  const u64 *is_dense_marks_poppy_layer_0,
  const u64 *is_dense_marks_poppy_layer_1_2,
  const u64 *dense_arrays_intervals,
  const u64 dense_arrays_intervals_width,
  const u64 dense_arrays_intervals_width_set_bits,
  u64 &arrays_start,
  u64 &arrays_end
) -> void;
__device__ auto
d_dense_get_next_color_present(u64 &array_idx, const u64 *dense_arrays) -> bool;
__device__ auto d_dense_get_min(u64 array_idx, const u64 *dense_arrays) -> u64;

__device__ auto d_sparse_get_arrays_start_end(
  const u64 color_set_idx,
  const u64 *is_dense_marks,
  const u64 *is_dense_marks_poppy_layer_0,
  const u64 *is_dense_marks_poppy_layer_1_2,
  const u64 *sparse_arrays_intervals,
  const u64 sparse_arrays_intervals_width,
  const u64 sparse_arrays_intervals_width_set_bits,
  u64 &arrays_start,
  u64 &arrays_end
) -> void;
__device__ auto d_sparse_get_next_color_present(
  const u64 color_idx,
  u64 &array_idx,
  const u64 *sparse_arrays,
  const u64 sparse_arrays_width,
  const u64 sparse_arrays_width_set_bits
) -> bool;
__device__ auto d_sparse_get_min(
  const u64 array_idx,
  const u64 *sparse_arrays,
  const u64 sparse_arrays_width,
  const u64 sparse_arrays_width_set_bits
) -> u64;

__global__ auto d_color_search(
  const u64 *sbwt_idxs,
  const u64 *key_kmer_marks,
  const u64 *key_kmer_marks_poppy_layer_0,
  const u64 *key_kmer_marks_poppy_layer_1_2,
  const u64 *color_set_idxs,
  const u32 color_set_idxs_width,
  const u64 color_set_idxs_width_set_bits,
  const u64 *is_dense_marks,
  const u64 *is_dense_marks_poppy_layer_0,
  const u64 *is_dense_marks_poppy_layer_1_2,
  const u64 *dense_arrays,
  const u64 *dense_arrays_intervals,
  const u32 dense_arrays_intervals_width,
  const u64 dense_arrays_intervals_width_set_bits,
  const u64 *sparse_arrays,
  const u32 sparse_arrays_width,
  const u64 sparse_arrays_width_set_bits,
  const u64 *sparse_arrays_intervals,
  const u32 sparse_arrays_intervals_width,
  const u64 sparse_arrays_intervals_width_set_bits,
  const u64 num_colors,
  u8 *results
) -> void {
  u64 thread_idx = get_idx();
  u64 sbwt_idx = sbwt_idxs[thread_idx];
  u64 array_idx = 0;
  bool is_dense = false;
  u64 arrays_start = 0;
  u64 arrays_end = 0;
  if (sbwt_idx == static_cast<u64>(-1)) { return; }
  u64 color_set_idxs_idx = d_rank(
    key_kmer_marks,
    key_kmer_marks_poppy_layer_0,
    key_kmer_marks_poppy_layer_1_2,
    sbwt_idx
  );
  u64 color_set_idx = d_variable_length_int_index(
    color_set_idxs,
    color_set_idxs_width,
    color_set_idxs_width_set_bits,
    color_set_idxs_idx
  );
  is_dense = d_get_bool_from_bit_vector(is_dense_marks, color_set_idx);
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

  // get min_color in this warp
  u64 min_color = is_dense ? d_dense_get_min(arrays_start, dense_arrays) :
                             d_sparse_get_min(
                               arrays_start,
                               sparse_arrays,
                               sparse_arrays_width,
                               sparse_arrays_width_set_bits
                             );
  for (int offset = gpu_warp_size / 2; offset > 0; offset /= 2) {
#if (defined(__HIP_CPU_RT__) || defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__))
    u64 shfld = __shfl_xor(min_color, offset);
    min_color = min_color < shfld ? min_color : shfld;
#elif (defined(__HIP_PLATFORM_NVCC__) || defined(__HIP_PLATFORM_NVIDIA__))
    min_color = llmin(min_color, __shfl_xor_sync(full_mask, min_color, offset));
#endif
  }
  array_idx = arrays_start + (is_dense ? min_color : 0);

  // get max_color in this warp (max_color is not included)
  u64 max_color = is_dense ?
    arrays_end - arrays_start :
    d_variable_length_int_index(
      sparse_arrays,
      sparse_arrays_width,
      sparse_arrays_width_set_bits,
      arrays_end - 1
    ) + 1;
  for (int offset = gpu_warp_size / 2; offset > 0; offset /= 2) {
#if (defined(__HIP_CPU_RT__) || defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__))
    u64 shfld = __shfl_xor(max_color, offset);
    max_color = max_color > shfld ? max_color : shfld;
#elif (defined(__HIP_PLATFORM_NVCC__) || defined(__HIP_PLATFORM_NVIDIA__))
    max_color = llmax(max_color, __shfl_xor_sync(full_mask, max_color, offset));
#endif
  }

  // fill colors
  for (u64 color_idx = min_color; color_idx < max_color; ++color_idx) {
    bool color_present = false;
    if (array_idx < arrays_end) {
      if (is_dense) {
        color_present = d_dense_get_next_color_present(array_idx, dense_arrays);
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
        = static_cast<u8>(__popcll(ballot_result));
    }
#elif (defined(__HIP_PLATFORM_NVCC__) || defined(__HIP_PLATFORM_NVIDIA__))
    int ballot_result = __ballot_sync(full_mask, color_present);
    if (thread_idx % gpu_warp_size == 0) {
      results[num_colors * thread_idx / gpu_warp_size + color_idx]
        = static_cast<u8>(__popc(ballot_result));
    }
#else
#error("No runtime defined");
#endif
  }
}

__device__ auto d_dense_get_arrays_start_end(
  const u64 color_set_idx,
  const u64 *is_dense_marks,
  const u64 *is_dense_marks_poppy_layer_0,
  const u64 *is_dense_marks_poppy_layer_1_2,
  const u64 *dense_arrays_intervals,
  const u64 dense_arrays_intervals_width,
  const u64 dense_arrays_intervals_width_set_bits,
  u64 &arrays_start,
  u64 &arrays_end
) -> void {
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

__device__ auto
d_dense_get_next_color_present(u64 &array_idx, const u64 *dense_arrays)
  -> bool {
  return d_get_bool_from_bit_vector(dense_arrays, array_idx++);
}

__device__ auto d_dense_get_min(const u64 array_idx, const u64 *dense_arrays)
  -> u64 {
  u64 result = array_idx;
  while (true) {
    if (result % u64_bits == 0) {
      u64 zeros = __clzll(__brevll(dense_arrays[result / u64_bits]));
      result += zeros;
      if (zeros < u64_bits) { break; }
    } else {
      bool is_1 = d_get_bool_from_bit_vector(dense_arrays, result);
      if (is_1) { break; }
      ++result;
    }
  }
  return result - array_idx;
}

__device__ auto d_sparse_get_arrays_start_end(
  const u64 color_set_idx,
  const u64 *is_dense_marks,
  const u64 *is_dense_marks_poppy_layer_0,
  const u64 *is_dense_marks_poppy_layer_1_2,
  const u64 *sparse_arrays_intervals,
  const u64 sparse_arrays_intervals_width,
  const u64 sparse_arrays_intervals_width_set_bits,
  u64 &arrays_start,
  u64 &arrays_end
) -> void {
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

__device__ auto d_sparse_get_next_color_present(
  const u64 color_idx,
  u64 &array_idx,
  const u64 *sparse_arrays,
  const u64 sparse_arrays_width,
  const u64 sparse_arrays_width_set_bits
) -> bool {
  u64 next_color = d_variable_length_int_index(
    sparse_arrays, sparse_arrays_width, sparse_arrays_width_set_bits, array_idx
  );
  if (color_idx == next_color) {
    ++array_idx;
    return true;
  }
  return false;
}

__device__ auto d_sparse_get_min(
  const u64 array_idx,
  const u64 *sparse_arrays,
  const u64 sparse_arrays_width,
  const u64 sparse_arrays_width_set_bits
) -> u64 {
  return d_variable_length_int_index(
    sparse_arrays, sparse_arrays_width, sparse_arrays_width_set_bits, array_idx
  );
}

}  // namespace sbwt_search

#endif
