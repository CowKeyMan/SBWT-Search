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

#include "Tools/KernelUtils.cuh"
#include "Tools/TypeDefinitions.h"
#include "UtilityKernels/GetBoolFromBitVector.cuh"
#include "UtilityKernels/Rank.cuh"
#include "UtilityKernels/VariableLengthIntegerIndex.cuh"

namespace sbwt_search {

using gpu_utils::get_idx;
using std::numeric_limits;

const int full_mask = 0xFFFFFFFF;

__device__ void dense(
  u64 color_set_idx,
  u64 *is_dense_marks,
  u64 *is_dense_marks_poppy_layer_0,
  u64 *is_dense_marks_poppy_layer_1_2,
  u64 *dense_arrays_intervals,
  u64 dense_arrays_intervals_width,
  u64 dense_arrays_intervals_width_set_bits,
  u64 *dense_arrays,
  u64 num_colors
);

__device__ void sparse(
  u64 color_set_idx,
  u64 *is_dense_marks,
  u64 *is_dense_marks_poppy_layer_0,
  u64 *is_dense_marks_poppy_layer_1_2,
  u64 *sparse_arrays_intervals,
  u64 sparse_arrays_intervals_width,
  u64 sparse_arrays_intervals_width_set_bits,
  u64 *sparse_arrays,
  u64 sparse_arrays_width,
  u64 sparse_arrays_width_set_bits,
  u64 num_colors
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
  u64 tidx = get_idx();
  u64 sbwt_index_idx = sbwt_index_idxs[tidx];
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
  if (is_dense) {
    dense(
      color_set_idx,
      is_dense_marks,
      is_dense_marks_poppy_layer_0,
      is_dense_marks_poppy_layer_1_2,
      dense_arrays_intervals,
      dense_arrays_intervals_width,
      dense_arrays_intervals_width_set_bits,
      dense_arrays,
      num_colors
    );
  } else {
    sparse(
      color_set_idx,
      is_dense_marks,
      is_dense_marks_poppy_layer_0,
      is_dense_marks_poppy_layer_1_2,
      sparse_arrays_intervals,
      sparse_arrays_intervals_width,
      sparse_arrays_intervals_width_set_bits,
      sparse_arrays,
      sparse_arrays_width,
      sparse_arrays_width_set_bits,
      num_colors
    );
    /* } */
    // TODO: start filling array, do shuffles. I need to set u64
    // TODO: case for padding
  }
}

__device__ void dense(
  u64 color_set_idx,
  u64 *is_dense_marks,
  u64 *is_dense_marks_poppy_layer_0,
  u64 *is_dense_marks_poppy_layer_1_2,
  u64 *dense_arrays_intervals,
  u64 dense_arrays_intervals_width,
  u64 dense_arrays_intervals_width_set_bits,
  u64 *dense_arrays,
  u64 num_colors
) {
  u64 starts_index = d_rank(
    is_dense_marks,
    is_dense_marks_poppy_layer_0,
    is_dense_marks_poppy_layer_1_2,
    color_set_idx
  );
  u64 arrays_start = d_variable_length_int_index(
    dense_arrays_intervals,
    dense_arrays_intervals_width,
    dense_arrays_intervals_width_set_bits,
    starts_index
  );
  u64 arrays_end = d_variable_length_int_index(
    dense_arrays_intervals,
    dense_arrays_intervals_width,
    dense_arrays_intervals_width_set_bits,
    starts_index + 1
  );
  u64 array_idx = arrays_start;
  for (u64 color_idx = 0; color_idx < num_colors; ++color_idx) {
    bool color_present = false;
    if (array_idx < arrays_end) {
      color_present = d_get_bool_from_bit_vector(dense_arrays, array_idx);
      ++array_idx;
    }
    u64 ballot_result = __ballot_sync(full_mask, color_present);
    /* if(thread_idx % 32 == 0) { */
    /*   results[num_colors * thread_idx / 32 + color_idx] = ballot_result; */
    /* } */
  }
}

__device__ void sparse(
  u64 color_set_idx,
  u64 *is_dense_marks,
  u64 *is_dense_marks_poppy_layer_0,
  u64 *is_dense_marks_poppy_layer_1_2,
  u64 *sparse_arrays_intervals,
  u64 sparse_arrays_intervals_width,
  u64 sparse_arrays_intervals_width_set_bits,
  u64 *sparse_arrays,
  u64 sparse_arrays_width,
  u64 sparse_arrays_width_set_bits,
  u64 num_colors
) {
  u64 starts_index = color_set_idx
    - d_rank(is_dense_marks,
             is_dense_marks_poppy_layer_0,
             is_dense_marks_poppy_layer_1_2,
             color_set_idx);
  u64 arrays_start = d_variable_length_int_index(
    sparse_arrays_intervals,
    sparse_arrays_intervals_width,
    sparse_arrays_intervals_width_set_bits,
    starts_index
  );
  u64 arrays_end = d_variable_length_int_index(
    sparse_arrays_intervals,
    sparse_arrays_intervals_width,
    sparse_arrays_intervals_width_set_bits,
    starts_index + 1
  );
  u64 array_idx = arrays_start;
  for (u64 color_idx = 0; color_idx < num_colors; ++color_idx) {
    bool color_present = false;
    if (array_idx < arrays_end) {
      u64 next_color = d_variable_length_int_index(
        sparse_arrays,
        sparse_arrays_width,
        sparse_arrays_width_set_bits,
        array_idx
      );
      if (color_idx == next_color) {
        ++array_idx;
        color_present = true;
      }
    }
    u64 ballot_result = __ballot_sync(full_mask, color_present);
    /* if(thread_idx % 32 == 0) { */
    /*   results[num_colors * thread_idx / 32 + color_idx] = ballot_result; */
    /* } */
  }
}

}  // namespace sbwt_search

#endif
