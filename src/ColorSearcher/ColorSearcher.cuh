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

using gpu_utils::get_idx;

namespace sbwt_search {

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
  u64 dense_arrays_intervals_width,
  u64 dense_arrays_intervals_width_set_bits,
  u64 *sparse_arrays,
  u64 sparse_arrays_width,
  u64 sparse_arrays_width_set_bits,
  u64 *sparse_arrays_intervals,
  u64 sparse_arrays_intervals_width,
  u64 sparse_arrays_intervals_width_set_bits
) {
  // set_bits contains an array where index x contains the first x bits as 1s,
  // and the rest are 0s
  auto tidx = get_idx();
  auto sbwt_index_idx = sbwt_index_idxs[tidx];
  auto color_set_idxs_idx = d_rank(
    core_kmer_marks,
    core_kmer_marks_poppy_layer_0,
    core_kmer_marks_poppy_layer_1_2,
    sbwt_index_idx
  );
  auto color_set_idx = d_variable_length_int_index(
    color_set_idxs,
    color_set_idxs_width,
    color_set_idxs_width_set_bits,
    color_set_idxs_idx
  );
  auto is_dense = d_get_bool_from_bit_vector(is_dense_marks, color_set_idx);
  auto starts_index = d_rank(
    is_dense_marks,
    is_dense_marks_poppy_layer_0,
    is_dense_marks_poppy_layer_1_2,
    color_set_idx
  );
  if (!is_dense) {
    starts_index = color_set_idx - starts_index;  // the rank of the 0s
  }
  auto arrays_start = is_dense ? d_variable_length_int_index(
                        dense_arrays_intervals,
                        dense_arrays_intervals_width,
                        dense_arrays_intervals_width_set_bits,
                        starts_index
                      ) :
                                 d_variable_length_int_index(
                                   sparse_arrays_intervals,
                                   sparse_arrays_intervals_width,
                                   sparse_arrays_intervals_width_set_bits,
                                   starts_index
                                 );
  auto arrays_end = is_dense ? d_variable_length_int_index(
                      dense_arrays_intervals,
                      dense_arrays_intervals_width,
                      dense_arrays_intervals_width_set_bits,
                      starts_index + 1
                    ) :
                               d_variable_length_int_index(
                                 sparse_arrays_intervals,
                                 sparse_arrays_intervals_width,
                                 sparse_arrays_intervals_width_set_bits,
                                 starts_index + 1
                               );
  // TODO: start filling array, do shuffles. I need to set u64
  // TODO: case for padding
}

}  // namespace sbwt_search

#endif
