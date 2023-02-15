#ifndef GPU_COLOR_INDEX_CONTAINER_H
#define GPU_COLOR_INDEX_CONTAINER_H

/**
 * @file GpuColorIndexContainer.h
 * @brief A container which holds items related to the color sets in the gpu
 */

#include <memory>

#include "Poppy/Poppy.h"
#include "Tools/GpuPointer.h"
#include "Tools/TypeDefinitions.h"
#include "sdsl/int_vector.hpp"

namespace sbwt_search {

using gpu_utils::GpuPointer;
using std::unique_ptr;

class GpuColorIndexContainer {
public:
  GpuPointer<u64> dense_arrays;
  GpuPointer<u64> dense_arrays_intervals;
  u64 dense_arrays_intervals_width;
  GpuPointer<u64> sparse_arrays;
  u64 sparse_arrays_width;
  GpuPointer<u64> sparse_arrays_intervals;
  u64 sparse_arrays_intervals_width;
  GpuPointer<u64> is_dense_marks;
  GpuPointer<u64> is_dense_marks_poppy_layer_0;
  GpuPointer<u64> is_dense_marks_poppy_layer_1_2;
  GpuPointer<u64> core_kmer_marks;
  GpuPointer<u64> core_kmer_marks_poppy_layer_0;
  GpuPointer<u64> core_kmer_marks_poppy_layer_1_2;
  GpuPointer<u64> color_set_idxs;
  u64 color_idxs_width;
  u64 num_color_sets;

  GpuColorIndexContainer(
    const sdsl::bit_vector &cpu_dense_arrays,
    const sdsl::int_vector<> &cpu_dense_arrays_intervals,
    const sdsl::int_vector<> &cpu_sparse_arrays,
    const sdsl::int_vector<> &cpu_sparse_arrays_intervals,
    const sdsl::bit_vector &cpu_is_dense_marks,
    const Poppy &cpu_is_dense_marks_poppy,
    const sdsl::bit_vector &cpu_core_kmer_marks,
    const Poppy &cpu_core_kmer_marks_poppy,
    const sdsl::int_vector<> &cpu_color_set_idxs,
    u64 num_color_sets_
  );
};

}  // namespace sbwt_search

#endif
