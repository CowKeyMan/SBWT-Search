#include "ColorIndexContainer/GpuColorIndexContainer.h"
#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

GpuColorIndexContainer::GpuColorIndexContainer(
  const sdsl::bit_vector &cpu_dense_arrays,
  const sdsl::int_vector<> &cpu_dense_arrays_intervals,
  const sdsl::int_vector<> &cpu_sparse_arrays,
  const sdsl::int_vector<> &cpu_sparse_arrays_intervals,
  const sdsl::bit_vector &cpu_is_dense_marks,
  const Poppy &cpu_is_dense_marks_poppy,
  const sdsl::bit_vector &cpu_core_kmer_marks,
  const Poppy &cpu_core_kmer_marks_poppy,
  const sdsl::int_vector<> &cpu_color_set_idxs,
  u64 largest_color_id_
):
    dense_arrays(
      cpu_dense_arrays.data(), cpu_dense_arrays.capacity() / u64_bits
    ),
    dense_arrays_intervals(
      cpu_dense_arrays_intervals.data(),
      cpu_dense_arrays_intervals.capacity() / u64_bits
    ),
    dense_arrays_intervals_width(cpu_dense_arrays.width()),
    sparse_arrays(
      cpu_sparse_arrays.data(), cpu_sparse_arrays.capacity() / u64_bits
    ),
    sparse_arrays_width(cpu_sparse_arrays.width()),
    sparse_arrays_intervals(
      cpu_sparse_arrays_intervals.data(),
      cpu_sparse_arrays_intervals.capacity() / u64_bits
    ),
    sparse_arrays_intervals_width(cpu_sparse_arrays_intervals.width()),
    is_dense_marks(
      cpu_is_dense_marks.data(), cpu_is_dense_marks.capacity() / u64_bits
    ),
    is_dense_marks_poppy_layer_0(cpu_is_dense_marks_poppy.layer_0),
    is_dense_marks_poppy_layer_1_2(cpu_is_dense_marks_poppy.layer_1_2),
    core_kmer_marks(
      cpu_core_kmer_marks.data(), cpu_core_kmer_marks.capacity() / u64_bits
    ),
    core_kmer_marks_poppy_layer_0(cpu_core_kmer_marks_poppy.layer_0),
    core_kmer_marks_poppy_layer_1_2(cpu_core_kmer_marks_poppy.layer_1_2),
    color_set_idxs(
      cpu_color_set_idxs.data(), cpu_color_set_idxs.capacity() / u64_bits
    ),
    color_idxs_width(cpu_color_set_idxs.width()),
    largest_color_id(largest_color_id_) {}

}  // namespace sbwt_search
