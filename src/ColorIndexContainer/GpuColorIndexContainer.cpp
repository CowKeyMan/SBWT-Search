#include "ColorIndexContainer/GpuColorIndexContainer.h"
#include "Tools/TypeDefinitions.h"

/* You will notice that for the variable length int vectors we add 1 more u64.
 * This is because when we are accessing this value we would want to access 2
 * u64 elements at a time (the current one and the next start), hence this helps
 * us to not have to insert any special code, and the cost is to instead use 64
 * * 4 more bits in memory (which is to say, there is negligible cost).
 * */

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
  u64 num_color_sets_,
  u64 num_colors_
):
    dense_arrays(
      cpu_dense_arrays.data(), cpu_dense_arrays.capacity() / u64_bits
    ),
    dense_arrays_intervals(
      // we add + 1 element to make accessing easier in the kernel
      cpu_dense_arrays_intervals.capacity() / u64_bits + 1
    ),
    dense_arrays_intervals_width(cpu_dense_arrays_intervals.width()),
    sparse_arrays(
      // we add + 1 element to make accessing easier in the kernel
      cpu_sparse_arrays.capacity() / u64_bits + 1
    ),
    sparse_arrays_width(cpu_sparse_arrays.width()),
    sparse_arrays_intervals(
      // we add + 1 element to make accessing easier in the kernel
      cpu_sparse_arrays_intervals.capacity() / u64_bits + 1
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
      // we add + 1 element to make accessing easier in the kernel
      cpu_color_set_idxs.capacity() / u64_bits + 1
    ),
    color_set_idxs_width(cpu_color_set_idxs.width()),
    num_color_sets(num_color_sets_),
    num_colors(num_colors_) {
  dense_arrays_intervals.memset(
    cpu_dense_arrays_intervals.capacity() / u64_bits, 1, 0
  );
  dense_arrays_intervals.set(
    cpu_dense_arrays_intervals.data(),
    cpu_dense_arrays_intervals.capacity() / u64_bits
  );

  sparse_arrays.memset(cpu_sparse_arrays.capacity() / u64_bits, 1, 0);
  sparse_arrays.set(
    cpu_sparse_arrays.data(), cpu_sparse_arrays.capacity() / u64_bits
  );

  sparse_arrays_intervals.memset(
    cpu_sparse_arrays_intervals.capacity() / u64_bits, 1, 0
  );
  sparse_arrays_intervals.set(
    cpu_sparse_arrays_intervals.data(),
    cpu_sparse_arrays_intervals.capacity() / u64_bits
  );

  color_set_idxs.memset(cpu_color_set_idxs.capacity() / u64_bits, 1, 0);
  color_set_idxs.set(
    cpu_color_set_idxs.data(), cpu_color_set_idxs.capacity() / u64_bits
  );
}

}  // namespace sbwt_search
