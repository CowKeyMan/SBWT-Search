#include "ColorIndexContainer/CpuColorIndexContainer.h"

namespace sbwt_search {

CpuColorIndexContainer::CpuColorIndexContainer(
  sdsl::bit_vector dense_arrays_,
  sdsl::int_vector<> dense_arrays_intervals_,
  sdsl::int_vector<> sparse_arrays_,
  sdsl::int_vector<> sparse_arrays_intervals_,
  sdsl::bit_vector is_dense_marks_,
  Poppy is_dense_marks_poppy_,
  sdsl::bit_vector core_kmer_marks_,
  Poppy core_kmer_marks_poppy_,
  sdsl::int_vector<> color_set_idxs_,
  u64 largest_color_id_
):
    dense_arrays(std::move(dense_arrays_)),
    dense_arrays_intervals(std::move(dense_arrays_intervals_)),
    sparse_arrays(std::move(sparse_arrays_)),
    sparse_arrays_intervals(std::move(sparse_arrays_intervals_)),
    is_dense_marks(std::move(is_dense_marks_)),
    is_dense_marks_poppy(std::move(is_dense_marks_poppy_)),
    core_kmer_marks(std::move(core_kmer_marks_)),
    core_kmer_marks_poppy(std::move(core_kmer_marks_poppy_)),
    color_set_idxs(std::move(color_set_idxs_)),
    largest_color_id(largest_color_id_){};

auto CpuColorIndexContainer::to_gpu() -> GpuColorIndexContainer {
  return {
    dense_arrays,
    dense_arrays_intervals,
    sparse_arrays,
    sparse_arrays_intervals,
    is_dense_marks,
    is_dense_marks_poppy,
    core_kmer_marks,
    core_kmer_marks_poppy,
    color_set_idxs,
    largest_color_id};
}

}  // namespace sbwt_search
