#include "ColorIndexContainer/CpuColorIndexContainer.h"

namespace sbwt_search {

auto CpuColorIndexContainer::to_gpu() const -> GpuColorIndexContainer {
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
    num_color_sets};
}

}  // namespace sbwt_search
