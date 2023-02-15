#ifndef CPU_COLOR_INDEX_CONTAINER_H
#define CPU_COLOR_INDEX_CONTAINER_H

/**
 * @file CpuColorIndexContainer.h
 * @brief A container which holds items related to the color sets in the cpu
 */

#include <memory>

#include "ColorIndexContainer/GpuColorIndexContainer.h"
#include "Poppy/Poppy.h"
#include "Tools/TypeDefinitions.h"
#include "sdsl/int_vector.hpp"

namespace sbwt_search {

using std::unique_ptr;

class CpuColorIndexContainer {
public:
  sdsl::bit_vector dense_arrays;
  sdsl::int_vector<> dense_arrays_intervals;
  sdsl::int_vector<> sparse_arrays;
  sdsl::int_vector<> sparse_arrays_intervals;
  sdsl::bit_vector is_dense_marks;
  Poppy is_dense_marks_poppy;
  sdsl::bit_vector core_kmer_marks;
  Poppy core_kmer_marks_poppy;
  sdsl::int_vector<> color_set_idxs;
  u64 num_color_sets;

  [[nodiscard]] auto to_gpu() const -> GpuColorIndexContainer;
};

}  // namespace sbwt_search

#endif
