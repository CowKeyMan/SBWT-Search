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

using std::shared_ptr;
using std::unique_ptr;

class CpuColorIndexContainer {
public:
  sdsl::bit_vector dense_arrays;
  sdsl::int_vector<> dense_arrays_intervals;
  sdsl::int_vector<> sparse_arrays;
  sdsl::int_vector<> sparse_arrays_intervals;
  sdsl::bit_vector is_dense_marks;
  Poppy is_dense_marks_poppy;
  sdsl::bit_vector key_kmer_marks;
  Poppy key_kmer_marks_poppy;
  sdsl::int_vector<> color_set_idxs;
  u64 num_color_sets;
  u64 num_colors;

  [[nodiscard]] auto to_gpu() const -> shared_ptr<GpuColorIndexContainer>;
};

}  // namespace sbwt_search

#endif
