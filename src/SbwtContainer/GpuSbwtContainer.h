#ifndef GPU_SBWT_CONTAINER_H
#define GPU_SBWT_CONTAINER_H

/**
 * @file GpuSbwtContainer.h
 * @brief Contains data class for SBWT index which reside on GPU
 */

#include <memory>
#include <vector>

#include "Poppy/Poppy.h"
#include "SbwtContainer/SbwtContainer.h"
#include "Tools/GpuPointer.h"
#include "sdsl/int_vector.hpp"

namespace sbwt_search {

using gpu_utils::GpuPointer;
using std::unique_ptr;
using std::vector;

class GpuSbwtContainer: public SbwtContainer {
private:
  vector<unique_ptr<GpuPointer<u64>>> acgt, layer_0, layer_1_2;
  unique_ptr<GpuPointer<u64>> c_map, presearch_left, presearch_right;
  unique_ptr<GpuPointer<u64 *>> acgt_pointers, layer_0_pointers,
    layer_1_2_pointers;
  unique_ptr<GpuPointer<u64>> key_kmer_marks;

public:
  GpuSbwtContainer(
    const vector<vector<u64>> &cpu_acgt,
    const vector<Poppy> &cpu_poppy,
    const vector<u64> &cpu_c_map,
    u64 bits_total,
    u64 bit_vector_size,
    u32 kmer_size,
    const sdsl::int_vector<> &cpu_key_kmer_marks
  );

  [[nodiscard]] auto get_c_map() const -> const GpuPointer<u64> &;
  [[nodiscard]] auto get_acgt_pointers() const -> const GpuPointer<u64 *> &;
  [[nodiscard]] auto get_layer_0_pointers() const -> const GpuPointer<u64 *> &;
  [[nodiscard]] auto get_layer_1_2_pointers() const
    -> const GpuPointer<u64 *> &;
  auto set_presearch(
    unique_ptr<GpuPointer<u64>> left, unique_ptr<GpuPointer<u64>> right
  ) -> void;
  [[nodiscard]] auto get_presearch_left() const -> GpuPointer<u64> &;
  [[nodiscard]] auto get_presearch_right() const -> GpuPointer<u64> &;
  [[nodiscard]] auto get_key_kmer_marks() const -> GpuPointer<u64> &;
};

}  // namespace sbwt_search

#endif
