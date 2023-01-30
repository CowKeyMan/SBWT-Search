#ifndef GPU_SBWT_CONTAINER_H
#define GPU_SBWT_CONTAINER_H

/**
 * @file GpuSbwtContainer.h
 * @brief Contains data class for SBWT index which reside on GPU
 */

#include <memory>
#include <vector>

#include "SbwtContainer/SbwtContainer.h"
#include "Tools/GpuPointer.h"

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

public:
  GpuSbwtContainer(
    const u64 *cpu_a,
    const u64 *cpu_c,
    const u64 *cpu_g,
    const u64 *cpu_t,
    u64 bits_total,
    u64 bit_vector_size,
    u32 kmer_size
  );

  void set_c_map(const vector<u64> &value);
  auto get_c_map() const -> const GpuPointer<u64> &;
  auto get_acgt_pointers() const -> const GpuPointer<u64 *> &;
  auto set_layer_0(const vector<vector<u64>> &value) -> void;
  auto set_layer_1_2(const vector<vector<u64>> &value) -> void;
  auto get_layer_0_pointers() const -> const GpuPointer<u64 *> &;
  auto get_layer_1_2_pointers() const -> const GpuPointer<u64 *> &;
  auto set_presearch(
    unique_ptr<GpuPointer<u64>> left, unique_ptr<GpuPointer<u64>> right
  ) -> void;
  auto get_presearch_left() const -> GpuPointer<u64> &;
  auto get_presearch_right() const -> GpuPointer<u64> &;
};

}  // namespace sbwt_search

#endif
