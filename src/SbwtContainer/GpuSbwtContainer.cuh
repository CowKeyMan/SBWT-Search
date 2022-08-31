#ifndef GPU_SBWT_CONTAINER_CUH
#define GPU_SBWT_CONTAINER_CUH

/**
 * @file GpuSbwtContainer.cuh
 * @brief Contains data class for SBWT index which reside on GPU
 * */

#include <memory>

#include "SbwtContainer/SbwtContainer.h"
#include "Utils/CudaUtils.cuh"

using gpu_utils::GpuPointer;
using std::unique_ptr;

namespace sbwt_search {

class GpuSbwtContainer: public SbwtContainer {
  private:
    vector<unique_ptr<GpuPointer<u64>>> acgt, layer_0, layer_1_2;
    unique_ptr<GpuPointer<u64>> c_map, presearch_left, presearch_right;
    unique_ptr<GpuPointer<u64 *>> acgt_pointers, layer_0_pointers,
      layer_1_2_pointers;
    u32 kmer_size;

  public:
    GpuSbwtContainer(
      const u64 *cpu_a,
      const u64 *cpu_c,
      const u64 *cpu_g,
      const u64 *cpu_t,
      const u64 bits_total,
      const u64 bit_vector_size,
      const u32 kmer_size = 30
    );

    void set_c_map(const vector<u64> &value);
    const GpuPointer<u64> &get_c_map() const;
    const GpuPointer<u64 *> &get_acgt_pointers() const;
    void set_layer_0(const vector<vector<u64>> &value);
    void set_layer_1_2(const vector<vector<u64>> &value);
    const GpuPointer<u64 *> &get_layer_0_pointers() const;
    const GpuPointer<u64 *> &get_layer_1_2_pointers() const;
    void set_presearch(
      unique_ptr<GpuPointer<u64>> left, unique_ptr<GpuPointer<u64>> right
    );
    GpuPointer<u64> &get_presearch_left() const;
    GpuPointer<u64> &get_presearch_right() const;
    u32 get_kmer_size() const;
};

}  // namespace sbwt_search

#endif
