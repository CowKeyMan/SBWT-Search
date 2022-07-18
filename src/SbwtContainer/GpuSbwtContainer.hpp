#ifndef GPU_SBWT_CONTAINER_HPP
#define GPU_SBWT_CONTAINER_HPP

/**
 * @file GpuSbwtContainer.hpp
 * @brief Contains data class for SBWT index which reside on GPU
 * */

#include <memory>

#include "SbwtContainer/SbwtContainer.hpp"

using std::make_unique;
using std::unique_ptr;

template <class T>
class CudaPointer;

namespace sbwt_search {

class GpuSbwtContainer: public SbwtContainer<GpuSbwtContainer> {
    friend SbwtContainer;
    vector<CudaPointer<u64>> acgt, layer_0, layer_1_2;
    unique_ptr<CudaPointer<u64>> c_map, presearch_left, presearch_right;
    unique_ptr<CudaPointer<u64 *>> acgt_pointers, layer_0_pointers,
      layer_1_2_pointers;

  public:
    GpuSbwtContainer(
      const u64 *cpu_a,
      const u64 *cpu_c,
      const u64 *cpu_g,
      const u64 *cpu_t,
      const u64 bits_total,
      const u64 bit_vector_size
    );

    void set_c_map(const vector<u64> &value);
    const CudaPointer<u64> &get_c_map() const;
    const CudaPointer<u64 *> &get_acgt_pointers() const;
    void set_layer_0(const vector<vector<u64>> &value);
    void set_layer_1_2(const vector<vector<u64>> &value);
    const CudaPointer<u64 *> &get_layer_0_pointers() const;
    const CudaPointer<u64 *> &get_layer_1_2_pointers() const;
    void set_presearch(CudaPointer<u64> &&left, CudaPointer<u64> &&right);
};

}

#endif
