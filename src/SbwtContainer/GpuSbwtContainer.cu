#include "SbwtContainer/GpuSbwtContainer.hpp"
#include "Utils/CudaUtils.cuh"
#include "Utils/TypeDefinitionUtils.h"

namespace sbwt_search {

GpuSbwtContainer::GpuSbwtContainer(
  const u64 *cpu_a,
  const u64 *cpu_c,
  const u64 *cpu_g,
  const u64 *cpu_t,
  const u64 bits_total,
  const u64 bit_vector_size
):
    SbwtContainer(bits_total, bit_vector_size),
    acgt{
      { cpu_a, bit_vector_size },
      { cpu_c, bit_vector_size },
      { cpu_g, bit_vector_size },
      { cpu_t, bit_vector_size },
    } {
  acgt_pointers = make_unique<CudaPointer<u64 *>>(vector<u64 *>(
    { acgt[0].get(), acgt[1].get(), acgt[2].get(), acgt[3].get() }
  ));
  layer_0.reserve(4);
  layer_1_2.reserve(4);
}

auto GpuSbwtContainer::set_c_map(const vector<u64> &value) -> void {
  c_map = make_unique<CudaPointer<u64>>(value);
}

auto GpuSbwtContainer::get_c_map() const -> const CudaPointer<u64> & {
  return *c_map.get();
}

auto GpuSbwtContainer::get_acgt_pointers() const -> const CudaPointer<u64 *> & {
  return *acgt_pointers.get();
};

auto GpuSbwtContainer::set_layer_0(const vector<vector<u64>> &value) -> void {
  for (int i = 0; i < value.size(); ++i) {
    layer_0.push_back(CudaPointer<u64>(value[i]));
  }
  layer_0_pointers = make_unique<CudaPointer<u64 *>>(vector<u64 *>(
    { layer_0[0].get(), acgt[1].get(), acgt[2].get(), acgt[3].get() }
  ));
}

auto GpuSbwtContainer::set_layer_1_2(const vector<vector<u64>> &value) -> void {
  for (int i = 0; i < value.size(); ++i) {
    layer_1_2.push_back(CudaPointer<u64>(value[i]));
  }
  layer_1_2_pointers = make_unique<CudaPointer<u64 *>>(vector<u64 *>(
    { layer_1_2[0].get(), acgt[1].get(), acgt[2].get(), acgt[3].get() }
  ));
}

auto GpuSbwtContainer::get_layer_0_pointers() const
  -> const CudaPointer<u64 *> & {
  return *layer_0_pointers.get();
}

auto GpuSbwtContainer::get_layer_1_2_pointers() const
  -> const CudaPointer<u64 *> & {
  return *layer_1_2_pointers.get();
}

auto GpuSbwtContainer::set_presearch(
  CudaPointer<u64> &&left, CudaPointer<u64> &&right
) -> void {
  presearch_left = make_unique<CudaPointer<u64>>(left);
  presearch_right = make_unique<CudaPointer<u64>>(right);
}

}
