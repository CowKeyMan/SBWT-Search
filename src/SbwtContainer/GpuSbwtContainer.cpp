#include <memory>
#include <utility>
#include <vector>

#include "SbwtContainer/GpuSbwtContainer.h"
#include "Tools/GpuPointer.h"
#include "Tools/TypeDefinitions.h"

using gpu_utils::GpuPointer;
using std::make_unique;
using std::unique_ptr;
using std::vector;

namespace sbwt_search {

GpuSbwtContainer::GpuSbwtContainer(
  const u64 *cpu_a,
  const u64 *cpu_c,
  const u64 *cpu_g,
  const u64 *cpu_t,
  const u64 bits_total,
  const u64 bit_vector_size,
  const u32 kmer_size
):
    SbwtContainer(bits_total, bit_vector_size, kmer_size) {
  acgt.reserve(4);
  acgt.push_back(make_unique<GpuPointer<u64>>(cpu_a, bit_vector_size));
  acgt.push_back(make_unique<GpuPointer<u64>>(cpu_c, bit_vector_size));
  acgt.push_back(make_unique<GpuPointer<u64>>(cpu_g, bit_vector_size));
  acgt.push_back(make_unique<GpuPointer<u64>>(cpu_t, bit_vector_size));
  auto v = vector<u64 *>(
    {acgt[0]->get(), acgt[1]->get(), acgt[2]->get(), acgt[3]->get()}
  );
  acgt_pointers = make_unique<GpuPointer<u64 *>>(vector<u64 *>(
    {acgt[0]->get(), acgt[1]->get(), acgt[2]->get(), acgt[3]->get()}
  ));
  layer_0.reserve(4);
  layer_1_2.reserve(4);
}

auto GpuSbwtContainer::set_c_map(const vector<u64> &value) -> void {
  c_map = make_unique<GpuPointer<u64>>(value);
}

auto GpuSbwtContainer::get_c_map() const -> const GpuPointer<u64> & {
  return *c_map;
}

auto GpuSbwtContainer::get_acgt_pointers() const -> const GpuPointer<u64 *> & {
  return *acgt_pointers;
};

auto GpuSbwtContainer::set_layer_0(const vector<vector<u64>> &value) -> void {
#pragma unroll 4
  for (const auto &v : value) {
    layer_0.push_back(make_unique<GpuPointer<u64>>(v));
  }
  layer_0_pointers = make_unique<GpuPointer<u64 *>>(vector<u64 *>(
    {layer_0[0]->get(), layer_0[1]->get(), layer_0[2]->get(), layer_0[3]->get()}
  ));
}

auto GpuSbwtContainer::set_layer_1_2(const vector<vector<u64>> &value) -> void {
#pragma unroll 4
  for (const auto &v : value) {
    layer_1_2.push_back(make_unique<GpuPointer<u64>>(v));
  }
  layer_1_2_pointers = make_unique<GpuPointer<u64 *>>(vector<u64 *>(
    {layer_1_2[0]->get(),
     layer_1_2[1]->get(),
     layer_1_2[2]->get(),
     layer_1_2[3]->get()}
  ));
}

auto GpuSbwtContainer::get_layer_0_pointers() const
  -> const GpuPointer<u64 *> & {
  return *layer_0_pointers;
}

auto GpuSbwtContainer::get_layer_1_2_pointers() const
  -> const GpuPointer<u64 *> & {
  return *layer_1_2_pointers;
}

auto GpuSbwtContainer::set_presearch(
  unique_ptr<GpuPointer<u64>> left, unique_ptr<GpuPointer<u64>> right
) -> void {
  presearch_left = std::move(left);
  presearch_right = std::move(right);
}

auto GpuSbwtContainer::get_presearch_left() const -> GpuPointer<u64> & {
  return *presearch_left;
}

auto GpuSbwtContainer::get_presearch_right() const -> GpuPointer<u64> & {
  return *presearch_right;
}

}  // namespace sbwt_search
