#include <memory>
#include <utility>

#include <ext/alloc_traits.h>

#include "SbwtContainer/SbwtContainer.h"

using std::move;
using std::unique_ptr;

namespace sbwt_search {

auto SbwtContainer::get_bit_vector_size() const -> u64 {
  return bit_vector_size;
}
auto SbwtContainer::get_bits_total() const -> u64 { return num_bits; }

CpuSbwtContainer::CpuSbwtContainer(
  u64 num_bits,
  unique_ptr<vector<u64>> &a,
  unique_ptr<vector<u64>> &c,
  unique_ptr<vector<u64>> &g,
  unique_ptr<vector<u64>> &t
):
    SbwtContainer(num_bits, a->size()) {
  acgt.push_back(move(a));
  acgt.push_back(move(c));
  acgt.push_back(move(g));
  acgt.push_back(move(t));
  layer_0.resize(4);
  layer_1_2.resize(4);
  c_map.resize(5);
  c_map[0] = 1;
}

auto CpuSbwtContainer::get_layer_0() const -> const vector<vector<u64>> {
  return layer_0;
}
auto CpuSbwtContainer::get_layer_0(ACGT letter) const -> const vector<u64> {
  return layer_0[static_cast<int>(letter)];
}

auto CpuSbwtContainer::get_layer_1_2() const -> const vector<vector<u64>> {
  return layer_1_2;
}
auto CpuSbwtContainer::get_layer_1_2(ACGT letter) const -> const vector<u64> {
  return layer_1_2[static_cast<int>(letter)];
}

auto CpuSbwtContainer::set_index(
  vector<u64> &&new_c_map,
  vector<vector<u64>> &&new_layer_0,
  vector<vector<u64>> &&new_layer_1_2
) -> void {
  c_map = new_c_map;
  layer_0 = new_layer_0;
  layer_1_2 = new_layer_1_2;
}
auto CpuSbwtContainer::get_c_map() const -> const vector<u64> { return c_map; }

auto CpuSbwtContainer::get_acgt(ACGT letter) const -> const u64 * {
  return acgt[static_cast<int>(letter)]->data();
}

}  // namespace sbwt_search
