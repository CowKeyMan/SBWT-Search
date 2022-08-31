#include <memory>

#include <sdsl/int_vector.hpp>

#include "SbwtContainer/SbwtContainer.h"

using sdsl::bit_vector;
using std::move;
using std::unique_ptr;

namespace sbwt_search {

auto SbwtContainer::get_bit_vector_size() const -> u64 {
  return bit_vector_size;
}
auto SbwtContainer::get_bits_total() const -> u64 { return bits_total; }

CpuSbwtContainer::CpuSbwtContainer(
  unique_ptr<bit_vector> &a,
  unique_ptr<bit_vector> &c,
  unique_ptr<bit_vector> &g,
  unique_ptr<bit_vector> &t
):
    SbwtContainer(a->size(), a->capacity() / 64) {
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
