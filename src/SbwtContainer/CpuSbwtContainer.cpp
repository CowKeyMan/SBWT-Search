#include <memory>

#include "SbwtContainer/CpuSbwtContainer.h"
#include "SbwtContainer/GpuSbwtContainer.h"

namespace sbwt_search {

using std::make_shared;

CpuSbwtContainer::CpuSbwtContainer(
  u64 num_bits,
  unique_ptr<vector<u64>> &a,
  unique_ptr<vector<u64>> &c,
  unique_ptr<vector<u64>> &g,
  unique_ptr<vector<u64>> &t,
  uint kmer_size
):
  SbwtContainer(num_bits, a->size(), kmer_size) {
  acgt.push_back(std::move(a));
  acgt.push_back(std::move(c));
  acgt.push_back(std::move(g));
  acgt.push_back(std::move(t));
  layer_0.resize(4);
  layer_1_2.resize(4);
  c_map.resize(cmap_size);
  c_map[0] = 1;
}

auto CpuSbwtContainer::get_layer_0() const -> const vector<vector<u64>> & {
  return layer_0;
}
auto CpuSbwtContainer::get_layer_0(ACGT letter) const -> const vector<u64> & {
  return layer_0[static_cast<int>(letter)];
}

auto CpuSbwtContainer::get_layer_1_2() const -> const vector<vector<u64>> & {
  return layer_1_2;
}
auto CpuSbwtContainer::get_layer_1_2(ACGT letter) const -> const vector<u64> & {
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
auto CpuSbwtContainer::get_c_map() const -> const vector<u64> & {
  return c_map;
}

auto CpuSbwtContainer::get_acgt(ACGT letter) const -> const u64 * {
  return acgt[static_cast<int>(letter)]->data();
}

auto CpuSbwtContainer::to_gpu() -> shared_ptr<GpuSbwtContainer> {
  auto result = make_shared<GpuSbwtContainer>(
    get_acgt(ACGT::A),
    get_acgt(ACGT::C),
    get_acgt(ACGT::G),
    get_acgt(ACGT::T),
    get_num_bits(),
    get_bit_vector_size(),
    get_kmer_size()
  );
  if (!layer_0.empty() && !layer_0.empty()) {
    result->set_c_map(c_map);
    result->set_layer_0(layer_0);
    result->set_layer_1_2(layer_1_2);
  }
  return result;
}

}  // namespace sbwt_search
