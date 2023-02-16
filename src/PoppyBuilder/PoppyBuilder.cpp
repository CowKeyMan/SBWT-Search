#include <algorithm>
#include <cstddef>
#include <numeric>
#include <utility>
#include <vector>

#include "Global/GlobalDefinitions.h"
#include "PoppyBuilder/PoppyBuilder.h"
#include "Tools/MathUtils.hpp"

namespace sbwt_search {

using math_utils::round_up;
using std::accumulate;
using std::fill;

PoppyBuilder::PoppyBuilder(const span<u64> bits_vector_, u64 num_bits_):
    bits_vector(bits_vector_), num_bits(num_bits_) {}

auto PoppyBuilder::get_poppy() -> Poppy {
  Poppy poppy;
  poppy.layer_0.reserve(round_up(num_bits, hyperblock_bits) / hyperblock_bits);
  poppy.layer_1_2.reserve(
    round_up(num_bits, superblock_bits) / superblock_bits
  );
  for (u64 i = 0, bits = 0; bits < round_up<u64>(num_bits, superblock_bits);
       bits += u64_bits, ++i) {
    if (bits % superblock_bits == 0) { do_divisble_by_superblock(bits, poppy); }
    auto condition = bits % basicblock_bits == 0 && bits % superblock_bits != 0;
    if (condition) { do_divisible_by_basicblock(poppy); }
    if (bits < round_up<u64>(num_bits, u64_bits)) {
      auto set_bits = __builtin_popcountll(bits_vector[i]);
      layer_0_count += set_bits;
      layer_1_count += set_bits;
      layer_2_count += set_bits;
    }
  }
  poppy.total_1s = layer_0_count;
  return poppy;
}

auto PoppyBuilder::do_divisble_by_superblock(const u64 bits, Poppy &poppy)
  -> void {
  if (bits % hyperblock_bits == 0) { do_divisble_by_hyperlock(poppy); }
  layer_2_temps_index = 0;
  layer_2_count = 0;
  fill(layer_2_temps.begin(), layer_2_temps.end(), 0);
}

auto PoppyBuilder::do_divisble_by_hyperlock(Poppy &poppy) -> void {
  poppy.layer_0.push_back(layer_0_count);
  layer_1_count = 0;
}

auto PoppyBuilder::do_divisible_by_basicblock(Poppy &poppy) -> void {
  layer_2_temps[layer_2_temps_index++] = layer_2_count;
  layer_2_count = 0;
  if (layer_2_temps_index == 3) { add_layer_1_2(poppy); }
}

auto PoppyBuilder::add_layer_1_2(Poppy &poppy) -> void {
  poppy.layer_1_2.push_back(
    (layer_1_count
     - accumulate(layer_2_temps.begin(), layer_2_temps.end(), 0ULL)
    ) << layer_1_bits
    | layer_2_temps[0] << (layer_2_bits * 2)
    | layer_2_temps[1] << (layer_2_bits * 1)
    | layer_2_temps[2] << (layer_2_bits * 0)
  );
}
}  // namespace sbwt_search
