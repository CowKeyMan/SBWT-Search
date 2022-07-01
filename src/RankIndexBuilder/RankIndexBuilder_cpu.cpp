#define popcount __builtin_popcountll

#include <algorithm>
#include <bit>
#include <cstddef>
#include <numeric>
#include <vector>

#include "MathUtils.hpp"
#include "RankIndexBuilder.h"
#include "TypeDefinitionUtils.h"

using std::accumulate;
using std::fill;
using std::vector;

namespace sbwt_search {

auto RankIndexBuilder::build_index() -> void {
  layer_0.reserve(round_up(bits_total, hyperblock_bits) / hyperblock_bits);
  layer_1_2.reserve(round_up(bits_total, superblock_bits) / superblock_bits);
  u64 layer_0_count = 0, layer_1_count = 0, layer_2_count = 0;
  auto layer_2_temps = vector<u64>(3, 0);
  size_t layer_2_temps_index = 0;
  for (u64 i = 0, bits = 0; bits < round_up(bits_total, superblock_bits);
       bits += 64, ++i) {
    if (bits % superblock_bits == 0) {
      if (bits % hyperblock_bits == 0) {
        layer_0.push_back(layer_0_count);
        layer_1_count = 0;
      }
      layer_2_temps_index = 0;
      layer_2_count = 0;
      fill(layer_2_temps.begin(), layer_2_temps.end(), 0);
    }
    auto set_bits = popcount(bits_vector[i]);
    if (bits % basicblock_bits == 0 and bits % superblock_bits != 0) {
      layer_2_temps[layer_2_temps_index++] = layer_2_count;
      layer_2_count = 0;
      if (layer_2_temps_index == 3) {
        layer_1_2.push_back(
          (layer_1_count
           - accumulate(layer_2_temps.begin(), layer_2_temps.end(), 0)
          ) << 32
          | layer_2_temps[0] << 20 | layer_2_temps[1] << 10
          | layer_2_temps[2] << 0
        );
      }
    }
    layer_0_count += set_bits;
    layer_1_count += set_bits;
    layer_2_count += set_bits;
  }
}

auto RankIndexBuilder::get_layer_0() -> const vector<u64> { return layer_0; }

auto RankIndexBuilder::get_layer_1_2() -> const vector<u64> {
  return layer_1_2;
}

}
