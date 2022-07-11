#define popcount __builtin_popcountll

#include <algorithm>
#include <bit>
#include <cstddef>
#include <numeric>
#include <vector>

#include "RankIndexBuilder/RankIndexBuilder.h"
#include "Utils/MathUtils.hpp"
#include "Utils/TypeDefinitionUtils.h"

using std::accumulate;
using std::fill;
using std::vector;

namespace sbwt_search {

class CPUIndexBuilder {
  private:
    const size_t bits_total;
    const u64 *bits_vector;
    const u64 basicblock_bits, superblock_bits, hyperblock_bits;
    vector<u64> layer_0, layer_1_2;
    vector<u64> layer_2_temps = vector<u64>(3, 0);
    ;
    size_t layer_2_temps_index = 0;
    u64 layer_0_count = 0, layer_1_count = 0, layer_2_count = 0;

  public:
    CPUIndexBuilder(
      const size_t bits_total,
      const u64 *bits_vector,
      const u64 basicblock_bits,
      const u64 superblock_bits,
      const u64 hyperblock_bits
    ):
        bits_total(bits_total),
        bits_vector(bits_vector),
        basicblock_bits(basicblock_bits),
        superblock_bits(superblock_bits),
        hyperblock_bits(hyperblock_bits) {
      layer_0.reserve(round_up(bits_total, hyperblock_bits) / hyperblock_bits);
      layer_1_2.reserve(
        round_up(bits_total, superblock_bits) / superblock_bits
      );
    }

    void build_index() {
      for (size_t i = 0, bits = 0; bits < round_up(bits_total, superblock_bits);
           bits += 64, ++i) {
        if (bits % superblock_bits == 0) { do_divisble_by_superblock(bits); }
        auto set_bits = popcount(bits_vector[i]);
        if (bits % basicblock_bits == 0 and bits % superblock_bits != 0) {
          do_divisible_by_basicblock();
        }
        layer_0_count += set_bits;
        layer_1_count += set_bits;
        layer_2_count += set_bits;
      }
    }

    void do_divisble_by_superblock(const u64 bits) {
      if (bits % hyperblock_bits == 0) { do_divisble_by_hyperlock(); }
      layer_2_temps_index = 0;
      layer_2_count = 0;
      fill(layer_2_temps.begin(), layer_2_temps.end(), 0);
    }

    void do_divisble_by_hyperlock() {
      layer_0.push_back(layer_0_count);
      layer_1_count = 0;
    }

    void do_divisible_by_basicblock() {
      layer_2_temps[layer_2_temps_index++] = layer_2_count;
      layer_2_count = 0;
      if (layer_2_temps_index == 3) { add_layer_1_2(); }
    }

    void add_layer_1_2() {
      layer_1_2.push_back(
        (layer_1_count
         - accumulate(layer_2_temps.begin(), layer_2_temps.end(), 0)
        ) << 32
        | layer_2_temps[0] << 20 | layer_2_temps[1] << 10
        | layer_2_temps[2] << 0
      );
    }

    const vector<u64> &&get_layer_0() { return move(layer_0); };

    const vector<u64> &&get_layer_1_2() { return move(layer_1_2); };
};

auto RankIndexBuilder::build_index() -> void {
  auto builder = CPUIndexBuilder(
    bits_total, bits_vector, basicblock_bits, superblock_bits, hyperblock_bits
  );
  builder.build_index();
  layer_0 = builder.get_layer_0();
  layer_1_2 = builder.get_layer_1_2();
}

auto RankIndexBuilder::get_layer_0() -> const vector<u64> & { return layer_0; }

auto RankIndexBuilder::get_layer_1_2() -> const vector<u64> & {
  return layer_1_2;
}

}
