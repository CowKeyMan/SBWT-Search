#ifndef RANK_INDEX_BUILDER_HPP
#define RANK_INDEX_BUILDER_HPP

/**
 * @file RankIndexBuilder.hpp
 * @brief Module responsible for building the rank index of the SBWT file
 *        Assumptions made:
 *          * basicblock_bits is a multiple of 64
 *          * superblock_bits is a multiple of basicblock_bits
 *          * hyperblock_bits is a multiple of hyperblock_bits
 * */

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <utility>
#include <vector>

#include "Utils/MathUtils.hpp"
#include "Utils/TypeDefinitions.h"

using math_utils::divisible_by_power_of_two;
using math_utils::round_up;
using std::accumulate;
using std::fill;
using std::vector;

namespace sbwt_search {

template <u64 superblock_bits, u64 hyperblock_bits>
class PoppyBuilder {
  private:
    const size_t bits_total;
    const u64 *bits_vector;
    vector<u64> layer_0, layer_1_2;
    vector<u64> layer_2_temps = vector<u64>(3, 0);
    size_t layer_2_temps_index = 0;
    u64 layer_0_count = 0, layer_1_count = 0, layer_2_count = 0;
    const u64 basicblock_bits;

  public:
    PoppyBuilder(const size_t bits_total, const u64 *bits_vector):
        bits_total(bits_total),
        bits_vector(bits_vector),
        basicblock_bits(superblock_bits / 4) {
      layer_0.reserve(round_up(bits_total, hyperblock_bits) / hyperblock_bits);
      layer_1_2.reserve(
        round_up(bits_total, superblock_bits) / superblock_bits
      );
    }

    vector<u64> &&get_layer_0() { return move(layer_0); };
    vector<u64> &&get_layer_1_2() { return move(layer_1_2); };
    u64 get_total_count() { return layer_0_count; };

    auto build() -> void {
      for (size_t i = 0, bits = 0;
           bits < round_up<u64>(bits_total, superblock_bits);
           bits += 64, ++i) {
        if (divisible_by_power_of_two(bits, superblock_bits)) {
          do_divisble_by_superblock(bits);
        }
        auto condition = divisible_by_power_of_two(bits, basicblock_bits)
                      && !divisible_by_power_of_two(bits, superblock_bits);
        if (condition) { do_divisible_by_basicblock(); }
        if (bits < round_up<u64>(bits_total, 64)) {
          auto set_bits = __builtin_popcountll(bits_vector[i]);
          layer_0_count += set_bits;
          layer_1_count += set_bits;
          layer_2_count += set_bits;
        }
      }
    }

  private:
    auto do_divisble_by_superblock(const u64 bits) -> void {
      if (divisible_by_power_of_two(bits, hyperblock_bits)) {
        do_divisble_by_hyperlock();
      }
      layer_2_temps_index = 0;
      layer_2_count = 0;
      fill(layer_2_temps.begin(), layer_2_temps.end(), 0);
    }

    auto do_divisble_by_hyperlock() -> void {
      layer_0.push_back(layer_0_count);
      layer_1_count = 0;
    }

    auto do_divisible_by_basicblock() -> void {
      layer_2_temps[layer_2_temps_index++] = layer_2_count;
      layer_2_count = 0;
      if (layer_2_temps_index == 3) { add_layer_1_2(); }
    }

    auto add_layer_1_2() -> void {
      layer_1_2.push_back(
        (layer_1_count
         - accumulate(layer_2_temps.begin(), layer_2_temps.end(), 0)
        ) << 32
        | layer_2_temps[0] << 20 | layer_2_temps[1] << 10
        | layer_2_temps[2] << 0
      );
    }
};

}  // namespace sbwt_search

#endif
