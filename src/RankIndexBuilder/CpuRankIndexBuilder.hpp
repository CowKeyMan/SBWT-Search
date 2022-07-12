#ifndef RANK_INDEX_BUILDER_HPP
#define RANK_INDEX_BUILDER_HPP

/**
 * @file CpuRankIndexBuilder.hpp
 * @brief Module responsible for building the rank index of the SBWT file on the CPU
          Inherits from RankIndexBuilder
 * */


#include <vector>

#include "RankIndexBuilder/RankIndexBuilder.hpp"
#include "Utils/MathUtils.hpp"
#include "Utils/TypeDefinitionUtils.h"

using std::accumulate;
using std::fill;
using std::vector;

namespace sbwt_search {

template <class Container, u64 superblock_bits, u64 hyperblock_bits>
class CpuRankIndexBuilder:
    public RankIndexBuilder<
      CpuRankIndexBuilder<Container, superblock_bits, hyperblock_bits>,
      Container,
      superblock_bits,
      hyperblock_bits> {
    using Base = RankIndexBuilder<
      CpuRankIndexBuilder,
      Container,
      superblock_bits,
      hyperblock_bits>;
    friend Base;

  private:
    class SingleIndexBuilder {
      private:
        const size_t bits_total;
        const u64 *bits_vector;
        vector<u64> layer_0, layer_1_2;
        vector<u64> layer_2_temps = vector<u64>(3, 0);
        size_t layer_2_temps_index = 0;
        u64 layer_0_count = 0, layer_1_count = 0, layer_2_count = 0;
        const u64 basicblock_bits;

      public:
        SingleIndexBuilder(
          const size_t bits_total,
          const u64 *bits_vector,
          const u64 basicblock_bits
        ):
            bits_total(bits_total),
            bits_vector(bits_vector),
            basicblock_bits(basicblock_bits) {
          layer_0.reserve(
            round_up(bits_total, hyperblock_bits) / hyperblock_bits
          );
          layer_1_2.reserve(
            round_up(bits_total, superblock_bits) / superblock_bits
          );
        }

        vector<u64> &&get_layer_0() { return move(layer_0); };
        vector<u64> &&get_layer_1_2() { return move(layer_1_2); };
        u64 get_total_count() { return layer_0_count; };

        auto build() -> void {
          for (size_t i = 0, bits = 0;
               bits < round_up(bits_total, superblock_bits);
               bits += 64, ++i) {
            if (bits % superblock_bits == 0) {
              do_divisble_by_superblock(bits);
            }
            auto set_bits = popcount(bits_vector[i]);
            if (bits % basicblock_bits == 0 and bits % superblock_bits != 0) {
              do_divisible_by_basicblock();
            }
            layer_0_count += set_bits;
            layer_1_count += set_bits;
            layer_2_count += set_bits;
          }
        }

      private:
        auto do_divisble_by_superblock(const u64 bits) -> void {
          if (bits % hyperblock_bits == 0) { do_divisble_by_hyperlock(); }
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

    auto do_build_index() -> void {
      vector<u64> c_map(5);
      c_map[0] = 1;
      for (auto i = 0; i < 4; ++i) {
        const u64 *vector_pointer
          = this->container.get_acgt(static_cast<ACGT>(i));
        auto single_builder = SingleIndexBuilder(
          this->container.get_bits_total(),
          vector_pointer,
          this->basicblock_bits
        );
        single_builder.build();
        this->container.set_layer_0(
          static_cast<ACGT>(i), single_builder.get_layer_0()
        );
        this->container.set_layer_1_2(
          static_cast<ACGT>(i), single_builder.get_layer_1_2()
        );
        c_map[i + 1] = single_builder.get_total_count();
      }
      for (auto i = 1; i < c_map.size(); ++i) { c_map[i] += c_map[i - 1]; }
      this->container.set_c_map(move(c_map));
    }

  public:
    CpuRankIndexBuilder(Container &container): Base(container) {}
};

}

#endif
