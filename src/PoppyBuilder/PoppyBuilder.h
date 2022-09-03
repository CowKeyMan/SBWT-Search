#ifndef POPPY_BUILDER_H
#define POPPY_BUILDER_H

/**
 * @file PoppyBuilder.h
 * @brief Module responsible for building the rank index of the SBWT file
 *        Assumptions made:
 *          * basicblock_bits is a multiple of 64
 *          * superblock_bits is a multiple of basicblock_bits
 *          * hyperblock_bits is a multiple of hyperblock_bits
 * */

#include <stddef.h>
#include <vector>

#include "Utils/TypeDefinitions.h"

using std::vector;

namespace sbwt_search {

class PoppyBuilder {
  private:
    const size_t bits_total;
    const u64 *bits_vector;
    vector<u64> layer_0, layer_1_2;
    vector<u64> layer_2_temps = vector<u64>(3, 0);
    size_t layer_2_temps_index = 0;
    u64 layer_0_count = 0, layer_1_count = 0, layer_2_count = 0;

  public:
    PoppyBuilder(const size_t bits_total, const u64 *bits_vector);

    auto get_layer_0() -> vector<u64> &&;
    auto get_layer_1_2() -> vector<u64> &&;
    auto get_total_count() -> u64;

    auto build() -> void;

  private:
    auto do_divisble_by_superblock(const u64 bits) -> void;
    auto do_divisble_by_hyperlock() -> void;
    auto do_divisible_by_basicblock() -> void;
    auto add_layer_1_2() -> void;
};

}  // namespace sbwt_search

#endif
