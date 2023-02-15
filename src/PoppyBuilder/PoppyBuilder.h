#ifndef POPPY_BUILDER_H
#define POPPY_BUILDER_H

/**
 * @file PoppyBuilder.h
 * @brief
 * | Module responsible for building the rank index of the SBWT file
 * | Assumptions made:
 * |   * basicblock_bits is a multiple of 64
 * |   * superblock_bits is a multiple of basicblock_bits
 * |   * hyperblock_bits is a multiple of hyperblock_bits
 */

#include <cstddef>
#include <span>
#include <vector>

#include "Poppy/Poppy.h"
#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using std::span;

using std::vector;

class PoppyBuilder {
private:
  span<u64> bits_vector;
  Poppy poppy;
  vector<u64> layer_2_temps = vector<u64>(3, 0);
  size_t layer_2_temps_index = 0;
  u64 layer_0_count = 0, layer_1_count = 0, layer_2_count = 0;
  u64 num_bits;

public:
  explicit PoppyBuilder(span<u64> bits_vector, u64 num_bits_);

  auto get_poppy() -> Poppy;

private:
  auto do_divisble_by_superblock(u64 bits, Poppy &poppy) -> void;
  auto do_divisble_by_hyperlock(Poppy &poppy) -> void;
  auto do_divisible_by_basicblock(Poppy &poppy) -> void;
  auto add_layer_1_2(Poppy &poppy) -> void;
};

}  // namespace sbwt_search

#endif
