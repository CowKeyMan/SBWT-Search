#ifndef BIT_DEFINITIONS_H
#define BIT_DEFINITIONS_H

/**
 * @file BitDefinitions.h
 * @brief Contains definitions for some useful commonly used bit related items.
 */

#include <array>

#include "TypeDefinitions.h"

namespace bit_utils {

constexpr auto _get_set_bits() -> std::array<u64, u64_bits + 1> {
  std::array<u64, u64_bits + 1> set_bits{};
  set_bits[0] = 0;
  for (u64 i = 0; i < u64_bits; ++i) {
    set_bits.at(i + 1) = set_bits.at(i) | (1ULL << i);
  }
  return set_bits;
}

constexpr std::array<u64, u64_bits + 1> set_bits = _get_set_bits();

constexpr auto two_1s = set_bits.at(2);
constexpr auto ten_1s = set_bits.at(10);
constexpr auto thirty_1s = set_bits.at(30);

}  // namespace bit_utils

#endif
