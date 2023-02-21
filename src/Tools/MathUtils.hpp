#ifndef MATH_UTILS_HPP
#define MATH_UTILS_HPP

/**
 * @file MathUtils.hpp
 * @brief A collection of utility scripts to extend math functionality
 */

#include <type_traits>

#include "Tools/TypeDefinitions.h"

namespace math_utils {

using std::is_unsigned;

template <class UnsignedNumber>
auto constexpr round_up(
  const UnsignedNumber value, const UnsignedNumber multiple
) -> UnsignedNumber {
  static_assert(
    is_unsigned<UnsignedNumber>::value,
    "Template class to round_up must be unsigned"
  );
  return (value + multiple - 1) / multiple * multiple;
}

template <class UnsignedNumber>
auto constexpr round_down(
  const UnsignedNumber value, const UnsignedNumber multiple
) -> UnsignedNumber {
  static_assert(
    is_unsigned<UnsignedNumber>::value,
    "Template class to round_down must be unsigned"
  );
  return (value / multiple) * multiple;
}

auto bits_to_gB(u64 bits) -> double;
auto gB_to_bits(double gB) -> u64;

}  // namespace math_utils

#endif
