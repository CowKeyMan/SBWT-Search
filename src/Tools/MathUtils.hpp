#ifndef MATH_UTILS_HPP
#define MATH_UTILS_HPP

/**
 * @file MathUtils.hpp
 * @brief A collection of utility scripts to extend math functionality
 */

#include <cmath>

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

template <class Return, class Real1, class Real2>
auto divide_and_round(Real1 a, Real2 b) -> Return {
  return static_cast<Return>(
    std::round(static_cast<double>(a) / static_cast<double>(b))
  );
}

template <class Return, class Real1, class Real2>
auto divide_and_ceil(Real1 a, Real2 b) -> Return {
  return static_cast<Return>(
    std::ceil(static_cast<double>(a) / static_cast<double>(b))
  );
}

}  // namespace math_utils

#endif
