#ifndef MATH_UTILS_HPP
#define MATH_UTILS_HPP

/**
 * @file MathUtils.hpp
 * @brief A collection of utility scripts to extend math functionality
 * */

#include <type_traits>

using std::is_unsigned;

template <class UnsignedNumber>
UnsignedNumber round_up(const UnsignedNumber value, const UnsignedNumber multiple) {
  static_assert(
    is_unsigned<UnsignedNumber>::value, "Template class to round_up must be unsigned"
  );
  return (value + multiple - 1) / multiple * multiple;
}

#endif
