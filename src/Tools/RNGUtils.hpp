#ifndef R_N_G_UTILS_HPP
#define R_N_G_UTILS_HPP

/**
 * @file RNGUtils.hpp
 * @brief Utilities for random number generation
 */

#include <functional>
#include <random>

using std::bind;
using std::mt19937;
using std::uniform_int_distribution;

namespace rng_utils {

template <class T>
auto get_uniform_generator(T min_value, T max_value) {
  return bind(uniform_int_distribution<T>(min_value, max_value), mt19937());
}

}  // namespace rng_utils

#endif
