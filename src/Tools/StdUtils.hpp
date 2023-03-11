#ifndef STD_UTILS_HPP
#define STD_UTILS_HPP

/**
 * @file StdUtils.hpp
 * @brief Functions to help out with standard library items
 */

#include <vector>

#include "Tools/MathUtils.hpp"
#include "Tools/TypeDefinitions.h"

namespace std_utils {

using math_utils::divide_and_round;
using std::vector;

template <class T>
auto split_vector(const vector<T> &v, u64 parts) -> vector<vector<T>> {
  vector<vector<T>> result;
  for (u64 i = 0; i < parts; ++i) {
    result.push_back(
      {v.begin() + divide_and_round<u64>(i * v.size(), parts),
       v.begin() + divide_and_round<u64>((i + 1) * v.size(), parts)}
    );
  }
  return result;
}

}  // namespace std_utils

#endif
