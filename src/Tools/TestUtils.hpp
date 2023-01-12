#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

/**
 * @file TestUtils.hpp
 * @brief Contains functions to make testing functions cleaner
 */

#include <span>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace test_utils {

using std::span;
using std::string;
using std::to_string;
using std::vector;

template <class T>
auto assert_arrays_equals(
  const span<T> v1, const span<T> v2, const string &filename, int line
) -> void {
  ASSERT_EQ(v1.size(), v2.size())
    << " unequal size at " << filename << ":" << to_string(line);
  for (size_t i = 0; i < v1.size(); ++i) {
    ASSERT_EQ(v1[i], v2[i]) << " unequal at index " << i << " at " << filename
                            << ":" << to_string(line);
  }
}

template <class T>
auto assert_vectors_equal(
  const vector<T> &v1, const vector<T> &v2, const string &filename, int line
) -> void {
  assert_arrays_equals(
    span{v1.data(), v1.size()}, span{v2.data(), v2.size()}, filename, line
  );
}

}  // namespace test_utils

#endif
