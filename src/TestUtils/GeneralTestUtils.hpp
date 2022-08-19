#ifndef GENERAL_TEST_UTILS_HPP
#define GENERAL_TEST_UTILS_HPP

/**
 * @file GeneralTestUtils.hpp
 * @brief Contains functions to make testing functions cleaner
 * */

#include <string>
#include <vector>

#include <gtest/gtest.h>

using std::string;
using std::to_string;
using std::vector;

template <class T>
auto assert_arrays_equals(const T *v1, const T *v2, const size_t size) -> void {
  for (size_t i = 0; i < size; ++i) {
    ASSERT_EQ(v1[i], v2[i]) << " unequal at index " << i;
  }
}

template <class T>
auto assert_arrays_equals(
  const T *v1, const T *v2, const size_t size, string filename, int line
) -> void {
  for (size_t i = 0; i < size; ++i) {
    ASSERT_EQ(v1[i], v2[i]) << " unequal at index " << i << " at " << filename
                            << ":" << to_string(line);
  }
}

template <class T>
auto assert_vectors_equal(const vector<T> &v1, const vector<T> &v2) -> void {
  ASSERT_EQ(v1.size(), v2.size());
  assert_arrays_equals(&v1[0], &v2[0], v1.size());
}

template <class T>
auto assert_vectors_equal(
  const vector<T> &v1, const vector<T> &v2, string filename, int line
) -> void {
  ASSERT_EQ(v1.size(), v2.size())
    << " unequal at " << filename << ":" << to_string(line);
  assert_arrays_equals(&v1[0], &v2[0], v1.size(), filename, line);
}

#endif
