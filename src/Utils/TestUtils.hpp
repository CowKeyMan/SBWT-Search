#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include <vector>

#include <gtest/gtest.h>

using std::vector;

template <class T>
auto assert_vectors_equal(const vector<T> &v1, const vector<T> &v2) -> void {
  ASSERT_EQ(v1.size(), v2.size());
  for (size_t i = 0; i < v1.size(); ++i) {
    ASSERT_EQ(v1[i], v2[i]) << " unequal at index " << i;
  }
}

#endif
