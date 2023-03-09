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

#include "Tools/TypeDefinitions.h"

namespace test_utils {

using std::span;
using std::string;
using std::to_string;
using std::vector;

template <class T>
auto assert_arrays_equals(
  const span<T> v1, const span<T> v2, const string &filename, int line
) -> void {
  EXPECT_EQ(v1.size(), v2.size())
    << " unequal size at " << filename << ":" << to_string(line);
  for (u64 i = 0; i < v1.size(); ++i) {
    EXPECT_EQ(v1[i], v2[i]) << " unequal at index " << i << " at " << filename
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

template <class Source>
auto to_u64s(const vector<vector<Source>> &int_vec)
  -> vector<vector<uint64_t>> {
  vector<vector<uint64_t>> ret_val;
  for (const auto &v : int_vec) {
    ret_val.emplace_back();
    for (const auto &element : v) { ret_val.back().emplace_back(element); }
  }
  return ret_val;
}

auto to_char_vec(const vector<string> &str_vec) -> vector<vector<char>>;

}  // namespace test_utils

#endif
