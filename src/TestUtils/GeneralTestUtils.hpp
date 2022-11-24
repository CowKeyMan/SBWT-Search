#ifndef GENERAL_TEST_UTILS_HPP
#define GENERAL_TEST_UTILS_HPP

/**
 * @file GeneralTestUtils.hpp
 * @brief Contains functions to make testing functions cleaner
 * */

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

using std::make_shared;
using std::shared_ptr;
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

template <class T>
class DummyBatchProducer {
    vector<shared_ptr<T>> batches;
    uint counter = 0;

  public:
    DummyBatchProducer(vector<shared_ptr<T>> _batches): batches(_batches) {}
    DummyBatchProducer(vector<T> _batches) {
      for (auto b: _batches) { batches.push_back(make_shared<T>(b)); }
    }

    auto operator>>(shared_ptr<T> &out) -> bool {
      if (counter == batches.size()) { return false; }
      out = batches[counter];
      ++counter;
      return true;
    }
};

#endif
