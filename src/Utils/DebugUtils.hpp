#ifndef DEBUG_UTILS_HPP
#define DEBUG_UTILS_HPP

/**
 * @file DebugUtils.hpp
 * @brief Utilities for debugging code
 * */

#include <iostream>
#include <vector>
#include <climits>

using namespace std;

template <class T>
void print(const vector<T> &v, size_t max = ULLONG_MAX) {
  size_t counter = 0;
  for (auto &x: v) {
    cout << x << ' ';
    if (counter++ >= max) { break; };
  }
  cout << endl;
}

#endif
