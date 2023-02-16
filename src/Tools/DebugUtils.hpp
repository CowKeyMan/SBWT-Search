#ifndef DEBUG_UTILS_HPP
#define DEBUG_UTILS_HPP

/**
 * @file DebugUtils.hpp
 * @brief Utilities for debugging code
 */

#include <climits>
#include <iostream>
#include <vector>

using namespace std;

template <class T>
void print(const vector<T> &v, u64 max = ULLONG_MAX) {
  u64 counter = 0;
  for (auto &x : v) {
    cout << x << ' ';
    if (counter++ >= max) { break; };
  }
  cout << endl;
}

#endif
