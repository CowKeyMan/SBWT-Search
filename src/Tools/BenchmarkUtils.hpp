#ifndef BENCHMARK_UTILS_HPP
#define BENCHMARK_UTILS_HPP

/**
 * @file BenchmarkUtils.hpp
 * @brief A collection of utility scripts used for benchmarking
 */

#include <chrono>

namespace benchmark_utils {

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using std::chrono::time_point;

// NOLINTBEGIN

// Macro for timing objects. The result is stored in TIME_IT_TOTAL
time_point<high_resolution_clock> TIME_IT_START_TIME, TIME_IT_END_TIME;
milliseconds::rep TIME_IT_TOTAL;
#define TIME_IT(...)                                                     \
  TIME_IT_START_TIME = high_resolution_clock::now();                     \
  __VA_ARGS__;                                                           \
  TIME_IT_END_TIME = high_resolution_clock::now();                       \
  TIME_IT_TOTAL                                                          \
    = duration_cast<milliseconds>(TIME_IT_END_TIME - TIME_IT_START_TIME) \
        .count();

// NOLINTEND

}  // namespace benchmark_utils

#endif
