#ifndef BENCHMARK_UTILS_HPP
#define BENCHMARK_UTILS_HPP

/**
 * @file BenchmarkUtils.hpp
 * @brief A collection of utility scripts used for benchmarking
 * */

#include <chrono>
#include <iostream>

using std::chrono::time_point;
using std::chrono::milliseconds;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;

// Macro for timing objects. The result is stored in TIME_IT_TOTAL
time_point<high_resolution_clock> TIME_IT_START_TIME, TIME_IT_END_TIME;
milliseconds::rep TIME_IT_TOTAL;
#define TIME_IT(code_block)                                              \
  TIME_IT_START_TIME = high_resolution_clock::now();                     \
  code_block;                                                            \
  TIME_IT_END_TIME = high_resolution_clock::now();                       \
  TIME_IT_TOTAL = duration_cast<milliseconds>(                           \
                    TIME_IT_END_TIME - TIME_IT_START_TIME                \
  )                                                                      \
                    .count();

#endif
