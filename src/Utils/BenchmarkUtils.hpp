#ifndef BENCHMARK_UTILS_HPP
#define BENCHMARK_UTILS_HPP

/**
 * @file BenchmarkUtils.hpp
 * @brief A collection of utility scripts used for benchmarking
 * */

#include <chrono>
#include <iostream>

// Macro for timing objects. The result is stored in TIME_IT_TOTAL
std::chrono::system_clock::time_point TIME_IT_START_TIME, TIME_IT_END_TIME;
std::chrono::milliseconds::rep TIME_IT_TOTAL;
#define TIME_IT(code_block)                                              \
  TIME_IT_START_TIME = std::chrono::high_resolution_clock::now();        \
  code_block;                                                            \
  TIME_IT_END_TIME = std::chrono::high_resolution_clock::now();          \
  TIME_IT_TOTAL = std::chrono::duration_cast<std::chrono::milliseconds>( \
                    TIME_IT_END_TIME - TIME_IT_START_TIME                \
  )                                                                      \
                    .count();

#endif
