#ifndef BENCHMARK_UTILS_H
#define BENCHMARK_UTILS_H

#ifndef NDEBUG
#include <chrono>
#include <iostream>
#endif

/**
 * @file BenchmarkUtils.h
 * @brief A collection of utility scripts used for benchmarking
 * */

// Macro for timing objects. The result is stored in TIME_IT_TOTAL
#ifndef NDEBUG
std::chrono::system_clock::time_point TIME_IT_START_TIME, TIME_IT_END_TIME;
unsigned long long TIME_IT_TOTAL;
#define TIME_IT(code_block)                                              \
  TIME_IT_START_TIME = std::chrono::high_resolution_clock::now();        \
  code_block;                                                            \
  TIME_IT_END_TIME = std::chrono::high_resolution_clock::now();          \
  TIME_IT_TOTAL = std::chrono::duration_cast<std::chrono::milliseconds>( \
                  TIME_IT_END_TIME - TIME_IT_START_TIME                  \
  )                                                                      \
                  .count();
#else
#define TIME_IT(code_block) code_block;
#endif

#endif
