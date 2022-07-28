#ifndef RANK_TEST_UTILS_HPP
#define RANK_TEST_UTILS_HPP

/**
 * @file RankTestUtils.hpp
 * @brief Contains a dmmmy Rank function which works on CPU
 * */

#include "Utils/TypeDefinitions.h"

namespace sbwt_search {

template <bool reversed_bits>
u64 dummy_cpu_rank(const u64 *v, const u64 index) {
  const u64 final_index = (index) / 64;
  u64 result = 0;
  for (int i = 0; i < final_index; ++i) {
    result += __builtin_popcountll(v[i]);
  }
  const u64 final_int = v[final_index];
  u64 required_bits;
  if (reversed_bits) {
    required_bits = (final_int << (64 - (index % 64))) & (-((index % 64) != 0));
  } else {
    required_bits = (final_int >> (64 - (index % 64))) & (-((index % 64) != 0));
  }
  result += __builtin_popcountll(required_bits);
  return result;
}

}

#endif
