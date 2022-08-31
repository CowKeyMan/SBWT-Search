#ifndef RANK__TEST_HPP
#define RANK__TEST_HPP

/**
 * @file Rank_test.hpp
 * @brief Header for compiling the test for the rank function
 * */

#include <vector>

#include "Utils/TypeDefinitions.h"

using std::vector;

namespace sbwt_search {

auto get_rank_output(
  const vector<u64> &bit_vector,
  const vector<u64> &layer_0,
  const vector<u64> &layer_1_2,
  const vector<u64> &test_indexes
) -> vector<u64>;

}  // namespace sbwt_search

#endif
