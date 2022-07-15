#ifndef BIT_VECTOR_TEST_UTILS
#define BIT_VECTOR_TEST_UTILS

/**
 * @file BitVectorTestUtils.hpp
 * @brief Contains functions and objects relating to bit vectors
 * */

#include <vector>

#include "Utils/TypeDefinitionUtils.h"

using std::vector;

namespace sbwt_search {

constexpr u64 get_ones(const int ones) {
  u64 result = 0;
  for (int i = 0; i < ones; ++i) { result = (result << 1) + 1; }
  return result;
}

// 2  u64s = 1 basic block
// 8  u64s = 1 super block
// 16 u64s = 1 hyper block
const vector<u64> bit_vector = {
  // # Hyper Block 1 - total: 130
  // ## Super Block 1 - total: 30
  // ### Basic Block 01 - total: 5
  get_ones(2),
  get_ones(3),
  // ### Basic Block 02 - total: 10
  get_ones(1),
  get_ones(9),
  // ### Basic Block 03 - total: 10
  get_ones(10),
  get_ones(0),
  // ### Basic Block 04 - total: 5
  get_ones(1),
  get_ones(4),

  // ## Super Block 2 - total: 100
  // ### Basic Block 05 - total: 10
  get_ones(5),
  get_ones(5),
  // ### Basic Block 06 - total: 65
  get_ones(60),
  get_ones(5),
  // ### Basic Block 07 - total: 20
  get_ones(13),
  get_ones(7),
  // ### Basic Block 08 - total: 5
  get_ones(0),
  get_ones(5),

  // # Hyper Block 2 - total: 2
  // ## Super Block 3 - total: 2
  // ### Basic Block 09 - total: 2
  get_ones(2),
  get_ones(0),  // PADDING TO NEXT SUPER BLOCK

  get_ones(0),  // PADDING TO NEXT SUPER BLOCK
  get_ones(0),  // PADDING TO NEXT SUPER BLOCK

  get_ones(0),  // PADDING TO NEXT SUPER BLOCK
  get_ones(0),  // PADDING TO NEXT SUPER BLOCK

  get_ones(0),  // PADDING TO NEXT SUPER BLOCK
  get_ones(0)  // PADDING TO NEXT SUPER BLOCK
};

const vector<u64> expected_layer_0 = { 0, 130 };
const vector<u64> expected_layer_1_2 = {
  (0ULL << 32) | (5ULL << 20) | (10ULL << 10) | (10ULL << 0),
  (30ULL << 32) | (10ULL << 20) | (65ULL << 10) | (20ULL << 0),
  (0ULL << 32) | (2ULL << 20) | (0ULL << 10) | (0ULL << 0),
};
const vector<u64> c_map = { 1, 132 + 1, 132 * 2 + 1, 132 * 3 + 1, 132 * 4 + 1 };

}

#endif
