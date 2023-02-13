#ifndef POPPY_H
#define POPPY_H

/**
 * @file Poppy.h
 * @brief The rank data structure container as discussed in the paper
 * "Space-Efficient, High-Performance Rank & Select Structures on Uncompressed
 * Bit Sequences" by Zhou et. al.
 */

#include <vector>

#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using std::vector;

const u64 layer_1_bits = 32;
const u64 layer_2_bits = 10;

class Poppy {
public:
  vector<u64> layer_0;
  vector<u64> layer_1_2;
  u64 total_1s = static_cast<u64>(-1);
};

}  // namespace sbwt_search

#endif
