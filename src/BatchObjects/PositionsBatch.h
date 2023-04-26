#ifndef POSITIONS_BATCH_H
#define POSITIONS_BATCH_H

/**
 * @file PositionsBatch.h
 * @brief Has data created by the positions builder, which is a list of valid
 * positions within a string sequence where the SBWT should search. A valid
 * sequence is one where the string length is equal to the k-mer size. For
 * example, given the string ABCDE and FGHIJ, and k-mer size 3, positions 0, 1,
 * 2 are valid positions, but position 3, which is the k-mer starting at D, is
 * not valid. Hence, our final position list will be: [0, 1, 2, 5, 6, 7]
 */

#include <vector>

#include "Tools/PinnedVector.h"
#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using gpu_utils::PinnedVector;

class PositionsBatch {
public:
  PinnedVector<u64> positions;
  explicit PositionsBatch(u64 positions_size): positions(positions_size) {}
};

}  // namespace sbwt_search

#endif
