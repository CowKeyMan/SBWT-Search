#ifndef CUMULATIVE_PROPERTIES_BATCH_H
#define CUMULATIVE_PROPERTIES_BATCH_H

/**
 * @file CumulativePropertiesBatch.h
 * @brief Data class for cumulative properties. cumsum_positions_per_string is
 * the cumulative sum of how many positions there are at every string while
 * cumsum_string_lengths is the cumulative sum of the the lengths string within
 * the current batch
 */

#include <vector>

#include "Utils/TypeDefinitions.h"

using std::vector;

namespace sbwt_search {

class CumulativePropertiesBatch {
  public:
    vector<u64> cumsum_positions_per_string;
    vector<u64> cumsum_string_lengths;
};

}  // namespace sbwt_search

#endif
