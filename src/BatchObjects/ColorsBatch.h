#ifndef COLORS_BATCH_H
#define COLORS_BATCH_H

/**
 * @file ColorsBatch.h
 * @brief Stores the colors contiguously for each colored sequence. A colored
 * sequence means that the sequence has found_idxs > 0.
 */

#include <vector>

#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using std::vector;

class ColorsBatch {
public:
  vector<u64> colors;
};

}  // namespace sbwt_search

#endif
