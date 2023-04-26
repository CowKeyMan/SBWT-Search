#ifndef COLORS_BATCH_H
#define COLORS_BATCH_H

/**
 * @file ColorsBatch.h
 * @brief Stores the colors contiguously for each colored sequence. A colored
 * sequence means that the sequence has found_idxs > 0.
 */

#include "Tools/PinnedVector.h"
#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using gpu_utils::PinnedVector;

class ColorsBatch {
public:
  PinnedVector<u64> colors;
  explicit ColorsBatch(u64 colors_size): colors(colors_size) {}
};

}  // namespace sbwt_search

#endif
