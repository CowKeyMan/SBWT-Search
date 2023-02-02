#ifndef INDEXES_INTERVALS_BATCH_H
#define INDEXES_INTERVALS_BATCH_H

/**
 * @file IndexesIntervalsBatch.h
 * @brief Holds a vector of integers which represents how many indexes below to
 * the current read before we need to start processing as a new read
 */

#include <vector>

#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using std::vector;

class IndexesIntervalsBatch {
public:
  vector<u64> indexes_intervals;
  auto reset() -> void { indexes_intervals.resize(0); }
};

}  // namespace sbwt_search

#endif
