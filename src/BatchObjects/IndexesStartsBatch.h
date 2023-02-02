#ifndef INDEXES_STARTS_BATCH_H
#define INDEXES_STARTS_BATCH_H

/**
 * @file IndexesStartsBatch.h
 * @brief Holds a vector of integers which represents the starting elements at
 * which we must start a new batch in the IndexesBatch's indexes
 */

#include <vector>

#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using std::vector;

class IndexesStartsBatch {
public:
  vector<u64> indexes_starts;
};

}  // namespace sbwt_search

#endif
