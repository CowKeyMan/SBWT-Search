#ifndef INDEXES_BATCH_H
#define INDEXES_BATCH_H

/**
 * @file IndexesBatch.h
 * @brief Contains the indexes of the search function which are output to disk
 * in the 'index' phase
 */

#include <vector>

#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using std::size_t;
using std::vector;

class IndexesBatch {
public:
  vector<u64> indexes;
};

}  // namespace sbwt_search

#endif
