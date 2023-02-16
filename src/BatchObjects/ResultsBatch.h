#ifndef RESULTS_BATCH_H
#define RESULTS_BATCH_H

/**
 * @file ResultsBatch.h
 * @brief Contains the vector with the results obtained after searching for the
 * k-mer within the SBWT index
 */

#include <vector>

#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using std::vector;

class ResultsBatch {
public:
  vector<u64> results;
};

}  // namespace sbwt_search

#endif
