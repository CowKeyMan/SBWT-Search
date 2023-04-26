#ifndef RESULTS_BATCH_H
#define RESULTS_BATCH_H

/**
 * @file ResultsBatch.h
 * @brief Contains the vector with the results obtained after searching for the
 * k-mer within the SBWT index
 */

#include "Tools/PinnedVector.h"
#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using gpu_utils::PinnedVector;

class ResultsBatch {
public:
  PinnedVector<u64> results;
  explicit ResultsBatch(u64 results_size): results(results_size) {}
};

}  // namespace sbwt_search

#endif
