#ifndef READ_STATISTICS_BATCH_H
#define READ_STATISTICS_BATCH_H

/**
 * @file ReadStatisticsBatch.h
 * @brief Stores statistics about each read. These include how many ids were
 * actually found within this read, how many were invalid and how many were not
 * found. Each of these are vectors, since a single batch can have many reads.
 * In the case of a read continuing from one batch to another, we set index 0 of
 * the next batch to be equal to the last element for each of our 3 vectors and
 * we continue adding to those elements were we left off from the previous
 * batch.
 */

#include <vector>

#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using std::vector;

class ReadStatisticsBatch {
public:
  vector<u64> found_idxs;
  vector<u64> invalid_idxs;
  vector<u64> not_found_idxs;

  auto reset() -> void {
    found_idxs.resize(0);
    invalid_idxs.resize(0);
    not_found_idxs.resize(0);
  }
};

}  // namespace sbwt_search

#endif
