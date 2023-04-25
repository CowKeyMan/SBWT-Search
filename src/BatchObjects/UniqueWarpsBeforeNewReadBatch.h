#ifndef UNIQUE_WARPS_BEFORE_NEW_READ_BATCH_H
#define UNIQUE_WARPS_BEFORE_NEW_READ_BATCH_H

/**
 * @file UniqueWarpsBeforeNewReadBatch.h
 * @brief Stores a vector which tells us after how many warps we need to start a
 * new read, that is, to know when one colour set ends and when another begins.
 * This is a cumulative count. This is unique values ones only with a 0 at the
 * start always.
 */

#include <memory>
#include <vector>

#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using std::shared_ptr;
using std::vector;

class UniqueWarpsBeforeNewReadBatch {
public:
  vector<u64> unique_warps_before_new_read;
  auto reset() -> void { unique_warps_before_new_read.resize(0); }
};

}  // namespace sbwt_search

#endif
