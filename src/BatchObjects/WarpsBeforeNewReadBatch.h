#ifndef WARPS_BEFORE_NEW_READ_BATCH_H
#define WARPS_BEFORE_NEW_READ_BATCH_H

/**
 * @file WarpsBeforeNewReadBatch.h
 * @brief Stores a vector which tells us after how many warps we need to start a
 * new read, that is, to know when one colour set ends and when another begins.
 * This is a cumulative count.
 */

#include <memory>
#include <vector>

#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using std::shared_ptr;
using std::vector;

class WarpsBeforeNewReadBatch {
public:
  shared_ptr<vector<u64>> warps_before_new_read;
  auto reset() -> void { warps_before_new_read->resize(0); }
};

}  // namespace sbwt_search

#endif
