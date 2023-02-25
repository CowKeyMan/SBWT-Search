#ifndef COLORS_INTERVAL_BATCH_H
#define COLORS_INTERVAL_BATCH_H

/**
 * @file ColorsIntervalBatch.h
 * @brief Stores items related to the intervals relating to indexes read from
 * the index file. `warps_before_new_read` is how many warps are read before a
 * new read starts. A u64::max() is added at the very end to signify that the
 * read continues in the next batch. `reads_before_newfile` stores how many
 * reads we need to process in the current batch before starting the next file.
 * Similar to `warps_before_new_read`, a u64::max() is added at the end to
 * signify that the current file continues in the next batch. The size of each
 * vector is (max_reads + 1) * 64 bits (meaning [max_reads + 1] elements)
 */

#include <memory>
#include <vector>

#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using std::shared_ptr;
using std::vector;

class ColorsIntervalBatch {
public:
  shared_ptr<vector<u64>> warps_before_new_read;
  vector<u64> reads_before_newfile;

  auto reset() -> void {
    warps_before_new_read->resize(0);
    reads_before_newfile.resize(0);
  }
};

}  // namespace sbwt_search

#endif
