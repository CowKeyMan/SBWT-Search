#ifndef COLORS_INTERVAL_BATCH_PRODUCER_H
#define COLORS_INTERVAL_BATCH_PRODUCER_H

/**
 * @file ColorsIntervalBatchProducer.h
 * @brief Produces the intervals of the colors indexes. These include the
 * warps_before_new_read and the reads_before_newfile.
 */

#include "BatchObjects/ColorsIntervalBatch.h"
#include "Tools/SharedBatchesProducer.hpp"

namespace sbwt_search {

using design_utils::SharedBatchesProducer;
using std::shared_ptr;

class ContinuousIndexFileParser;

class ColorsIntervalBatchProducer:
    public SharedBatchesProducer<ColorsIntervalBatch> {
  friend ContinuousIndexFileParser;

private:
  u64 max_reads;

public:
  ColorsIntervalBatchProducer(
    u64 max_batches,
    u64 max_reads,
    vector<shared_ptr<vector<u64>>> &warps_before_new_read
  );

private:
  auto get_default_value() -> shared_ptr<ColorsIntervalBatch> override;
  auto do_at_batch_start() -> void override;
};

}  // namespace sbwt_search

#endif
