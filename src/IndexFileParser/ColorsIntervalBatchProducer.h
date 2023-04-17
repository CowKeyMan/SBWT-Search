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

public:
  ColorsIntervalBatchProducer(
    u64 max_batches,
    const vector<shared_ptr<vector<u64>>> &warps_before_new_read
  );

  auto static get_bits_per_read() -> u64;

private:
  auto get_default_value() -> shared_ptr<ColorsIntervalBatch> override;
  auto do_at_batch_finish() -> void override;
};

}  // namespace sbwt_search

#endif
