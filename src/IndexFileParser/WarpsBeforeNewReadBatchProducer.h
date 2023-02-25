#ifndef WARPS_BEFORE_NEW_READ_BATCH_PRODUCER_H
#define WARPS_BEFORE_NEW_READ_BATCH_PRODUCER_H

/**
 * @file WarpsBeforeNewReadBatchProducer.h
 * @brief Produces the warps_before_new_read vector for the current batch.
 */
#include "BatchObjects/WarpsBeforeNewReadBatch.h"
#include "Tools/SharedBatchesProducer.hpp"

namespace sbwt_search {

using design_utils::SharedBatchesProducer;

class ContinuousIndexFileParser;

class WarpsBeforeNewReadBatchProducer:
    SharedBatchesProducer<WarpsBeforeNewReadBatch> {
  friend ContinuousIndexFileParser;

public:
  explicit WarpsBeforeNewReadBatchProducer(
    u64 max_batches, vector<shared_ptr<vector<u64>>> &warps_before_new_read
  );

protected:
  auto do_at_batch_start() -> void override;
};

}  // namespace sbwt_search

#endif
