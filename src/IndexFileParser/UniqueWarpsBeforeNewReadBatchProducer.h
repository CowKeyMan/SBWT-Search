#ifndef UNIQUE_WARPS_BEFORE_NEW_READ_BATCH_PRODUCER_H
#define UNIQUE_WARPS_BEFORE_NEW_READ_BATCH_PRODUCER_H

/**
 * @file WarpsBeforeNewReadBatchProducer.h
 * @brief Produces the warps_before_new_read vector for the current batch.
 */
#include "BatchObjects/UniqueWarpsBeforeNewReadBatch.h"
#include "Tools/SharedBatchesProducer.hpp"

namespace sbwt_search {

using design_utils::SharedBatchesProducer;

class ContinuousIndexFileParser;

class UniqueWarpsBeforeNewReadBatchProducer:
    public SharedBatchesProducer<UniqueWarpsBeforeNewReadBatch> {
  friend ContinuousIndexFileParser;

public:
  explicit UniqueWarpsBeforeNewReadBatchProducer(u64 max_batches);

protected:
  auto get_default_value()
    -> shared_ptr<UniqueWarpsBeforeNewReadBatch> override;
};

}  // namespace sbwt_search

#endif
