#ifndef READ_STATISTICS_BATCH_PRODUCER_H
#define READ_STATISTICS_BATCH_PRODUCER_H

/**
 * @file ReadStatisticsBatchProducer.h
 * @brief Producer for read statistics batch, which includes the count of
 * found_idxs, not_found_idxs and invalid_idxs for each read in the current
 * batch.
 */

#include "BatchObjects/ReadStatisticsBatch.h"
#include "Tools/SharedBatchesProducer.hpp"

namespace sbwt_search {

using design_utils::SharedBatchesProducer;

class ContinuousIndexFileParser;

class ReadStatisticsBatchProducer: SharedBatchesProducer<ReadStatisticsBatch> {
  friend ContinuousIndexFileParser;

public:
  ReadStatisticsBatchProducer(u64 max_batches);

protected:
  auto do_at_batch_start() -> void override;
};

}  // namespace sbwt_search

#endif
