#ifndef READ_STATISTICS_BATCH_PRODUCER_H
#define READ_STATISTICS_BATCH_PRODUCER_H

/**
 * @file ReadStatisticsBatchProducer.h
 * @brief Producer for read statistics batch, which includes the count of
 * found_idxs, not_found_idxs and invalid_idxs for each read in the current
 * batch.
 */

#include <memory>

#include "BatchObjects/ReadStatisticsBatch.h"
#include "Tools/SharedBatchesProducer.hpp"

namespace sbwt_search {

using design_utils::SharedBatchesProducer;
using std::shared_ptr;

class ContinuousIndexFileParser;

class ReadStatisticsBatchProducer:
    public SharedBatchesProducer<ReadStatisticsBatch> {
  friend ContinuousIndexFileParser;

public:
  explicit ReadStatisticsBatchProducer(u64 max_batches);

  auto static get_bits_per_read() -> u64;

protected:
  auto get_default_value() -> shared_ptr<ReadStatisticsBatch> override;
};

}  // namespace sbwt_search

#endif
