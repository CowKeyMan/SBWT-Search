#ifndef INDEXES_BATCH_PRODUCER_H
#define INDEXES_BATCH_PRODUCER_H

/**
 * @file IndexesBatchProducer.h
 * @brief Simple class used by the IndexFileParser which stores the index batch
 * and serves it to its consumers. The current_write batch can be obtained and
 * written to
 */

#include <memory>

#include "BatchObjects/IndexesBatch.h"
#include "Tools/SharedBatchesProducer.hpp"

namespace sbwt_search {

using design_utils::SharedBatchesProducer;
using std::shared_ptr;

class ContinuousIndexFileParser;

class IndexesBatchProducer: public SharedBatchesProducer<IndexesBatch> {
  friend ContinuousIndexFileParser;

private:
  u64 max_indexes_per_batch;

public:
  IndexesBatchProducer(u64 max_indexes_per_batch_, u64 max_batches);

private:
  auto get_default_value() -> shared_ptr<IndexesBatch> override;
};

}  // namespace sbwt_search

#endif
