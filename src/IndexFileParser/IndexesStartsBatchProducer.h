#ifndef INDEXES_STARTS_BATCH_PRODUCER_H
#define INDEXES_STARTS_BATCH_PRODUCER_H

/**
 * @file IndexesStartsBatchProducer.h
 * @brief Simple class used by the IndexFileParser which stores the index starts
 * batch and serves it to its consumers. The current_write batch can be obtained
 * and written to
 */

#include <memory>

#include "BatchObjects/IndexesStartsBatch.h"
#include "Tools/SharedBatchesProducer.hpp"

namespace sbwt_search {

using design_utils::SharedBatchesProducer;
using std::shared_ptr;

class ContinuousIndexFileParser;

class IndexesStartsBatchProducer:
    public SharedBatchesProducer<IndexesStartsBatch> {
  friend ContinuousIndexFileParser;

public:
  explicit IndexesStartsBatchProducer(u64 max_batches);

private:
  auto get_default_value() -> shared_ptr<IndexesStartsBatch> override;
  auto start_new_batch() -> void;
  auto get_current_write() -> shared_ptr<IndexesStartsBatch>;
};

}  // namespace sbwt_search

#endif
