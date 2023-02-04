#ifndef INDEXES_BEFORE_NEWFILE_BATCH_PRODUCER_H
#define INDEXES_BEFORE_NEWFILE_BATCH_PRODUCER_H

/**
 * @file IndexesBeforeNewfileBatchProducer.h
 * @brief Stores and serves the IndexesBeforeNewfileBatch. Used by the
 * ContinuousIndexFileParser and also populated by it
 */

#include <memory>

#include "BatchObjects/IndexesBeforeNewfileBatch.h"
#include "Tools/SharedBatchesProducer.hpp"

namespace sbwt_search {

using design_utils::SharedBatchesProducer;
using std::shared_ptr;

class ContinuousIndexFileParser;

class IndexesBeforeNewfileBatchProducer:
    public SharedBatchesProducer<IndexesBeforeNewfileBatch> {
  friend ContinuousIndexFileParser;

public:
  explicit IndexesBeforeNewfileBatchProducer(size_t max_batches);

private:
  auto get_default_value() -> shared_ptr<IndexesBeforeNewfileBatch> override;
  auto start_new_batch() -> void;
  auto add(size_t element) -> void;
};

}  // namespace sbwt_search

#endif
