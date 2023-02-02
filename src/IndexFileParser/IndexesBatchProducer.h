#ifndef INDEXES_BATCH_PRODUCER_H
#define INDEXES_BATCH_PRODUCER_H

/**
 * @file IndexesBatchProducer.h
 * @brief Simple class sued by the IndexFileParser which stores the index batch
 * and serves it to its consumers. The class provides an interfact to fill the
 * batch
 */

#include <memory>

#include "BatchObjects/IndexesBatch.h"
#include "Tools/SharedBatchesProducer.hpp"

namespace sbwt_search {

using design_utils::SharedBatchesProducer;
using std::shared_ptr;

class IndexFileParser;

class IndexesBatchProducer: SharedBatchesProducer<IndexesBatch> {
  friend IndexFileParser;

private:
  size_t max_indexes_per_batch;

public:
  IndexesBatchProducer(size_t max_indexes_per_batch_, uint max_batches);

private:
  auto get_default_value() -> shared_ptr<IndexesBatch> override;
  auto start_new_batch() -> void;
  auto add(size_t index) -> void;
};

}  // namespace sbwt_search

#endif
