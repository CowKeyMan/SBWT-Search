#ifndef INDEXES_BATCH_PRODUCER_H
#define INDEXES_BATCH_PRODUCER_H

/**
 * @file IndexesBatchProducer.h
 * @brief Simple class used by the IndexFileParser which stores the index batch
 * and serves it to its consumers.
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
  u64 max_seqs_per_batch;

public:
  IndexesBatchProducer(
    u64 max_seqs_per_batch_, u64 max_indexes_per_batch_, u64 max_batches
  );

  auto static get_bits_per_element() -> u64;
  auto static get_bits_per_seq() -> u64;

private:
  auto get_default_value() -> shared_ptr<IndexesBatch> override;
};

}  // namespace sbwt_search

#endif
