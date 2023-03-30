#ifndef CONTINUOUS_INDEX_SEARCHER_H
#define CONTINUOUS_INDEX_SEARCHER_H

/**
 * @file ContinuousIndexSearcher.h
 * @brief Search implementation with threads
 */

#include <memory>

#include "BatchObjects/BitSeqBatch.h"
#include "BatchObjects/PositionsBatch.h"
#include "BatchObjects/ResultsBatch.h"
#include "IndexSearcher/IndexSearcher.h"
#include "Tools/SharedBatchesProducer.hpp"

namespace sbwt_search {

using design_utils::SharedBatchesProducer;
using std::shared_ptr;

class ContinuousIndexSearcher: public SharedBatchesProducer<ResultsBatch> {
  IndexSearcher searcher;
  shared_ptr<SharedBatchesProducer<BitSeqBatch>> bit_seq_producer;
  shared_ptr<SharedBatchesProducer<PositionsBatch>> positions_producer;
  shared_ptr<BitSeqBatch> bit_seq_batch;
  shared_ptr<PositionsBatch> positions_batch;
  u64 max_chars_per_batch;
  u64 stream_id;

public:
  ContinuousIndexSearcher(
    u64 stream_id,
    shared_ptr<GpuSbwtContainer> container,
    shared_ptr<SharedBatchesProducer<BitSeqBatch>> bit_seq_producer_,
    shared_ptr<SharedBatchesProducer<PositionsBatch>> positions_producer_,
    u64 max_batches,
    u64 max_positions_per_batch,
    bool move_to_key_kmer
  );

  auto get_default_value() -> shared_ptr<ResultsBatch> override;
  auto continue_read_condition() -> bool override;
  auto generate() -> void override;
  auto do_at_batch_start() -> void override;
  auto do_at_batch_finish() -> void override;
};

}  // namespace sbwt_search
#endif
