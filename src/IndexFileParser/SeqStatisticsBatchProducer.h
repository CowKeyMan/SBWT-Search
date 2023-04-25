#ifndef SEQ_STATISTICS_BATCH_PRODUCER_H
#define SEQ_STATISTICS_BATCH_PRODUCER_H

/**
 * @file SeqStatisticsBatchProducer.h
 * @brief Producer for SeqStatisticsBatch, which includes the count of
 * found_idxs, not_found_idxs and invalid_idxs for each read in the current
 * batch, as well as when each file starts and ends and other information about
 * the sequence list.
 */

#include <memory>

#include "BatchObjects/SeqStatisticsBatch.h"
#include "Tools/SharedBatchesProducer.hpp"

namespace sbwt_search {

using design_utils::SharedBatchesProducer;
using std::shared_ptr;

class ContinuousIndexFileParser;

class SeqStatisticsBatchProducer:
    public SharedBatchesProducer<SeqStatisticsBatch> {
  friend ContinuousIndexFileParser;

  u64 max_seqs_per_batch;

public:
  explicit SeqStatisticsBatchProducer(u64 max_seqs_per_batch_, u64 max_batches);

  auto static get_bits_per_seq() -> u64;

protected:
  auto get_default_value() -> shared_ptr<SeqStatisticsBatch> override;
};

}  // namespace sbwt_search

#endif
