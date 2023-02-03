#ifndef INTERVAL_BATCH_PRODUCER_H
#define INTERVAL_BATCH_PRODUCER_H

/**
 * @file IntervalBatchProducer.h
 * @brief Builds the IntervalBatch
 */

#include <memory>

#include "BatchObjects/IntervalBatch.h"
#include "Tools/SharedBatchesProducer.hpp"
#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

class ContinuousSequenceFileParser;

using design_utils::SharedBatchesProducer;
using std::shared_ptr;

class IntervalBatchProducer: public SharedBatchesProducer<IntervalBatch> {
  friend ContinuousSequenceFileParser;

public:
  explicit IntervalBatchProducer(uint max_batches);

private:
  auto add_file_start(size_t newlines) -> void;
  auto do_at_batch_start() -> void override;
  auto do_at_batch_finish() -> void override;
  auto get_default_value() -> shared_ptr<IntervalBatch> override;
  auto set_chars_before_newline(const vector<size_t> &chars_before_newline)
    -> void;
};

}  // namespace sbwt_search

#endif
