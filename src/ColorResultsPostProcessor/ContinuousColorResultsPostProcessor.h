#ifndef CONTINUOUS_COLOR_RESULTS_POST_PROCESSOR_H
#define CONTINUOUS_COLOR_RESULTS_POST_PROCESSOR_H

/**
 * @file ContinuousColorResultsPostProcessor.h
 * @brief Takes results from the searcher and does some cpu post processing on
 * them before they are printed to disk
 */

#include <memory>

#include "BatchObjects/ColorSearchResultsBatch.h"
#include "BatchObjects/WarpsBeforeNewReadBatch.h"
#include "Tools/SharedBatchesProducer.hpp"

namespace sbwt_search {
using std::shared_ptr;

using design_utils::SharedBatchesProducer;

class ContinuousColorResultsPostProcessor:
    public SharedBatchesProducer<ColorSearchResultsBatch> {
private:
  shared_ptr<SharedBatchesProducer<ColorSearchResultsBatch>>
    results_batch_producer;
  shared_ptr<ColorSearchResultsBatch> results_batch;
  shared_ptr<SharedBatchesProducer<WarpsBeforeNewReadBatch>>
    warps_before_new_read_batch_producer;
  shared_ptr<WarpsBeforeNewReadBatch> warps_before_new_read_batch;
  u64 num_colors;
  u64 stream_id;

public:
  ContinuousColorResultsPostProcessor(
    u64 stream_id_,
    shared_ptr<SharedBatchesProducer<ColorSearchResultsBatch>>
      results_batch_producer_,
    shared_ptr<SharedBatchesProducer<WarpsBeforeNewReadBatch>>
      warps_before_new_read_batch_producer_,
    u64 max_batches,
    u64 num_colors_
  );

private:
  auto get_default_value() -> shared_ptr<ColorSearchResultsBatch> override;
  auto continue_read_condition() -> bool override;
  auto generate() -> void override;
  auto squeeze_results() -> void;
  auto do_at_batch_start() -> void override;
  auto do_at_batch_finish() -> void override;
};

}  // namespace sbwt_search

#endif
