#ifndef CONTINUOUS_COLOR_SEARCHER_H
#define CONTINUOUS_COLOR_SEARCHER_H

/**
 * @file ContinuousColorSearcher.h
 * @brief
 */

#include <memory>

#include "BatchObjects/ColorSearchResultsBatch.h"
#include "ColorIndexContainer/GpuColorIndexContainer.h"
#include "ColorSearcher/ColorSearcher.h"
#include "IndexFileParser/IndexesBatchProducer.h"
#include "Tools/SharedBatchesProducer.hpp"

namespace sbwt_search {

using std::shared_ptr;
using std::unique_ptr;

using design_utils::SharedBatchesProducer;

class ContinuousColorSearcher:
    public SharedBatchesProducer<ColorSearchResultsBatch> {
private:
  shared_ptr<IndexesBatchProducer> indexes_batch_producer;
  shared_ptr<IndexesBatch> indexes_batch;
  ColorSearcher searcher;

public:
  ContinuousColorSearcher(
    shared_ptr<GpuColorIndexContainer> color_index_container_,
    shared_ptr<IndexesBatchProducer> indexes_batch_producer_,
    u64 max_indexes_per_batch,
    u64 max_batches
  );

private:
  auto get_default_value() -> shared_ptr<ColorSearchResultsBatch> override;
  auto start_new_batch() -> void;
  auto continue_read_condition() -> bool override;
  auto generate() -> void override;
  auto do_at_batch_start() -> void override;
  auto do_at_batch_finish() -> void override;
};

}  // namespace sbwt_search

#endif