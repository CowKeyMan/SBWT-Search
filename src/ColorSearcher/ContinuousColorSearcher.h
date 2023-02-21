#ifndef CONTINUOUS_COLOR_SEARCHER_H
#define CONTINUOUS_COLOR_SEARCHER_H

/**
 * @file ContinuousColorSearcher.h
 * @brief
 */

#include <memory>

#include "BatchObjects/ColorSearchResultsBatch.h"
#include "ColorIndexContainer/GpuColorIndexContainer.h"
#include "IndexFileParser/IndexesBatchProducer.h"
#include "Tools/SharedBatchesProducer.hpp"

namespace sbwt_search {

using std::shared_ptr;

using design_utils::SharedBatchesProducer;

class ContinuousColorSearcher:
    public SharedBatchesProducer<ColorSearchResultsBatch> {
private:
  shared_ptr<GpuColorIndexContainer> color_index_container;
  shared_ptr<IndexesBatchProducer> indexes_batch_producer;

public:
  ContinuousColorSearcher(
    shared_ptr<GpuColorIndexContainer> color_index_container_,
    shared_ptr<IndexesBatchProducer> indexes_batch_producer_,
    u64 max_batches
  );
};

}  // namespace sbwt_search

#endif
