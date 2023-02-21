#include "ColorSearcher/ContinuousColorSearcher.h"

namespace sbwt_search {

ContinuousColorSearcher::ContinuousColorSearcher(
  shared_ptr<GpuColorIndexContainer> color_index_container_,
  shared_ptr<IndexesBatchProducer> indexes_batch_producer_,
  u64 max_batches
):
    SharedBatchesProducer<ColorSearchResultsBatch>(max_batches),
    color_index_container(std::move(color_index_container_)),
    indexes_batch_producer(std::move(indexes_batch_producer_)) {}

}  // namespace sbwt_search
