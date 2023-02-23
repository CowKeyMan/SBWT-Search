#include <memory>

#include "ColorSearcher/ContinuousColorSearcher.h"
#include "Tools/Logger.h"
#include "fmt/core.h"

namespace sbwt_search {

using fmt::format;
using log_utils::Logger;
using std::make_shared;

ContinuousColorSearcher::ContinuousColorSearcher(
  shared_ptr<GpuColorIndexContainer> color_index_container_,
  shared_ptr<IndexesBatchProducer> indexes_batch_producer_,
  u64 max_indexes_per_batch,
  u64 max_batches
):
    SharedBatchesProducer<ColorSearchResultsBatch>(max_batches),
    searcher(std::move(color_index_container_), max_indexes_per_batch),
    indexes_batch_producer(std::move(indexes_batch_producer_)) {
  initialise_batches();
}

auto ContinuousColorSearcher::get_default_value()
  -> shared_ptr<ColorSearchResultsBatch> {
  auto batch = make_shared<ColorSearchResultsBatch>();
  batch->results.reserve(1024 * 1024);  // TODO: correct this
  return batch;
}

auto ContinuousColorSearcher::continue_read_condition() -> bool {
  bool a = (*indexes_batch_producer) >> indexes_batch;
  return a;
}

auto ContinuousColorSearcher::generate() -> void {
  searcher.search(
    indexes_batch->indexes, current_write()->results, get_batch_id()
  );
}

auto ContinuousColorSearcher::do_at_batch_start() -> void {
  SharedBatchesProducer<ColorSearchResultsBatch>::do_at_batch_start();
  Logger::log_timed_event(
    "Searcher", Logger::EVENT_STATE::START, format("batch {}", get_batch_id())
  );
}

auto ContinuousColorSearcher::do_at_batch_finish() -> void {
  Logger::log_timed_event(
    "Searcher", Logger::EVENT_STATE::STOP, format("batch {}", get_batch_id())
  );
  SharedBatchesProducer<ColorSearchResultsBatch>::do_at_batch_finish();
}

}  // namespace sbwt_search
