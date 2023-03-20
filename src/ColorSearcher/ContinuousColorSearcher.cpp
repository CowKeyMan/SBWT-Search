#include <memory>

#include "ColorSearcher/ContinuousColorSearcher.h"
#include "Global/GlobalDefinitions.h"
#include "Tools/Logger.h"
#include "fmt/core.h"

namespace sbwt_search {

using fmt::format;
using log_utils::Logger;
using std::make_shared;

ContinuousColorSearcher::ContinuousColorSearcher(
  u64 stream_id_,
  shared_ptr<GpuColorIndexContainer> color_index_container_,
  shared_ptr<SharedBatchesProducer<IndexesBatch>> indexes_batch_producer_,
  u64 max_indexes_per_batch_,
  u64 max_batches,
  u64 num_colors_
):
    SharedBatchesProducer<ColorSearchResultsBatch>(max_batches),
    searcher(
      stream_id_, std::move(color_index_container_), max_indexes_per_batch_
    ),
    indexes_batch_producer(std::move(indexes_batch_producer_)),
    max_indexes_per_batch(max_indexes_per_batch_),
    num_colors(num_colors_),
    stream_id(stream_id_) {
  initialise_batches();
}

auto ContinuousColorSearcher::get_default_value()
  -> shared_ptr<ColorSearchResultsBatch> {
  auto batch = make_shared<ColorSearchResultsBatch>();
  batch->results = make_shared<vector<u64>>();
  batch->results->reserve(max_indexes_per_batch / gpu_warp_size * num_colors);
  return batch;
}

auto ContinuousColorSearcher::continue_read_condition() -> bool {
  return (*indexes_batch_producer) >> indexes_batch;
}

auto ContinuousColorSearcher::generate() -> void {
  searcher.search(
    indexes_batch->indexes, *current_write()->results, get_batch_id()
  );
}

auto ContinuousColorSearcher::do_at_batch_start() -> void {
  SharedBatchesProducer<ColorSearchResultsBatch>::do_at_batch_start();
  Logger::log_timed_event(
    format("Searcher_{}", stream_id),
    Logger::EVENT_STATE::START,
    format("batch {}", get_batch_id())
  );
}

auto ContinuousColorSearcher::do_at_batch_finish() -> void {
  Logger::log_timed_event(
    format("Searcher_{}", stream_id),
    Logger::EVENT_STATE::STOP,
    format("batch {}", get_batch_id())
  );
  SharedBatchesProducer<ColorSearchResultsBatch>::do_at_batch_finish();
}

}  // namespace sbwt_search
