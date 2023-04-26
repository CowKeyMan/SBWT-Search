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
  u64 max_seqs_per_batch,
  u64 max_batches,
  u64 num_colors_
):
    SharedBatchesProducer<ColorsBatch>(max_batches),
    searcher(
      stream_id_,
      std::move(color_index_container_),
      max_indexes_per_batch_,
      max_seqs_per_batch
    ),
    indexes_batch_producer(std::move(indexes_batch_producer_)),
    max_indexes_per_batch(max_indexes_per_batch_),
    num_colors(num_colors_),
    stream_id(stream_id_) {
  initialise_batches();
}

auto ContinuousColorSearcher::get_bits_per_seq_cpu(u64 num_colors) -> u64 {
  const u64 bits_required_per_result = 64;
  return num_colors * bits_required_per_result;
}

auto ContinuousColorSearcher::get_bits_per_element_gpu() -> u64 {
  const u64 bits_required_per_index = 64;
  return bits_required_per_index;
}

auto ContinuousColorSearcher::get_bits_per_warp_gpu(u64 num_colors) -> u64 {
  const u64 bits_required_per_fat_result = 8;
  return num_colors * bits_required_per_fat_result;
}

auto ContinuousColorSearcher::get_bits_per_seq_gpu(u64 num_colors) -> u64 {
  const u64 bits_required_per_result = 64;
  return num_colors * bits_required_per_result;
}

auto ContinuousColorSearcher::get_default_value() -> shared_ptr<ColorsBatch> {
  return make_shared<ColorsBatch>(
    max_indexes_per_batch / gpu_warp_size * num_colors
  );
}

auto ContinuousColorSearcher::continue_read_condition() -> bool {
  return *indexes_batch_producer >> indexes_batch;
}

auto ContinuousColorSearcher::generate() -> void {
  searcher.search(
    indexes_batch->warped_indexes,
    indexes_batch->warps_intervals,
    current_write()->colors,
    get_batch_id()
  );
}

auto ContinuousColorSearcher::do_at_batch_start() -> void {
  SharedBatchesProducer<ColorsBatch>::do_at_batch_start();
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
  SharedBatchesProducer<ColorsBatch>::do_at_batch_finish();
}

}  // namespace sbwt_search
