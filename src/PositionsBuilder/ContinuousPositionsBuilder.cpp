#include <memory>

#include "PositionsBuilder/ContinuousPositionsBuilder.h"
#include "Tools/Logger.h"
#include "fmt/core.h"

namespace sbwt_search {

using fmt::format;
using log_utils::Logger;
using std::make_shared;

ContinuousPositionsBuilder::ContinuousPositionsBuilder(
  u64 stream_id_,
  shared_ptr<SharedBatchesProducer<StringBreakBatch>> _producer,
  u64 kmer_size_,
  u64 max_chars_per_batch_,
  u64 max_batches
):
    producer(std::move(_producer)),
    max_chars_per_batch(max_chars_per_batch_),
    builder(kmer_size_),
    SharedBatchesProducer<PositionsBatch>(max_batches),
    stream_id(stream_id_) {
  initialise_batches();
}

auto ContinuousPositionsBuilder::get_default_value()
  -> shared_ptr<PositionsBatch> {
  auto batch = make_shared<PositionsBatch>();
  batch->positions.resize(max_chars_per_batch);
  return batch;
}

auto ContinuousPositionsBuilder::continue_read_condition() -> bool {
  return (*producer) >> read_batch;
}

auto ContinuousPositionsBuilder::generate() -> void {
  builder.build_positions(
    *read_batch->chars_before_newline,
    read_batch->string_size,
    current_write()->positions
  );
}

auto ContinuousPositionsBuilder::do_at_batch_start() -> void {
  SharedBatchesProducer<PositionsBatch>::do_at_batch_start();
  Logger::log_timed_event(
    format("PositionsBuilder_{}", stream_id),
    Logger::EVENT_STATE::START,
    format("batch {}", get_batch_id())
  );
}

auto ContinuousPositionsBuilder::do_at_batch_finish() -> void {
  Logger::log_timed_event(
    format("PositionsBuilder_{}", stream_id),
    Logger::EVENT_STATE::STOP,
    format("batch {}", get_batch_id())
  );
  SharedBatchesProducer<PositionsBatch>::do_at_batch_finish();
}

}  // namespace sbwt_search
