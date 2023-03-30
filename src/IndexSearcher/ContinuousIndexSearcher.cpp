#include <memory>

#include "BatchObjects/BitSeqBatch.h"
#include "BatchObjects/PositionsBatch.h"
#include "BatchObjects/ResultsBatch.h"
#include "Global/GlobalDefinitions.h"
#include "IndexSearcher/ContinuousIndexSearcher.h"
#include "IndexSearcher/IndexSearcher.h"
#include "SbwtContainer/GpuSbwtContainer.h"
#include "Tools/Logger.h"
#include "Tools/MathUtils.hpp"
#include "Tools/SharedBatchesProducer.hpp"
#include "Tools/TypeDefinitions.h"
#include "fmt/core.h"

namespace sbwt_search {

using design_utils::SharedBatchesProducer;
using fmt::format;
using log_utils::Logger;
using math_utils::round_up;
using std::make_shared;
using std::shared_ptr;

ContinuousIndexSearcher::ContinuousIndexSearcher(
  u64 stream_id_,
  shared_ptr<GpuSbwtContainer> container,
  shared_ptr<SharedBatchesProducer<BitSeqBatch>> bit_seq_producer_,
  shared_ptr<SharedBatchesProducer<PositionsBatch>> positions_producer_,
  u64 max_batches,
  u64 max_chars_per_batch_,
  bool move_to_key_kmer
):
    searcher(stream_id_, std::move(container), max_chars_per_batch_, move_to_key_kmer),
    bit_seq_producer(std::move(bit_seq_producer_)),
    positions_producer(std::move(positions_producer_)),
    max_chars_per_batch(max_chars_per_batch_),
    SharedBatchesProducer<ResultsBatch>(max_batches),
    stream_id(stream_id_) {
  initialise_batches();
}

auto ContinuousIndexSearcher::get_default_value() -> shared_ptr<ResultsBatch> {
  auto batch = make_shared<ResultsBatch>();
  batch->results.reserve(max_chars_per_batch);
  return batch;
}

auto ContinuousIndexSearcher::continue_read_condition() -> bool {
  return static_cast<bool>(
    static_cast<u64>(*positions_producer >> positions_batch)
    & static_cast<u64>(*bit_seq_producer >> bit_seq_batch)
  );
}

auto ContinuousIndexSearcher::generate() -> void {
  searcher.search(
    bit_seq_batch->bit_seq,
    positions_batch->positions,
    current_write()->results,
    get_batch_id()
  );
}

auto ContinuousIndexSearcher::do_at_batch_start() -> void {
  SharedBatchesProducer<ResultsBatch>::do_at_batch_start();
  Logger::log_timed_event(
    format("Searcher_{}", stream_id),
    Logger::EVENT_STATE::START,
    format("batch {}", get_batch_id())
  );
}

auto ContinuousIndexSearcher::do_at_batch_finish() -> void {
  Logger::log_timed_event(
    format("Searcher_{}", stream_id),
    Logger::EVENT_STATE::STOP,
    format("batch {}", get_batch_id())
  );
  SharedBatchesProducer<ResultsBatch>::do_at_batch_finish();
}

}  // namespace sbwt_search
