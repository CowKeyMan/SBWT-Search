#include <memory>

#include "BatchObjects/BitSeqBatch.h"
#include "BatchObjects/PositionsBatch.h"
#include "BatchObjects/ResultsBatch.h"
#include "Global/GlobalDefinitions.h"
#include "SbwtContainer/GpuSbwtContainer.h"
#include "Searcher/ContinuousSearcher.h"
#include "Searcher/Searcher.h"
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

ContinuousSearcher::ContinuousSearcher(
  shared_ptr<GpuSbwtContainer> container,
  shared_ptr<SharedBatchesProducer<BitSeqBatch>> bit_seq_producer_,
  shared_ptr<SharedBatchesProducer<PositionsBatch>> positions_producer_,
  u64 max_batches,
  u64 max_chars_per_batch_
):
    searcher(std::move(container), max_chars_per_batch_),
    bit_seq_producer(std::move(bit_seq_producer_)),
    positions_producer(std::move(positions_producer_)),
    max_chars_per_batch(max_chars_per_batch_),
    SharedBatchesProducer<ResultsBatch>(max_batches) {
  initialise_batches();
}

auto ContinuousSearcher::get_default_value() -> shared_ptr<ResultsBatch> {
  return make_shared<ResultsBatch>(vector<u64>(max_chars_per_batch));
}

auto ContinuousSearcher::continue_read_condition() -> bool {
  return static_cast<bool>(
    static_cast<u64>(*positions_producer >> positions_batch)
    & static_cast<u64>(*bit_seq_producer >> bit_seq_batch)
  );
}

auto ContinuousSearcher::generate() -> void {
  searcher.search(
    bit_seq_batch->bit_seq,
    positions_batch->positions,
    current_write()->results,
    get_batch_id()
  );
}

auto ContinuousSearcher::do_at_batch_start() -> void {
  SharedBatchesProducer<ResultsBatch>::do_at_batch_start();
  Logger::log_timed_event(
    "Searcher", Logger::EVENT_STATE::START, format("batch {}", get_batch_id())
  );
}

auto ContinuousSearcher::do_at_batch_finish() -> void {
  Logger::log_timed_event(
    "Searcher", Logger::EVENT_STATE::STOP, format("batch {}", get_batch_id())
  );
  SharedBatchesProducer<ResultsBatch>::do_at_batch_finish();
}

}  // namespace sbwt_search
