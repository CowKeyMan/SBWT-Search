#include <memory>
#include <string>
#include <vector>

#include "BatchObjects/CumulativePropertiesBatch.hpp"
#include "SequenceFileParser/CumulativePropertiesBatchProducer.h"

using std::make_shared;
using std::shared_ptr;
using std::string;

namespace sbwt_search {

CumulativePropertiesBatchProducer::CumulativePropertiesBatchProducer(
  u64 max_batches, u64 max_strings_per_batch, uint kmer_size
):
    batches(max_batches + 1), semaphore(0, max_batches), kmer_size(kmer_size) {
  for (int i = 0; i < batches.size(); ++i) {
    batches.set(i, get_empty_cumsum_batch(max_strings_per_batch));
  }
}

auto CumulativePropertiesBatchProducer::get_empty_cumsum_batch(
  const u64 max_strings_per_batch
) -> shared_ptr<CumulativePropertiesBatch> {
  auto batch = make_shared<CumulativePropertiesBatch>();
  batch->cumsum_positions_per_string.reserve(max_strings_per_batch);
  batch->cumsum_positions_per_string.push_back(0);
  batch->cumsum_string_lengths.reserve(max_strings_per_batch);
  batch->cumsum_string_lengths.push_back(0);
  return batch;
}

auto CumulativePropertiesBatchProducer::add_string(const string &s) -> void {
  auto &batch = batches.current_write();
  auto new_positions = 0;
  if (s.size() > kmer_size) { new_positions = s.size() - kmer_size + 1; }
  batch->cumsum_positions_per_string.push_back(
    batch->cumsum_positions_per_string.back() + new_positions
  );
  batch->cumsum_string_lengths.push_back(
    batch->cumsum_string_lengths.back() + s.size()
  );
}

auto CumulativePropertiesBatchProducer::terminate_batch() -> void {
  batches.step_write();
  semaphore.release();
}

auto CumulativePropertiesBatchProducer::start_new_batch() -> void {
  reset_batch(batches.current_write());
}

auto CumulativePropertiesBatchProducer::reset_batch(
  shared_ptr<CumulativePropertiesBatch> &batch
) -> void {
  batch->cumsum_positions_per_string.resize(1);
  batch->cumsum_string_lengths.resize(1);
}

auto CumulativePropertiesBatchProducer::operator>>(
  shared_ptr<CumulativePropertiesBatch> &batch
) -> bool {
  semaphore.acquire();
  if (no_more_sequences()) { return false; }
  batch = batches.current_read();
  batches.step_read();
  return true;
}

auto CumulativePropertiesBatchProducer::CumulativePropertiesBatchProducer::
  set_finished_reading() -> void {
  finished_reading = true;
  semaphore.release();
}

auto CumulativePropertiesBatchProducer::no_more_sequences() -> bool {
  return finished_reading && batches.empty();
}

}  // namespace sbwt_search
