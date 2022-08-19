#include <memory>
#include <string>
#include <vector>

#include "BatchObjects/IntervalBatch.h"
#include "SequenceFileParser/IntervalBatchProducer.h"

using std::make_shared;
using std::shared_ptr;
using std::string;

namespace sbwt_search {

IntervalBatchProducer::IntervalBatchProducer(
  u64 max_batches, u64 max_strings_per_batch
):
    batches(max_batches + 1), semaphore(0, max_batches) {
  for (int i = 0; i < batches.size(); ++i) {
    batches.set(i, get_empty_batch(max_strings_per_batch));
  }
}

auto IntervalBatchProducer::get_empty_batch(const u64 max_strings_per_batch)
  -> shared_ptr<IntervalBatch> {
  auto batch = make_shared<IntervalBatch>();
  batch->string_lengths.reserve(max_strings_per_batch);
  batch->strings_before_newfile.reserve(max_strings_per_batch + 1);
  return batch;
}

auto IntervalBatchProducer::add_string(const string &s) -> void {
  batches.current_write()->string_lengths.push_back(s.size());
  string_counter++;
}

auto IntervalBatchProducer::file_end() -> void {
  batches.current_write()->strings_before_newfile.push_back(string_counter);
  string_counter = 0;
}

auto IntervalBatchProducer::terminate_batch() -> void {
  batches.current_write()->strings_before_newfile.push_back(u64(-1));
  batches.step_write();
  semaphore.release();
}

auto IntervalBatchProducer::start_new_batch() -> void {
  batches.current_write()->string_lengths.resize(0);
  batches.current_write()->strings_before_newfile.resize(0);
  string_counter = 0;
}

bool IntervalBatchProducer::operator>>(shared_ptr<IntervalBatch> &batch) {
  semaphore.acquire();
  if (no_more_sequences()) { return false; }
  batch = batches.current_read();
  batches.step_read();
  return true;
}

auto IntervalBatchProducer::set_finished_reading() -> void {
  finished_reading = true;
  semaphore.release();
}

auto IntervalBatchProducer::no_more_sequences() -> bool {
  return finished_reading && batches.empty();
}
}  // namespace sbwt_search
