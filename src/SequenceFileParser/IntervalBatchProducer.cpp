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
    SharedBatchesProducer<IntervalBatch>(max_batches),
    max_strings_per_batch(max_strings_per_batch) {
  initialise_batches();
}

auto IntervalBatchProducer::get_default_value()
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

auto IntervalBatchProducer::do_at_batch_finish(unsigned int batch_id) -> void {
  batches.current_write()->strings_before_newfile.push_back(u64(-1));
  SharedBatchesProducer<IntervalBatch>::do_at_batch_finish();
}

auto IntervalBatchProducer::do_at_batch_start(unsigned int batch_id) -> void {
  SharedBatchesProducer<IntervalBatch>::do_at_batch_start();
  batches.current_write()->string_lengths.resize(0);
  batches.current_write()->strings_before_newfile.resize(0);
  string_counter = 0;
}

}  // namespace sbwt_search
