#include <memory>
#include <string>
#include <vector>

#include "BatchObjects/IntervalBatch.h"
#include "SequenceFileParser/IntervalBatchProducer.h"

using std::make_shared;
using std::shared_ptr;
using std::string;

namespace sbwt_search {

IntervalBatchProducer::IntervalBatchProducer(uint max_batches):
    SharedBatchesProducer<IntervalBatch>(max_batches) {
  initialise_batches();
}

auto IntervalBatchProducer::get_default_value() -> shared_ptr<IntervalBatch> {
  return make_shared<IntervalBatch>();
}

auto IntervalBatchProducer::add_file_end(size_t newlines) -> void {
  batches.current_write()->newlines_before_newfile.push_back(newlines);
}

auto IntervalBatchProducer::set_string_breaks(
  const vector<size_t> &string_breaks
) -> void {
  batches.current_write()->string_breaks = &string_breaks;
}

auto IntervalBatchProducer::do_at_batch_start() -> void {
  SharedBatchesProducer<IntervalBatch>::do_at_batch_start();
  batches.current_write()->newlines_before_newfile.resize(0);
}

auto IntervalBatchProducer::do_at_batch_finish() -> void {
  batches.current_write()->newlines_before_newfile.push_back(size_t(-1));
  SharedBatchesProducer<IntervalBatch>::do_at_batch_finish();
}

}  // namespace sbwt_search
