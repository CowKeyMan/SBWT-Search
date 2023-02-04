#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "BatchObjects/IntervalBatch.h"
#include "SequenceFileParser/IntervalBatchProducer.h"

using std::make_shared;
using std::shared_ptr;
using std::string;

namespace sbwt_search {

using std::numeric_limits;

IntervalBatchProducer::IntervalBatchProducer(uint max_batches):
    SharedBatchesProducer<IntervalBatch>(max_batches) {
  initialise_batches();
}

auto IntervalBatchProducer::get_default_value() -> shared_ptr<IntervalBatch> {
  return make_shared<IntervalBatch>();
}

auto IntervalBatchProducer::add_file_start(size_t newlines) -> void {
  get_batches().current_write()->newlines_before_newfile.push_back(newlines);
}

auto IntervalBatchProducer::set_chars_before_newline(
  const vector<size_t> &chars_before_newline
) -> void {
  get_batches().current_write()->chars_before_newline = &chars_before_newline;
}

auto IntervalBatchProducer::do_at_batch_start() -> void {
  SharedBatchesProducer<IntervalBatch>::do_at_batch_start();
  get_batches().current_write()->newlines_before_newfile.resize(0);
}

auto IntervalBatchProducer::do_at_batch_finish() -> void {
  get_batches().current_write()->newlines_before_newfile.push_back(
    numeric_limits<size_t>::max()
  );
  SharedBatchesProducer<IntervalBatch>::do_at_batch_finish();
}

}  // namespace sbwt_search
