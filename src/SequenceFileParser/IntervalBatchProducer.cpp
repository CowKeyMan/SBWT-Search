#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "BatchObjects/IntervalBatch.h"
#include "SequenceFileParser/IntervalBatchProducer.h"
#include "Tools/TypeDefinitions.h"

using std::make_shared;
using std::shared_ptr;
using std::string;

namespace sbwt_search {

using std::numeric_limits;

IntervalBatchProducer::IntervalBatchProducer(u64 max_batches):
    SharedBatchesProducer<IntervalBatch>(max_batches) {
  initialise_batches();
}

auto IntervalBatchProducer::get_default_value() -> shared_ptr<IntervalBatch> {
  return make_shared<IntervalBatch>();
}

auto IntervalBatchProducer::add_file_start(u64 newlines) -> void {
  current_write()->newlines_before_newfile.push_back(newlines);
}

auto IntervalBatchProducer::set_chars_before_newline(
  const vector<u64> &chars_before_newline
) -> void {
  current_write()->chars_before_newline = &chars_before_newline;
}

auto IntervalBatchProducer::do_at_batch_start() -> void {
  SharedBatchesProducer<IntervalBatch>::do_at_batch_start();
  current_write()->newlines_before_newfile.resize(0);
}

auto IntervalBatchProducer::do_at_batch_finish() -> void {
  current_write()->newlines_before_newfile.push_back(
    numeric_limits<u64>::max()
  );
  SharedBatchesProducer<IntervalBatch>::do_at_batch_finish();
}

}  // namespace sbwt_search
