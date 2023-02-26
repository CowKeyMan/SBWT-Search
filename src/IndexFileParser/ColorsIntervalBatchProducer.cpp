#include <cassert>
#include <limits>
#include <memory>

#include "IndexFileParser/ColorsIntervalBatchProducer.h"

namespace sbwt_search {

using std::make_shared;
using std::numeric_limits;

ColorsIntervalBatchProducer::ColorsIntervalBatchProducer(
  u64 max_batches,
  u64 max_reads_,
  const vector<shared_ptr<vector<u64>>> &warps_before_new_read
):
    SharedBatchesProducer<ColorsIntervalBatch>(max_batches),
    max_reads(max_reads_) {
  initialise_batches();
  assert(max_batches == warps_before_new_read.size());
  for (int i = 0; i < max_batches; ++i) {
    get_batches().get(i)->warps_before_new_read = warps_before_new_read[i];
  }
}

auto ColorsIntervalBatchProducer::get_default_value()
  -> shared_ptr<ColorsIntervalBatch> {
  auto batch = make_shared<ColorsIntervalBatch>();
  batch->reads_before_newfile.reserve(max_reads + 1);
  batch->warps_before_new_read = nullptr;
  return batch;
}

auto ColorsIntervalBatchProducer::do_at_batch_start() -> void {
  SharedBatchesProducer<ColorsIntervalBatch>::do_at_batch_start();
  current_write()->reset();
}

auto ColorsIntervalBatchProducer::do_at_batch_finish() -> void {
  current_write()->reads_before_newfile.push_back(numeric_limits<u64>::max());
  current_write()->warps_before_new_read->push_back(numeric_limits<u64>::max());
  SharedBatchesProducer<ColorsIntervalBatch>::do_at_batch_finish();
}

}  // namespace sbwt_search
