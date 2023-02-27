#include <cassert>
#include <limits>
#include <memory>

#include "IndexFileParser/ColorsIntervalBatchProducer.h"

namespace sbwt_search {

using std::make_shared;
using std::numeric_limits;

ColorsIntervalBatchProducer::ColorsIntervalBatchProducer(
  u64 max_batches, const vector<shared_ptr<vector<u64>>> &warps_before_new_read
):
    SharedBatchesProducer<ColorsIntervalBatch>(max_batches) {
  initialise_batches();
  assert(max_batches == warps_before_new_read.size());
  for (int i = 0; i < max_batches; ++i) {
    get_batches().get(i)->warps_before_new_read = warps_before_new_read[i];
  }
}

auto ColorsIntervalBatchProducer::get_default_value()
  -> shared_ptr<ColorsIntervalBatch> {
  auto batch = make_shared<ColorsIntervalBatch>();
  batch->warps_before_new_read = nullptr;
  return batch;
}

auto ColorsIntervalBatchProducer::do_at_batch_finish() -> void {
  SharedBatchesProducer<ColorsIntervalBatch>::do_at_batch_finish();
}

}  // namespace sbwt_search
