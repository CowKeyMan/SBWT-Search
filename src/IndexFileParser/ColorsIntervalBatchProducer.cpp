#include <memory>

#include "IndexFileParser/ColorsIntervalBatchProducer.h"

namespace sbwt_search {

using std::make_shared;

ColorsIntervalBatchProducer::ColorsIntervalBatchProducer(
  u64 max_batches,
  u64 max_reads_,
  vector<shared_ptr<vector<u64>>> &warps_before_new_read
):
    SharedBatchesProducer<ColorsIntervalBatch>(max_batches),
    max_reads(max_reads_) {
  initialise_batches();
  for (int i = 0; i < get_batches().size(); ++i) {
    get_batches().get(i)->warps_before_new_read = warps_before_new_read[i];
  }
}

auto ColorsIntervalBatchProducer::get_default_value()
  -> shared_ptr<ColorsIntervalBatch> {
  auto batch = make_shared<ColorsIntervalBatch>();
  batch->reads_before_newfile.reserve(max_reads + 1);
  return batch;
}

auto ColorsIntervalBatchProducer::do_at_batch_start() -> void {
  SharedBatchesProducer<ColorsIntervalBatch>::do_at_batch_start();
  current_write()->reset();
}

}  // namespace sbwt_search
