#include "IndexFileParser/WarpsBeforeNewReadBatchProducer.h"

namespace sbwt_search {

WarpsBeforeNewReadBatchProducer::WarpsBeforeNewReadBatchProducer(
  u64 max_batches, vector<shared_ptr<vector<u64>>> &warps_before_new_read
):
    SharedBatchesProducer<WarpsBeforeNewReadBatch>(max_batches) {
  initialise_batches();
  for (int i = 0; i < get_batches().size(); ++i) {
    get_batches().get(i)->warps_before_new_read = warps_before_new_read[i];
  }
}

auto WarpsBeforeNewReadBatchProducer::do_at_batch_start() -> void {
  SharedBatchesProducer<WarpsBeforeNewReadBatch>::do_at_batch_start();
  current_write()->reset();
}

}  // namespace sbwt_search
