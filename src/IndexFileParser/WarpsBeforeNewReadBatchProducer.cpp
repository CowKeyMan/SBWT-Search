#include <cassert>
#include <memory>

#include "IndexFileParser/WarpsBeforeNewReadBatchProducer.h"

namespace sbwt_search {

using std::make_shared;

WarpsBeforeNewReadBatchProducer::WarpsBeforeNewReadBatchProducer(
  u64 max_batches, const vector<shared_ptr<vector<u64>>> &warps_before_new_read
):
    SharedBatchesProducer<WarpsBeforeNewReadBatch>(max_batches) {
  initialise_batches();
  assert(max_batches == warps_before_new_read.size());
  for (int i = 0; i < max_batches; ++i) {
    get_batches().get(i)->warps_before_new_read = warps_before_new_read[i];
  }
}

auto WarpsBeforeNewReadBatchProducer::do_at_batch_start() -> void {
  SharedBatchesProducer<WarpsBeforeNewReadBatch>::do_at_batch_start();
  current_write()->reset();
}

auto WarpsBeforeNewReadBatchProducer::get_default_value()
  -> shared_ptr<WarpsBeforeNewReadBatch> {
  return make_shared<WarpsBeforeNewReadBatch>();
}

}  // namespace sbwt_search
