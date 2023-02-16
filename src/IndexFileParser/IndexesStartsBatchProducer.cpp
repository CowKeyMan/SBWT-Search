#include <memory>

#include "IndexFileParser/IndexesStartsBatchProducer.h"

namespace sbwt_search {

using std::make_shared;

IndexesStartsBatchProducer::IndexesStartsBatchProducer(u64 max_batches):
    SharedBatchesProducer<IndexesStartsBatch>(max_batches) {
  initialise_batches();
}

auto IndexesStartsBatchProducer::get_default_value()
  -> shared_ptr<IndexesStartsBatch> {
  return make_shared<IndexesStartsBatch>();
}

auto IndexesStartsBatchProducer::start_new_batch() -> void {
  SharedBatchesProducer<IndexesStartsBatch>::do_at_batch_start();
  current_write()->indexes_starts.resize(0);
}

auto IndexesStartsBatchProducer::get_current_write()
  -> shared_ptr<IndexesStartsBatch> {
  return current_write();
}

}  // namespace sbwt_search
