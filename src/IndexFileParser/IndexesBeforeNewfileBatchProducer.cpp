#include <memory>

#include "IndexFileParser/IndexesBeforeNewfileBatchProducer.h"

namespace sbwt_search {

using std::make_shared;

IndexesBeforeNewfileBatchProducer::IndexesBeforeNewfileBatchProducer(
  size_t max_batches
):
  SharedBatchesProducer<IndexesBeforeNewfileBatch>(max_batches) {
  initialise_batches();
}

auto IndexesBeforeNewfileBatchProducer::get_default_value()
  -> shared_ptr<IndexesBeforeNewfileBatch> {
  return make_shared<IndexesBeforeNewfileBatch>();
}

auto IndexesBeforeNewfileBatchProducer::start_new_batch() -> void {
  SharedBatchesProducer<IndexesBeforeNewfileBatch>::do_at_batch_start();
  current_write()->indexes_before_newfile.resize(0);
}

auto IndexesBeforeNewfileBatchProducer::add(size_t element) -> void {
  current_write()->indexes_before_newfile.push_back(element);
}

}  // namespace sbwt_search
