#include <memory>

#include "IndexFileParser/IndexesBatchProducer.h"

namespace sbwt_search {

using std::make_shared;

IndexesBatchProducer::IndexesBatchProducer(
  size_t max_indexes_per_batch_, size_t max_batches
):
    SharedBatchesProducer<IndexesBatch>(max_batches),
    max_indexes_per_batch(max_indexes_per_batch_) {
  initialise_batches();
}

auto IndexesBatchProducer::get_default_value() -> shared_ptr<IndexesBatch> {
  auto batch = make_shared<IndexesBatch>();
  batch->indexes.reserve(max_indexes_per_batch);
  return batch;
}

auto IndexesBatchProducer::start_new_batch() -> void {
  SharedBatchesProducer<IndexesBatch>::do_at_batch_start();
  current_write()->indexes.resize(0);
}

auto IndexesBatchProducer::get_current_write() -> shared_ptr<IndexesBatch> {
  return current_write();
}

}  // namespace sbwt_search
