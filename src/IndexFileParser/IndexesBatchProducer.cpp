#include <memory>

#include "IndexFileParser/IndexesBatchProducer.h"

namespace sbwt_search {

using std::make_shared;

IndexesBatchProducer::IndexesBatchProducer(
  u64 max_indexes_per_batch_, u64 max_batches
):
    SharedBatchesProducer<IndexesBatch>(max_batches),
    max_indexes_per_batch(max_indexes_per_batch_) {
  initialise_batches();
}

auto IndexesBatchProducer::get_bits_per_element() -> u64 {
  const u64 bits_required_per_index = 64;
  return bits_required_per_index;
}

auto IndexesBatchProducer::get_default_value() -> shared_ptr<IndexesBatch> {
  auto batch = make_shared<IndexesBatch>();
  batch->indexes.reserve(max_indexes_per_batch);
  return batch;
}

}  // namespace sbwt_search
