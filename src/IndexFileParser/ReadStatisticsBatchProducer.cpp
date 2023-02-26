#include <memory>

#include "IndexFileParser/ReadStatisticsBatchProducer.h"

namespace sbwt_search {

using std::make_shared;

ReadStatisticsBatchProducer::ReadStatisticsBatchProducer(
  u64 max_batches, u64 max_reads_
):
    SharedBatchesProducer<ReadStatisticsBatch>(max_batches),
    max_reads(max_reads_) {
  initialise_batches();
}

auto ReadStatisticsBatchProducer::do_at_batch_start() -> void {
  SharedBatchesProducer<ReadStatisticsBatch>::do_at_batch_start();
  current_write()->reset();
}

auto ReadStatisticsBatchProducer::get_default_value()
  -> shared_ptr<ReadStatisticsBatch> {
  auto batch = make_shared<ReadStatisticsBatch>();
  batch->found_idxs.reserve(max_reads);
  batch->not_found_idxs.reserve(max_reads);
  batch->invalid_idxs.reserve(max_reads);
  return batch;
}

}  // namespace sbwt_search
