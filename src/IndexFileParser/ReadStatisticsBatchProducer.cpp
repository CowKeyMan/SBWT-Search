#include "IndexFileParser/ReadStatisticsBatchProducer.h"

namespace sbwt_search {

ReadStatisticsBatchProducer::ReadStatisticsBatchProducer(u64 max_batches):
    SharedBatchesProducer<ReadStatisticsBatch>(max_batches) {}

auto ReadStatisticsBatchProducer::do_at_batch_start() -> void {
  SharedBatchesProducer<ReadStatisticsBatch>::do_at_batch_start();
  current_write()->reset();
}

}  // namespace sbwt_search
