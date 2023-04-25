#include <memory>

#include "IndexFileParser/ReadStatisticsBatchProducer.h"

namespace sbwt_search {

using std::make_shared;

ReadStatisticsBatchProducer::ReadStatisticsBatchProducer(u64 max_batches):
    SharedBatchesProducer<ReadStatisticsBatch>(max_batches) {
  initialise_batches();
}

auto ReadStatisticsBatchProducer::get_bits_per_read() -> u64 {
  const u64 bits_required_per_result = 64ULL * 5;
  return bits_required_per_result;
}

auto ReadStatisticsBatchProducer::get_default_value()
  -> shared_ptr<ReadStatisticsBatch> {
  auto batch = make_shared<ReadStatisticsBatch>();
  return batch;
}

}  // namespace sbwt_search
