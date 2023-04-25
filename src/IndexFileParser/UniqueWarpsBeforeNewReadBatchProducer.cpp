#include <cassert>
#include <memory>

#include "IndexFileParser/UniqueWarpsBeforeNewReadBatchProducer.h"

namespace sbwt_search {

using std::make_shared;

UniqueWarpsBeforeNewReadBatchProducer::UniqueWarpsBeforeNewReadBatchProducer(
  u64 max_batches
):
    SharedBatchesProducer<UniqueWarpsBeforeNewReadBatch>(max_batches) {
  initialise_batches();
}

auto UniqueWarpsBeforeNewReadBatchProducer::get_default_value()
  -> shared_ptr<UniqueWarpsBeforeNewReadBatch> {
  return make_shared<UniqueWarpsBeforeNewReadBatch>();
}

}  // namespace sbwt_search
