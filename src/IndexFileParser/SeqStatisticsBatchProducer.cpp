#include <memory>

#include "IndexFileParser/SeqStatisticsBatchProducer.h"

namespace sbwt_search {

using std::make_shared;

SeqStatisticsBatchProducer::SeqStatisticsBatchProducer(
  u64 max_seqs_per_batch_, u64 max_batches
):
    max_seqs_per_batch(max_seqs_per_batch_),
    SharedBatchesProducer<SeqStatisticsBatch>(max_batches) {
  initialise_batches();
}

auto SeqStatisticsBatchProducer::get_bits_per_seq() -> u64 {
  const u64 bits_required_per_result = 64ULL * 5;
  return bits_required_per_result;
}

auto SeqStatisticsBatchProducer::get_default_value()
  -> shared_ptr<SeqStatisticsBatch> {
  auto batch = make_shared<SeqStatisticsBatch>();
  batch->found_idxs.reserve(max_seqs_per_batch);
  batch->not_found_idxs.reserve(max_seqs_per_batch);
  batch->invalid_idxs.reserve(max_seqs_per_batch);
  batch->colored_seq_id.reserve(max_seqs_per_batch);
  batch->seqs_before_new_file.reserve(max_seqs_per_batch);
  return batch;
}

}  // namespace sbwt_search
