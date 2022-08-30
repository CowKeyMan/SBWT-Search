#include <memory>
#include <string>
#include <vector>

#include "BatchObjects/CumulativePropertiesBatch.h"
#include "SequenceFileParser/CumulativePropertiesBatchProducer.h"

using std::make_shared;
using std::shared_ptr;
using std::string;

namespace sbwt_search {

CumulativePropertiesBatchProducer::CumulativePropertiesBatchProducer(
  u64 max_batches, u64 max_strings_per_batch, uint kmer_size
):
    kmer_size(kmer_size),
    SharedBatchesProducer<CumulativePropertiesBatch>(max_batches),
    max_strings_per_batch(max_strings_per_batch) {
  initialise_batches();
}

auto CumulativePropertiesBatchProducer::get_default_value()
  -> shared_ptr<CumulativePropertiesBatch> {
  auto batch = make_shared<CumulativePropertiesBatch>();
  batch->cumsum_positions_per_string.reserve(max_strings_per_batch);
  batch->cumsum_positions_per_string.push_back(0);
  batch->cumsum_string_lengths.reserve(max_strings_per_batch);
  batch->cumsum_string_lengths.push_back(0);
  return batch;
}

auto CumulativePropertiesBatchProducer::add_string(const string &s) -> void {
  auto &batch = batches.current_write();
  auto new_positions = 0;
  if (s.size() > kmer_size) { new_positions = s.size() - kmer_size + 1; }
  batch->cumsum_positions_per_string.push_back(
    batch->cumsum_positions_per_string.back() + new_positions
  );
  batch->cumsum_string_lengths.push_back(
    batch->cumsum_string_lengths.back() + s.size()
  );
}

auto CumulativePropertiesBatchProducer::do_at_batch_start(unsigned int batch_id)
  -> void {
  SharedBatchesProducer<CumulativePropertiesBatch>::do_at_batch_start();
  reset_batch(batches.current_write());
}

auto CumulativePropertiesBatchProducer::reset_batch(
  shared_ptr<CumulativePropertiesBatch> &batch
) -> void {
  batch->cumsum_positions_per_string.resize(1);
  batch->cumsum_string_lengths.resize(1);
}

}  // namespace sbwt_search
