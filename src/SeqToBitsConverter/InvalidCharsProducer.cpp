#include <algorithm>
#include <memory>

#include "SeqToBitsConverter/InvalidCharsProducer.h"
#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using std::fill;
using std::make_shared;

InvalidCharsProducer::InvalidCharsProducer(
  uint kmer_size_, size_t max_chars_per_batch_, uint max_batches
):
  kmer_size(kmer_size_),
  max_chars_per_batch(max_chars_per_batch_),
  SharedBatchesProducer<InvalidCharsBatch>(max_batches) {
  initialise_batches();
}

auto InvalidCharsProducer::get_default_value()
  -> shared_ptr<InvalidCharsBatch> {
  auto batch = make_shared<InvalidCharsBatch>();
  batch->invalid_chars.resize(max_chars_per_batch + kmer_size);
  return batch;
}

auto InvalidCharsProducer::start_new_batch(size_t num_chars) -> void {
  SharedBatchesProducer<InvalidCharsBatch>::do_at_batch_start();
  get_batches().current_write()->invalid_chars.resize(num_chars + kmer_size);
  fill(
    get_batches().current_write()->invalid_chars.begin(),
    get_batches().current_write()->invalid_chars.end(),
    0
  );
}

auto InvalidCharsProducer::set(size_t index) -> void {
  get_batches().current_write()->invalid_chars[index] = 1;
}

}  // namespace sbwt_search
