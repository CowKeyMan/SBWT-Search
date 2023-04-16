#include <algorithm>
#include <memory>

#include "SeqToBitsConverter/InvalidCharsProducer.h"
#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using std::fill;
using std::make_shared;

InvalidCharsProducer::InvalidCharsProducer(
  u64 kmer_size_, u64 max_chars_per_batch_, u64 max_batches
):
    kmer_size(kmer_size_),
    max_chars_per_batch(max_chars_per_batch_),
    SharedBatchesProducer<InvalidCharsBatch>(max_batches) {
  initialise_batches();
}

auto InvalidCharsProducer::get_bits_per_element() -> u64 {
  u64 bits_required_per_entry = 8;
  return 8;
}

auto InvalidCharsProducer::get_default_value()
  -> shared_ptr<InvalidCharsBatch> {
  auto batch = make_shared<InvalidCharsBatch>();
  batch->invalid_chars.reserve(max_chars_per_batch + kmer_size);
  return batch;
}

auto InvalidCharsProducer::start_new_batch(u64 num_chars) -> void {
  SharedBatchesProducer<InvalidCharsBatch>::do_at_batch_start();
  current_write()->invalid_chars.resize(num_chars + kmer_size);
  fill(
    current_write()->invalid_chars.begin(),
    current_write()->invalid_chars.end(),
    0
  );
}

auto InvalidCharsProducer::set(u64 index) -> void {
  current_write()->invalid_chars[index] = 1;
}

}  // namespace sbwt_search
