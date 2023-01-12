#include <cmath>

#include "SeqToBitsConverter/BitsProducer.h"

namespace sbwt_search {

const u64 chars_per_u64 = 32;

BitsProducer::BitsProducer(u64 max_chars_per_batch_, uint max_batches):
  max_chars_per_batch(max_chars_per_batch_),
  SharedBatchesProducer<BitSeqBatch>(max_batches) {
  initialise_batches();
}

auto BitsProducer::get_default_value() -> shared_ptr<BitSeqBatch> {
  auto batch = make_shared<BitSeqBatch>();
  batch->bit_seq.resize(
    round_up<u64>(max_chars_per_batch, chars_per_u64) / chars_per_u64
  );
  return batch;
}

auto BitsProducer::start_new_batch(u64 num_chars) -> void {
  SharedBatchesProducer<BitSeqBatch>::do_at_batch_start();
  get_batches().current_write()->bit_seq.resize(
    ceil(static_cast<double>(num_chars) / chars_per_u64)
  );
}

auto BitsProducer::set(u64 index, u64 value) -> void {
  get_batches().current_write()->bit_seq[index] = value;
}

}  // namespace sbwt_search
