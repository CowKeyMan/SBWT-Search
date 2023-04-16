#include <algorithm>
#include <cmath>

#include "SeqToBitsConverter/ContinuousSeqToBitsConverter.h"
#include "Tools/Logger.h"
#include "fmt/core.h"

namespace sbwt_search {

using fmt::format;
using log_utils::Logger;
using std::min;

ContinuousSeqToBitsConverter::ContinuousSeqToBitsConverter(
  u64 stream_id_,
  shared_ptr<SharedBatchesProducer<StringSequenceBatch>> producer,
  u64 threads,
  u64 kmer_size,
  u64 max_chars_per_batch,
  u64 invalid_chars_producer_max_batches,
  u64 bits_producer_max_batches
):
    producer(std::move(producer)),
    threads(threads),
    invalid_chars_producer(make_shared<InvalidCharsProducer>(
      kmer_size, max_chars_per_batch, invalid_chars_producer_max_batches
    )),
    bits_producer(
      make_shared<BitsProducer>(max_chars_per_batch, bits_producer_max_batches)
    ),
    stream_id(stream_id_) {}

auto ContinuousSeqToBitsConverter::get_invalid_chars_producer() const
  -> const shared_ptr<InvalidCharsProducer> & {
  return invalid_chars_producer;
}
auto ContinuousSeqToBitsConverter::get_bits_producer() const
  -> const shared_ptr<BitsProducer> & {
  return bits_producer;
}

auto ContinuousSeqToBitsConverter::read_and_generate() -> void {
  shared_ptr<StringSequenceBatch> read_batch;
  for (u64 batch_idx = 0; (*producer) >> read_batch; ++batch_idx) {
    invalid_chars_producer->start_new_batch(read_batch->seq->size());
    bits_producer->start_new_batch(read_batch->seq->size());
    Logger::log_timed_event(
      format("SeqToBitsConverter_{}", stream_id),
      Logger::EVENT_STATE::START,
      format("batch {}", batch_idx)
    );
    parallel_generate(*read_batch);
    Logger::log_timed_event(
      format("SeqToBitsConverter_{}", stream_id),
      Logger::EVENT_STATE::STOP,
      format("batch {}", batch_idx)
    );
    invalid_chars_producer->do_at_batch_finish();
    bits_producer->do_at_batch_finish();
  }
  invalid_chars_producer->do_at_generate_finish();
  bits_producer->do_at_generate_finish();
}

auto ContinuousSeqToBitsConverter::parallel_generate(
  StringSequenceBatch &read_batch
) -> void {
  const u64 chars_per_u64 = 32;
  auto seq_size = read_batch.seq->size();
  u64 chars_per_thread = static_cast<u64>(ceil(
                           (ceil(static_cast<double>(seq_size) / chars_per_u64))
                           / static_cast<double>(threads)
                         ))
    * chars_per_u64;
#pragma omp parallel num_threads(threads) shared(read_batch)
  {
    u64 idx = omp_get_thread_num();
    u64 start_index = min(idx * chars_per_thread, seq_size);
    u64 end_index = min((idx + 1) * chars_per_thread, seq_size);
    for (u64 index = start_index; index < end_index; index += chars_per_u64) {
      bits_producer->set(
        index / chars_per_u64,
        convert_int(
          *read_batch.seq, index, min(index + chars_per_u64, end_index)
        )
      );
    }
  }
}

auto ContinuousSeqToBitsConverter::convert_int(
  const vector<char> &str, u64 start_index, u64 end_index
) -> u64 {
  const u64 bits_per_character = 2;
  u64 result = 0;
  for (u64 internal_shift = u64_bits - bits_per_character, index = 0;
       internal_shift < u64_bits && index + start_index < end_index;
       internal_shift -= bits_per_character, ++index) {
    u64 c = char_to_bits(str[index + start_index]);
    if (c == invalid_char_to_bits_value) {
      invalid_chars_producer->set(index + start_index);
      continue;
    }
    result |= c << internal_shift;
  }
  return result;
}

};  // namespace sbwt_search
