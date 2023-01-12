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
  shared_ptr<SharedBatchesProducer<StringSequenceBatch>> producer,
  shared_ptr<InvalidCharsProducer> invalid_chars_producer_,
  shared_ptr<BitsProducer> bits_producer_,
  uint threads
):
  producer(std::move(producer)),
  threads(threads),
  invalid_chars_producer(std::move(invalid_chars_producer_)),
  bits_producer(std::move(bits_producer_)) {}

auto ContinuousSeqToBitsConverter::read_and_generate() -> void {
  shared_ptr<StringSequenceBatch> read_batch;
  for (uint batch_idx = 0; (*producer) >> read_batch; ++batch_idx) {
    invalid_chars_producer->start_new_batch(read_batch->seq->size());
    bits_producer->start_new_batch(read_batch->seq->size());
    Logger::log_timed_event(
      "SeqToBitsConverter",
      Logger::EVENT_STATE::START,
      format("batch {}", batch_idx)
    );
    parallel_generate(*read_batch);
    Logger::log_timed_event(
      "SeqToBitsConverter",
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
  const size_t chars_per_u64 = 32;
  auto seq_size = read_batch.seq->size();
  size_t chars_per_thread
    = static_cast<size_t>(
        ceil((ceil(static_cast<double>(seq_size) / chars_per_u64)) / threads)
      )
    * chars_per_u64;
#pragma omp parallel num_threads(threads) shared(read_batch)
  {
    uint idx = omp_get_thread_num();
    size_t start_index = min(idx * chars_per_thread, seq_size);
    size_t end_index = min((idx + 1) * chars_per_thread, seq_size);
    for (size_t index = start_index; index < end_index;
         index += chars_per_u64) {
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
  const string &string, size_t start_index, size_t end_index
) -> u64 {
  const u64 bits_per_character = 2;
  u64 result = 0;
  for (u64 internal_shift = u64_bits - bits_per_character, index = 0;
       internal_shift < u64_bits && index + start_index < end_index;
       internal_shift -= bits_per_character, ++index) {
    u64 c = char_to_bits(string[index + start_index]);
    if (c == invalid_char_to_bits_value) {
      invalid_chars_producer->set(index + start_index);
      continue;
    }
    result |= c << internal_shift;
  }
  return result;
}

auto ContinuousSeqToBitsConverter::operator>>(shared_ptr<BitSeqBatch> &batch)
  -> bool {
  return (*bits_producer) >> batch;
}

auto ContinuousSeqToBitsConverter::operator>>(
  shared_ptr<InvalidCharsBatch> &batch
) -> bool {
  return (*invalid_chars_producer) >> batch;
}

};  // namespace sbwt_search
