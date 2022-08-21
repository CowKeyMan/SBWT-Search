#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "BatchObjects/StringSequenceBatch.h"
#include "SequenceFileParser/StringSequenceBatchProducer.h"
#include "Utils/MathUtils.hpp"
#include "Utils/TypeDefinitions.h"

using math_utils::round_up;
using std::make_shared;
using std::shared_ptr;
using std::string;

namespace sbwt_search {

StringSequenceBatchProducer::StringSequenceBatchProducer(
  const size_t max_strings_per_batch,
  const size_t max_chars_per_batch,
  const uint max_batches,
  const uint num_readers,
  const uint bits_split
):
    max_strings_per_batch(max_strings_per_batch),
    chars_per_reader(
      round_up<u64>(max_chars_per_batch / num_readers, bits_split / 2)
    ),
    semaphore(0, max_batches),
    batches(max_batches + 1),
    num_readers(num_readers),
    bits_split(bits_split) {
  for (int i = 0; i < batches.size(); ++i) {
    batches.set(i, get_empty_sequence_batch());
  };
}

auto StringSequenceBatchProducer::get_empty_sequence_batch()
  -> shared_ptr<StringSequenceBatch> {
  auto batch = make_shared<StringSequenceBatch>();
  batch->buffer.reserve(max_strings_per_batch);
  batch->string_indexes.reserve(num_readers + 1);
  batch->string_indexes.push_back(0);
  batch->char_indexes.reserve(num_readers + 1);
  batch->char_indexes.push_back(0);
  batch->cumulative_char_indexes.reserve(num_readers + 1);
  batch->cumulative_char_indexes.push_back(0);
  return batch;
}

auto StringSequenceBatchProducer::add_string(const string &s) -> void {
  auto &batch = *batches.current_write();
  if (s.size() > chars_to_next_index) {
    batch.char_indexes.push_back(chars_to_next_index);
    batch.string_indexes.push_back(batch.buffer.size());
    batch.cumulative_char_indexes.push_back(
      current_batch_size + chars_to_next_index
    );
    chars_to_next_index += chars_per_reader - s.size();
  } else {
    chars_to_next_index -= s.size();
  }
  current_batch_size += s.size();
  batch.buffer.push_back(move(s));
}

auto StringSequenceBatchProducer::terminate_batch() -> void {
  auto &batch = batches.current_write();
  for (uint i = batch->string_indexes.size(); i < num_readers + 1; ++i) {
    batch->string_indexes.push_back(batch->buffer.size());
    batch->char_indexes.push_back(0);
    batch->cumulative_char_indexes.push_back(current_batch_size);
  }
  batches.step_write();
  semaphore.release();
}

auto StringSequenceBatchProducer::start_new_batch() -> void {
  reset_batch(batches.current_write());
  current_batch_size = 0;
  chars_to_next_index = chars_per_reader;
}

auto StringSequenceBatchProducer::reset_batch(
  shared_ptr<StringSequenceBatch> &batch
) -> void {
  batch->buffer.resize(0);
  batch->string_indexes.resize(1);
  batch->char_indexes.resize(1);
  batch->cumulative_char_indexes.resize(1);
}

auto StringSequenceBatchProducer::operator>>(
  shared_ptr<StringSequenceBatch> &batch
) -> bool {
  semaphore.acquire();
  if (no_more_sequences()) { return false; }
  batch = batches.current_read();
  batches.step_read();
  return true;
}

auto StringSequenceBatchProducer::set_finished_reading() -> void {
  finished_reading = true;
  semaphore.release();
}

auto StringSequenceBatchProducer::no_more_sequences() -> bool {
  return finished_reading && batches.empty();
}

}  // namespace sbwt_search
