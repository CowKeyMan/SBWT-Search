#ifndef STRING_SEQUENCE_BATCH_PRODUCER_HPP
#define STRING_SEQUENCE_BATCH_PRODUCER_HPP

/**
 * @file StringSequenceBatchProducer.hpp
 * @brief takes care of building and sending the stringsequencebatch
 * */

#include <iostream>
#include <queue>
#include <string>
#include <vector>

#include "BatchObjects/CumulativePropertiesBatch.hpp"
#include "BatchObjects/StringSequenceBatch.hpp"
#include "SequenceFileParser/SequenceFileParser.h"
#include "Utils/BoundedSemaphore.hpp"
#include "Utils/CircularBuffer.hpp"
#include "Utils/MathUtils.hpp"
#include "Utils/TypeDefinitions.h"

using std::make_shared;
using std::queue;
using std::shared_ptr;
using std::string;
using std::to_string;
using std::vector;
using structure_utils::CircularBuffer;
using threading_utils::BoundedSemaphore;

namespace sbwt_search {

class StringSequenceBatchProducer {
  private:
    const uint bits_split = 64;
    CircularBuffer<shared_ptr<StringSequenceBatch>> batches;
    const uint max_strings_per_batch;
    const uint num_readers;
    BoundedSemaphore semaphore;
    uint current_batch_size = 0;
    bool finished_reading = false;
    uint chars_to_next_index;
    const uint chars_per_reader;

  public:
    StringSequenceBatchProducer(
      const uint max_strings_per_batch,
      const uint max_chars_per_batch,
      const uint max_batches,
      const uint num_readers,
      const uint bits_split = 64
    ):
        max_strings_per_batch(max_strings_per_batch),
        chars_per_reader(
          round_up<u64>(max_chars_per_batch / num_readers, bits_split / 2)
        ),
        semaphore(0, max_batches),
        batches(max_batches + 2),
        num_readers(num_readers),
        bits_split(bits_split) {
      for (int i = 0; i < batches.size(); ++i) {
        batches.set(i, get_empty_sequence_batch());
      };
    }

  private:
    auto get_empty_sequence_batch() -> shared_ptr<StringSequenceBatch> {
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

  public:
    auto add_string(const string &s) -> void {
      auto &batch = *batches.current_write();
      if (s.size() > chars_to_next_index) {
        batch.char_indexes.push_back(chars_to_next_index);
        batch.string_indexes.push_back(batch.buffer.size());
        chars_to_next_index += chars_per_reader - s.size();
      } else {
        chars_to_next_index -= s.size();
      }
      current_batch_size += s.size();
      batch.buffer.push_back(move(s));
    }

    auto terminate_batch() -> void {
      auto &batch = batches.current_write();
      for (uint i = batch->string_indexes.size(); i < num_readers + 1; ++i) {
        batch->string_indexes.push_back(batch->buffer.size());
        batch->char_indexes.push_back(0);
        batch->cumulative_char_indexes.push_back(current_batch_size);
      }
      batches.step_write();
      reset_sequence_batch(batches.current_write());
      semaphore.release();
    }

    auto start_new_batch() -> void {
      current_batch_size = 0;
      chars_to_next_index = chars_per_reader;
    }

  private:
    auto reset_sequence_batch(shared_ptr<StringSequenceBatch> &batch) -> void {
      batch->buffer.resize(0);
      batch->string_indexes.resize(1);
      batch->char_indexes.resize(1);
      batch->cumulative_char_indexes.resize(1);
    }

  public:
    bool operator>>(shared_ptr<const StringSequenceBatch> &sb) {
      semaphore.acquire();
      if (no_more_sequences()) { return false; }
      sb = batches.current_read();
      batches.step_read();
      return true;
    }

    auto set_finished_reading() -> void {
      finished_reading = true;
      semaphore.release();
    }

  private:
    auto no_more_sequences() -> bool {
      return finished_reading && batches.empty();
    }
};

}
#endif
