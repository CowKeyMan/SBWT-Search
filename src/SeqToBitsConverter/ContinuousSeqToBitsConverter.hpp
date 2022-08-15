#ifndef CONTINUOUS_SEQ_TO_BITS_CONVERTER_HPP
#define CONTINUOUS_SEQ_TO_BITS_CONVERTER_HPP

/**
 * @file ContinuousSeqToBitsConverter.hpp
 * @brief Class for converting char sequences continuously, with parallel
 * capabilities
 * */

#include <algorithm>
#include <iterator>
#include <list>
#include <memory>
#include <tuple>

#include "BatchObjects/StringSequenceBatch.hpp"
#include "SeqToBitsConverter/CharToBits.h"
#include "Utils/BoundedSemaphore.hpp"
#include "Utils/CircularBuffer.hpp"
#include "Utils/MathUtils.hpp"
#include "Utils/TypeDefinitions.h"

using std::list;
using std::make_tuple;
using std::next;
using std::shared_ptr;
using std::string;
using std::unique_ptr;
using std::vector;
using structure_utils::CircularBuffer;
using threading_utils::BoundedSemaphore;

namespace sbwt_search {

template <class StringSequenceBatchProducer>
class ContinuousSeqToBitsConverter {
  private:
    shared_ptr<StringSequenceBatchProducer> producer;
    CircularBuffer<vector<u64>> write_batches;
    BoundedSemaphore batch_semaphore;
    const uint threads;
    const CharToBits char_to_bits;
    bool finished = false;

  public:
    ContinuousSeqToBitsConverter(
      shared_ptr<StringSequenceBatchProducer> producer,
      uint threads,
      u64 max_ints_per_batch = 999,
      u64 max_batches = 10
    ):
        producer(producer),
        threads(threads),
        batch_semaphore(0, max_batches),
        char_to_bits(),
        write_batches(max_batches + 1, vector<u64>(max_ints_per_batch)) {}

  public:
    void read_and_generate() {
      omp_set_nested(1);
      shared_ptr<const StringSequenceBatch> read_batch;
      while (*producer >> read_batch) {
        vector<u64> &write_batch = write_batches.current_write();
        write_batch.resize(
          round_up<u64>(read_batch->cumulative_char_indexes.back(), 32) / 32
        );
#pragma omp parallel num_threads(threads) default(none) \
  shared(read_batch, write_batch)
        {
          uint idx = omp_get_thread_num();
          auto char_index = read_batch->char_indexes[idx];
          auto string_index = read_batch->string_indexes[idx];
          auto cumulative_char_index = read_batch->cumulative_char_indexes[idx];
          const auto next_cumulative_char_index
            = read_batch->cumulative_char_indexes[idx + 1];
          auto write_index = cumulative_char_index / 32;
          while (cumulative_char_index < next_cumulative_char_index) {
            write_batch[write_index]
              = convert_int(read_batch->buffer, string_index, char_index);
            cumulative_char_index += 32;
            ++write_index;
          }
        }
        write_batches.step_write();
        batch_semaphore.release();
      }
      finished = true;
      batch_semaphore.release();
    }

    u64 convert_int(
      const vector<string> &buffer, u64 &string_index, u64 &char_index
    ) {
      u64 result = 0;
      for (u64 internal_shift = 62; internal_shift < 64; internal_shift -= 2) {
        if (end_of_string(buffer, string_index, char_index)) {
          ++string_index;
          char_index = 0;
        }
        if (string_index == buffer.size()) { return result; }
        char c = buffer[string_index][char_index++];
        result |= (char_to_bits(c) << internal_shift);
      }
      return result;
    }

  private:
    bool end_of_string(
      const vector<string> &buffer, const u64 string_index, const u64 char_index
    ) {
      return char_index == buffer[string_index].size();
    }

  public:
    bool operator>>(vector<u64> &batch) {
      batch_semaphore.acquire();
      if (finished && write_batches.empty()) { return false; }
      batch = write_batches.current_read();
      write_batches.step_read();
      return true;
    }
};
}

#endif
