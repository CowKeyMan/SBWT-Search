#ifndef CONTINUOUS_SEQ_TO_BITS_CONVERTER_HPP
#define CONTINUOUS_SEQ_TO_BITS_CONVERTER_HPP

/**
 * @file ContinuousSeqToBitsConverter.hpp
 * @brief Class for converting character sequences continuously, with parallel
 * capabilities
 * */

#include <algorithm>
#include <iterator>
#include <list>
#include <memory>
#include <queue>
#include <tuple>

#include "BatchObjects/StringSequenceBatch.hpp"
#include "SeqToBitsConverter/CharToBits.h"
#include "Utils/BoundedSemaphore.hpp"
#include "Utils/MathUtils.hpp"
#include "Utils/Semaphore.hpp"
#include "Utils/TypeDefinitions.h"

using std::list;
using std::make_tuple;
using std::next;
using std::queue;
using std::unique_ptr;
using std::shared_ptr;
using std::string;
using std::vector;
using threading_utils::BoundedSemaphore;

namespace sbwt_search {

template <class StringSequenceBatchProducer>
class ContinuousSeqToBitsConverter {
  private:
    shared_ptr<StringSequenceBatchProducer> producer;
    queue<vector<u64>> write_batches;
    BoundedSemaphore batch_semaphore;
    const uint threads;
    const CharToBits char_to_bits;
    const u64 max_ints_per_batch;
    bool finished = 0;

  public:
    ContinuousSeqToBitsConverter(
      shared_ptr<StringSequenceBatchProducer> producer,
      uint threads,
      u64 max_ints_per_batch = 999,
      u64 max_batches = UINT_MAX
    ):
        producer(producer),
        threads(threads),
        max_ints_per_batch(max_ints_per_batch),
        batch_semaphore(0, max_batches),
        char_to_bits() {}

  public:
    void read_and_generate() {
      omp_set_nested(1);
      unique_ptr<StringSequenceBatch> read_batch;
      vector<u64> write_batch;
      while (*producer >> read_batch) {
        write_batch = vector<u64>(max_ints_per_batch);
#pragma omp parallel num_threads(threads) default(none) \
  shared(read_batch, write_batch)
        {
          uint idx = omp_get_thread_num();
          auto character_index = read_batch->character_indexes[idx];
          auto string_index = read_batch->string_indexes[idx];
          auto cumulative_character_index
            = read_batch->cumulative_character_indexes[idx];
          const auto next_cumulative_character_index
            = read_batch->cumulative_character_indexes[idx + 1];
          auto write_index = cumulative_character_index / 32;
          while (cumulative_character_index < next_cumulative_character_index) {
            write_batch[write_index]
              = convert_int(read_batch->buffer, string_index, character_index);
            cumulative_character_index += 32;
            ++write_index;
          }
        }
        write_batch.resize(
          round_up<u64>(read_batch->cumulative_character_indexes.back(), 32)
          / 32
        );
        write_batches.push(move(write_batch));
        batch_semaphore.release();
      }
      finished = true;
      batch_semaphore.release();
    }

    u64 convert_int(
      vector<string> &buffer, u64 &string_index, u64 &character_index
    ) {
      u64 result = 0;
      for (u64 internal_shift = 62; internal_shift < 64; internal_shift -= 2) {
        if (end_of_string(buffer, string_index, character_index)) {
          ++string_index;
          character_index = 0;
        }
        if (string_index == buffer.size()) { return result; }
        char c = buffer[string_index][character_index++];
        result |= (char_to_bits(c) << internal_shift);
      }
      return result;
    }

  private:
    bool end_of_string(
      vector<string> &buffer, u64 string_index, u64 character_index
    ) {
      return character_index == buffer[string_index].size();
    }

  public:
    bool operator>>(vector<u64> &batch) {
      batch_semaphore.acquire();
      if (finished && write_batches.size() == 0) { return false; }
      batch = move(write_batches.front());
      write_batches.pop();
      return true;
    }
};
}

#endif
