#ifndef CONTINUOUS_SEQ_TO_BITS_CONVERTER_HPP
#define CONTINUOUS_SEQ_TO_BITS_CONVERTER_HPP

/**
 * @file ContinuousSeqToBitsConverter.hpp
 * @brief Class for converting char sequences continuously, with parallel
 * capabilities
 * */

#include <algorithm>
#include <memory>
#include <omp.h>
#include <string>
#include <vector>

#include <ext/alloc_traits.h>

#include "BatchObjects/StringSequenceBatch.h"
#include "SeqToBitsConverter/CharToBits.h"
#include "Utils/BoundedSemaphore.hpp"
#include "Utils/CircularBuffer.hpp"
#include "Utils/MathUtils.hpp"
#include "Utils/TypeDefinitions.h"

using math_utils::round_up;
using std::fill;
using std::make_shared;
using std::shared_ptr;
using std::string;
using std::vector;
using structure_utils::CircularBuffer;
using threading_utils::BoundedSemaphore;

namespace sbwt_search {

template <class StringSequenceBatchProducer>
class ContinuousSeqToBitsConverter {
  private:
    shared_ptr<StringSequenceBatchProducer> producer;
    CircularBuffer<shared_ptr<vector<u64>>> bit_batches;
    CircularBuffer<shared_ptr<vector<char>>> invalid_batches;
    BoundedSemaphore bit_semaphore;
    BoundedSemaphore invalid_semaphore;
    const uint threads;
    const uint kmer_size;
    const CharToBits char_to_bits;
    bool finished = false;

  public:
    ContinuousSeqToBitsConverter(
      shared_ptr<StringSequenceBatchProducer> producer,
      uint threads,
      uint kmer_size,
      u64 max_ints_per_batch = 999,
      u64 max_batches = 10
    ):
        producer(producer),
        threads(threads),
        kmer_size(kmer_size),
        bit_semaphore(0, max_batches),
        invalid_semaphore(0, max_batches),
        char_to_bits(),
        bit_batches(max_batches + 1),
        invalid_batches(max_batches + 1) {
      for (uint i = 0; i < bit_batches.size(); ++i) {
        bit_batches.set(i, make_shared<vector<u64>>(max_ints_per_batch));
        invalid_batches.set(
          i, make_shared<vector<char>>(max_ints_per_batch + kmer_size)
        );
      }
    }

  public:
    auto read_and_generate() -> void {
      shared_ptr<StringSequenceBatch> read_batch;
      while (*producer >> read_batch) {
        bit_batches.current_write()->resize(
          round_up<u64>(read_batch->cumulative_char_indexes.back(), 32) / 32
        );
        invalid_batches.current_write()->resize(
          read_batch->cumulative_char_indexes.back()
        );
        fill(
          invalid_batches.current_write()->begin(),
          invalid_batches.current_write()->end(),
          0
        );
#pragma omp parallel num_threads(threads) default(none) shared(read_batch)
        {
          uint idx = omp_get_thread_num();
          auto char_index = read_batch->char_indexes[idx];
          auto string_index = read_batch->string_indexes[idx];
          auto cumulative_char_index = read_batch->cumulative_char_indexes[idx];
          const auto next_cumulative_char_index
            = read_batch->cumulative_char_indexes[idx + 1];
          auto write_index = cumulative_char_index / 32;
          while (cumulative_char_index < next_cumulative_char_index) {
            (*bit_batches.current_write())[write_index] = convert_int(
              read_batch->buffer,
              string_index,
              char_index,
              cumulative_char_index
            );
            cumulative_char_index += 32;
            ++write_index;
          }
        }
        invalid_batches.step_write();
        invalid_semaphore.release();
        bit_batches.step_write();
        bit_semaphore.release();
      }
      finished = true;
      bit_semaphore.release();
      invalid_semaphore.release();
    }

    auto convert_int(
      const vector<string> &buffer,
      u64 &string_index,
      u64 &char_index,
      u64 start_index
    ) -> u64 {
      u64 result = 0;
      for (u64 internal_shift = 62, index = 0; internal_shift < 64;
           internal_shift -= 2, ++index) {
        if (end_of_string(buffer, string_index, char_index)) {
          ++string_index;
          char_index = 0;
        }
        if (string_index >= buffer.size()) { return result; }
        u64 c = char_to_bits(buffer[string_index][char_index++]);
        if (c == invalid_char_to_bits_value) {
          (*invalid_batches.current_write())[index + start_index] = 1;
          continue;
        }
        result |= c << internal_shift;
      }
      return result;
    }

  private:
    auto end_of_string(
      const vector<string> &buffer, const u64 string_index, const u64 char_index
    ) -> bool {
      return char_index == buffer[string_index].size();
    }

  public:
    auto operator>>(shared_ptr<vector<u64>> &batch) -> bool {
      bit_semaphore.acquire();
      if (finished && bit_batches.empty()) { return false; }
      batch = bit_batches.current_read();
      bit_batches.step_read();
      return true;
    }

    auto operator>>(shared_ptr<vector<char>> &batch) -> bool {
      invalid_semaphore.acquire();
      if (finished && invalid_batches.empty()) { return false; }
      batch = invalid_batches.current_read();
      invalid_batches.step_read();
      return true;
    }
};
}  // namespace sbwt_search

#endif
