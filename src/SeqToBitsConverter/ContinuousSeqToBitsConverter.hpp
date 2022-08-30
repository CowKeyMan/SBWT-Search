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

#include "BatchObjects/StringSequenceBatch.h"
#include "SeqToBitsConverter/BitsProducer.hpp"
#include "SeqToBitsConverter/CharToBits.h"
#include "SeqToBitsConverter/InvalidCharsProducer.hpp"
#include "Utils/Logger.h"
#include "Utils/MathUtils.hpp"
#include "Utils/SharedBatchesProducer.hpp"
#include "Utils/TypeDefinitions.h"
#include "fmt/core.h"

using design_utils::SharedBatchesProducer;
using fmt::format;
using log_utils::Logger;
using math_utils::round_up;
using std::fill;
using std::make_shared;
using std::shared_ptr;
using std::string;
using std::vector;

namespace sbwt_search {

template <class StringSequenceBatchProducer>
class ContinuousSeqToBitsConverter {
  private:
    InvalidCharsProducer<StringSequenceBatchProducer> invalid_chars_producer;
    BitsProducer<StringSequenceBatchProducer> bits_producer;

    shared_ptr<StringSequenceBatchProducer> producer;
    const uint threads;
    const CharToBits char_to_bits;

  public:
    ContinuousSeqToBitsConverter(
      shared_ptr<StringSequenceBatchProducer> producer,
      uint threads,
      uint kmer_size,
      u64 max_chars_per_batch = 999,
      u64 max_batches = 10
    ):
        producer(producer),
        threads(threads),
        char_to_bits(),
        invalid_chars_producer(kmer_size, max_chars_per_batch, max_batches),
        bits_producer(max_chars_per_batch, max_batches) {}

  public:
    auto read_and_generate() -> void {
      shared_ptr<StringSequenceBatch> read_batch;
      for (uint batch_idx = 0; *producer >> read_batch; ++batch_idx) {
        invalid_chars_producer.start_new_batch(
          read_batch->cumulative_char_indexes.back()
        );
        bits_producer.start_new_batch(read_batch->cumulative_char_indexes.back()
        );
        Logger::log_timed_event(
          "SeqToBitsConverter",
          Logger::EVENT_STATE::START,
          format("batch {}", batch_idx)
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
            bits_producer.set(
              write_index,
              convert_int(
                read_batch->buffer,
                string_index,
                char_index,
                cumulative_char_index
              )
            );
            cumulative_char_index += 32;
            ++write_index;
          }
        }
        Logger::log_timed_event(
          "SeqToBitsConverter",
          Logger::EVENT_STATE::STOP,
          format("batch {}", batch_idx)
        );
        invalid_chars_producer.do_at_batch_finish();
        bits_producer.do_at_batch_finish();
      }
      invalid_chars_producer.do_at_generate_finish();
      bits_producer.do_at_generate_finish();
    }

    auto operator>>(shared_ptr<vector<u64>> &batch) -> bool {
      return bits_producer >> batch;
    }

    auto operator>>(shared_ptr<vector<char>> &batch) -> bool {
      return invalid_chars_producer >> batch;
    }

  private:
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
          invalid_chars_producer.set(index + start_index, 1);
          continue;
        }
        result |= c << internal_shift;
      }
      return result;
    }

    auto end_of_string(
      const vector<string> &buffer, const u64 string_index, const u64 char_index
    ) -> bool {
      return char_index == buffer[string_index].size();
    }
};
}  // namespace sbwt_search

#endif
