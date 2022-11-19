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
#include <cmath>

#include "BatchObjects/BitSeqBatch.h"
#include "BatchObjects/InvalidCharsBatch.h"
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
using std::min;
using std::shared_ptr;
using std::string;
using std::vector;

namespace sbwt_search {

template <class StringSequenceBatchProducer>
class ContinuousSeqToBitsConverter {
  private:
    shared_ptr<InvalidCharsProducer<StringSequenceBatchProducer>>
      invalid_chars_producer;
    shared_ptr<BitsProducer<StringSequenceBatchProducer>> bits_producer;

    shared_ptr<StringSequenceBatchProducer> producer;
    const uint threads;
    const CharToBits char_to_bits;

  public:
    ContinuousSeqToBitsConverter(
      shared_ptr<StringSequenceBatchProducer> producer,
      shared_ptr<InvalidCharsProducer<StringSequenceBatchProducer>>
        _invalid_chars_producer,
      shared_ptr<BitsProducer<StringSequenceBatchProducer>> _bits_producer,
      uint threads,
      u64 max_chars_per_batch,
      u64 max_batches
    ):
        producer(producer),
        threads(threads),
        char_to_bits(),
        invalid_chars_producer(_invalid_chars_producer),
        bits_producer(_bits_producer) {}

  public:
    auto read_and_generate() -> void {
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

  private:
    auto parallel_generate(StringSequenceBatch &read_batch) -> void {
      auto seq_size = read_batch.seq->size();
      size_t chars_per_thread = ceil(round_up<size_t>(seq_size, 32) / 32.0 / threads) * 32ULL;
#pragma omp parallel num_threads(threads) shared(read_batch)
      {
        uint idx = omp_get_thread_num();
        size_t start_index = min(idx * chars_per_thread, seq_size);
        size_t end_index = min((idx + 1) * chars_per_thread, seq_size);
        for (size_t index = start_index; index < end_index; index += 32) {
          auto x = convert_int(*read_batch.seq, start_index, end_index);
          bits_producer->set(
            index / 32, convert_int(*read_batch.seq, start_index, end_index)
          );
        }
      }
    }

    auto convert_int(const string &string, size_t start_index, size_t end_index)
      -> u64 {
      u64 result = 0;
      for (u64 internal_shift = 62, index = 0;
           internal_shift < 64 && index + start_index < end_index;
           internal_shift -= 2, ++index) {
        u64 c = char_to_bits(string[index + start_index]);
        if (c == invalid_char_to_bits_value) {
          invalid_chars_producer->set(index + start_index);
          continue;
        }
        result |= c << internal_shift;
      }
      return result;
    }


  public:
    auto operator>>(shared_ptr<BitSeqBatch> &batch) -> bool {
      return (*bits_producer) >> batch;
    }

    auto operator>>(shared_ptr<InvalidCharsBatch> &batch) -> bool {
      return (*invalid_chars_producer) >> batch;
    }

};  // namespace sbwt_search
}  // namespace sbwt_search

#endif
