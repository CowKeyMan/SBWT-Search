#ifndef INVALID_CHARS_PRODUCER_HPP
#define INVALID_CHARS_PRODUCER_HPP

/**
 * @file InvalidCharsProducer.hpp
 * @brief Produces a list of booleans which tell wether a character is valid or
 * not
 * */

#include <algorithm>

#include "BatchObjects/InvalidCharsBatch.h"
#include "Utils/SharedBatchesProducer.hpp"

using design_utils::SharedBatchesProducer;
using std::fill;

namespace sbwt_search {
template <class StringSequenceBatchProducer>
class ContinuousSeqToBitsConverter;
}  // namespace sbwt_search

namespace sbwt_search {

template <class StringSequenceBatchProducer>
class InvalidCharsProducer: public SharedBatchesProducer<InvalidCharsBatch> {
    friend ContinuousSeqToBitsConverter<StringSequenceBatchProducer>;
    const uint kmer_size;
    const size_t max_chars_per_batch;

    public:
        InvalidCharsProducer(
          const uint kmer_size,
          const size_t max_chars_per_batch,
          const uint max_batches
        ):
        kmer_size(kmer_size),
        max_chars_per_batch(max_chars_per_batch),
        SharedBatchesProducer<InvalidCharsBatch>(max_batches) {
      initialise_batches();
    }

  private:
    auto get_default_value() -> shared_ptr<InvalidCharsBatch> override {
      auto batch = make_shared<InvalidCharsBatch>();
      batch->invalid_chars.resize(max_chars_per_batch + kmer_size);
      return batch;
    }

    auto start_new_batch(size_t num_chars) -> void {
      SharedBatchesProducer<InvalidCharsBatch>::do_at_batch_start();
      batches.current_write()->invalid_chars.resize(num_chars + kmer_size);
      fill(
        batches.current_write()->invalid_chars.begin(),
        batches.current_write()->invalid_chars.end(),
        0
      );
    }

    auto set(size_t index) {
      batches.current_write()->invalid_chars[index] = true;
    }
};

}  // namespace sbwt_search

#endif
