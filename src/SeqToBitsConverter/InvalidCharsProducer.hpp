#ifndef INVALID_CHARS_PRODUCER_HPP
#define INVALID_CHARS_PRODUCER_HPP

/**
 * @file InvalidCharsProducer.hpp
 * @brief Produces a list of booleans which tell wether a character is valid or
 * not
 * */

#include <algorithm>

#include "Utils/SharedBatchesProducer.hpp"
#include "Utils/TypeDefinitions.h"

using design_utils::SharedBatchesProducer;
using std::fill;

namespace sbwt_search {
template <class StringSequenceBatchProducer>
class ContinuousSeqToBitsConverter;
}  // namespace sbwt_search

namespace sbwt_search {

template <class StringSequenceBatchProducer>
class InvalidCharsProducer: public SharedBatchesProducer<vector<char>> {
    friend ContinuousSeqToBitsConverter<StringSequenceBatchProducer>;
    const uint kmer_size;
    const u64 max_chars_per_batch;

    InvalidCharsProducer(
      const uint kmer_size,
      const u64 max_chars_per_batch,
      const uint max_batches
    ):
        kmer_size(kmer_size),
        max_chars_per_batch(max_chars_per_batch),
        SharedBatchesProducer<vector<char>>(max_batches) {
      initialise_batches();
    }

    auto get_default_value() -> shared_ptr<vector<char>> override {
      return make_shared<vector<char>>(max_chars_per_batch + kmer_size);
    }

    auto start_new_batch(u64 num_chars) -> void {
      SharedBatchesProducer<vector<char>>::do_at_batch_start();
      batches.current_write()->resize(num_chars);
      fill(batches.current_write()->begin(), batches.current_write()->end(), 0);
    }

    auto set(u64 index) {
      (*batches.current_write())[index] = true;
    }
};

}  // namespace sbwt_search

#endif
