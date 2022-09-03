#ifndef BITS_PRODUCER_HPP
#define BITS_PRODUCER_HPP

/**
 * @file BitsProducer.hpp
 * @brief Produces a list of u64 bit vectors
 * not
 * */

#include <algorithm>

#include "Utils/MathUtils.hpp"
#include "Utils/SharedBatchesProducer.hpp"
#include "Utils/TypeDefinitions.h"

using design_utils::SharedBatchesProducer;
using math_utils::round_up;
using std::fill;

namespace sbwt_search {
template <class StringSequenceBatchProducer>
class ContinuousSeqToBitsConverter;
}  // namespace sbwt_search

namespace sbwt_search {

template <class StringSequenceBatchProducer>
class BitsProducer: public SharedBatchesProducer<vector<u64>> {
    friend ContinuousSeqToBitsConverter<StringSequenceBatchProducer>;
    const u64 max_chars_per_batch;

    BitsProducer(const u64 max_chars_per_batch, const uint max_batches):
        max_chars_per_batch(max_chars_per_batch),
        SharedBatchesProducer<vector<u64>>(max_batches) {
      initialise_batches();
    }

    auto get_default_value() -> shared_ptr<vector<u64>> override {
      return make_shared<vector<u64>>(
        round_up<u64>(max_chars_per_batch, 32) / 32
      );
    }

    auto start_new_batch(u64 num_chars) -> void {
      SharedBatchesProducer<vector<u64>>::do_at_batch_start();
      batches.current_write()->resize(round_up<u64>(num_chars, 32) / 32);
    }

    auto set(u64 index, u64 value) {
      (*batches.current_write())[index] = value;
    }
};

}  // namespace sbwt_search

#endif
