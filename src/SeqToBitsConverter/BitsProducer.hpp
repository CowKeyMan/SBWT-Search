#ifndef BITS_PRODUCER_HPP
#define BITS_PRODUCER_HPP

/**
 * @file BitsProducer.hpp
 * @brief Produces a list of u64 bit vectors
 * not
 * */

#include <algorithm>

#include "BatchObjects/BitSeqBatch.h"
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
class BitsProducer: public SharedBatchesProducer<BitSeqBatch> {
    friend ContinuousSeqToBitsConverter<StringSequenceBatchProducer>;
    const u64 max_chars_per_batch;

    BitsProducer(const u64 max_chars_per_batch, const uint max_batches):
        max_chars_per_batch(max_chars_per_batch),
        SharedBatchesProducer<BitSeqBatch>(max_batches) {
      initialise_batches();
    }

    auto get_default_value() -> shared_ptr<BitSeqBatch> override {
      auto batch = make_shared<BitSeqBatch>();
      batch->bit_seq.resize(round_up<u64>(max_chars_per_batch, 32) / 32);
      return batch;
    }

    auto start_new_batch(u64 num_chars) -> void {
      SharedBatchesProducer<BitSeqBatch>::do_at_batch_start();
      batches.current_write()->bit_seq.resize(
        round_up<u64>(num_chars, 32) / 32
      );
    }

    auto set(u64 index, u64 value) {
      batches.current_write()->bit_seq[index] = value;
    }
};

}  // namespace sbwt_search

#endif
