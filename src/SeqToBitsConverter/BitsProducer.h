#ifndef BITS_PRODUCER_H
#define BITS_PRODUCER_H

/**
 * @file BitsProducer.h
 * @brief Produces a list of u64 bit vectors
 */

#include <algorithm>
#include <memory>

#include "BatchObjects/BitSeqBatch.h"
#include "Tools/MathUtils.hpp"
#include "Tools/SharedBatchesProducer.hpp"
#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

class ContinuousSeqToBitsConverter;

using design_utils::SharedBatchesProducer;
using math_utils::round_up;
using std::fill;
using std::make_shared;
using std::shared_ptr;

class BitsProducer: public SharedBatchesProducer<BitSeqBatch> {
  friend ContinuousSeqToBitsConverter;
  u64 max_chars_per_batch;

public:
  BitsProducer(u64 max_chars_per_batch_, u64 max_batches);

  auto static get_bits_per_element() -> u64;

private:
  auto get_default_value() -> shared_ptr<BitSeqBatch> override;
  auto start_new_batch(u64 num_chars) -> void;
  auto set(u64 index, u64 value) -> void;
};

}  // namespace sbwt_search

#endif
