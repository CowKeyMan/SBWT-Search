#ifndef INVALID_CHARS_PRODUCER_H
#define INVALID_CHARS_PRODUCER_H

/**
 * @file InvalidCharsProducer.h
 * @brief Produces a list of booleans which tell wether a character is valid or
 * not
 */

#include <memory>

#include "BatchObjects/InvalidCharsBatch.h"
#include "Tools/SharedBatchesProducer.hpp"
#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

class ContinuousSeqToBitsConverter;

using design_utils::SharedBatchesProducer;
using std::shared_ptr;

class InvalidCharsProducer: public SharedBatchesProducer<InvalidCharsBatch> {
  friend ContinuousSeqToBitsConverter;
  u64 kmer_size;
  u64 max_chars_per_batch;

public:
  InvalidCharsProducer(
    u64 kmer_size_, u64 max_chars_per_batch_, u64 max_batches
  );

  auto static get_bits_per_element() -> u64;

private:
  auto get_default_value() -> shared_ptr<InvalidCharsBatch> override;
  auto start_new_batch(u64 num_chars) -> void;
  auto set(u64 index) -> void;
};

}  // namespace sbwt_search

#endif
