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

namespace sbwt_search {

class ContinuousSeqToBitsConverter;

using design_utils::SharedBatchesProducer;
using std::shared_ptr;

class InvalidCharsProducer: public SharedBatchesProducer<InvalidCharsBatch> {
  friend ContinuousSeqToBitsConverter;
  uint kmer_size;
  size_t max_chars_per_batch;

public:
  InvalidCharsProducer(
    uint kmer_size_, size_t max_chars_per_batch_, uint max_batches
  );

private:
  auto get_default_value() -> shared_ptr<InvalidCharsBatch> override;
  auto start_new_batch(size_t num_chars) -> void;
  auto set(size_t index) -> void;
};

}  // namespace sbwt_search

#endif
