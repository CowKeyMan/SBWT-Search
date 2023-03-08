#ifndef CONTINUOUS_SEQ_TO_BITS_CONVERTER_H
#define CONTINUOUS_SEQ_TO_BITS_CONVERTER_H

/**
 * @file ContinuousSeqToBitsConverter.h
 * @brief Class for converting char sequences continuously, with parallel
 * capabilities
 */

#include <string>

#include "BatchObjects/StringSequenceBatch.h"
#include "SeqToBitsConverter/BitsProducer.h"
#include "SeqToBitsConverter/CharToBits.h"
#include "SeqToBitsConverter/InvalidCharsProducer.h"
#include "SequenceFileParser/StringSequenceBatchProducer.h"
#include "Tools/SharedBatchesProducer.hpp"

namespace sbwt_search {

using design_utils::SharedBatchesProducer;
using std::string;

class ContinuousSeqToBitsConverter {
private:
  shared_ptr<SharedBatchesProducer<StringSequenceBatch>> producer;
  shared_ptr<InvalidCharsProducer> invalid_chars_producer;
  shared_ptr<BitsProducer> bits_producer;
  u64 threads;
  CharToBits char_to_bits;

public:
  ContinuousSeqToBitsConverter(
    shared_ptr<SharedBatchesProducer<StringSequenceBatch>> producer,
    u64 threads,
    u64 kmer_size,
    u64 max_chars_per_batch,
    u64 max_batches
  );

  [[nodiscard]] auto get_invalid_chars_producer() const
    -> const shared_ptr<InvalidCharsProducer> &;
  [[nodiscard]] auto get_bits_producer() const
    -> const shared_ptr<BitsProducer> &;
  auto read_and_generate() -> void;

private:
  auto parallel_generate(StringSequenceBatch &read_batch) -> void;
  auto convert_int(const vector<char> &str, u64 start_index, u64 end_index)
    -> u64;
};

}  // namespace sbwt_search

#endif
