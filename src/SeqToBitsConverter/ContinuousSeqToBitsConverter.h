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
  uint threads;
  CharToBits char_to_bits;

public:
  ContinuousSeqToBitsConverter(
    shared_ptr<SharedBatchesProducer<StringSequenceBatch>> producer,
    shared_ptr<InvalidCharsProducer> invalid_chars_producer_,
    shared_ptr<BitsProducer> bits_producer_,
    uint threads
  );

  auto read_and_generate() -> void;

private:
  auto parallel_generate(StringSequenceBatch &read_batch) -> void;
  auto convert_int(const string &string, size_t start_index, size_t end_index)
    -> u64;

public:
  auto operator>>(shared_ptr<BitSeqBatch> &batch) -> bool;
  auto operator>>(shared_ptr<InvalidCharsBatch> &batch) -> bool;
};

}  // namespace sbwt_search

#endif
