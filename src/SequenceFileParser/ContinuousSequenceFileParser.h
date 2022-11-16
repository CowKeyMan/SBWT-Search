#ifndef CONTINUOUS_SEQUENCE_FILE_PARSER_H
#define CONTINUOUS_SEQUENCE_FILE_PARSER_H

/**
 * @file ContinuousSequenceFileParser.h
 * @brief Continuously reads sequences from a file or multiple files into a
 * buffer. Then it can serve these sequences to its consumers
 * */

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "Utils/SharedBatchesProducer.hpp"
#include "Utils/TypeDefinitions.h"
#include "kseqpp_read.hpp"

using reklibpp::Seq;
using reklibpp::SeqStreamIn;
using std::shared_ptr;
using std::string;
using std::unique_ptr;
using std::vector;
using structure_utils::CircularBuffer;

namespace sbwt_search {

class StringSequenceBatchProducer;
class StringBreakBatchProducer;
class IntervalBatchProducer;

class ContinuousSequenceFileParser {
  private:
    const size_t max_chars_per_batch;
    const vector<string> filenames;
    unique_ptr<SeqStreamIn> stream;
    vector<string>::const_iterator filename_iterator;
    uint batch_id = 0;
    uint kmer_size = 0;
    bool fail = false;
    shared_ptr<StringSequenceBatchProducer> string_sequence_batch_producer;
    shared_ptr<StringBreakBatchProducer> string_break_batch_producer;
    shared_ptr<IntervalBatchProducer> interval_batch_producer;
    CircularBuffer<shared_ptr<Seq>> batches;

  public:
    ContinuousSequenceFileParser(
      const vector<string> &_filenames,
      const uint _kmer_size,
      const size_t _max_chars_per_batch,
      const size_t max_batches,
      shared_ptr<StringSequenceBatchProducer> string_sequence_batch_producer,
      shared_ptr<StringBreakBatchProducer> string_break_batch_producer,
      shared_ptr<IntervalBatchProducer> interval_batch_producer
    );
    auto read_and_generate() -> void;

  private:
    auto start_next_file() -> bool;
    auto read_next() -> void;
    auto reset_rec() -> void;
    auto do_at_batch_start() -> void;
    auto do_at_batch_finish() -> void;
    auto do_at_generate_finish() -> void;
};

}  // namespace sbwt_search
#endif
