#ifndef CONTINUOUS_SEQUENCE_FILE_PARSER_H
#define CONTINUOUS_SEQUENCE_FILE_PARSER_H

/**
 * @file ContinuousSequenceFileParser.h
 * @brief Continuously reads sequences from a file or multiple files into a
 * buffer. Then it can serve these sequences to its consumers. The reading is
 * done in such a way that a single batch can contain characters from multiple
 * lines. kseqpp_REad is used for parsing the files and getting the list of
 * where each line break is.
 */

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "SequenceFileParser/IntervalBatchProducer.h"
#include "SequenceFileParser/StringBreakBatchProducer.h"
#include "SequenceFileParser/StringSequenceBatchProducer.h"
#include "Tools/SharedBatchesProducer.hpp"
#include "Tools/TypeDefinitions.h"
#include "kseqpp_read.hpp"

namespace sbwt_search {

using reklibpp::Seq;
using reklibpp::SeqStreamIn;
using std::shared_ptr;
using std::string;
using std::unique_ptr;
using std::vector;
using structure_utils::CircularBuffer;

class ContinuousSequenceFileParser {
private:
  u64 max_chars_per_batch;
  u64 max_seqs_per_batch;
  vector<string> filenames;
  unique_ptr<SeqStreamIn> stream;
  vector<string>::const_iterator filename_iterator;
  u64 batch_id = 0;
  u64 kmer_size = 0;
  bool fail = false;
  shared_ptr<StringSequenceBatchProducer> string_sequence_batch_producer;
  shared_ptr<StringBreakBatchProducer> string_break_batch_producer;
  shared_ptr<IntervalBatchProducer> interval_batch_producer;
  CircularBuffer<shared_ptr<Seq>> batches;
  u64 stream_id;

public:
  ContinuousSequenceFileParser(
    u64 stream_id,
    const vector<string> &_filenames,
    u64 _kmer_size,
    u64 max_chars_per_batch_,
    u64 max_seqs_per_batch_,
    u64 string_sequence_batch_producer_max_batches,
    u64 string_break_batch_producer_max_batches,
    u64 interval_batch_producer_max_batches
  );
  auto read_and_generate() -> void;
  [[nodiscard]] auto get_string_sequence_batch_producer() const
    -> const shared_ptr<StringSequenceBatchProducer> &;
  [[nodiscard]] auto get_string_break_batch_producer() const
    -> const shared_ptr<StringBreakBatchProducer> &;
  [[nodiscard]] auto get_interval_batch_producer() const
    -> const shared_ptr<IntervalBatchProducer> &;

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
