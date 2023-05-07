#ifndef CONTINUOUS_INDEX_FILE_PARSER_H
#define CONTINUOUS_INDEX_FILE_PARSER_H

/**
 * @file ContinuousIndexFileParser.h
 * @brief Reads a list of files one by one, filling in the batches producer as
 * it goes along. Uses the sub IndexFileParsers to do its parsing for it.
 * Indexes are padded to the next warp and sequence statistics are counted as
 * well.
 */

#include <memory>

#include "IndexFileParser/IndexFileParser.h"
#include "IndexFileParser/IndexesBatchProducer.h"
#include "IndexFileParser/SeqStatisticsBatchProducer.h"

namespace sbwt_search {

using std::shared_ptr;
using std::unique_ptr;
using std::vector;

class ContinuousIndexFileParser {
private:
  shared_ptr<SeqStatisticsBatchProducer> seq_statistics_batch_producer;
  shared_ptr<IndexesBatchProducer> indexes_batch_producer;

  vector<string> filenames;
  vector<string>::iterator filename_iterator;
  u64 batch_id = 0;
  bool fail = false;
  unique_ptr<IndexFileParser> index_file_parser;
  u64 max_indexes_per_batch;
  u64 max_seqs_per_batch;
  u64 warp_size;
  u64 stream_id;

public:
  ContinuousIndexFileParser(
    u64 stream_id_,
    u64 max_indexes_per_batch_,
    u64 max_seqs_per_batch_,
    u64 warp_size_,
    vector<string> filenames_,
    u64 seq_statistics_batch_producer_max_batches,
    u64 indexes_batch_producer_max_batches
  );

  [[nodiscard]] auto get_seq_statistics_batch_producer() const
    -> const shared_ptr<SeqStatisticsBatchProducer> &;
  [[nodiscard]] auto get_indexes_batch_producer() const
    -> const shared_ptr<IndexesBatchProducer> &;

  auto read_and_generate() -> void;

private:
  auto do_at_batch_start() -> void;
  auto do_at_batch_finish() -> void;
  auto do_at_generate_finish() -> void;
  auto read_next() -> void;
  auto start_next_file() -> bool;
  auto start_new_file(const string &filename) -> void;
  auto reset_batches() -> void;
};

}  // namespace sbwt_search

#endif
