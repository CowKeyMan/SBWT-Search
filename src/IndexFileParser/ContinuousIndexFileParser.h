#ifndef CONTINUOUS_INDEX_FILE_PARSER_H
#define CONTINUOUS_INDEX_FILE_PARSER_H

/**
 * @file ContinuousIndexFileParser.h
 * @brief Reads a list of files one by one, filling in the batches producer as
 * it goes along. Uses the sub IndexFileParsers to do its parsing for it
 */

#include <memory>
#include <span>

#include "IndexFileParser/IndexFileParser.h"
#include "IndexFileParser/IndexesBatchProducer.h"
#include "IndexFileParser/IndexesBeforeNewfileBatchProducer.h"
#include "IndexFileParser/IndexesStartsBatchProducer.h"

namespace sbwt_search {

using std::shared_ptr;
using std::span;
using std::unique_ptr;

class ContinuousIndexFileParser {
private:
  shared_ptr<IndexesBatchProducer> indexes_batch_producer;
  shared_ptr<IndexesStartsBatchProducer> indexes_starts_batch_producer;
  shared_ptr<IndexesBeforeNewfileBatchProducer>
    indexes_before_newfile_batch_producer;
  span<const string> filenames;
  span<const string>::iterator filename_iterator;
  u64 batch_id = 0;
  bool fail = false;
  unique_ptr<IndexFileParser> index_file_parser;
  u64 max_indexes_per_batch;
  u64 read_padding;

public:
  ContinuousIndexFileParser(
    u64 max_indexes_per_batch_,
    u64 max_batches,
    span<const string> filenames_,
    u64 read_padding_
  );

  [[nodiscard]] auto get_indexes_batch_producer() const
    -> const shared_ptr<IndexesBatchProducer> &;
  [[nodiscard]] auto get_indexes_starts_batch_producer() const
    -> const shared_ptr<IndexesStartsBatchProducer> &;
  [[nodiscard]] auto get_indexes_before_newfile_batch_producer() const
    -> const shared_ptr<IndexesBeforeNewfileBatchProducer> &;

  auto read_and_generate() -> void;
  auto do_at_batch_start() -> void;
  auto reset() -> void;
  auto do_at_batch_finish() -> void;
  auto do_at_generate_finish() -> void;
  auto read_next() -> void;
  auto start_next_file() -> bool;
  auto open_parser(const string &filename) -> void;
};

}  // namespace sbwt_search

#endif
