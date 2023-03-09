#ifndef CONTINUOUS_INDEX_FILE_PARSER_H
#define CONTINUOUS_INDEX_FILE_PARSER_H

/**
 * @file ContinuousIndexFileParser.h
 * @brief Reads a list of files one by one, filling in the batches producer as
 * it goes along. Uses the sub IndexFileParsers to do its parsing for it
 */

#include <memory>

#include "IndexFileParser/ColorsIntervalBatchProducer.h"
#include "IndexFileParser/IndexFileParser.h"
#include "IndexFileParser/IndexesBatchProducer.h"
#include "IndexFileParser/ReadStatisticsBatchProducer.h"
#include "IndexFileParser/WarpsBeforeNewReadBatchProducer.h"

namespace sbwt_search {

using std::shared_ptr;
using std::unique_ptr;
using std::vector;

class ContinuousIndexFileParser {
private:
  vector<shared_ptr<vector<u64>>> warps_before_new_read;
  shared_ptr<ColorsIntervalBatchProducer> colors_interval_batch_producer;
  shared_ptr<ReadStatisticsBatchProducer> read_statistics_batch_producer;
  shared_ptr<WarpsBeforeNewReadBatchProducer>
    warps_before_new_read_batch_producer;
  shared_ptr<IndexesBatchProducer> indexes_batch_producer;

  vector<string> filenames;
  vector<string>::iterator filename_iterator;
  u64 batch_id = 0;
  bool fail = false;
  unique_ptr<IndexFileParser> index_file_parser;
  u64 max_indexes_per_batch;
  u64 max_reads_per_batch;
  u64 warp_size;

public:
  ContinuousIndexFileParser(
    u64 max_batches,
    u64 max_indexes_per_batch_,
    u64 max_reads_per_batch_,
    u64 warp_size_,
    vector<string> filenames_
  );

  [[nodiscard]] auto get_colors_interval_batch_producer() const
    -> const shared_ptr<ColorsIntervalBatchProducer> &;
  [[nodiscard]] auto get_read_statistics_batch_producer() const
    -> const shared_ptr<ReadStatisticsBatchProducer> &;
  [[nodiscard]] auto get_warps_before_new_read_batch_producer() const
    -> const shared_ptr<WarpsBeforeNewReadBatchProducer> &;
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
  [[nodiscard]] auto create_warps_before_new_read(u64 amount) const
    -> vector<shared_ptr<vector<u64>>>;
  auto reset_batches() -> void;
};

}  // namespace sbwt_search

#endif
