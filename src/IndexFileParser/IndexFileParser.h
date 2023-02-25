#ifndef INDEX_FILE_PARSER_H
#define INDEX_FILE_PARSER_H

/**
 * @file IndexFileParser.h
 * @brief Parent template class for reading the list of integers
 * provided by the indexing function. Provides a padded list of integers per
 * read and another list of indexes to indicate where each read starts in our
 * list of integers. Note: these classes expect the input to have the version
 * number as the first item, and then the contents later. The format encoded in
 * the file's header is read by another part of the code
 */

#include <fstream>
#include <memory>

#include "BatchObjects/IndexesBatch.h"
#include "BatchObjects/ReadStatisticsBatch.h"
#include "BatchObjects/WarpsBeforeNewReadBatch.h"
#include "Tools/IOUtils.h"
#include "Tools/SharedBatchesProducer.hpp"

namespace sbwt_search {

const u64 sixteen_kB = 16ULL * 8ULL * 1024ULL;
const u64 pad = static_cast<u64>(-1);

using design_utils::SharedBatchesProducer;
using io_utils::ThrowingIfstream;
using std::shared_ptr;

class IndexFileParser {
private:
  shared_ptr<ThrowingIfstream> in_stream;
  shared_ptr<ReadStatisticsBatch> read_statistics_batch;
  shared_ptr<WarpsBeforeNewReadBatch> warps_before_new_read_batch;
  shared_ptr<IndexesBatch> indexes_batch;
  u64 max_indexes;
  u64 read_padding;

protected:
  [[nodiscard]] auto get_istream() const -> ThrowingIfstream &;
  [[nodiscard]] auto get_indexes() const -> vector<u64> &;
  [[nodiscard]] auto get_max_indexes() const -> u64;
  [[nodiscard]] auto get_read_padding() const -> u64;
  IndexFileParser(
    shared_ptr<ThrowingIfstream> in_stream_, u64 max_indexes_, u64 read_padding_
  );

public:
  // return true if we manage to read from the file
  virtual auto generate_batch(
    shared_ptr<ReadStatisticsBatch> read_statistics_batch_,
    shared_ptr<WarpsBeforeNewReadBatch> warps_before_new_read_batch_,
    shared_ptr<IndexesBatch> indexes_batch_
  ) -> bool;
  virtual ~IndexFileParser() = default;
  IndexFileParser(IndexFileParser &) = delete;
  IndexFileParser(IndexFileParser &&) = delete;
  auto operator=(IndexFileParser &) = delete;
  auto operator=(IndexFileParser &&) = delete;
  auto pad_read() -> void;

protected:
  [[nodiscard]] auto get_read_statistics_batch() const
    -> const shared_ptr<ReadStatisticsBatch> &;
  [[nodiscard]] auto get_warps_before_new_read_batch() const
    -> const shared_ptr<WarpsBeforeNewReadBatch> &;
  [[nodiscard]] auto get_indexes_batch() const
    -> const shared_ptr<IndexesBatch> &;

  auto begin_new_read() -> void;
};

}  // namespace sbwt_search

#endif
