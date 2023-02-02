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
#include "BatchObjects/IndexesStartsBatch.h"
#include "Tools/IOUtils.h"
#include "Tools/SharedBatchesProducer.hpp"

namespace sbwt_search {

using design_utils::SharedBatchesProducer;
using io_utils::ThrowingIfstream;
using std::shared_ptr;

class IndexFileParser {
private:
  shared_ptr<ThrowingIfstream> in_stream;
  shared_ptr<IndexesBatch> indexes;
  shared_ptr<IndexesStartsBatch> indexes_starts_batch;
  size_t max_indexes;
  size_t read_padding;

protected:
  [[nodiscard]] auto get_istream() const -> ThrowingIfstream &;
  [[nodiscard]] auto get_indexes() const -> vector<u64> &;
  [[nodiscard]] auto get_max_indexes() const -> u64;
  [[nodiscard]] auto get_starts() const -> vector<u64> &;
  [[nodiscard]] auto get_read_padding() const -> u64;

public:
  IndexFileParser(
    shared_ptr<ThrowingIfstream> in_stream_,
    shared_ptr<IndexesBatch> indexes_,
    shared_ptr<IndexesStartsBatch> indexes_starts_batch_,
    size_t max_indexes_,
    size_t read_padding_
  );
  virtual auto generate_batch() -> void = 0;
  virtual ~IndexFileParser() = default;
  IndexFileParser(IndexFileParser &) = delete;
  IndexFileParser(IndexFileParser &&) = delete;
  auto operator=(IndexFileParser &) = delete;
  auto operator=(IndexFileParser &&) = delete;
};

}  // namespace sbwt_search

#endif
