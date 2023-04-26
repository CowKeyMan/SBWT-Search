#ifndef INDEX_FILE_PARSER_H
#define INDEX_FILE_PARSER_H

/**
 * @file IndexFileParser.h
 * @brief Parent template class for reading the list of integers
 * provided by the indexing function. Provides a padded list of integers per
 * seq and another list of indexes to indicate where each seq starts in our
 * list of integers. Note: these classes expect the input to have the version
 * number as the first item, and then the contents later. The format encoded
 * in the file's header is read by another part of the code
 */

#include <fstream>
#include <memory>

#include "BatchObjects/IndexesBatch.h"
#include "BatchObjects/SeqStatisticsBatch.h"
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
  shared_ptr<SeqStatisticsBatch> seq_statistics_batch;
  shared_ptr<IndexesBatch> indexes_batch;
  u64 max_indexes;
  u64 max_seqs;
  u64 warp_size;
  u64 colored_seq_id;

protected:
  [[nodiscard]] auto get_istream() const -> ThrowingIfstream &;
  [[nodiscard]] auto get_indexes() const -> PinnedVector<u64> &;
  [[nodiscard]] auto get_max_indexes() const -> u64;
  [[nodiscard]] auto get_max_seqs() const -> u64;
  IndexFileParser(
    shared_ptr<ThrowingIfstream> in_stream_,
    u64 max_indexes_,
    u64 max_seqs_,
    u64 warp_size
  );

public:
  // return true if we manage to seq from the file
  virtual auto generate_batch(
    shared_ptr<SeqStatisticsBatch> seq_statistics_batch_,
    shared_ptr<IndexesBatch> indexes_batch_
  ) -> bool;
  virtual ~IndexFileParser() = default;
  IndexFileParser(IndexFileParser &) = delete;
  IndexFileParser(IndexFileParser &&) = delete;
  auto operator=(IndexFileParser &) = delete;
  auto operator=(IndexFileParser &&) = delete;

protected:
  [[nodiscard]] auto get_seq_statistics_batch() const
    -> const shared_ptr<SeqStatisticsBatch> &;
  [[nodiscard]] auto get_indexes_batch() const
    -> const shared_ptr<IndexesBatch> &;

  auto end_seq() -> void;
  auto get_num_seqs() -> u64;
  auto add_warp_interval() -> void;

private:
  auto pad_warp() -> void;
  auto begin_new_seq() -> void;
};

}  // namespace sbwt_search

#endif
