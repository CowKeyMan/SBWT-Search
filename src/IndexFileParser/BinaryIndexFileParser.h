#ifndef BINARY_INDEX_FILE_PARSER_H
#define BINARY_INDEX_FILE_PARSER_H

/**
 * @file BinaryIndexFileParser.h
 * @brief Reads the input binary file integer by integer, and pads each
 * line/read to the given parameter. It also takes note of the starting index of
 * where each read starts in our vector of integers.
 */

#include <memory>
#include <sstream>

#include "IndexFileParser/IndexFileParser.h"
#include "Tools/IOUtils.h"

namespace sbwt_search {

using io_utils::ThrowingIfstream;
using std::shared_ptr;
using std::string;
using std::stringstream;

class BinaryIndexFileParser: public IndexFileParser {
private:
  vector<u64> buffer;
  u64 buffer_size = 0;
  u64 buffer_index = 0;
  bool new_read = false;

public:
  BinaryIndexFileParser(
    shared_ptr<ThrowingIfstream> in_stream_,
    u64 max_indexes_,
    u64 warp_size_,
    u64 buffer_size = sixteen_kB / sizeof(u64)
  );
  auto generate_batch(
    shared_ptr<ReadStatisticsBatch> read_statistics_batch_,
    shared_ptr<WarpsBeforeNewReadBatch> warps_before_new_read_batch_,
    shared_ptr<IndexesBatch> indexes_batch_
  ) -> bool override;

private:
  auto load_buffer() -> void;
  auto assert_version() -> void;
  auto get_next() -> u64;
};

}  // namespace sbwt_search

#endif
