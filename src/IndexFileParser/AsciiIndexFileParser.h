#ifndef ASCII_INDEX_FILE_PARSER_H
#define ASCII_INDEX_FILE_PARSER_H

/**
 * @file AsciiIndexFileParser.h
 * @brief Reads the input text file integer by integer, and pads each line/read
 * to the given parameter. It also takes note of the starting index of where
 * each read starts in our vector of integers.
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

class AsciiIndexFileParser: public IndexFileParser {
private:
  string buffer;
  u64 buffer_size = 0;
  u64 buffer_index = 0;
  bool new_read = false;

public:
  AsciiIndexFileParser(
    shared_ptr<ThrowingIfstream> in_stream_,
    u64 max_indexes_,
    u64 warp_size_,
    u64 buffer_size = sixteen_kB
  );
  auto generate_batch(
    shared_ptr<ReadStatisticsBatch> read_statistics_batch_,
    shared_ptr<WarpsBeforeNewReadBatch> warps_before_new_read_batch_,
    shared_ptr<IndexesBatch> indexes_batch_
  ) -> bool override;

private:
  auto load_buffer() -> void;
  auto assert_version() -> void;
  auto skip_until_next_whitespace() -> char;
  auto getc() -> char;
  auto read_number(u64 starting_number) -> u64;
};

}  // namespace sbwt_search

#endif
