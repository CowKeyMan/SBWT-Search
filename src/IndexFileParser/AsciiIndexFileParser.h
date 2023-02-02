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

#include "BatchObjects/IndexesBatch.h"
#include "BatchObjects/IndexesIntervalsBatch.h"
#include "IndexFileParser/IndexFileParser.h"
#include "Tools/IOUtils.h"

namespace sbwt_search {

using io_utils::ThrowingIfstream;
using std::shared_ptr;
using std::string;
using std::stringstream;

const size_t sixteen_kB = 16ULL * 8ULL * 1024ULL;

class AsciiIndexFileParser: IndexFileParser {
private:
  string buffer;
  size_t buffer_size = 0;
  size_t buffer_index = 0;
  size_t current_index = 0;

public:
  AsciiIndexFileParser(
    shared_ptr<ThrowingIfstream> in_stream_,
    shared_ptr<IndexesBatch> indexes_,
    shared_ptr<IndexesIntervalsBatch> indexes_intervals_batch_,
    size_t read_padding_,
    size_t buffer_size = sixteen_kB
  );
  auto generate_batch() -> void override;

private:
  auto load_buffer(uint num_copy_from_end = 0) -> void;
  auto assert_version() -> void;
  auto skip_until_next_whitespace() -> char;
  auto getc() -> char;
  auto read_number(u64 starting_number) -> u64;
  auto skip_to_next_read() -> void;
};

}  // namespace sbwt_search

#endif
