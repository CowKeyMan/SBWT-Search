#include <stdexcept>
#include <string>

#include "IndexFileParser/AsciiIndexFileParser.h"
#include "Tools/IOUtils.h"

namespace sbwt_search {

using std::runtime_error;
using std::string;

AsciiIndexFileParser::AsciiIndexFileParser(
  shared_ptr<ThrowingIfstream> in_stream_,
  u64 max_indexes_,
  u64 read_padding_,
  u64 buffer_size_
):
    IndexFileParser(std::move(in_stream_), max_indexes_, read_padding_),
    buffer_size(buffer_size_) {
  assert_version();
  buffer.resize(buffer_size);
  load_buffer();
}

auto AsciiIndexFileParser::assert_version() -> void {
  auto version = get_istream().read_string_with_size();
  if (version != "v1.0") {
    throw runtime_error("The file has an incompatible version number");
  }
}

auto AsciiIndexFileParser::generate_batch(
  shared_ptr<IndexesBatch> indexes_batch_,
  shared_ptr<IndexesStartsBatch> indexes_starts_batch_
) -> bool {
  IndexFileParser::generate_batch(
    std::move(indexes_batch_), std::move(indexes_starts_batch_)
  );
  const u64 initial_size = get_starts().size() + get_indexes().size();
  char c = '\0';
  while (get_indexes().size() < get_max_indexes()
         && (!get_istream().eof() || buffer_index != buffer_size)) {
    c = getc();
    if (c == '\0') { break; }  // EOF
    if (new_read) {
      get_starts().push_back(get_indexes().size());
      new_read = false;
    }
    if (c == '-') {
      c = skip_until_next_whitespace();
      ++get_indexes_batch()->skipped;
    }
    if (c == '\n') {
      pad_read();
      new_read = true;
    }
    // if it is a number (note: all special characters are smaller than '0')
    if (c >= '0') {
      ++get_indexes_batch()->true_indexes;
      get_indexes().push_back(read_number(c - '0'));
    }
  }
  pad_read();
  return get_starts().size() > initial_size;
}

inline auto AsciiIndexFileParser::load_buffer() -> void {
  get_istream().read(buffer.data(), static_cast<std::streamsize>(buffer_size));
  buffer_size = get_istream().gcount();
  buffer_index = 0;
}

inline auto AsciiIndexFileParser::getc() -> char {
  if (buffer_index >= buffer_size) { load_buffer(); }
  if (buffer_size == 0) { return 0; }
  return buffer[buffer_index++];
}

inline auto AsciiIndexFileParser::skip_until_next_whitespace() -> char {
  char c = '\0';
  // while we are still getting digits
  while ((c = getc()) >= '0') {}
  return c;
}

inline auto AsciiIndexFileParser::read_number(u64 starting_number) -> u64 {
  auto result = starting_number;
  char c = '\0';
  const u64 base = 10;
  while ((c = getc()) >= '0') { result = result * base + c - '0'; }
  buffer_index--;
  return result;
}

}  // namespace sbwt_search
