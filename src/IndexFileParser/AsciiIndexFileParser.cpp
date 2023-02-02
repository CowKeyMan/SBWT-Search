#include <stdexcept>
#include <string>

#include "IndexFileParser/AsciiIndexFileParser.h"
#include "Tools/IOUtils.h"

namespace sbwt_search {

using io_utils::read_string_with_size;
using std::runtime_error;
using std::string;

AsciiIndexFileParser::AsciiIndexFileParser(
  shared_ptr<ThrowingIfstream> in_stream_,
  shared_ptr<IndexesBatch> indexes_,
  shared_ptr<IndexesStartsBatch> indexes_starts_batch_,
  size_t max_indexes_,
  size_t read_padding_,
  size_t buffer_size_
):
  IndexFileParser(
    std::move(in_stream_),
    std::move(indexes_),
    std::move(indexes_starts_batch_),
    max_indexes_,
    read_padding_
  ),
  buffer_size(buffer_size_) {
  assert_version();
  buffer.resize(buffer_size);
  load_buffer();
}

auto AsciiIndexFileParser::assert_version() -> void {
  auto version = read_string_with_size(get_istream());
  if (version != "v1.0") {
    throw runtime_error("The file has an incompatible version number");
  }
}

auto AsciiIndexFileParser::generate_batch(size_t start_index) -> void {
  current_index = start_index;
  char c = '\0';
  while (get_indexes().size() < get_max_indexes()
         && (!get_istream().eof() || buffer_index != buffer_size)) {
    c = getc();
    if (c == '\0') { return; }  // EOF
    if (new_read) {
      get_starts().push_back(current_index);
      new_read = false;
    }
    if (c == '-') { c = skip_until_next_whitespace(); }
    if (c == '\n') { end_read(); }
    // if it is a number (note: all special characters are smaller than '0')
    if (c >= '0') {
      get_indexes().push_back(read_number(c - '0'));
      ++current_index;
    }
  }
}

inline auto AsciiIndexFileParser::load_buffer(uint num_copy_from_end) -> void {
  std::copy(buffer.end() - num_copy_from_end, buffer.end(), buffer.begin());
  get_istream().read(buffer.data(), static_cast<int>(buffer_size));
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

inline auto AsciiIndexFileParser::end_read() -> void {
  while (current_index % get_read_padding() != 0) {
    get_indexes().push_back(static_cast<u64>(-1));
    ++current_index;
  }
  new_read = true;
}

}  // namespace sbwt_search
