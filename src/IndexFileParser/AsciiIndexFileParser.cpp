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
  u64 max_seqs_,
  u64 warp_size_,
  u64 buffer_size_
):
    IndexFileParser(std::move(in_stream_), max_indexes_, max_seqs_, warp_size_),
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
  shared_ptr<SeqStatisticsBatch> seq_statistics_batch_,
  shared_ptr<IndexesBatch> indexes_batch_
) -> bool {
  IndexFileParser::generate_batch(
    std::move(seq_statistics_batch_), std::move(indexes_batch_)
  );
  const u64 initial_size = get_indexes_batch()->warped_indexes.size()
    + get_seq_statistics_batch()->colored_seq_id.size();
  char c = '\0';
  while (get_indexes().size() < get_max_indexes()
         && get_num_seqs() < get_max_seqs()
         && (!get_istream().eof() || buffer_index != buffer_size)) {
    c = getc();
    if (c == '\0') {  // EOF
      break;
    }
    if (c == '-') {
      c = getc();
      if (c == '1') {
        ++get_seq_statistics_batch()->not_found_idxs.back();
      } else if (c == '2') {
        ++get_seq_statistics_batch()->invalid_idxs.back();
      }
      c = skip_until_next_whitespace();
    }
    if (c == '\n') { end_seq(); }
    // if it is a number (note: all special characters are smaller than '0')
    if (c >= '0') {
      ++get_seq_statistics_batch()->found_idxs.back();
      get_indexes().push_back(parse_number(c - '0'));
    }
  }
  add_warp_interval();
  return (get_indexes_batch()->warped_indexes.size()
          + get_seq_statistics_batch()->colored_seq_id.size())
    > initial_size;
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

inline auto AsciiIndexFileParser::parse_number(u64 starting_number) -> u64 {
  auto result = starting_number;
  char c = '\0';
  const u64 base = 10;
  while ((c = getc()) >= '0') { result = result * base + c - '0'; }
  buffer_index--;
  return result;
}

}  // namespace sbwt_search
