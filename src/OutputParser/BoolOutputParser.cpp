#include "OutputParser/BoolOutputParser.h"

using std::stoull;

namespace sbwt_search {

BoolOutputParser::BoolOutputParser(string filename):
    OutputParser(filename),
    seqsize_stream(filename + "_seq_sizes", ios::binary | ios::in) {
  read_next_seq_size();
}


auto BoolOutputParser::get_next() -> ITEM_TYPE {
  if (is_eof) { return ITEM_TYPE::EOF_T; }
  if (seq_idx == current_seq_size) {
    read_next_seq_size();
    seq_idx = 0;
    return ITEM_TYPE::NEWLINE;
  }
  read_next_bool();
  ++seq_idx;
  return ITEM_TYPE::VALUE;
}

auto BoolOutputParser::get_value() -> size_t { return current_value; }

auto BoolOutputParser::read_next_seq_size() -> void {
  is_eof = !seqsize_stream.read(
    reinterpret_cast<char *>(&current_seq_size), sizeof(size_t)
  );
}

auto BoolOutputParser::read_next_bool() -> void {
  if (shift == 0) {
    if (!stream.read(
          reinterpret_cast<char *>(&bits_buffer), sizeof(u64)
        )) {
      is_eof = true;
      return;
    }
    shift = 64;
  }
  --shift;
  current_value = (bits_buffer & (1ULL << shift)) > 0;
}

}  // namespace sbwt_search
