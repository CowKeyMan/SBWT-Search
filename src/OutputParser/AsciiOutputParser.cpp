#include "OutputParser/AsciiOutputParser.h"

using std::stoull;

namespace sbwt_search {

AsciiOutputParser::AsciiOutputParser(string filename): OutputParser(filename) {}

auto AsciiOutputParser::get_next() -> ITEM_TYPE {
  if (at_end_of_line) {
    at_end_of_line = false;
    if (!getline(stream, line_buffer)) { return ITEM_TYPE::EOF_T; }
    line_stream = stringstream(line_buffer);
  }
  if (line_stream >> result_buffer) {
    current_value = stoull(result_buffer);
    return ITEM_TYPE::VALUE;
  }
  at_end_of_line = true;
  return ITEM_TYPE::NEWLINE;
}

auto AsciiOutputParser::get_value() -> size_t { return current_value; }

}  // namespace sbwt_search
