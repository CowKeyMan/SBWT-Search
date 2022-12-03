#include "OutputParser/BinaryOutputParser.h"

using std::stoull;

namespace sbwt_search {

BinaryOutputParser::BinaryOutputParser(string filename):
    OutputParser(filename) {}

auto BinaryOutputParser::get_next() -> ITEM_TYPE {
  if (stream.read(reinterpret_cast<char *>(&current_value), sizeof(size_t))) {
    if (current_value == size_t(-3)) { return ITEM_TYPE::NEWLINE; }
    return ITEM_TYPE::VALUE;
  }
  return ITEM_TYPE::EOF_T;
}

auto BinaryOutputParser::get_value() -> size_t { return current_value; }

}  // namespace sbwt_search
