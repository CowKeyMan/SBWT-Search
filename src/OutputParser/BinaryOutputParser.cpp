#include <bit>

#include "OutputParser/BinaryOutputParser.h"

using std::bit_cast;

namespace sbwt_search {

BinaryOutputParser::BinaryOutputParser(const string &filename):
  OutputParser(filename) {}

auto BinaryOutputParser::get_next() -> ITEM_TYPE {
  if (get_stream().read(bit_cast<char *>(&current_value), sizeof(size_t))) {
    if (current_value == static_cast<size_t>(-3)) { return ITEM_TYPE::NEWLINE; }
    return ITEM_TYPE::VALUE;
  }
  return ITEM_TYPE::EOF_T;
}

auto BinaryOutputParser::get_value() -> size_t { return current_value; }

}  // namespace sbwt_search
