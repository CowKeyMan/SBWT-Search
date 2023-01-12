#ifndef ASCII_OUTPUT_PARSER_H
#define ASCII_OUTPUT_PARSER_H

/**
 * @file AsciiOutputParser.h
 * @brief Parses the ascii output file
 */

#include <sstream>
#include <string>

#include "OutputParser/OutputParser.h"

namespace sbwt_search {

using std::stringstream;

class AsciiOutputParser: public OutputParser {
private:
  size_t current_value;
  bool at_end_of_line = true;
  string line_buffer, result_buffer;
  stringstream line_stream;

public:
  explicit AsciiOutputParser(string filename);
  auto get_next() -> ITEM_TYPE override;
  auto get_value() -> size_t override;
};

}  // namespace sbwt_search

#endif
