#ifndef BINARY_OUTPUT_PARSER_H
#define BINARY_OUTPUT_PARSER_H

/**
 * @file BinaryOutputParser.h
 * @brief Parses the binary output file
 */

#include <sstream>
#include <string>

#include "OutputParser/OutputParser.h"

using std::stringstream;

namespace sbwt_search {

class BinaryOutputParser: public OutputParser {
  private:
    size_t current_value;

  public:
    BinaryOutputParser(string filename);
    auto get_next() -> ITEM_TYPE override;
    auto get_value() -> size_t override;
};

}  // namespace sbwt_search

#endif
