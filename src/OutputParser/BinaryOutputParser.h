#ifndef BINARY_OUTPUT_PARSER_H
#define BINARY_OUTPUT_PARSER_H

/**
 * @file BinaryOutputParser.h
 * @brief Parses the binary output file
 */

#include <memory>
#include <string>

#include "OutputParser/OutputParser.h"

namespace sbwt_search {

class BinaryOutputParser: public OutputParser {
private:
  size_t current_value = 0;

public:
  explicit BinaryOutputParser(const string &filename);
  auto get_next() -> ITEM_TYPE override;
  auto get_value() -> size_t override;
};

}  // namespace sbwt_search

#endif
