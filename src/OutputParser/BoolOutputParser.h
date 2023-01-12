#ifndef BOOL_OUTPUT_PARSER_H
#define BOOL_OUTPUT_PARSER_H

/**
 * @file BoolOutputParser.h
 * @brief Parses the bool output file
 */

#include <sstream>
#include <string>

#include "OutputParser/OutputParser.h"
#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using std::stringstream;

class BoolOutputParser: public OutputParser {
private:
  bool is_eof = false;
  size_t current_value = 0, current_seq_size = 0, seq_idx = 0;
  u64 bits_buffer = 0;
  ThrowingIfstream seqsize_stream;
  uint shift = 0;

public:
  explicit BoolOutputParser(const string &filename);
  auto get_next() -> ITEM_TYPE override;
  auto get_value() -> size_t override;

private:
  auto read_next_seq_size() -> void;
  auto read_next_bool() -> void;
};

}  // namespace sbwt_search

#endif
