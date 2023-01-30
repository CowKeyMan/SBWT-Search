#ifndef OUTPUT_PARSER_H
#define OUTPUT_PARSER_H

/**
 * @file OutputParser.h
 * @brief Parses the output files
 */

#include <string>

#include "Tools/IOUtils.h"

namespace sbwt_search {

using io_utils::ThrowingIfstream;
using std::ios;
using std::string;

enum class ITEM_TYPE { NEWLINE, VALUE, EOF_T };

class OutputParser {
private:
  ThrowingIfstream stream;

protected:
  auto get_stream() -> ThrowingIfstream & { return stream; }

public:
  OutputParser(const OutputParser &) = delete;
  OutputParser(const OutputParser &&) = delete;
  auto operator=(const OutputParser &) = delete;
  auto operator=(const OutputParser &&) = delete;
  explicit OutputParser(const string &filename):
    stream(filename, ios::binary | ios::in) {}
  virtual auto get_next() -> ITEM_TYPE = 0;
  virtual auto get_value() -> size_t = 0;
  virtual ~OutputParser() = default;
};

}  // namespace sbwt_search

#endif
