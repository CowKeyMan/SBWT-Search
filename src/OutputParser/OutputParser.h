#ifndef OUTPUT_PARSER_H
#define OUTPUT_PARSER_H

/**
 * @file OutputParser.h
 * @brief Parses the output files
 */

#include <string>

#include "Utils/IOUtils.h"

using io_utils::ThrowingIfstream;
using std::ios;
using std::string;

namespace sbwt_search {

enum class ITEM_TYPE { NEWLINE, VALUE, EOF_T };

class OutputParser {
  protected:
    ThrowingIfstream stream;

  public:
    OutputParser(string filename): stream(filename, ios::binary | ios::in) {}
    virtual auto get_next() -> ITEM_TYPE = 0;
    virtual auto get_value() -> size_t = 0;
};

}  // namespace sbwt_search

#endif
