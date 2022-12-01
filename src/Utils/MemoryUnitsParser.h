#ifndef MEMORY_UNITS_PARSER_H
#define MEMORY_UNITS_PARSER_H

/**
 * @file MemoryUnitsParser.h
 * @brief Given a string such as "1KB" or "1000" (bits) or "10 GB", this module
 * converts it to bits.
 */

#include <cctype>
#include <string>

#include <unordered_map>

using std::unordered_map;
using std::string;

namespace units_parser {

class MemoryUnitsParser {
    static unordered_map<string, size_t> units_to_multiplier;

  public:
    static auto convert(const string &s) -> size_t;
};

}  // namespace units_parser

#endif
