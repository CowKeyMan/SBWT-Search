#ifndef MEMORY_UNITS_PARSER_H
#define MEMORY_UNITS_PARSER_H

/**
 * @file MemoryUnitsParser.h
 * @brief Given a string such as "1KB" or "1000" (bits) or "10 GB", this module
 * converts it to bits.
 */

#include <cctype>
#include <regex>
#include <string>
#include <vector>

#include "Tools/TypeDefinitions.h"

namespace units_parser {

using std::pair;
using std::regex;
using std::string;
using std::vector;

class MemoryUnitsParser {
  static auto units_to_multiplier() -> const vector<pair<regex, u64>>;

public:
  static auto convert(const string &s) -> u64;
};

}  // namespace units_parser

#endif
