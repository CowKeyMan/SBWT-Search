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

#include "Tools/TypeDefinitions.h"

namespace units_parser {

using std::string;
using std::unordered_map;

class MemoryUnitsParser {
  const static unordered_map<string, u64> units_to_multiplier;

public:
  static auto convert(const string &s) -> u64;
};

}  // namespace units_parser

#endif
