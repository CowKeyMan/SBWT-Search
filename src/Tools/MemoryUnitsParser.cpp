#include <regex>
#include <stdexcept>

#include "Tools/MemoryUnitsParser.h"

namespace units_parser {

using std::regex;
using std::regex_match;
using std::runtime_error;
using std::stoull;
using std::string;
using std::toupper;
using std::vector;

auto MemoryUnitsParser::units_to_multiplier()
  -> const vector<pair<regex, u64>> {
  return {
    {regex(R"(^([0-9]+)$)"), 1ULL},
    {regex(R"(^([0-9]+)[ ]?B$)"), 8ULL},
    {regex(R"(^([0-9]+)[ ]?K[B]?$)"), 8ULL * 1024},
    {regex(R"(^([0-9]+)[ ]?M[B]?$)"), 8ULL * 1024 * 1024},
    {regex(R"(^([0-9]+)[ ]?G[B]?$)"), 8ULL * 1024 * 1024 * 1024}};
}

auto MemoryUnitsParser::convert(const string &s) -> u64 {
  u64 str_size = s.size();
  string str_number = s;
  std::smatch match;
  for (auto &utm : units_to_multiplier()) {
    if (std::regex_search(s.begin(), s.end(), match, utm.first)) {
      return stoull(match[0]) * utm.second;
    }
  }
  throw runtime_error("Unable to infer bits from " + s);
}

}  // namespace units_parser
