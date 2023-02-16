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

const unordered_map<string, u64> MemoryUnitsParser::units_to_multiplier = {
  {"B", 8ULL},
  {"KB", 8 * 1024ULL},
  {"MB", 8 * 1024 * 1024ULL},
  {"GB", 8 * 1024 * 1024 * 1024ULL}};

auto MemoryUnitsParser::convert(const string &s) -> u64 {
  u64 str_size = s.size();
  u64 multiplier = 1;
  string str_number = s;
  if (units_to_multiplier.find(s.substr(str_size - 2, 2)) != units_to_multiplier.end()) {
    auto sub = s.substr(str_size - 2, 2);
    for (auto &c : sub) { c = toupper(c); }
    multiplier = units_to_multiplier.at(sub);
    str_number = s.substr(0, str_size - 2);
  } else if (toupper(s.back()) == 'B') {
    multiplier = units_to_multiplier.at("B");
    str_number = s.substr(0, str_size - 1);
  }
  auto re = regex(R"(\s*[0-9]+\s*)");
  if (!regex_match(str_number.begin(), str_number.end(), re)) {
    throw runtime_error("Unable to infer bits from " + s);
  }
  return stoull(str_number) * multiplier;
}

}  // namespace units_parser
