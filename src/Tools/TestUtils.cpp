#include "Tools/TestUtils.hpp"

namespace test_utils {

auto to_char_vec(const vector<string> &str_vec) -> vector<vector<char>> {
  vector<vector<char>> result;
  for (const auto &s : str_vec) {
    result.push_back({});
    result.back().resize(s.size());
    std::copy(s.begin(), s.end(), result.back().begin());
  }
  return result;
}

}  // namespace test_utils
