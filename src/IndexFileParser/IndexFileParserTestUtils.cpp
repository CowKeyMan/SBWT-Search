#include <ios>

#include "Global/GlobalDefinitions.h"
#include "IndexFileParser/IndexFileParserTestUtils.h"
#include "Tools/IOUtils.h"
#include "Tools/TestUtils.hpp"

namespace sbwt_search {

using io_utils::ThrowingOfstream;
using std::ios;
using test_utils::to_u64s;

auto write_fake_binary_results_to_file(
  const string &filename, const vector<vector<int>> &results_ints
) -> void {
  const auto results = to_u64s(results_ints);
  ThrowingOfstream out_stream(filename, ios::binary | ios::out);
  out_stream.write_string_with_size("binary");
  out_stream.write_string_with_size("v1.0");
  for (u64 i = 0; i < results.size(); ++i) {
    out_stream.write(results[i]);
    out_stream.write(static_cast<u64>(-3));
  }
}

}  // namespace sbwt_search
