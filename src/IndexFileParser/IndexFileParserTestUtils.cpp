#include <ios>
#include <vector>

#include "Global/GlobalDefinitions.h"
#include "IndexFileParser/IndexFileParserTestUtils.h"
#include "Tools/IOUtils.h"
#include "Tools/TestUtils.hpp"

namespace sbwt_search {

using io_utils::ThrowingOfstream;
using std::ios;
using std::vector;
using test_utils::to_u64s;

auto get_binary_index_output_filename() -> string {
  return "test_objects/tmp/BinaryIndexFileParserTest.bin";
}

auto write_fake_binary_results_to_file() -> void {
  const vector<vector<int>> results_ints = {
    {-2, 39, 164, 216, 59, -1, -2},
    {-2, -1, -1, -1, -1, -1, -2},
    {1, 2, 3, 4},
    {},
    {0, 1, 2, 4, 5, 6},
  };
  const auto results = to_u64s(results_ints);
  ThrowingOfstream out_stream(
    get_binary_index_output_filename(), ios::binary | ios::out
  );
  out_stream.write_string_with_size("binary");
  out_stream.write_string_with_size("v1.0");
  for (u64 i = 0; i < results.size(); ++i) {
    out_stream.write(results[i]);
    if (i < results.size() - 1) { out_stream.write(static_cast<u64>(-3)); }
  }
}

}  // namespace sbwt_search
