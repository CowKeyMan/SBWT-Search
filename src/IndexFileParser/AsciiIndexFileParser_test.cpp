#include <ios>

#include <gtest/gtest.h>

#include "IndexFileParser/AsciiIndexFileParser.h"
#include "Tools/IOUtils.h"
#include "Tools/RNGUtils.hpp"
#include "Tools/TestUtils.hpp"

namespace sbwt_search {

using io_utils::read_string_with_size;
using std::ios_base;
using std::make_shared;

class AsciiIndexFileParserTest: public ::testing::Test {
protected:
  auto run_test(
    const string &filename,
    size_t max_indexes_per_batch,
    size_t padding,
    const vector<vector<u64>> &expected_indexes,
    const vector<vector<size_t>> &expected_intervals,
    size_t buffer_size
  ) -> void {
    auto in_stream = make_shared<ThrowingIfstream>(filename, ios_base::in);
    auto format_name = read_string_with_size(*in_stream);
    ASSERT_EQ(format_name, "ascii");
    vector<vector<u64>> actual_indexes;
    vector<vector<size_t>> actual_intervals;
    auto indexes_batch = make_shared<IndexesBatch>(max_indexes_per_batch);
    auto indexes_intervals_batch = make_shared<IndexesIntervalsBatch>();
    auto host = AsciiIndexFileParser(
      in_stream, indexes_batch, indexes_intervals_batch, padding, buffer_size
    );
    for (int i = 0; i < expected_indexes.size(); ++i) {
      host.generate_batch();
      actual_indexes.push_back(indexes_batch->indexes);
      actual_intervals.push_back(indexes_intervals_batch->indexes_intervals);
      indexes_batch.reset();
      indexes_intervals_batch.reset();
    }
    EXPECT_EQ(expected_indexes, actual_indexes);
    EXPECT_EQ(expected_intervals, actual_intervals);
  }
};

auto to_u64s(const vector<vector<int>> &int_vec) -> vector<vector<u64>> {
  vector<vector<u64>> ret_val;
  for (const auto &v : int_vec) {
    ret_val.emplace_back();
    for (const auto &element : v) { ret_val.back().emplace_back(element); }
  }
  return ret_val;
}

TEST_F(AsciiIndexFileParserTest, OneBatch) {
  const size_t max_indexes_per_batch = 999;
  const size_t padding = 4;
  const int pad = -1;
  const vector<vector<int>> ints = {{
    39,
    164,
    216,
    59,  // end of 1st read
         // 2nd read is empty
    1,
    2,
    3,
    4,  // end of 3rd read
    0,
    1,
    2,
    4,
    5,
    6,
    pad,
    pad  // end of 4th read
  }};

  const vector<vector<u64>> intervals = {{4, 4, 8, 8, 16}};
  // 22 is how many characters are on the first line, 23 includes '\n'
  // 63 is how many characters are in entire file, 24 includes EOF
  for (auto buffer_size : {1, 2, 3, 4, 22, 23, 63, 64, 999}) {
    run_test(
      "test_objects/example_index_search_result.txt",
      max_indexes_per_batch,
      padding,
      to_u64s(ints),
      intervals,
      buffer_size
    );
  }
}

}  // namespace sbwt_search
