#include <filesystem>
#include <memory>

#include <gtest/gtest.h>

#include "IndexFileParser/BinaryIndexFileParser.h"
#include "IndexFileParser/IndexFileParserTestUtils.h"
#include "Tools/IOUtils.h"
#include "Tools/TestUtils.hpp"

namespace sbwt_search {

using std::ios;
using std::make_shared;
using std::filesystem::remove;

class BinaryIndexFileParserTest: public ::testing::Test {
protected:
  auto run_test(
    const string &filename,
    u64 max_indexes,
    u64 padding,
    const vector<vector<u64>> &expected_indexes,
    const vector<vector<u64>> &expected_starts,
    u64 buffer_size,
    const vector<u64> &expected_true_indexes,
    const vector<u64> &expected_skipped
  ) -> void {
    write_fake_binary_results_to_file();
    auto in_stream = make_shared<ThrowingIfstream>(filename, ios::in);
    auto format_name = in_stream->read_string_with_size();
    ASSERT_EQ(format_name, "binary");
    auto indexes_batch = make_shared<IndexesBatch>();
    auto indexes_starts_batch = make_shared<IndexesStartsBatch>();
    auto host
      = BinaryIndexFileParser(in_stream, max_indexes, padding, buffer_size);
    for (int i = 0; i < expected_indexes.size(); ++i) {
      indexes_batch->indexes.resize(0);
      indexes_batch->true_indexes = 0;
      indexes_batch->skipped = 0;
      indexes_starts_batch->indexes_starts.resize(0);
      host.generate_batch(indexes_batch, indexes_starts_batch);
      EXPECT_EQ(indexes_batch->indexes, expected_indexes[i]);
      EXPECT_EQ(indexes_starts_batch->indexes_starts, expected_starts[i]);
      EXPECT_EQ(indexes_batch->true_indexes, expected_true_indexes[i]);
      EXPECT_EQ(indexes_batch->skipped, expected_skipped[i]);
    }
    remove(get_binary_index_output_filename());
  }
};

TEST_F(BinaryIndexFileParserTest, OneBatch) {
  const u64 max_indexes = 999;
  const u64 padding = 4;
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
  const vector<vector<u64>> starts = {{0, 4, 4, 8, 8}};
  const vector<u64> true_indexes = {14};
  const vector<u64> skipped_indexes = {10};
  // 22 is how many characters are on the first line, 23 includes '\n'
  // 63 is how many characters are in entire file, 24 includes EOF
  for (auto buffer_size : {1, 2, 3, 4, 22, 23, 63, 64, 999}) {
    run_test(
      get_binary_index_output_filename(),
      max_indexes,
      padding,
      test_utils::to_u64s(ints),
      starts,
      buffer_size,
      true_indexes,
      skipped_indexes
    );
  }
}

TEST_F(BinaryIndexFileParserTest, MultipleBatches) {
  const u64 max_indexes = 4;
  const u64 padding = 4;
  const int pad = -1;
  const vector<vector<int>> ints = {
    {39, 164, 216, 59},  // end of 1st read
                         // 2nd read is empty
    {1, 2, 3, 4},        // end of 3rd read
                         // empty line
    {0, 1, 2, 4},
    {5, 6, pad, pad}     // end of 4th read
  };
  const vector<vector<u64>> starts = {{0}, {0, 0}, {0, 0}, {}};
  const vector<u64> true_indexes = {4, 4, 4, 2};
  const vector<u64> skipped_indexes = {1, 9, 0, 0};
  // 22 is how many characters are on the first line, 23 includes '\n'
  // 63 is how many characters are in entire file, 24 includes EOF
  for (auto buffer_size : {1, 2, 3, 4, 22, 23, 63, 64, 999}) {
    run_test(
      get_binary_index_output_filename(),
      max_indexes,
      padding,
      test_utils::to_u64s(ints),
      starts,
      buffer_size,
      true_indexes,
      skipped_indexes
    );
  }
}

}  // namespace sbwt_search
