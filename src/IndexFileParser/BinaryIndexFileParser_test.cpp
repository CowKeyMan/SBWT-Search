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

class BinaryIndexFileParserTest: public ::testing::Test {
protected:
  auto run_test(
    const string &filename,
    u64 max_indexes,
    u64 max_reads,
    u64 padding,
    u64 buffer_size,
    const vector<vector<u64>> &expected_indexes,
    const vector<vector<u64>> &expected_warps_before_new_reads,
    const vector<vector<u64>> &expected_found_idxs,
    const vector<vector<u64>> &expected_not_found_idxs,
    const vector<vector<u64>> &expected_invalid_idxs
  ) -> void {
    write_fake_binary_results_to_file();
    auto in_stream = make_shared<ThrowingIfstream>(filename, ios::in);
    auto format_name = in_stream->read_string_with_size();
    ASSERT_EQ(format_name, "binary");
    auto read_statistics_batch = make_shared<ReadStatisticsBatch>();
    auto warps_before_new_read_batch = make_shared<WarpsBeforeNewReadBatch>();
    warps_before_new_read_batch->warps_before_new_read
      = make_shared<vector<u64>>();
    auto indexes_batch = make_shared<IndexesBatch>();
    auto host = BinaryIndexFileParser(
      in_stream, max_indexes, max_reads, padding, buffer_size
    );
    for (int i = 0; i < expected_indexes.size(); ++i) {
      read_statistics_batch->reset();
      warps_before_new_read_batch->reset();
      indexes_batch->reset();
      host.generate_batch(
        read_statistics_batch, warps_before_new_read_batch, indexes_batch
      );
      EXPECT_EQ(indexes_batch->indexes, expected_indexes[i]);
      EXPECT_EQ(read_statistics_batch->found_idxs, expected_found_idxs[i]);
      EXPECT_EQ(
        read_statistics_batch->not_found_idxs, expected_not_found_idxs[i]
      );
      EXPECT_EQ(read_statistics_batch->invalid_idxs, expected_invalid_idxs[i]);
      EXPECT_EQ(
        *warps_before_new_read_batch->warps_before_new_read,
        expected_warps_before_new_reads[i]
      );
    }
  }
};

TEST_F(BinaryIndexFileParserTest, OneBatch) {
  const u64 max_indexes = 999;
  const u64 max_reads = 999;
  const u64 padding = 4;
  const int pad = -1;
  const vector<vector<int>> expected_indexes = {{
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
  const vector<vector<u64>> expected_warps_before_new_reads = {{1, 1, 2, 2}};
  const vector<vector<u64>> expected_found_idxs = {{4, 0, 4, 0, 6}};
  const vector<vector<u64>> expected_not_found_idxs = {{1, 5, 0, 0, 0}};
  const vector<vector<u64>> expected_invalid_idxs = {{2, 2, 0, 0, 0}};

  // 22 is how many characters are on the first line, 23 includes '\n'
  // 63 is how many characters are in entire file, 24 includes EOF
  for (auto buffer_size : {1, 2, 3, 4, 22, 23, 63, 64, 999}) {
    run_test(
      get_binary_index_output_filename(),
      max_indexes,
      max_reads,
      padding,
      buffer_size,
      test_utils::to_u64s(expected_indexes),
      expected_warps_before_new_reads,
      expected_found_idxs,
      expected_not_found_idxs,
      expected_invalid_idxs
    );
  }
}

TEST_F(BinaryIndexFileParserTest, MaxReads) {
  const u64 max_indexes = 999;
  const u64 max_reads = 3;
  const u64 warp_size = 4;
  const int pad = -1;
  const vector<vector<int>> expected_indexes = {
    {39,
     164,
     216,
     59,  // end of 1st read
          // 2nd read is empty
     1,
     2,
     3,
     4},  // end of 3rd read
    {
      0,
      1,
      2,
      4,
      5,
      6,
      pad,
      pad  // end of 4th read
    }};
  const vector<vector<u64>> expected_warps_before_new_reads = {{1, 1}, {0, 0}};
  const vector<vector<u64>> expected_found_idxs = {{4, 0, 4}, {0, 0, 6}};
  const vector<vector<u64>> expected_not_found_idxs = {{1, 5, 0}, {0, 0, 0}};
  const vector<vector<u64>> expected_invalid_idxs = {{2, 2, 0}, {0, 0, 0}};

  // 22 is how many characters are on the first line, 23 includes '\n'
  // 63 is how many characters are in entire file, 24 includes EOF
  for (auto buffer_size : {1, 2, 3, 4, 22, 23, 63, 64, 999}) {
    run_test(
      get_binary_index_output_filename(),
      max_indexes,
      max_reads,
      warp_size,
      buffer_size,
      test_utils::to_u64s(expected_indexes),
      expected_warps_before_new_reads,
      expected_found_idxs,
      expected_not_found_idxs,
      expected_invalid_idxs
    );
  }
}

TEST_F(BinaryIndexFileParserTest, BreakInMiddle) {
  const u64 max_indexes = 12;
  const u64 max_reads = 999;
  const u64 warp_size = 4;
  const int pad = -1;
  const vector<vector<int>> expected_indexes = {
    {39,
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
     4},
    {
      5,
      6,
      pad,
      pad  // end of 4th read
    }};
  const vector<vector<u64>> expected_warps_before_new_reads
    = {{1, 1, 2, 2}, {}};
  const vector<vector<u64>> expected_found_idxs = {{4, 0, 4, 0, 4}, {2}};
  const vector<vector<u64>> expected_not_found_idxs = {{1, 5, 0, 0, 0}, {0}};
  const vector<vector<u64>> expected_invalid_idxs = {{2, 2, 0, 0, 0}, {0}};

  // 22 is how many characters are on the first line, 23 includes '\n'
  // 63 is how many characters are in entire file, 24 includes EOF
  for (auto buffer_size : {1, 2, 3, 4, 22, 23, 63, 64, 999}) {
    run_test(
      get_binary_index_output_filename(),
      max_indexes,
      max_reads,
      warp_size,
      buffer_size,
      test_utils::to_u64s(expected_indexes),
      expected_warps_before_new_reads,
      expected_found_idxs,
      expected_not_found_idxs,
      expected_invalid_idxs
    );
  }
}

TEST_F(BinaryIndexFileParserTest, MultipleBatches) {
  const u64 max_indexes = 8;
  const u64 max_reads = 999;
  const u64 padding = 4;
  const int pad = -1;
  const vector<vector<int>> expected_indexes = {
    {39,
     164,
     216,
     59,  // end of 1st read
          // 2nd read is empty
     1,
     2,
     3,
     4},                          // end of 3rd read
                                  // empty line
    {0, 1, 2, 4, 5, 6, pad, pad}  // end of 4th read
  };
  const vector<vector<u64>> expected_warps_before_new_reads = {{1, 1}, {0, 0}};
  // below, the first 0 of the second element is from the previous batch,
  // since the reader will not know that the batch has finished
  const vector<vector<u64>> expected_found_idxs = {{4, 0, 4}, {0, 0, 6}};
  const vector<vector<u64>> expected_not_found_idxs = {{1, 5, 0}, {0, 0, 0}};
  const vector<vector<u64>> expected_invalid_idxs = {{2, 2, 0}, {0, 0, 0}};

  // 22 is how many characters are on the first line, 23 includes '\n'
  // 63 is how many characters are in entire file, 24 includes EOF
  for (auto buffer_size : {1, 2, 3, 4, 22, 23, 63, 64, 999}) {
    run_test(
      get_binary_index_output_filename(),
      max_indexes,
      max_reads,
      padding,
      buffer_size,
      test_utils::to_u64s(expected_indexes),
      expected_warps_before_new_reads,
      expected_found_idxs,
      expected_not_found_idxs,
      expected_invalid_idxs
    );
  }
}
}  // namespace sbwt_search
