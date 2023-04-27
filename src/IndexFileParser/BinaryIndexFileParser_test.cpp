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
private:
  string temp_filename = "test_objects/tmp/BinaryIndexFileParserTest.bin";

protected:
  auto get_results_ints() -> vector<vector<int>> {
    const vector<vector<int>> result = {
      {-2, 39, 164, 216, 59, -1, -2},
      {-2, -1, -1, -1, -1, -1, -2},
      {1, 2, 3, 4},
      {},
      {0, 1, 2, 4, 5, 6},
    };
    return result;
  }
  auto get_results_ints_with_newlines() -> vector<vector<int>> {
    const vector<vector<int>> result = {
      {},
      {},
      {-2, 39, 164, 216, 59, -1, -2},
      {-2, -1, -1, -1, -1, -1, -2},
      {1, 2, 3, 4},
      {},
      {0, 1, 2, 4, 5, 6},
    };
    return result;
  }
  auto run_test(
    const vector<vector<int>> &results_ints,
    u64 max_indexes,
    u64 max_seqs,
    u64 warp_size,
    u64 buffer_size,
    const vector<vector<u64>> &expected_indexes,
    const vector<vector<u64>> &expected_warps_intervals,
    const vector<vector<u64>> &expected_found_idxs,
    const vector<vector<u64>> &expected_not_found_idxs,
    const vector<vector<u64>> &expected_invalid_idxs,
    const vector<vector<u64>> &expected_colored_seq_id
  ) -> void {
    write_fake_binary_results_to_file(temp_filename, results_ints);
    auto in_stream = make_shared<ThrowingIfstream>(temp_filename, ios::in);
    auto format_name = in_stream->read_string_with_size();
    ASSERT_EQ(format_name, "binary");
    auto seq_statistics_batch = make_shared<SeqStatisticsBatch>();
    auto indexes_batch = make_shared<IndexesBatch>();
    auto host = BinaryIndexFileParser(
      in_stream, max_indexes, max_seqs, warp_size, buffer_size
    );
    for (int i = 0; i < expected_indexes.size(); ++i) {
      seq_statistics_batch->reset();
      indexes_batch->reset();
      host.generate_batch(seq_statistics_batch, indexes_batch);
      EXPECT_EQ(indexes_batch->warped_indexes, expected_indexes[i]);
      EXPECT_EQ(indexes_batch->warp_intervals, expected_warps_intervals[i]);
      EXPECT_EQ(seq_statistics_batch->found_idxs, expected_found_idxs[i]);
      EXPECT_EQ(
        seq_statistics_batch->not_found_idxs, expected_not_found_idxs[i]
      );
      EXPECT_EQ(seq_statistics_batch->invalid_idxs, expected_invalid_idxs[i]);
      EXPECT_EQ(
        seq_statistics_batch->colored_seq_id, expected_colored_seq_id[i]
      );
    }
    remove(temp_filename);
  }
};

TEST_F(BinaryIndexFileParserTest, OneBatch) {
  const u64 max_indexes = 999;
  const u64 max_seqs = 999;
  const u64 warp_size = 4;
  const int pad = -1;
  const vector<vector<int>> expected_indexes = {{
    39,
    164,
    216,
    59,  // end of 1st seq
         // 2nd seq is empty
    1,
    2,
    3,
    4,  // end of 3rd seq
    0,
    1,
    2,
    4,
    5,
    6,
    pad,
    pad  // end of 4th seq
  }};
  const vector<vector<u64>> expected_warps_intervals = {{0, 1, 2, 4}};
  const vector<vector<u64>> expected_found_idxs = {{4, 0, 4, 0, 6, 0}};
  const vector<vector<u64>> expected_not_found_idxs = {{1, 5, 0, 0, 0, 0}};
  const vector<vector<u64>> expected_invalid_idxs = {{2, 2, 0, 0, 0, 0}};
  const vector<vector<u64>> expected_colored_seq_id = {{0, 1, 1, 2, 2, 3}};

  // 22 is how many characters are on the first line, 23 includes '\n'
  // 63 is how many characters are in entire file, 24 includes EOF
  for (auto buffer_size : {1, 2, 3, 4, 22, 23, 63, 64, 999}) {
    run_test(
      get_results_ints(),
      max_indexes,
      max_seqs,
      warp_size,
      buffer_size,
      test_utils::to_u64s(expected_indexes),
      expected_warps_intervals,
      expected_found_idxs,
      expected_not_found_idxs,
      expected_invalid_idxs,
      expected_colored_seq_id
    );
  }
}

TEST_F(BinaryIndexFileParserTest, MaxSeqs) {
  const u64 max_indexes = 999;
  const u64 max_seqs = 4;
  const u64 warp_size = 4;
  const int pad = -1;
  const vector<vector<int>> expected_indexes = {
    {39,
     164,
     216,
     59,  // end of 1st seq
          // 2nd seq is empty
     1,
     2,
     3,
     4},  // end of 3rd seq + empty seq
    {
      0,
      1,
      2,
      4,
      5,
      6,
      pad,
      pad  // end of 4th seq
    }};
  const vector<vector<u64>> expected_warps_intervals = {{0, 1, 2}, {0, 2}};
  const vector<vector<u64>> expected_found_idxs = {{4, 0, 4, 0}, {0, 6, 0}};
  const vector<vector<u64>> expected_not_found_idxs = {{1, 5, 0, 0}, {0, 0, 0}};
  const vector<vector<u64>> expected_invalid_idxs = {{2, 2, 0, 0}, {0, 0, 0}};
  const vector<vector<u64>> expected_colored_seq_id = {{0, 1, 1, 2}, {0, 0, 1}};

  // 22 is how many characters are on the first line, 23 includes '\n'
  // 63 is how many characters are in entire file, 24 includes EOF
  for (auto buffer_size : {1, 2, 3, 4, 22, 23, 63, 64, 999}) {
    run_test(
      get_results_ints(),
      max_indexes,
      max_seqs,
      warp_size,
      buffer_size,
      test_utils::to_u64s(expected_indexes),
      expected_warps_intervals,
      expected_found_idxs,
      expected_not_found_idxs,
      expected_invalid_idxs,
      expected_colored_seq_id
    );
  }
}

TEST_F(BinaryIndexFileParserTest, BreakInMiddle) {
  const u64 max_indexes = 12;
  const u64 max_seqs = 999;
  const u64 warp_size = 4;
  const int pad = -1;
  const vector<vector<int>> expected_indexes = {
    {39,
     164,
     216,
     59,  // end of 1st seq
          // 2nd seq is empty
     1,
     2,
     3,
     4,  // end of 3rd seq
     0,
     1,
     2,
     4},
    {
      5,
      6,
      pad,
      pad  // end of 4th seq
    }};
  const vector<vector<u64>> expected_warps_intervals = {{0, 1, 2, 3}, {0, 1}};
  const vector<vector<u64>> expected_found_idxs = {{4, 0, 4, 0, 4}, {2, 0}};
  const vector<vector<u64>> expected_not_found_idxs = {{1, 5, 0, 0, 0}, {0, 0}};
  const vector<vector<u64>> expected_invalid_idxs = {{2, 2, 0, 0, 0}, {0, 0}};
  const vector<vector<u64>> expected_colored_seq_id = {{0, 1, 1, 2, 2}, {0, 1}};

  // 22 is how many characters are on the first line, 23 includes '\n'
  // 63 is how many characters are in entire file, 24 includes EOF
  for (auto buffer_size : {1, 2, 3, 4, 22, 23, 63, 64, 999}) {
    run_test(
      get_results_ints(),
      max_indexes,
      max_seqs,
      warp_size,
      buffer_size,
      test_utils::to_u64s(expected_indexes),
      expected_warps_intervals,
      expected_found_idxs,
      expected_not_found_idxs,
      expected_invalid_idxs,
      expected_colored_seq_id
    );
  }
}

TEST_F(BinaryIndexFileParserTest, NewlinesAtStart) {
  const u64 max_indexes = 12;
  const u64 max_seqs = 999;
  const u64 warp_size = 4;
  const int pad = -1;
  const vector<vector<int>> expected_indexes = {
    {39,
     164,
     216,
     59,  // end of 1st seq
          // 2nd seq is empty
     1,
     2,
     3,
     4,  // end of 3rd seq
     0,
     1,
     2,
     4},
    {
      5,
      6,
      pad,
      pad  // end of 4th seq
    }};
  const vector<vector<u64>> expected_warps_intervals = {{0, 1, 2, 3}, {0, 1}};
  const vector<vector<u64>> expected_found_idxs
    = {{0, 0, 4, 0, 4, 0, 4}, {2, 0}};
  const vector<vector<u64>> expected_not_found_idxs
    = {{0, 0, 1, 5, 0, 0, 0}, {0, 0}};
  const vector<vector<u64>> expected_invalid_idxs
    = {{0, 0, 2, 2, 0, 0, 0}, {0, 0}};
  const vector<vector<u64>> expected_colored_seq_id
    = {{0, 0, 0, 1, 1, 2, 2}, {0, 1}};

  // 22 is how many characters are on the first line, 23 includes '\n'
  // 63 is how many characters are in entire file, 24 includes EOF
  for (auto buffer_size : {1, 2, 3, 4, 22, 23, 63, 64, 999}) {
    run_test(
      get_results_ints_with_newlines(),
      max_indexes,
      max_seqs,
      warp_size,
      buffer_size,
      test_utils::to_u64s(expected_indexes),
      expected_warps_intervals,
      expected_found_idxs,
      expected_not_found_idxs,
      expected_invalid_idxs,
      expected_colored_seq_id
    );
  }
}

TEST_F(BinaryIndexFileParserTest, MultipleBatches) {
  const u64 max_indexes = 8;
  const u64 max_seqs = 999;
  const u64 warp_size = 4;
  const int pad = -1;
  const vector<vector<int>> expected_indexes = {
    {39,
     164,
     216,
     59,  // end of 1st seq
          // 2nd seq is empty
     1,
     2,
     3,
     4},                          // end of 3rd seq
                                  // empty line
    {0, 1, 2, 4, 5, 6, pad, pad}  // end of 4th seq
  };
  const vector<vector<u64>> expected_warps_intervals = {{0, 1, 2}, {0, 2}};
  // below, the first 0 of the second element is from the previous batch,
  // since the reader will not know that the batch has finished
  const vector<vector<u64>> expected_found_idxs = {{4, 0, 4}, {0, 0, 6, 0}};
  const vector<vector<u64>> expected_not_found_idxs = {{1, 5, 0}, {0, 0, 0, 0}};
  const vector<vector<u64>> expected_invalid_idxs = {{2, 2, 0}, {0, 0, 0, 0}};
  const vector<vector<u64>> expected_colored_seq_id = {{0, 1, 1}, {0, 0, 0, 1}};

  // 22 is how many characters are on the first line, 23 includes '\n'
  // 63 is how many characters are in entire file, 24 includes EOF
  for (auto buffer_size : {1, 2, 3, 4, 22, 23, 63, 64, 999}) {
    run_test(
      get_results_ints(),
      max_indexes,
      max_seqs,
      warp_size,
      buffer_size,
      test_utils::to_u64s(expected_indexes),
      expected_warps_intervals,
      expected_found_idxs,
      expected_not_found_idxs,
      expected_invalid_idxs,
      expected_colored_seq_id
    );
  }
}

}  // namespace sbwt_search
