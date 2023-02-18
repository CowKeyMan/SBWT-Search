#include <chrono>
#include <filesystem>
#include <limits>
#include <memory>
#include <span>
#include <thread>

#include <gtest/gtest.h>

#include "IndexFileParser/ContinuousIndexFileParser.h"
#include "IndexFileParser/IndexFileParserTestUtils.h"
#include "Tools/RNGUtils.hpp"
#include "Tools/TestUtils.hpp"

namespace sbwt_search {

using rng_utils::get_uniform_generator;
using std::make_unique;
using std::numeric_limits;
using std::span;
using std::chrono::milliseconds;
using std::filesystem::remove;
using std::this_thread::sleep_for;

const u64 default_max_batches = 7;
const auto max = numeric_limits<u64>::max();
const u64 time_to_wait = 100;

class ContinuousIndexFileParserTest: public ::testing::Test {
protected:
  auto run_test(
    u64 max_indexes_per_batch,
    u64 max_batches,
    span<const string> filenames,
    u64 read_padding,
    const vector<vector<u64>> &expected_indexes,
    const vector<u64> &expected_true_indexes,
    const vector<u64> &expected_skipped,
    const vector<vector<u64>> &expected_indexes_starts,
    const vector<vector<u64>> &expected_indexes_before_newfile
  ) {
    write_fake_binary_results_to_file();
    auto host = make_unique<ContinuousIndexFileParser>(
      max_indexes_per_batch, max_batches, filenames, read_padding
    );
#pragma omp parallel sections num_threads(4)
    {
#pragma omp section
      { host_generate(*host); }
#pragma omp section
      {
        auto indexes_batch_producer = host->get_indexes_batch_producer();
        assert_indexes_correct(
          *indexes_batch_producer,
          expected_indexes,
          expected_true_indexes,
          expected_skipped
        );
      }
#pragma omp section
      {
        auto indexes_starts_batch_producer
          = host->get_indexes_starts_batch_producer();
        assert_indexes_starts_correct(
          *indexes_starts_batch_producer, expected_indexes_starts
        );
      }
#pragma omp section
      {
        auto indexes_before_newfile_batch_producer
          = host->get_indexes_before_newfile_batch_producer();
        assert_indexes_before_newfile_correct(
          *indexes_before_newfile_batch_producer,
          expected_indexes_before_newfile
        );
      }
    }
    remove(get_binary_index_output_filename());
  }

  auto host_generate(ContinuousIndexFileParser &host) const -> void {
    auto rng = get_uniform_generator(0UL, time_to_wait);
    sleep_for(milliseconds(rng()));
    host.read_and_generate();
  }

  auto assert_indexes_correct(
    IndexesBatchProducer &indexes_batch_producer,
    const vector<vector<u64>> &expected_indexes,
    const vector<u64> &expected_true_indexes,
    const vector<u64> &expected_skipped
  ) -> void {
    shared_ptr<IndexesBatch> batch;
    auto rng = get_uniform_generator(0UL, time_to_wait);
    u64 batches = 0;
    for (batches = 0; indexes_batch_producer >> batch; ++batches) {
      sleep_for(milliseconds(rng()));
      EXPECT_EQ(expected_indexes[batches], batch->indexes);
      EXPECT_EQ(expected_true_indexes[batches], batch->true_indexes);
      EXPECT_EQ(expected_skipped[batches], batch->skipped);
    }
    EXPECT_EQ(batches, expected_indexes.size());
  }

  auto assert_indexes_starts_correct(
    IndexesStartsBatchProducer &indexes_starts_batch_producer,
    const vector<vector<u64>> &expected_indexes_starts
  ) -> void {
    shared_ptr<IndexesStartsBatch> batch;
    auto rng = get_uniform_generator(0UL, time_to_wait);
    u64 batches = 0;
    for (batches = 0; indexes_starts_batch_producer >> batch; ++batches) {
      sleep_for(milliseconds(rng()));
      EXPECT_EQ(expected_indexes_starts[batches], batch->indexes_starts);
    }
    EXPECT_EQ(batches, expected_indexes_starts.size());
  }

  auto assert_indexes_before_newfile_correct(
    IndexesBeforeNewfileBatchProducer &indexes_before_newfile_batch_producer,
    const vector<vector<u64>> &expected_indexes_before_newfile
  ) -> void {
    shared_ptr<IndexesBeforeNewfileBatch> batch;
    auto rng = get_uniform_generator(0UL, time_to_wait);
    u64 batches = 0;
    for (batches = 0; indexes_before_newfile_batch_producer >> batch;
         ++batches) {
      sleep_for(milliseconds(rng()));
      EXPECT_EQ(
        expected_indexes_before_newfile[batches], batch->indexes_before_newfile
      );
    }
    EXPECT_EQ(batches, expected_indexes_before_newfile.size());
  }
};

TEST_F(ContinuousIndexFileParserTest, TestAll) {
  const u64 max_indexes_per_batch = 4;
  const vector<string> filenames = {
    "test_objects/example_index_search_result.txt",
    get_binary_index_output_filename()};
  span<const string> filenames_span = {filenames.data(), filenames.size()};
  const u64 read_padding = 4;
  int pad = -1;
  const vector<vector<int>> ints = {
    {39, 164, 216, 59},  // end of 1st read
                         // 2nd read is empty
    {1, 2, 3, 4},        // end of 3rd read
                         // empty line
    {0, 1, 2, 4},
    {5, 6, pad, pad},    // end of 4th read
                         // end of first file
    {39, 164, 216, 59},  // end of 1st read
                         // 2nd read is empty
    {1, 2, 3, 4},        // end of 3rd read
                         // empty line
    {0, 1, 2, 4},
    {5, 6, pad, pad}     // end of 4th read
  };
  const vector<vector<u64>> starts
    = {{0}, {0, 0}, {0, 0}, {}, {0}, {0, 0}, {0, 0}, {}};
  const vector<u64> true_indexes = {4, 4, 4, 2, 4, 4, 4, 2};
  const vector<u64> skipped_indexes = {1, 9, 0, 0, 1, 9, 0, 0};
  const vector<vector<u64>> expected_indexes_before_newfile
    = {{}, {}, {}, {4}, {}, {}, {}, {}};
  for (auto max_batches : {1, 2, 3, 4, 5, 7}) {
    run_test(
      max_indexes_per_batch,
      max_batches,
      filenames_span,
      read_padding,
      test_utils::to_u64s(ints),
      true_indexes,
      skipped_indexes,
      starts,
      expected_indexes_before_newfile
    );
  }
}

TEST_F(ContinuousIndexFileParserTest, TestOneBatch) {
  const u64 max_indexes_per_batch = 999;
  const vector<string> filenames = {
    "test_objects/example_index_search_result.txt",
    get_binary_index_output_filename()};
  span<const string> filenames_span = {filenames.data(), filenames.size()};
  const u64 read_padding = 4;
  int pad = -1;
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
        // empty line
    0,
    1,
    2,
    4,
    5,
    6,
    pad,
    pad,  // end of 4th read
          // end of first file
    39,
    164,
    216,
    59,  // end of 1st read
         // 2nd read is empty
    1,
    2,
    3,
    4,  // end of 3rd read
        // empty line
    0,
    1,
    2,
    4,
    5,
    6,
    pad,
    pad  // end of 4th read
  }};
  const vector<vector<u64>> starts = {{0, 4, 4, 8, 8, 16, 20, 20, 24, 24}};
  const vector<u64> true_indexes = {14ULL * 2};
  const vector<u64> skipped_indexes = {10ULL * 2};
  const vector<vector<u64>> expected_indexes_before_newfile = {{16}};
  for (auto max_batches : {1, 2, 3, 4, 5, 7}) {
    run_test(
      max_indexes_per_batch,
      max_batches,
      filenames_span,
      read_padding,
      test_utils::to_u64s(ints),
      true_indexes,
      skipped_indexes,
      starts,
      expected_indexes_before_newfile
    );
  }
}

}  // namespace sbwt_search
