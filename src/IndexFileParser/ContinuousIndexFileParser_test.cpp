#include <chrono>
#include <filesystem>
#include <limits>
#include <memory>
#include <thread>

#include <gtest/gtest.h>

#include "IndexFileParser/ContinuousIndexFileParser.h"
#include "IndexFileParser/IndexFileParserTestUtils.h"
#include "Tools/RNGUtils.hpp"
#include "Tools/TestUtils.hpp"

namespace sbwt_search {

using rng_utils::get_uniform_int_generator;
using std::numeric_limits;
using std::chrono::milliseconds;
using std::filesystem::remove;
using std::this_thread::sleep_for;
using test_utils::to_u64s;

const u64 default_max_batches = 7;
const auto max = numeric_limits<u64>::max();
const u64 time_to_wait = 50;

class ContinuousIndexFileParserTest: public ::testing::Test {
private:
  auto get_results_ints() -> vector<vector<int>> {
    return vector<vector<int>>{
      {-2, 39, 164, 216, 59, -1, -2},
      {-2, -1, -1, -1, -1, -1, -2},
      {1, 2, 3, 4},
      {},
      {0, 1, 2, 4, 5, 6},
    };
  }

protected:
  auto get_binary_filename() {
    return "test_objects/tmp/BinaryIndexFileParserTest.bin";
  }
  auto run_test(
    u64 max_batches,
    u64 max_indexes_per_batch,
    u64 max_reads_per_batch,
    u64 read_padding,
    const vector<string> &filenames,
    const vector<vector<u64>> &expected_indexes,
    const vector<vector<u64>> &expected_warps_intervals,
    const vector<vector<u64>> &expected_found_idxs,
    const vector<vector<u64>> &expected_not_found_idxs,
    const vector<vector<u64>> &expected_invalid_idxs,
    const vector<vector<u64>> &expected_colored_seq_id,
    const vector<vector<u64>> &expected_seqs_before_newfile
  ) {
    write_fake_binary_results_to_file(
      get_binary_filename(), get_results_ints()
    );
    auto host = ContinuousIndexFileParser(
      0,
      max_indexes_per_batch,
      max_reads_per_batch,
      read_padding,
      filenames,
      max_batches,
      max_batches
    );
    const auto num_sections = 3;
#pragma omp parallel sections num_threads(num_sections)
    {
#pragma omp section
      { host_generate(host); }
#pragma omp section
      {
        const auto &producer = host.get_seq_statistics_batch_producer();
        assert_seq_statistics_correct(
          *producer,
          expected_found_idxs,
          expected_not_found_idxs,
          expected_invalid_idxs,
          expected_colored_seq_id,
          expected_seqs_before_newfile
        );
      }
#pragma omp section
      {
        const auto &producer = host.get_indexes_batch_producer();
        assert_indexes_correct(
          *producer, expected_indexes, expected_warps_intervals
        );
      }
    }
    remove(get_binary_filename());
  }

  auto host_generate(ContinuousIndexFileParser &host) const -> void {
    auto rng = get_uniform_int_generator(0UL, time_to_wait);
    sleep_for(milliseconds(rng()));
    host.read_and_generate();
  }

  auto assert_seq_statistics_correct(
    SeqStatisticsBatchProducer &producer,
    const vector<vector<u64>> &expected_found_idxs,
    const vector<vector<u64>> &expected_not_found_idxs,
    const vector<vector<u64>> &expected_invalid_idxs,
    const vector<vector<u64>> &expected_colored_seq_id,
    const vector<vector<u64>> &expected_seqs_before_newfile
  ) -> void {
    shared_ptr<SeqStatisticsBatch> batch;
    auto rng = get_uniform_int_generator(0UL, time_to_wait);
    u64 batches = 0;
    for (batches = 0; producer >> batch; ++batches) {
      sleep_for(milliseconds(rng()));
      EXPECT_EQ(expected_found_idxs[batches], batch->found_idxs)
        << "at index " << batches;
      EXPECT_EQ(expected_not_found_idxs[batches], batch->not_found_idxs)
        << "at index " << batches;
      EXPECT_EQ(expected_invalid_idxs[batches], batch->invalid_idxs)
        << "at index " << batches;
      EXPECT_EQ(expected_colored_seq_id[batches], batch->colored_seq_id)
        << "at index " << batches;
      EXPECT_EQ(
        expected_seqs_before_newfile[batches], batch->seqs_before_new_file
      ) << "at index "
        << batches;
    }
    EXPECT_EQ(batches, expected_found_idxs.size());
  }

  auto assert_indexes_correct(
    IndexesBatchProducer &producer,
    const vector<vector<u64>> &expected_indexes,
    const vector<vector<u64>> &expected_warps_intervals
  ) -> void {
    shared_ptr<IndexesBatch> batch;
    auto rng = get_uniform_int_generator(0UL, time_to_wait);
    u64 batches = 0;
    for (batches = 0; producer >> batch; ++batches) {
      sleep_for(milliseconds(rng()));
      EXPECT_EQ(expected_indexes[batches], batch->warped_indexes)
        << "at index " << batches;
      EXPECT_EQ(expected_warps_intervals[batches], batch->warps_intervals)
        << "at index " << batches;
    }
    EXPECT_EQ(batches, expected_indexes.size());
  }
};

TEST_F(ContinuousIndexFileParserTest, TestAll) {
  const u64 max_indexes_per_batch = 4;
  const u64 max_reads_per_batch = 4;
  const vector<string> filenames
    = {"test_objects/example_index_search_result.txt", get_binary_filename()};
  const u64 read_padding = 4;
  int pad = -1;
  const vector<vector<int>> expected_indexes = {
    {39, 164, 216, 59},  // end of 1st read
                         // 2nd read is empty
    {1, 2, 3, 4},        // end of 3rd read
                         // empty line
    {0, 1, 2, 4},
    {5, 6, pad, pad},    // end of 4th read
    {39, 164, 216, 59},  // end of 1st read
                         // 2nd read is empty
    {1, 2, 3, 4},        // end of 3rd read
                         // empty line
    {0, 1, 2, 4},
    {5, 6, pad, pad},    // end of 4th read
    {}};
  u64 max = numeric_limits<u64>::max();
  const vector<vector<u64>> expected_warps_intervals
    = {{0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0}};

  const vector<vector<u64>> expected_seqs_before_newfile
    = {{max}, {max}, {max}, {max}, {1, max}, {max}, {max}, {max}, {max}};

  const vector<vector<u64>> expected_found_idxs = {
    {4},
    {0, 0, 4},
    {0, 0, 4},
    {2, 0},
    {0, 0, 4},
    {0, 0, 4},
    {0, 0, 4},
    {2, 0},
    {0}};
  const vector<vector<u64>> expected_not_found_idxs = {
    {0},
    {1, 5, 0},
    {0, 0, 0},
    {0, 0},
    {0, 0, 0},
    {1, 5, 0},
    {0, 0, 0},
    {0, 0},
    {0}};
  const vector<vector<u64>> expected_invalid_idxs = {
    {1},
    {1, 2, 0},
    {0, 0, 0},
    {0, 0},
    {0, 0, 1},
    {1, 2, 0},
    {0, 0, 0},
    {0, 0},
    {0}};
  const vector<vector<u64>> expected_colored_seq_id = {
    {0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 1},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 1},
    {0}};
  /* for (auto max_batches : {1, 2, 3, 4, 5, 7, 99}) { */
  for (auto max_batches : {7}) {
    run_test(
      max_batches,
      max_indexes_per_batch,
      max_reads_per_batch,
      read_padding,
      filenames,
      to_u64s(expected_indexes),
      expected_warps_intervals,
      expected_found_idxs,
      expected_not_found_idxs,
      expected_invalid_idxs,
      expected_colored_seq_id,
      expected_seqs_before_newfile
    );
  }
}

/* TEST_F(ContinuousIndexFileParserTest, TestOneBatch) { */
/*   const u64 max_indexes_per_batch = 999; */
/*   const u64 max_reads_per_batch = 999; */
/*   const vector<string> filenames = { */
/*     "test_objects/example_index_search_result.txt", */
/*     get_binary_index_output_filename()}; */
/*   const u64 read_padding = 4; */
/*   int pad = -1; */
/*   const vector<vector<int>> expected_indexes = {{ */
/*     39, */
/*     164, */
/*     216, */
/*     59,  // end of 1st read */
/*          // 2nd read is empty */
/*     1, */
/*     2, */
/*     3, */
/*     4,  // end of 3rd read */
/*         // empty line */
/*     0, */
/*     1, */
/*     2, */
/*     4, */
/*     5, */
/*     6, */
/*     pad, */
/*     pad,  // end of 4th read */
/*           // end of first file */
/*     39, */
/*     164, */
/*     216, */
/*     59,  // end of 1st read */
/*          // 2nd read is empty */
/*     1, */
/*     2, */
/*     3, */
/*     4,  // end of 3rd read */
/*         // empty line */
/*     0, */
/*     1, */
/*     2, */
/*     4, */
/*     5, */
/*     6, */
/*     pad, */
/*     pad  // end of 4th read */
/*   }}; */
/*   const vector<vector<u64>> expected_warps_before_new_reads */
/*     = {{1, 1, 2, 2, 4, 5, 5, 6, 6, max}}; */
/*   const vector<vector<u64>> expected_reads_before_newfile = {{5, max}}; */
/*   const vector<vector<u64>> expected_found_idxs */
/*     = {{4, 0, 4, 0, 6, 4, 0, 4, 0, 6}}; */
/*   const vector<vector<u64>> expected_not_found_idxs */
/*     = {{1, 5, 0, 0, 0, 1, 5, 0, 0, 0}}; */
/*   const vector<vector<u64>> expected_invalid_idxs */
/*     = {{2, 2, 0, 0, 0, 2, 2, 0, 0, 0}}; */
/*   for (auto max_batches : {1, 2, 3, 4, 5, 7, 99}) { */
/*     run_test( */
/*       max_batches, */
/*       max_indexes_per_batch, */
/*       max_reads_per_batch, */
/*       read_padding, */
/*       filenames, */
/*       to_u64s(expected_indexes), */
/*       expected_warps_before_new_reads, */
/*       expected_reads_before_newfile, */
/*       expected_found_idxs, */
/*       expected_not_found_idxs, */
/*       expected_invalid_idxs */
/*     ); */
/*   } */
/* } */

}  // namespace sbwt_search
