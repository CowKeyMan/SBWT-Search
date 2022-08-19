#include <chrono>
#include <climits>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>

#include "gtest/gtest_pred_impl.h"
#include <gtest/gtest-message.h>
#include <gtest/gtest-test-part.h>

#include "BatchObjects/CumulativePropertiesBatch.hpp"
#include "BatchObjects/IntervalBatch.hpp"
#include "BatchObjects/StringSequenceBatch.hpp"
#include "SequenceFileParser/ContinuousSequenceFileParser.h"
#include "TestUtils/GeneralTestUtils.hpp"

using std::cerr;
using std::shared_ptr;
using std::stringstream;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using std::this_thread::sleep_for;

namespace sbwt_search {

class ContinuousSequenceFileParserTest: public ::testing::Test {
  protected:
    const uint kmer_size = 3;
    vector<string> filenames
      = { "test_objects/test_query.fna", "test_objects/test_query.fna" };
    u64 max_chars_per_batch = UINT_MAX;
    u64 max_strings_per_batch = 1000;
    uint num_readers = 3;
    uint max_batches = 20;
    vector<vector<u64>> expected_string_indexes = { { 0, 8, 8, 8 } };
    vector<vector<u64>> expected_char_indexes = { { 0, 0, 0, 0 } };
    vector<vector<u64>> expected_cumulative_char_indexes
      = { { 0, 30, 30, 30 } };
    vector<vector<string>> expected_sequence_batches
      = { { "GACTG", "AA", "GATCGA", "TA", "GACTG", "AA", "GATCGA", "TA" } };
    vector<shared_ptr<vector<string>>> sequence_batches;
    vector<vector<u64>> expected_cumsum_positions_per_string
      = { { 0, 3, 3, 7, 7, 10, 10, 14, 14 } };
    vector<vector<u64>> expected_cumsum_string_lengths
      = { { 0, 5, 7, 13, 15, 20, 22, 28, 30 } };
    vector<vector<u64>> expected_string_lengths
      = { { 5, 2, 6, 2, 5, 2, 6, 2 } };
    vector<vector<u64>> expected_strings_before_newfile
      = { { 4, 4, ULLONG_MAX } };

  protected:
    inline auto get_host() -> ContinuousSequenceFileParser {
      return ContinuousSequenceFileParser(
        filenames,
        kmer_size,
        max_chars_per_batch,
        max_strings_per_batch,
        num_readers,
        max_batches
      );
    }

    auto shared_tests() -> void {
      auto host = get_host();
      host.read_and_generate();
      vector<StringSequenceBatch> sequence_batches;
      vector<CumulativePropertiesBatch> cumsum_batches;
      shared_ptr<StringSequenceBatch> sequence_batch;
      shared_ptr<CumulativePropertiesBatch> cumsum_batch;
      shared_ptr<IntervalBatch> interval_batch;
      for (uint i = 0; host >> sequence_batch & host >> cumsum_batch
                       & host >> interval_batch;
           ++i) {
        assert_vectors_equal(
          expected_string_indexes[i],
          sequence_batch->string_indexes,
          __FILE__,
          __LINE__
        );
        assert_vectors_equal(
          expected_char_indexes[i],
          sequence_batch->char_indexes,
          __FILE__,
          __LINE__
        );
        sequence_batches.push_back(*sequence_batch);
        assert_vectors_equal(
          expected_cumsum_positions_per_string[i],
          cumsum_batch->cumsum_positions_per_string,
          __FILE__,
          __LINE__
        );
        assert_vectors_equal(
          expected_cumsum_string_lengths[i],
          cumsum_batch->cumsum_string_lengths,
          __FILE__,
          __LINE__
        );
        assert_vectors_equal(
          expected_string_lengths[i],
          interval_batch->string_lengths,
          __FILE__,
          __LINE__
        );
        assert_vectors_equal(
          expected_strings_before_newfile[i],
          interval_batch->strings_before_newfile,
          __FILE__,
          __LINE__
        );
      }
      ASSERT_EQ(expected_sequence_batches.size(), sequence_batches.size());
      for (auto i = 0; i < expected_sequence_batches.size(); ++i) {
        assert_vectors_equal(
          expected_sequence_batches[i],
          sequence_batches[i].buffer,
          __FILE__,
          __LINE__
        );
      }
    }
};

TEST_F(ContinuousSequenceFileParserTest, GetSimple) { shared_tests(); }

TEST_F(ContinuousSequenceFileParserTest, GetMaxCharsPerBatchEqualToFileSize) {
  filenames = { "test_objects/test_query_with_long_string.fna",
                "test_objects/test_query_with_long_string.fna" };
  max_chars_per_batch = 32 * 3;
  expected_string_indexes = { { 0, 1, 2, 4 }, { 0, 1, 2, 4 } };
  expected_char_indexes = { { 0, 12, 12, 0 }, { 0, 12, 12, 0 } };
  expected_cumulative_char_indexes = { { 0, 32, 64, 87 }, { 0, 32, 64, 87 } };
  expected_cumsum_positions_per_string
    = { { 0, 18, 48, 79, 79 }, { 0, 18, 48, 79, 79 } };
  expected_cumsum_string_lengths
    = { { 0, 20, 52, 85, 87 }, { 0, 20, 52, 85, 87 } };
  expected_sequence_batches = { { "AAAAAAAAAAAAAAAAAAAA",
                                  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                                  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                                  "TA" },
                                { "AAAAAAAAAAAAAAAAAAAA",
                                  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                                  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                                  "TA" } };
  expected_string_lengths = { { 20, 32, 33, 2 }, { 20, 32, 33, 2 } };
  expected_strings_before_newfile = { { 4, ULLONG_MAX }, { 4, ULLONG_MAX } };
  shared_tests();
}

TEST_F(ContinuousSequenceFileParserTest, TestInvalidFile) {
  filenames = { "test_objects/test_query.fna",
                "invalid_file__",
                "test_objects/test_query.fna" };
  expected_strings_before_newfile = { { 4, 0, 4, ULLONG_MAX } };
  stringstream mybuffer;
  auto *old_buf = cerr.rdbuf();
  cerr.rdbuf(mybuffer.rdbuf());
  shared_tests();
  cerr.rdbuf(old_buf);
  ASSERT_EQ("The input file invalid_file__ cannot be opened\n", mybuffer.str());
}

TEST_F(ContinuousSequenceFileParserTest, TestStringTooLong) {
  filenames = { "test_objects/test_query_with_long_string.fna" };
  max_chars_per_batch = 5;  // rounds up to 32
  expected_sequence_batches = {
    { "AAAAAAAAAAAAAAAAAAAA" },
    { "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" },
    { "TA" },
  };
  expected_string_indexes = { { 0, 1, 1, 1 }, { 0, 1, 1, 1 }, { 0, 1, 1, 1 } };
  expected_char_indexes = { { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 } };
  expected_cumulative_char_indexes
    = { { 0, 20, 20, 20 }, { 0, 32, 32, 32 }, { 0, 2, 2, 2 } };
  expected_cumsum_positions_per_string = { { 0, 18 }, { 0, 30 }, { 0, 0 } };
  expected_cumsum_string_lengths = { { 0, 20 }, { 0, 32 }, { 0, 2 } };
  expected_string_lengths = { { 20 }, { 32, 0 }, { 2 } };
  expected_strings_before_newfile
    = { { ULLONG_MAX }, { ULLONG_MAX }, { 1, ULLONG_MAX } };
  stringstream mybuffer;
  auto *old_buf = cerr.rdbuf();
  cerr.rdbuf(mybuffer.rdbuf());
  shared_tests();
  cerr.rdbuf(old_buf);
  ASSERT_EQ(
    "The string in file test_objects/test_query_with_long_string.fna at "
    "position 2 is too "
    "large\n",
    mybuffer.str()
  );
}

TEST_F(ContinuousSequenceFileParserTest, TestParallel) {
  auto sleep_time = 300;
  filenames = { "test_objects/test_query_with_long_string.fna",
                "test_objects/test_query_with_long_string.fna",
                "test_objects/test_query_with_long_string.fna" };
  max_chars_per_batch = 32 * 3;
  expected_string_indexes = { { 0, 1, 2, 4 }, { 0, 1, 2, 4 }, { 0, 1, 2, 4 } };
  expected_char_indexes
    = { { 0, 12, 12, 0 }, { 0, 12, 12, 0 }, { 0, 12, 12, 0 } };
  expected_cumulative_char_indexes
    = { { 0, 87, 87, 87 }, { 0, 87, 87, 87 }, { 0, 87, 87, 87 } };
  expected_cumsum_positions_per_string
    = { { 0, 18, 48, 79, 79 }, { 0, 18, 48, 79, 79 }, { 0, 18, 48, 79, 79 } };
  expected_cumsum_string_lengths
    = { { 0, 20, 52, 85, 87 }, { 0, 20, 52, 85, 87 }, { 0, 20, 52, 85, 87 } };
  max_batches = 2;
  expected_sequence_batches = { { "AAAAAAAAAAAAAAAAAAAA",
                                  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                                  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                                  "TA" },
                                { "AAAAAAAAAAAAAAAAAAAA",
                                  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                                  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                                  "TA" },
                                { "AAAAAAAAAAAAAAAAAAAA",
                                  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                                  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                                  "TA" } };
  auto host = get_host();
  vector<StringSequenceBatch> sequence_batches;
  vector<CumulativePropertiesBatch> cumsum_batches;
  vector<IntervalBatch> interval_batches;
  shared_ptr<StringSequenceBatch> sequence_batch;
  milliseconds::rep read_time;
#pragma omp parallel sections
  {
#pragma omp section
    {
      auto start_time = high_resolution_clock::now();
      host.read_and_generate();
      auto end_time = high_resolution_clock::now();
      read_time = duration_cast<milliseconds>(end_time - start_time).count();
    }
#pragma omp section
    {
      sleep_for(milliseconds(sleep_time));
      shared_ptr<StringSequenceBatch> sequence_batch;
      shared_ptr<CumulativePropertiesBatch> cumsum_batch;
      shared_ptr<IntervalBatch> interval_batch;
      while (host >> sequence_batch & host >> cumsum_batch
             & host >> interval_batch) {
        sequence_batches.push_back(*sequence_batch);
        cumsum_batches.push_back(*cumsum_batch);
        interval_batches.push_back(*interval_batch);
      }
    }
  }
  ASSERT_EQ(expected_sequence_batches.size(), sequence_batches.size());
  for (int i = 0; i < expected_sequence_batches.size(); ++i) {
    assert_vectors_equal(
      expected_sequence_batches[i],
      sequence_batches[i].buffer,
      __FILE__,
      __LINE__
    );
    assert_vectors_equal(
      expected_string_indexes[i],
      sequence_batches[i].string_indexes,
      __FILE__,
      __LINE__
    );
    assert_vectors_equal(
      expected_char_indexes[i],
      sequence_batches[i].char_indexes,
      __FILE__,
      __LINE__
    );
    assert_vectors_equal(
      expected_cumsum_positions_per_string[i],
      cumsum_batches[i].cumsum_positions_per_string,
      __FILE__,
      __LINE__
    );
    assert_vectors_equal(
      expected_cumsum_string_lengths[i],
      cumsum_batches[i].cumsum_string_lengths,
      __FILE__,
      __LINE__
    );
  }
  ASSERT_GE(read_time, sleep_time);
}
}  // namespace sbwt_search
