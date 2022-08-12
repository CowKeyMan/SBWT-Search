#include <chrono>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <thread>

#include <gtest/gtest.h>

#include "SequenceFileParser/ContinuousSequenceFileParser.hpp"
#include "TestUtils/GeneralTestUtils.hpp"

using std::cerr;
using std::fill;
using std::make_shared;
using std::out_of_range;
using std::shared_ptr;
using std::stringstream;
using std::tie;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using std::this_thread::sleep_for;

#include <iostream>
using namespace std;

namespace sbwt_search {

class ContinuousSequenceFileParserTest: public ::testing::Test {
  protected:
    vector<string> filenames
      = { "test_objects/test_query.fna", "test_objects/test_query.fna" };
    u64 max_characters_per_batch = UINT_MAX;
    uint readers_amount = 3;
    uint max_batches = UINT_MAX;
    vector<vector<u64>> expected_string_indexes = { { 0, 8, 8, 8 } };
    vector<vector<u64>> expected_character_indexes = { { 0, 0, 0, 0 } };
    vector<vector<string>> expected_batches
      = { { "GACTG", "AA", "GATCGA", "TA", "GACTG", "AA", "GATCGA", "TA" } };
    vector<shared_ptr<vector<string>>> batches;

  protected:
    inline auto get_host() -> ContinuousSequenceFileParser {
      return ContinuousSequenceFileParser(
        filenames, max_characters_per_batch, readers_amount, max_batches
      );
    }

    auto shared_tests() -> void {
      auto host = get_host();
      host.read_and_generate();
      vector<unique_ptr<StringSequenceBatch>> batches;
      unique_ptr<StringSequenceBatch> batch;
      for (uint i = 0; host >> batch; ++i) {
        assert_vectors_equal(expected_string_indexes[i], batch->string_indexes);
        assert_vectors_equal(
          expected_character_indexes[i], batch->character_indexes
        );
        batches.push_back(move(batch));
      }
      ASSERT_EQ(expected_batches.size(), batches.size());
      for (auto i = 0; i < expected_batches.size(); ++i) {
        assert_vectors_equal(expected_batches[i], batches[i]->buffer);
      }
    }
};

TEST_F(ContinuousSequenceFileParserTest, GetSimple) { shared_tests(); }

TEST_F(ContinuousSequenceFileParserTest, GetMaxCharsPerBatchEqualToFileSize) {
  filenames = { "test_objects/test_query_with_long_string.fna",
                "test_objects/test_query_with_long_string.fna" };
  max_characters_per_batch = 32 * 3;
  expected_string_indexes = { { 0, 1, 2, 4 }, { 0, 1, 2, 4 } };
  expected_character_indexes = { { 0, 12, 12, 0 }, { 0, 12, 12, 0 } };
  expected_batches = { { "AAAAAAAAAAAAAAAAAAAA",
                         "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                         "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                         "TA" },
                       { "AAAAAAAAAAAAAAAAAAAA",
                         "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                         "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                         "TA" } };
  shared_tests();
}

TEST_F(ContinuousSequenceFileParserTest, TestInvalidFile) {
  filenames = { "test_objects/test_query.fna",
                "invalid_file__",
                "test_objects/test_query.fna" };
  stringstream mybuffer;
  auto *old_buf = cerr.rdbuf();
  cerr.rdbuf(mybuffer.rdbuf());
  shared_tests();
  cerr.rdbuf(old_buf);
  ASSERT_EQ("The input file invalid_file__ cannot be opened\n", mybuffer.str());
}

TEST_F(ContinuousSequenceFileParserTest, TestStringTooLong) {
  filenames = { "test_objects/test_query_with_long_string.fna" };
  max_characters_per_batch = 5;  // rounds up to 32
  expected_batches = {
    { "AAAAAAAAAAAAAAAAAAAA" },
    { "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" },
    { "TA" },
  };
  expected_string_indexes = { { 0, 1, 1, 1 }, { 0, 1, 1, 1 }, { 0, 1, 1, 1 } };
  expected_character_indexes
    = { { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 } };
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
  max_characters_per_batch = 32 * 3;
  expected_string_indexes = { { 0, 1, 2, 4 }, { 0, 1, 2, 4 }, { 0, 1, 2, 4 } };
  expected_character_indexes
    = { { 0, 12, 12, 0 }, { 0, 12, 12, 0 }, { 0, 12, 12, 0 } };
  max_batches = 2;
  expected_batches = { { "AAAAAAAAAAAAAAAAAAAA",
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
  vector<unique_ptr<StringSequenceBatch>> batches;
  unique_ptr<StringSequenceBatch> batch;
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
      u64 string_index, character_index;
      u32 batch_index, in_batch_index;
      sleep_for(milliseconds(sleep_time));
      while (host >> batch) { batches.push_back(move(batch)); }
    }
  }
  ASSERT_EQ(expected_batches.size(), batches.size());
  for (int i = 0; i < expected_batches.size(); ++i) {
    assert_vectors_equal(expected_batches[i], batches[i]->buffer);
    assert_vectors_equal(expected_string_indexes[i], batches[i]->string_indexes);
    assert_vectors_equal(expected_character_indexes[i], batches[i]->character_indexes);
  }
  ASSERT_GE(read_time, sleep_time);
}
}
