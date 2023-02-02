#include <chrono>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <thread>

#include <gtest/gtest.h>

#include "BatchObjects/IntervalBatch.h"
#include "BatchObjects/StringBreakBatch.h"
#include "BatchObjects/StringSequenceBatch.h"
#include "SequenceFileParser/ContinuousSequenceFileParser.h"
#include "SequenceFileParser/IntervalBatchProducer.h"
#include "SequenceFileParser/StringBreakBatchProducer.h"
#include "SequenceFileParser/StringSequenceBatchProducer.h"
#include "Tools/RNGUtils.hpp"

namespace sbwt_search {

using rng_utils::get_uniform_generator;
using std::make_unique;
using std::numeric_limits;
using std::shared_ptr;
using std::chrono::milliseconds;
using std::this_thread::sleep_for;

const uint default_max_batches = 7;
const auto max = numeric_limits<size_t>::max();

class ContinuousSequenceFileParserTest: public ::testing::Test {
private:
  const uint time_to_wait = 100;

protected:
  auto run_test(
    const vector<string> &filenames,
    uint kmer_size,
    size_t max_chars_per_batch,
    const vector<string> &seq,
    const vector<vector<size_t>> &chars_before_newline,
    const vector<vector<size_t>> &newlines_before_newfile,
    uint max_batches
  ) {
    auto host = make_unique<ContinuousSequenceFileParser>(
      filenames, kmer_size, max_chars_per_batch, max_batches
    );
    auto string_sequence_batch_producer
      = host->get_string_sequence_batch_producer();
    auto string_break_batch_producer = host->get_string_break_batch_producer();
    auto interval_batch_producer = host->get_interval_batch_producer();
    size_t expected_batches = seq.size();
#pragma omp parallel sections num_threads(4)
    {
#pragma omp section
      host_generate(*host);
#pragma omp section
      assert_string_break_batch_correct(
        *string_break_batch_producer,
        seq,
        expected_batches,
        chars_before_newline
      );
#pragma omp section
      assert_string_sequence_batch_correct(
        *string_sequence_batch_producer, seq, expected_batches
      );
#pragma omp section
      assert_interval_batch_correct(
        *interval_batch_producer,
        chars_before_newline,
        newlines_before_newfile,
        expected_batches
      );
    }
  }

private:
  auto host_generate(ContinuousSequenceFileParser &host) const -> void {
    auto rng = get_uniform_generator(0U, time_to_wait);
    sleep_for(milliseconds(rng()));
    host.read_and_generate();
  }

  auto assert_string_break_batch_correct(
    StringBreakBatchProducer &string_break_batch_producer,
    const vector<string> &seq,
    size_t expected_batches,
    const vector<vector<size_t>> &chars_before_newline
  ) const -> void {
    auto rng = get_uniform_generator(0U, time_to_wait);
    shared_ptr<StringBreakBatch> string_break_batch;
    size_t batches = 0;
    for (batches = 0; string_break_batch_producer >> string_break_batch;
         ++batches) {
      sleep_for(milliseconds(rng()));
      EXPECT_EQ(
        chars_before_newline[batches], *string_break_batch->chars_before_newline
      );
      EXPECT_EQ(seq[batches].size(), string_break_batch->string_size);
    }
    EXPECT_EQ(batches, expected_batches);
  }

  auto assert_string_sequence_batch_correct(
    StringSequenceBatchProducer &string_sequence_batch_producer,
    const vector<string> &seq,
    size_t expected_batches
  ) const -> void {
    auto rng = get_uniform_generator(0U, time_to_wait);
    shared_ptr<StringSequenceBatch> string_sequence_batch;
    size_t batches = 0;
    for (batches = 0; string_sequence_batch_producer >> string_sequence_batch;
         ++batches) {
      sleep_for(milliseconds(rng()));
      EXPECT_EQ(seq[batches], *string_sequence_batch->seq);
    }
    EXPECT_EQ(batches, expected_batches);
  }

  auto assert_interval_batch_correct(
    IntervalBatchProducer &interval_batch_producer,
    const vector<vector<size_t>> &chars_before_newline,
    const vector<vector<size_t>> &newlines_before_newfile,
    size_t expected_batches
  ) const -> void {
    auto rng = get_uniform_generator(0U, time_to_wait);
    shared_ptr<IntervalBatch> interval_batch;
    size_t batches = 0;
    for (batches = 0; interval_batch_producer >> interval_batch; ++batches) {
      sleep_for(milliseconds(rng()));
      EXPECT_EQ(
        chars_before_newline[batches], *interval_batch->chars_before_newline
      );
      EXPECT_EQ(
        newlines_before_newfile[batches],
        interval_batch->newlines_before_newfile
      );
    }
    EXPECT_EQ(batches, expected_batches);
  }
};

TEST_F(ContinuousSequenceFileParserTest, GetAllInOneBatch) {
  const uint kmer_size = 10;
  const size_t max_chars_per_batch = 999;
  const vector<string> seq
    = {"1ACTGCAATGGGCAATATGTCTCTGTGTGGATTAC23TCTAGCTACTACTACTGA"
       "TGGATGGAATGTGATG45TGAGTGAGATGAGGTGATAGTGACGTAGTGAGGA61A"
       "CTGCAATGGGCAATATGTCTCTGTGTGGATTAC23TCTAGCTACTACTACTGATG"
       "GATGGAATGTGATG45TGAGTGAGATGAGGTGATAGTGACGTAGTGAGGA6"};
  const vector<vector<size_t>> chars_before_newline
    = {{36UL, 36UL * 2UL, 36UL * 3UL, 36UL * 4UL, 36UL * 5UL, 36UL * 6UL, max}};
  const vector<vector<size_t>> newlines_before_newfile = {{3, 6, max}};
  for (auto max_batches : {1, 2, 3, 7}) {
    run_test(
      {"test_objects/small_fasta.fna", "test_objects/small_fastq.fnq"},
      kmer_size,
      max_chars_per_batch,
      seq,
      chars_before_newline,
      newlines_before_newfile,
      max_batches
    );
  }
}

TEST_F(ContinuousSequenceFileParserTest, GetSplitBatchesBigMaxChar) {
  const uint kmer_size = 15;
  const size_t max_chars_per_batch = 36 * 3 + 10;
  const vector<string> seq = {
    "1ACTGCAATGGGCAATATGTCTCTGTGTGGATTAC23TCTAGCTACTACTACTGATGGATGGAATGTGATG45T"
    "GAGTGAGATGAGGTGATAGTGACGTAGTGAGGA61ACTGCAATG",
    "1ACTGCAATGGGCAATATGTCTCTGTGTGGATTAC23TCTAGCTACTACTACTGATGGATGGAATGTGATG45T"
    "GAGTGAGATGAGGTGATAGTGACGTAGTGAGGA6",
  };
  const vector<vector<size_t>> chars_before_newline
    = {{36, 36ULL * 2, 36ULL * 3, max}, {36, 36ULL * 2, 36ULL * 3, max}};
  vector<vector<size_t>> newlines_before_newfile = {{3}, {3}};
  for (auto &v : newlines_before_newfile) { v.push_back(max); }
  for (auto max_batches : {1, 2, 3, 7}) {
    run_test(
      {"test_objects/small_fasta.fna", "test_objects/small_fastq.fnq"},
      kmer_size,
      max_chars_per_batch,
      seq,
      chars_before_newline,
      newlines_before_newfile,
      max_batches
    );
  }
}

TEST_F(ContinuousSequenceFileParserTest, GetSplitBatchesSmallMaxChar) {
  uint kmer_size = 3;
  const size_t max_chars_per_batch = 36 * 3 - 10;
  const vector<string> seq = {
    "1ACTGCAATGGGCAATATGTCTCTGTGTGGATTAC23TCTAGCTACTACTACT"
    "GATGGATGGAATGTGATG45TGAGTGAGATGAGGTGATAGTGACG",
    "CGTAGTGAGGA61ACTGCAATGGGCAATATGTCTCTGTGTGGATTAC23TCTA"
    "GCTACTACTACTGATGGATGGAATGTGATG45TGAGTGAGATGAG",
    "AGGTGATAGTGACGTAGTGAGGA6"};
  const vector<vector<size_t>> chars_before_newline
    = {{36, 36ULL * 2, max}, {12, 12 + 36, 12 + 36ULL * 2, max}, {24, max}};
  const vector<vector<size_t>> newlines_before_newfile
    = {{max}, {1, max}, {1, max}};
  for (auto max_batches : {1, 2, 3, 7}) {
    run_test(
      {"test_objects/small_fasta.fna", "test_objects/small_fastq.fnq"},
      kmer_size,
      max_chars_per_batch,
      seq,
      chars_before_newline,
      newlines_before_newfile,
      max_batches
    );
  }
}

TEST_F(ContinuousSequenceFileParserTest, IncorrectFileAndVerySmallMaxChar) {
  const uint kmer_size = 3;
  const size_t max_chars_per_batch = 30;
  const vector<string> seq = {
    {"1ACTGCAATGGGCAATATGTCTCTGTGTGG"},
    {"GGATTAC23TCTAGCTACTACTACTGATGG"},
    {"GGATGGAATGTGATG45TGAGTGAGATGAG"},
    {"AGGTGATAGTGACGTAGTGAGGA6"}};
  const vector<vector<size_t>> chars_before_newline
    = {{max}, {8, max}, {16, max}, {24, max}};
  const vector<vector<size_t>> newlines_before_newfile
    = {{max}, {max}, {max}, {1, 1, max}};
  for (auto max_batches : {1, 2, 3, 7}) {
    run_test(
      {"test_objects/small_fasta.fna", "garbage_filename"},
      kmer_size,
      max_chars_per_batch,
      seq,
      chars_before_newline,
      newlines_before_newfile,
      max_batches
    );
  }
}

}  // namespace sbwt_search
