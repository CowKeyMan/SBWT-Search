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
#include "Tools/TestUtils.hpp"

namespace sbwt_search {

using rng_utils::get_uniform_int_generator;
using std::make_unique;
using std::numeric_limits;
using std::shared_ptr;
using std::chrono::milliseconds;
using std::this_thread::sleep_for;
using test_utils::to_char_vec;

const auto max = numeric_limits<u64>::max();
const u64 time_to_wait = 100;

class ContinuousSequenceFileParserTest: public ::testing::Test {
protected:
  auto run_test(
    const vector<string> &filenames,
    u64 kmer_size,
    u64 max_chars_per_batch,
    u64 max_reads_per_batch,
    const vector<vector<char>> &seq,
    const vector<vector<u64>> &chars_before_newline,
    const vector<vector<u64>> &newlines_before_newfile,
    u64 max_batches
  ) {
    auto host = make_unique<ContinuousSequenceFileParser>(
      0,
      filenames,
      kmer_size,
      max_chars_per_batch,
      max_reads_per_batch,
      max_batches,
      max_batches,
      max_batches
    );
    u64 expected_batches = seq.size();
#pragma omp parallel sections num_threads(4)
    {
#pragma omp section
      host_generate(*host);
#pragma omp section
      {
        auto string_break_batch_producer
          = host->get_string_break_batch_producer();
        assert_string_break_batch_correct(
          *string_break_batch_producer,
          seq,
          expected_batches,
          chars_before_newline
        );
      }
#pragma omp section
      {
        auto string_sequence_batch_producer
          = host->get_string_sequence_batch_producer();
        assert_string_sequence_batch_correct(
          *string_sequence_batch_producer, seq, expected_batches
        );
      }
#pragma omp section
      {
        auto interval_batch_producer = host->get_interval_batch_producer();
        assert_interval_batch_correct(
          *interval_batch_producer,
          chars_before_newline,
          newlines_before_newfile,
          expected_batches
        );
      }
    }
  }

private:
  auto host_generate(ContinuousSequenceFileParser &host) const -> void {
    auto rng = get_uniform_int_generator(0UL, time_to_wait);
    sleep_for(milliseconds(rng()));
    host.read_and_generate();
  }

  auto assert_string_break_batch_correct(
    StringBreakBatchProducer &string_break_batch_producer,
    const vector<vector<char>> &seq,
    u64 expected_batches,
    const vector<vector<u64>> &chars_before_newline
  ) const -> void {
    auto rng = get_uniform_int_generator(0UL, time_to_wait);
    shared_ptr<StringBreakBatch> string_break_batch;
    u64 batches = 0;
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
    const vector<vector<char>> &seq,
    u64 expected_batches
  ) const -> void {
    auto rng = get_uniform_int_generator(0UL, time_to_wait);
    shared_ptr<StringSequenceBatch> string_sequence_batch;
    u64 batches = 0;
    for (batches = 0; string_sequence_batch_producer >> string_sequence_batch;
         ++batches) {
      sleep_for(milliseconds(rng()));
      EXPECT_EQ(seq[batches], *string_sequence_batch->seq);
    }
    EXPECT_EQ(batches, expected_batches);
  }

  auto assert_interval_batch_correct(
    IntervalBatchProducer &interval_batch_producer,
    const vector<vector<u64>> &chars_before_newline,
    const vector<vector<u64>> &newlines_before_newfile,
    u64 expected_batches
  ) const -> void {
    auto rng = get_uniform_int_generator(0UL, time_to_wait);
    shared_ptr<IntervalBatch> interval_batch;
    u64 batches = 0;
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
  const u64 kmer_size = 10;
  const u64 max_chars_per_batch = 999;
  const u64 max_reads_per_batch = 999;
  const vector<string> str_seq
    = {"1ACTGCAATGGGCAATATGTCTCTGTGTGGATTAC23TCTAGCTACTACTACTGA"
       "TGGATGGAATGTGATG45TGAGTGAGATGAGGTGATAGTGACGTAGTGAGGA61A"
       "CTGCAATGGGCAATATGTCTCTGTGTGGATTAC23TCTAGCTACTACTACTGATG"
       "GATGGAATGTGATG45TGAGTGAGATGAGGTGATAGTGACGTAGTGAGGA6"};
  auto seq = to_char_vec(str_seq);
  const vector<vector<u64>> chars_before_newline
    = {{36UL, 36UL * 2UL, 36UL * 3UL, 36UL * 4UL, 36UL * 5UL, 36UL * 6UL, max}};
  const vector<vector<u64>> newlines_before_newfile = {{3, max}};
  for (auto max_batches : {1, 2, 3, 7}) {
    run_test(
      {"test_objects/small_fasta.fna", "test_objects/small_fastq.fnq"},
      kmer_size,
      max_chars_per_batch,
      max_reads_per_batch,
      seq,
      chars_before_newline,
      newlines_before_newfile,
      max_batches
    );
  }
}

TEST_F(ContinuousSequenceFileParserTest, GetSplitBatchesBigMaxChar) {
  const u64 kmer_size = 15;
  const u64 max_chars_per_batch = 36 * 3 + 10;
  const u64 max_reads_per_batch = 999;
  const vector<string> str_seq = {
    "1ACTGCAATGGGCAATATGTCTCTGTGTGGATTAC23TCTAGCTACTACTACTGATGGATGGAATGTGATG45T"
    "GAGTGAGATGAGGTGATAGTGACGTAGTGAGGA61ACTGCAATG",
    "1ACTGCAATGGGCAATATGTCTCTGTGTGGATTAC23TCTAGCTACTACTACTGATGGATGGAATGTGATG45T"
    "GAGTGAGATGAGGTGATAGTGACGTAGTGAGGA6",
  };
  auto seq = to_char_vec(str_seq);
  const vector<vector<u64>> chars_before_newline
    = {{36, 36ULL * 2, 36ULL * 3, max}, {36, 36ULL * 2, 36ULL * 3, max}};
  vector<vector<u64>> newlines_before_newfile = {{3, max}, {max}};
  for (auto max_batches : {1, 2, 3, 7}) {
    run_test(
      {"test_objects/small_fasta.fna", "test_objects/small_fastq.fnq"},
      kmer_size,
      max_chars_per_batch,
      max_reads_per_batch,
      seq,
      chars_before_newline,
      newlines_before_newfile,
      max_batches
    );
  }
}

TEST_F(ContinuousSequenceFileParserTest, GetSplitBatchesSmallMaxChar) {
  u64 kmer_size = 3;
  const u64 max_chars_per_batch = 36 * 3 - 10;
  const u64 max_reads_per_batch = 999;
  const vector<string> str_seq = {
    "1ACTGCAATGGGCAATATGTCTCTGTGTGGATTAC23TCTAGCTACTACTACT"
    "GATGGATGGAATGTGATG45TGAGTGAGATGAGGTGATAGTGACG",
    "CGTAGTGAGGA61ACTGCAATGGGCAATATGTCTCTGTGTGGATTAC23TCTA"
    "GCTACTACTACTGATGGATGGAATGTGATG45TGAGTGAGATGAG",
    "AGGTGATAGTGACGTAGTGAGGA6"};
  auto seq = to_char_vec(str_seq);
  const vector<vector<u64>> chars_before_newline
    = {{36, 36ULL * 2, max}, {12, 12 + 36, 12 + 36ULL * 2, max}, {24, max}};
  const vector<vector<u64>> newlines_before_newfile = {{max}, {1, max}, {max}};
  for (auto max_batches : {1, 2, 3, 7}) {
    run_test(
      {"test_objects/small_fasta.fna", "test_objects/small_fastq.fnq"},
      kmer_size,
      max_chars_per_batch,
      max_reads_per_batch,
      seq,
      chars_before_newline,
      newlines_before_newfile,
      max_batches
    );
  }
}

TEST_F(ContinuousSequenceFileParserTest, IncorrectFileAndVerySmallMaxChar) {
  const u64 kmer_size = 3;
  const u64 max_chars_per_batch = 30;
  const u64 max_reads_per_batch = 999;
  const vector<string> str_seq = {
    {"1ACTGCAATGGGCAATATGTCTCTGTGTGG"},
    {"GGATTAC23TCTAGCTACTACTACTGATGG"},
    {"GGATGGAATGTGATG45TGAGTGAGATGAG"},
    {"AGGTGATAGTGACGTAGTGAGGA6"}};
  auto seq = to_char_vec(str_seq);
  const vector<vector<u64>> chars_before_newline
    = {{max}, {8, max}, {16, max}, {24, max}};
  const vector<vector<u64>> newlines_before_newfile
    = {{max}, {max}, {max}, {1, max}};
  for (auto max_batches : {1, 2, 3, 7}) {
    run_test(
      {"test_objects/small_fasta.fna", "garbage_filename"},
      kmer_size,
      max_chars_per_batch,
      max_reads_per_batch,
      seq,
      chars_before_newline,
      newlines_before_newfile,
      max_batches
    );
  }
}

}  // namespace sbwt_search
