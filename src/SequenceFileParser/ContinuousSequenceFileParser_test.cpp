#include <chrono>
#include <functional>
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
#include "Utils/RNGUtils.h"

using rng_utils::get_uniform_generator;
using std::make_unique;
using std::shared_ptr;
using std::unique_ptr;
using std::chrono::milliseconds;
using std::this_thread::sleep_for;

namespace sbwt_search {

class ContinuousSequenceFileParserTest: public ::testing::Test {
  protected:
    vector<string> fasta_and_fastq
      = { "test_objects/small_fasta.fna", "test_objects/small_fastq.fnq" };
    shared_ptr<StringSequenceBatchProducer> string_sequence_batch_producer;
    shared_ptr<StringBreakBatchProducer> string_break_batch_producer;
    shared_ptr<IntervalBatchProducer> interval_batch_producer;
    unique_ptr<ContinuousSequenceFileParser> host;

  private:
    auto set_host(
      vector<string> filenames,
      uint kmer_size,
      size_t max_chars_per_batch,
      uint max_batches = 7
    ) -> void {
      string_sequence_batch_producer
        = make_shared<StringSequenceBatchProducer>(max_batches);
      string_break_batch_producer
        = make_shared<StringBreakBatchProducer>(max_batches);
      interval_batch_producer = make_shared<IntervalBatchProducer>(max_batches);
      host = make_unique<ContinuousSequenceFileParser>(
        filenames,
        kmer_size,
        max_chars_per_batch,
        max_batches,
        string_sequence_batch_producer,
        string_break_batch_producer,
        interval_batch_producer
      );
    }

  protected:
    auto run_test(
      vector<string> filenames,
      uint kmer_size,
      size_t max_chars_per_batch,
      vector<string> &seq,
      vector<vector<size_t>> &string_breaks,
      vector<vector<size_t>> &newlines_before_newfile,
      uint max_batches
    ) {
      set_host(filenames, kmer_size, max_chars_per_batch, max_batches);
      size_t expected_batches = seq.size();
      size_t batches = 0;
      uint time_to_wait = 100;
#pragma omp parallel sections private(batches)
      {
#pragma omp section
        {
          auto rng = get_uniform_generator(0, 100);
          sleep_for(milliseconds(rng()));
          host->read_and_generate();
        }
#pragma omp section
        {
          auto rng = get_uniform_generator(0, 100);
          shared_ptr<StringBreakBatch> string_break_batch;
          for (batches = 0;
               (*string_break_batch_producer) >> string_break_batch;
               ++batches) {
            sleep_for(milliseconds(rng()));
            EXPECT_EQ(
              string_breaks[batches], *string_break_batch->string_breaks
            );
            EXPECT_EQ(seq[batches].size(), string_break_batch->string_size);
          }
          EXPECT_EQ(batches, expected_batches);
        }
#pragma omp section
        {
          auto rng = get_uniform_generator(0, 100);
          shared_ptr<StringSequenceBatch> string_sequence_batch;
          for (batches = 0;
               (*string_sequence_batch_producer) >> string_sequence_batch;
               ++batches) {
            sleep_for(milliseconds(rng()));
            EXPECT_EQ(seq[batches], *string_sequence_batch->seq);
          }
          EXPECT_EQ(batches, expected_batches);
        }
#pragma omp section
        {
          auto rng = get_uniform_generator(0, 100);
          shared_ptr<IntervalBatch> interval_batch;
          for (batches = 0; (*interval_batch_producer) >> interval_batch;
               ++batches) {
            sleep_for(milliseconds(rng()));
            EXPECT_EQ(string_breaks[batches], *interval_batch->string_breaks);
            EXPECT_EQ(
              newlines_before_newfile[batches],
              interval_batch->newlines_before_newfile
            );
          }
          EXPECT_EQ(batches, expected_batches);
        }
      }
    }
};

TEST_F(ContinuousSequenceFileParserTest, GetAllInOneBatch) {
  uint kmer_size = 10;
  size_t max_chars_per_batch = 999;
  vector<string> seq
    = { "1ACTGCAATGGGCAATATGTCTCTGTGTGGATTAC23TCTAGCTACTACTACTGA"
        "TGGATGGAATGTGATG45TGAGTGAGATGAGGTGATAGTGACGTAGTGAGGA61A"
        "CTGCAATGGGCAATATGTCTCTGTGTGGATTAC23TCTAGCTACTACTACTGATG"
        "GATGGAATGTGATG45TGAGTGAGATGAGGTGATAGTGACGTAGTGAGGA6" };
  vector<vector<size_t>> string_breaks
    = { { 36, 36 * 2, 36 * 3, 36 * 4, 36 * 5, 36 * 6 } };
  vector<vector<size_t>> newlines_before_newfile = { { 3, 6 } };
  for (auto &v: newlines_before_newfile) { v.push_back(size_t(-1)); }
  for (auto max_batches: { 1, 2, 3, 7 }) {
    run_test(
      fasta_and_fastq,
      kmer_size,
      max_chars_per_batch,
      seq,
      string_breaks,
      newlines_before_newfile,
      max_batches
    );
  }
}

TEST_F(ContinuousSequenceFileParserTest, GetSplitBatchesBigMaxChar) {
  uint kmer_size = 15;
  size_t max_chars_per_batch = 36 * 3 + 10;
  vector<string> seq = {
    "1ACTGCAATGGGCAATATGTCTCTGTGTGGATTAC23TCTAGCTACTACTACTGATGGATGGAATGTGATG45T"
    "GAGTGAGATGAGGTGATAGTGACGTAGTGAGGA61ACTGCAATG",
    "1ACTGCAATGGGCAATATGTCTCTGTGTGGATTAC23TCTAGCTACTACTACTGATGGATGGAATGTGATG45T"
    "GAGTGAGATGAGGTGATAGTGACGTAGTGAGGA6",
  };
  vector<vector<size_t>> string_breaks
    = { { 36, 36 * 2, 36 * 3 }, { 36, 36 * 2, 36 * 3 } };
  vector<vector<size_t>> newlines_before_newfile = { { 3 }, { 3 } };
  for (auto &v: newlines_before_newfile) { v.push_back(size_t(-1)); }
  for (auto max_batches: { 1, 2, 3, 7 }) {
    run_test(
      fasta_and_fastq,
      kmer_size,
      max_chars_per_batch,
      seq,
      string_breaks,
      newlines_before_newfile,
      max_batches
    );
  }
}

TEST_F(ContinuousSequenceFileParserTest, GetSplitBatchesSmallMaxChar) {
  uint kmer_size = 3;
  size_t max_chars_per_batch = 36 * 3 - 10;
  vector<string> seq = { "1ACTGCAATGGGCAATATGTCTCTGTGTGGATTAC23TCTAGCTACTACTACT"
                         "GATGGATGGAATGTGATG45TGAGTGAGATGAGGTGATAGTGACG",
                         "CGTAGTGAGGA61ACTGCAATGGGCAATATGTCTCTGTGTGGATTAC23TCTA"
                         "GCTACTACTACTGATGGATGGAATGTGATG45TGAGTGAGATGAG",
                         "AGGTGATAGTGACGTAGTGAGGA6" };
  vector<vector<size_t>> string_breaks
    = { { 36, 36 * 2 }, { 12, 12 + 36, 12 + 36 * 2 }, { 24 } };
  vector<vector<size_t>> newlines_before_newfile = { {}, { 1 }, { 1 } };
  for (auto &v: newlines_before_newfile) { v.push_back(size_t(-1)); }
  for (auto max_batches: { 1, 2, 3, 7 }) {
    run_test(
      fasta_and_fastq,
      kmer_size,
      max_chars_per_batch,
      seq,
      string_breaks,
      newlines_before_newfile,
      max_batches
    );
  }
}

TEST_F(ContinuousSequenceFileParserTest, IncorrectFileAndVerySmallMaxChar) {
  uint kmer_size = 3;
  size_t max_chars_per_batch = 30;
  vector<string> seq = { { "1ACTGCAATGGGCAATATGTCTCTGTGTGG" },
                         { "GGATTAC23TCTAGCTACTACTACTGATGG" },
                         { "GGATGGAATGTGATG45TGAGTGAGATGAG" },
                         { "AGGTGATAGTGACGTAGTGAGGA6" } };
  vector<vector<size_t>> string_breaks = { {}, { 8 }, { 16 }, { 24 } };
  vector<vector<size_t>> newlines_before_newfile = { {}, {}, {}, { 1, 1 } };
  for (auto &v: newlines_before_newfile) { v.push_back(size_t(-1)); }
  for (auto max_batches: { 1, 2, 3, 7 }) {
    run_test(
      { "test_objects/small_fasta.fna", "garbage_filename" },
      kmer_size,
      max_chars_per_batch,
      seq,
      string_breaks,
      newlines_before_newfile,
      max_batches
    );
  }
}

}  // namespace sbwt_search
