#include <chrono>
#include <cmath>
#include <memory>
#include <omp.h>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "SeqToBitsConverter/BitsProducer.hpp"
#include "SeqToBitsConverter/ContinuousSeqToBitsConverter.hpp"
#include "SeqToBitsConverter/InvalidCharsProducer.hpp"
#include "TestUtils/GeneralTestUtils.hpp"
#include "Utils/RNGUtils.h"
#include "Utils/TypeDefinitions.h"

using rng_utils::get_uniform_generator;
using std::make_shared;
using std::string;
using std::unique_ptr;
using std::vector;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using std::this_thread::sleep_for;

namespace sbwt_search {

class DummyStringSequenceBatchProducer {
  private:
    int counter = 0;
    vector<string> seqs;

  public:
    DummyStringSequenceBatchProducer(vector<string> &_seqs): seqs(_seqs) {}

    auto operator>>(shared_ptr<StringSequenceBatch> &batch) -> bool {
      if (counter < seqs.size()) {
        batch = make_shared<StringSequenceBatch>();
        batch->seq = &seqs[counter];
        ++counter;
        return true;
      }
      return false;
    }
};

class ContinuousSeqToBitsConverterTest: public ::testing::Test {
  protected:
    auto run_test(
      uint kmer_size,
      vector<string> &seqs,
      vector<vector<u64>> &bits,
      vector<vector<char>> &invalid_chars,
      size_t max_chars_per_batch,
      size_t max_batches
    ) {
      omp_set_nested(1);
      uint threads;
#pragma omp parallel
#pragma omp single
      { threads = omp_get_num_threads(); }
      auto producer = make_shared<DummyStringSequenceBatchProducer>(seqs);
      auto invalid_chars_producer
        = make_shared<InvalidCharsProducer<DummyStringSequenceBatchProducer>>(
          kmer_size, max_chars_per_batch, max_batches
        );
      auto bits_producer
        = make_shared<BitsProducer<DummyStringSequenceBatchProducer>>(
          max_chars_per_batch, max_batches
        );
      auto host = make_shared<
        ContinuousSeqToBitsConverter<DummyStringSequenceBatchProducer>>(
        producer,
        invalid_chars_producer,
        bits_producer,
        threads,
        max_chars_per_batch,
        max_batches
      );
      size_t expected_batches = seqs.size();
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
          shared_ptr<BitSeqBatch> bit_seq_batch;
          for (batches = 0; (*bits_producer) >> bit_seq_batch; ++batches) {
            sleep_for(milliseconds(rng()));
            EXPECT_EQ(bits[batches], bit_seq_batch->bit_seq);
          }
          EXPECT_EQ(batches, expected_batches);
        }
#pragma omp section
        {
          auto rng = get_uniform_generator(0, 100);
          shared_ptr<InvalidCharsBatch> invalid_chars_batch;
          for (batches = 0; (*invalid_chars_producer) >> invalid_chars_batch;
               ++batches) {
            sleep_for(milliseconds(rng()));
            EXPECT_EQ(
              invalid_chars[batches], invalid_chars_batch->invalid_chars
            );
          }
          EXPECT_EQ(batches, expected_batches);
        }
      }
    }
};

u64 convert_binary(string bin) {
  u64 total = 0;
  for (int i = bin.size(); i > 0; --i) {
    total += (bin[i - 1] == '1') * pow(2, bin.size() - i);
  }
  return total;
}

TEST_F(ContinuousSeqToBitsConverterTest, TestAll) {
  vector<string> seqs
    = { "ACgTgnGAtGtCa"  // A00 C01 g10 T11 g10 n00 G10 A00 t11 G10 t11 C01 a00
        "AAAAaAAaAAAAAAAaAAAAAAAAAAAAAAAA"  // 32 As = 64 0s
        "GC",  // 1001
        "nTAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAATn"
        "nAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAnG" };

  vector<vector<u64>> expected_bits
    = { {
          convert_binary(
            "0001101110001000111011010000000000000000000000000000000000000000"
          ),
          convert_binary(
            "0000000000000000000000000010010000000000000000000000000000000000"
          )  // We apply 0 padding on the right to get decimal equivalent
        },
        {
          convert_binary(
            "0011000000000000000000000000000000000000000000000000000000000000"
          ),
          convert_binary(
            "0000000000000000000000000000000000000000000000000000000000001100"
          ),
          convert_binary(
            "0000000000000000000000000000000000000000000000000000000000000000"
          ),
          convert_binary(
            "0000000000000000000000000000000000000000000000000000000000000010"
          )  // We apply 0 padding on the right to get decimal equivalent };
        } };

  const uint max_chars_per_batch = 200;
  for (auto kmer_size: { 3 }) {
    vector<vector<char>> expected_invalid_chars
      = { vector<char>(47 + kmer_size, 0), vector<char>(128 + kmer_size, 0) };
    expected_invalid_chars[0][5] = 1;
    for (auto i: { 0, 63, 64, 126 }) { expected_invalid_chars[1][i] = 1; }
    for (auto max_batches: { 1 }) {
      run_test(
        kmer_size,
        seqs,
        expected_bits,
        expected_invalid_chars,
        max_chars_per_batch,
        max_batches
      );
    }
  }
}

}  // namespace sbwt_search
