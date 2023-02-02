#include <chrono>
#include <cmath>
#include <memory>
#include <omp.h>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "BatchObjects/StringSequenceBatch.h"
#include "SeqToBitsConverter/BitsProducer.h"
#include "SeqToBitsConverter/ContinuousSeqToBitsConverter.h"
#include "SeqToBitsConverter/InvalidCharsProducer.h"
#include "Tools/DummyBatchProducer.hpp"
#include "Tools/RNGUtils.hpp"
#include "Tools/TestUtils.hpp"
#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using rng_utils::get_uniform_generator;
using std::make_shared;
using std::string;
using std::unique_ptr;
using std::vector;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using std::this_thread::sleep_for;
using test_utils::DummyBatchProducer;

using DummyStringSequenceBatchProducer
  = DummyBatchProducer<StringSequenceBatch>;

class ContinuousSeqToBitsConverterTest: public ::testing::Test {
protected:
  auto run_test(
    uint kmer_size,
    const vector<string> &seqs,
    const vector<vector<u64>> &bits,
    const vector<vector<char>> &invalid_chars,
    size_t max_chars_per_batch,
    size_t max_batches
  ) {
    omp_set_nested(1);
    uint threads = 0;
#pragma omp parallel
#pragma omp single
    { threads = omp_get_num_threads(); }
    vector<shared_ptr<StringSequenceBatch>> shared_seqs;
    for (auto &seq : seqs) {
      auto a = make_shared<StringSequenceBatch>(StringSequenceBatch({&seq}));
      shared_seqs.push_back(a);
    }
    auto producer = make_shared<DummyStringSequenceBatchProducer>(shared_seqs);
    auto host = make_shared<ContinuousSeqToBitsConverter>(
      producer, threads, kmer_size, max_chars_per_batch, max_batches
    );
    auto bits_producer = host->get_bits_producer();
    auto invalid_chars_producer = host->get_invalid_chars_producer();
    size_t expected_batches = seqs.size();
    size_t batches = 0;
    const uint time_to_wait = 100;
#pragma omp parallel sections private(batches) num_threads(3)
    {
#pragma omp section
      {
        auto rng = get_uniform_generator(0U, time_to_wait);
        sleep_for(milliseconds(rng()));
        host->read_and_generate();
      }
#pragma omp section
      {
        auto rng = get_uniform_generator(0U, time_to_wait);
        shared_ptr<BitSeqBatch> bit_seq_batch;
        for (batches = 0; (*bits_producer) >> bit_seq_batch; ++batches) {
          sleep_for(milliseconds(rng()));
          EXPECT_EQ(bits[batches], bit_seq_batch->bit_seq);
        }
        EXPECT_EQ(batches, expected_batches);
      }
#pragma omp section
      {
        auto rng = get_uniform_generator(0U, time_to_wait);
        shared_ptr<InvalidCharsBatch> invalid_chars_batch;
        for (batches = 0; (*invalid_chars_producer) >> invalid_chars_batch;
             ++batches) {
          sleep_for(milliseconds(rng()));
          EXPECT_EQ(invalid_chars[batches], invalid_chars_batch->invalid_chars);
        }
        EXPECT_EQ(batches, expected_batches);
      }
    }
  }
};

auto convert_binary(string bin) -> u64 {
  u64 total = 0;
  for (size_t i = bin.size(); i > 0; --i) {
    total += static_cast<u64>(bin.at(i - 1) == '1')
      * (1ULL << (bin.size() - i));  // 2^(bin.size() - i)
  }
  return total;
}

TEST_F(ContinuousSeqToBitsConverterTest, TestAll) {
  vector<string> seqs = {
    "ACgTgnGAtGtCa"  // A00 C01 g10 T11 g10 n00 G10 A00 t11 G10 t11 C01 a00
    "AAAAaAAaAAAAAAAaAAAAAAAAAAAAAAAA"  // 32 As = 64 0s
    "GC",  // 1001
    "nTAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAATn"
    "nAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAnG"};

  vector<vector<u64>> expected_bits = {
    {
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
    }};

  const uint max_chars_per_batch = 200;
  for (auto kmer_size : {3}) {
    const vector<vector<char>> expected_invalid_chars = [&] {
      vector<vector<char>> ret_val
        = {vector<char>(47 + kmer_size, 0), vector<char>(128 + kmer_size, 0)};
      ret_val[0][5] = 1;
      for (auto i : {0, 63, 64, 126}) { ret_val[1][i] = 1; }
      return ret_val;
    }();
    for (auto max_batches : {1}) {
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
