#include <chrono>
#include <memory>
#include <thread>

#include "gtest/gtest.h"

#include "BatchObjects/CumulativePropertiesBatch.h"
#include "PositionsBuilder/ContinuousPositionsBuilder.hpp"
#include "TestUtils/GeneralTestUtils.hpp"

using std::make_shared;
using std::shared_ptr;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using std::this_thread::sleep_for;

namespace sbwt_search {

class DummyProducer {
  private:
    int counter = 0;
    vector<vector<u64>> cumsum_string_lengths, cumsum_positions_per_string;

  public:
    DummyProducer(
      vector<vector<u64>> &cumsum_string_lengths,
      vector<vector<u64>> &cumsum_positions_per_string
    ):
        cumsum_string_lengths(cumsum_string_lengths),
        cumsum_positions_per_string(cumsum_positions_per_string) {}

    auto operator>>(shared_ptr<CumulativePropertiesBatch> &batch) -> bool {
      if (counter < cumsum_string_lengths.size()) {
        batch = make_shared<CumulativePropertiesBatch>();
        batch->cumsum_string_lengths = cumsum_string_lengths[counter];
        batch->cumsum_positions_per_string
          = cumsum_positions_per_string[counter];
        ++counter;
        return true;
      }
      return false;
    }
};

// string lengths: 4, 2 ,5
vector<u64> cumsum_string_lengths_1 = { 0, 4, 6, 11 };
vector<u64> cumsum_positions_per_string_1 = { 0, 2, 2, 5 };
vector<u64> expected_positions_1 = { 0, 1, 6, 7, 8 };
// string lengths: 6, 1, 4
vector<u64> cumsum_string_lengths_2 = { 0, 6, 7, 11 };
vector<u64> cumsum_positions_per_string_2 = { 0, 4, 4, 6 };
vector<u64> expected_positions_2 = { 0, 1, 2, 3, 7, 8 };

class ContinuousPositionsBuilderTest: public ::testing::Test {
  protected:
    vector<vector<u64>> cumsum_string_lengths, cumsum_positions_per_string,
      expected_positions;
    uint kmer_size = 3;

    auto shared_tests() -> void {
      auto parser = make_shared<DummyProducer>(
        cumsum_string_lengths, cumsum_positions_per_string
      );
      auto host = ContinuousPositionsBuilder<DummyProducer>(parser, kmer_size);
      host.read_and_generate();
      shared_ptr<vector<u64>> output;
      for (int i = 0; host >> output; ++i) {
        assert_vectors_equal(expected_positions[i], *output);
      }
    }
};

TEST_F(ContinuousPositionsBuilderTest, SingleBatch) {
  cumsum_string_lengths = { { cumsum_string_lengths_1 } };
  cumsum_positions_per_string = { { cumsum_positions_per_string_1 } };
  expected_positions = { { expected_positions_1 } };
  shared_tests();
}

TEST_F(ContinuousPositionsBuilderTest, MultipleBatches) {
  cumsum_string_lengths = { cumsum_string_lengths_1, cumsum_string_lengths_2 };
  cumsum_positions_per_string
    = { cumsum_positions_per_string_1, cumsum_positions_per_string_2 };
  expected_positions = { expected_positions_1, expected_positions_2 };
  shared_tests();
}

TEST_F(ContinuousPositionsBuilderTest, Parallel) {
  const uint threads = 2, iterations = 60;
  auto sleep_amount = 200;
  auto max_positions_per_batch = 999;
  auto max_batches = 3;
  milliseconds::rep read_time;
  cumsum_string_lengths = {};
  cumsum_positions_per_string = {};
  for (uint i = 0; i < iterations / 2; ++i) {
    cumsum_string_lengths.push_back(cumsum_string_lengths_1);
    cumsum_positions_per_string.push_back(cumsum_positions_per_string_1);
    expected_positions.push_back(expected_positions_1);

    cumsum_string_lengths.push_back(cumsum_string_lengths_2);
    cumsum_positions_per_string.push_back(cumsum_positions_per_string_2);
    expected_positions.push_back(expected_positions_2);
  }
  auto producer = make_shared<DummyProducer>(
    cumsum_string_lengths, cumsum_positions_per_string
  );
  auto host = ContinuousPositionsBuilder<DummyProducer>(
    producer, kmer_size, max_positions_per_batch, max_batches
  );
  vector<vector<u64>> outputs;
  int counter = 0;
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
      sleep_for(milliseconds(sleep_amount));
      shared_ptr<vector<u64>> output;
      for (uint i = 0; host >> output; ++i) { outputs.push_back(*output); };
    }
  }
  ASSERT_EQ(outputs.size(), iterations);
  for (uint i = 0; i < iterations; ++i) {
    assert_vectors_equal(expected_positions[i], outputs[i]);
  }
  ASSERT_GE(read_time, sleep_amount);
}

}  // namespace sbwt_search
