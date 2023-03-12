#include <chrono>
#include <memory>
#include <thread>

#include "gtest/gtest.h"

#include "BatchObjects/StringBreakBatch.h"
#include "PositionsBuilder/ContinuousPositionsBuilder.h"
#include "Tools/DummyBatchProducer.hpp"
#include "Tools/RNGUtils.hpp"

namespace sbwt_search {

using rng_utils::get_uniform_int_generator;
using std::make_shared;
using std::shared_ptr;
using std::chrono::milliseconds;
using std::this_thread::sleep_for;
using test_utils::DummyBatchProducer;

using DummyStringBreakBatchProducer = DummyBatchProducer<StringBreakBatch>;

auto get_producer(
  vector<vector<u64>> &chars_before_newline, vector<u64> string_sizes
) -> shared_ptr<DummyStringBreakBatchProducer> {
  vector<shared_ptr<StringBreakBatch>> b;
  for (u64 i = 0; i < string_sizes.size(); ++i) {
    b.push_back(make_shared<StringBreakBatch>(
      StringBreakBatch({&chars_before_newline[i], string_sizes[i]})
    ));
  }
  return make_shared<DummyStringBreakBatchProducer>(b);
}

class ContinuousPositionsBuilderTest: public ::testing::Test {
protected:
  vector<vector<u64>> cumsum_string_lengths, cumsum_positions_per_string,
    expected_positions;

  auto run_test(
    u64 kmer_size,
    vector<vector<u64>> chars_before_newline,
    vector<u64> string_sizes,
    vector<vector<u64>> expected_positions,
    u64 max_batches = 7
  ) {
    const u64 time_to_wait = 200;
    const auto max_chars_per_batch = 999;
    auto producer = get_producer(chars_before_newline, string_sizes);
    auto host = ContinuousPositionsBuilder(
      0, producer, kmer_size, max_chars_per_batch, max_batches
    );
    u64 expected_batches = chars_before_newline.size();
    u64 batches = 0;
#pragma omp parallel sections private(batches) num_threads(2)
    {
#pragma omp section
      {
        auto rng = get_uniform_int_generator(0UL, time_to_wait);
        sleep_for(milliseconds(rng()));
        host.read_and_generate();
      }
#pragma omp section
      {
        auto rng = get_uniform_int_generator(0UL, time_to_wait);
        shared_ptr<PositionsBatch> positions_batch;
        for (batches = 0; host >> positions_batch; ++batches) {
          sleep_for(milliseconds(rng()));
          EXPECT_EQ(expected_positions[batches], positions_batch->positions);
        }
        EXPECT_EQ(batches, expected_batches);
      }
    }
  }
};

TEST_F(ContinuousPositionsBuilderTest, Basic) {
  const vector<vector<u64>> chars_before_newline
    = {{8, 9, 11, 16, u64(-1)}, {8, 9, 11, 17, u64(-1)}};
  const vector<u64> string_sizes = {20, 15};
  const vector<vector<u64>> expected_positions = {
    {0, 1, 2, 3, 4, 5, 11, 12, 13, 16, 17}, {0, 1, 2, 3, 4, 5, 11, 12, 13, 14}};
  u64 kmer_size = 3;
  for (auto max_batches : {1, 2, 3, 7}) {
    run_test(
      kmer_size,
      chars_before_newline,
      string_sizes,
      expected_positions,
      max_batches
    );
  }
}

}  // namespace sbwt_search
