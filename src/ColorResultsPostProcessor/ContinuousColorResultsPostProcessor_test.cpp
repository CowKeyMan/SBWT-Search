#include <limits>

#include "gtest/gtest.h"

#include "ColorResultsPostProcessor/ContinuousColorResultsPostProcessor.h"
#include "Tools/DummyBatchProducer.hpp"

namespace sbwt_search {

using std::make_shared;
using std::numeric_limits;
using test_utils::DummyBatchProducer;

const u64 na = static_cast<u64>(-1);

class ContinuousColorResultsPostProcessorTest: public ::testing::Test {
protected:
  auto run_test(
    const vector<vector<u64>> &results,
    const vector<vector<u64>> &warps_before_new_reads,
    const vector<vector<u64>> &expected_results,
    u64 max_batches,
    u64 num_colors
  ) -> void {
    // populate batches
    assert(results.size() == warps_before_new_reads.size());
    assert(results.size() == expected_results.size());
    vector<ColorSearchResultsBatch> results_batches(results.size());
    vector<WarpsBeforeNewReadBatch> warps_before_new_read_batches(results.size()
    );
    for (u64 i = 0; i < results.size(); ++i) {
      results_batches[i].results = make_shared<vector<u64>>(results[i]);
      warps_before_new_read_batches[i].warps_before_new_read
        = make_shared<vector<u64>>(warps_before_new_reads[i]);
    }
    // create producers and host
    auto results_producer
      = make_shared<DummyBatchProducer<ColorSearchResultsBatch>>(results_batches
      );
    auto warps_before_new_read_batch_producer
      = make_shared<DummyBatchProducer<WarpsBeforeNewReadBatch>>(
        warps_before_new_read_batches
      );
    auto host = ContinuousColorResultsPostProcessor(
      results_producer,
      warps_before_new_read_batch_producer,
      max_batches,
      num_colors
    );
    // assertions
    omp_set_nested(1);
#pragma omp parallel sections num_threads(2)
    {
#pragma omp section
      { host.read_and_generate(); }
#pragma omp section
      {
        shared_ptr<ColorSearchResultsBatch> results_batch;
        shared_ptr<ColorSearchResultsBatch> warps_before_new_read_batch;
        for (int batch_idx = 0; host >> results_batch; ++batch_idx) {
          for (u64 i = 0; i < expected_results[batch_idx].size(); ++i) {
            if (expected_results[batch_idx][i] != na) {
              EXPECT_EQ(
                (*results_batch->results)[i], expected_results[batch_idx][i]
              ) << "unequal at batch "
                << batch_idx << " index " << i;
            }
          }
          EXPECT_EQ(
            results_batch->results->size(), expected_results[batch_idx].size()
          );
        }
      }
    }
  }
};

TEST_F(ContinuousColorResultsPostProcessorTest, TestAll) {
  const u64 max = numeric_limits<u64>::max();
  const u64 num_colors = 3;
  const vector<vector<u64>> results = {
    {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6},
    {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}};
  const vector<vector<u64>> warps_before_new_reads
    = {{1, 3, 4, max}, {2, 4, max}};
  const vector<vector<u64>> squeezed_results = {
    {1, 1, 1, 5, 5, 5, na, na, na, 4, 4, 4, 11, 11, 11, na, na, na},
    {2, 4, 6, na, na, na, 2, 4, 6, na, na, na, 2, 4, 6, na, na, na}};
  run_test(results, warps_before_new_reads, squeezed_results, 1, num_colors);
}

}  // namespace sbwt_search
