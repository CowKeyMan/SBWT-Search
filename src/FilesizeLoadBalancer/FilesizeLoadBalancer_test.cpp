#include <gtest/gtest.h>

#include "FilesizeLoadBalancer/FilesizeLoadBalancer.h"

namespace sbwt_search {

class FilesizeLoadBalancerTest: public ::testing::Test {
  vector<string> in_files = {
    "test_objects/example_index_search_result.txt",  // 90b
    "test_objects/example_index_search_result_with_newlines_at_start.txt",  // 92b
    "test_objects/filenames.list",  // 32b
    "test_objects/small_fasta.fna"  // 250b
  };
  vector<string> out_files = {"90", "92", "32", "250"};

protected:
  auto run_test(
    u64 partitions,
    const vector<vector<string>> &expected_in_files,
    const vector<vector<string>> &expected_out_files
  ) -> void {
    auto [actual_in_files, actual_out_files]
      = FilesizeLoadBalancer(in_files, out_files).partition(partitions);
    EXPECT_EQ(actual_in_files, expected_in_files);
    EXPECT_EQ(actual_out_files, expected_out_files);
  }
};

TEST_F(FilesizeLoadBalancerTest, TwoPartitions) {
  const vector<vector<string>> in_files = {
    {"test_objects/small_fasta.fna"},
    {"test_objects/example_index_search_result_with_newlines_at_start.txt",
     "test_objects/example_index_search_result.txt",
     "test_objects/filenames.list"}};
  const vector<vector<string>> out_files = {{"250"}, {"92", "90", "32"}};
  run_test(2, in_files, out_files);
}

TEST_F(FilesizeLoadBalancerTest, ThreePartitions) {
  const vector<vector<string>> in_files = {
    {"test_objects/small_fasta.fna"},
    {"test_objects/example_index_search_result_with_newlines_at_start.txt"},
    {"test_objects/example_index_search_result.txt",
     "test_objects/filenames.list"}};
  const vector<vector<string>> out_files = {{"250"}, {"92"}, {"90", "32"}};
  run_test(3, in_files, out_files);
}

/* TEST(FilesizeLoadBalancerTestIndividual, InvalidFile) {} */

}  // namespace sbwt_search
