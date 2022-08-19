#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest_pred_impl.h"

#include "SequenceFileParser/SequenceFileParser.h"
#include "TestUtils/GeneralTestUtils.hpp"

using std::out_of_range;
using std::string;
using std::unique_ptr;
using std::vector;

namespace sbwt_search {

class SequenceFileParserTest: public ::testing::Test {
  protected:
    SequenceFileParser host;
    vector<string> expected_seqs = { "GACTG", "AA", "GATCGA", "TA" };

    SequenceFileParserTest(): host("test_objects/test_query.fna") {}

    void shared_tests(const vector<string> &seqs) {
      assert_vectors_equal(expected_seqs, seqs);
    }
};

TEST_F(SequenceFileParserTest, ParseAll) {
  const vector<string> seqs = host.get_all();
  shared_tests(seqs);
}

TEST_F(SequenceFileParserTest, ParseOneByOne) {
  auto seqs = vector<string>();
  string s;
  while (host >> s) { seqs.push_back(move(s)); }
  shared_tests(seqs);
}

TEST(SequenceFileParserTest_NoClass, FileWithEmptyString) {
  SequenceFileParser host("test_objects/test_query_with_empty_line.fna");
  vector<string> expected_seqs = { "GACTG", "AA", "", "GATCGA", "TA" };
  const vector<string> seqs = host.get_all();
  assert_vectors_equal(expected_seqs, seqs);
}

}  // namespace sbwt_search
