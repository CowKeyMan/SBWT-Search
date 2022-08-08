#include <memory>
#include <stdexcept>

#include <gtest/gtest.h>

#include "SequenceFileParser/SequenceFileParser.h"

using std::make_unique;
using std::out_of_range;
using std::unique_ptr;

namespace sbwt_search {

const auto seq_0 = "GACTG";
const auto seq_3 = "TA";

class SequenceFileParserTest: public ::testing::Test {
  protected:
    SequenceFileParser host;

    SequenceFileParserTest(): host("test_objects/test_query.fna", 3) {}

    void shared_tests(const vector<string> &seqs) {
      ASSERT_EQ(seq_0, (seqs)[0]);
      ASSERT_EQ(seq_3, (seqs)[3]);
      ASSERT_EQ(4, seqs.size());
    }
};

TEST_F(SequenceFileParserTest, ParseAll) {
  const vector<string> vs = host.get_all();
  shared_tests(vs);
}

TEST_F(SequenceFileParserTest, ParseOneByOne) {
  auto seqs = make_unique<vector<string>>();
  while (!host.eof()) { seqs->push_back(move(host.get_next())); }
  shared_tests(*seqs);
}

}
