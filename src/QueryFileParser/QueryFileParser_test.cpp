#include <memory>
#include <stdexcept>

#include <gtest/gtest.h>

#include "QueryFileParser/QueryFileParser.h"

using std::make_unique;
using std::unique_ptr;

namespace sbwt_search {

const auto seq_0 = "GACTG";
const auto seq_3 = "TA";

class QueryFileParserTest: public ::testing::Test {
  protected:
    unique_ptr<QueryFileParser> host;

    QueryFileParserTest() {
      host = make_unique<QueryFileParser>("test_objects/test_query.fna", 3);
    }

    void shared_tests() {
      ASSERT_EQ(seq_0, host->get_seqs()[0]);
      ASSERT_EQ(seq_3, host->get_seqs()[3]);
      ASSERT_EQ(4, host->get_seqs().size());
      ASSERT_EQ(15, host->get_total_letters());
      ASSERT_EQ(7, host->get_total_positions());
    }
};

TEST_F(QueryFileParserTest, ParseKseqppStreams) {
  host->parse_kseqpp_streams();
  shared_tests();
}

TEST_F(QueryFileParserTest, ParseKseqppRead) {
  host->parse_kseqpp_read();
  shared_tests();
}

TEST_F(QueryFileParserTest, ParseKseqppGzStream) {
  host->parse_kseqpp_gz_stream();
  shared_tests();
}

TEST_F(QueryFileParserTest, ParseKseqppGzRead) {
  host->parse_kseqpp_gz_read();
  shared_tests();
}

}
