#include <memory>
#include <stdexcept>

#include <gtest/gtest.h>

#include "SequenceFileParser/SequenceFileParser.h"

using std::make_unique;
using std::unique_ptr;

namespace sbwt_search {

const auto seq_0 = "GACTG";
const auto seq_3 = "TA";

class SequenceFileParserTest: public ::testing::Test {
  protected:
    SequenceFileParser host;

    SequenceFileParserTest(): host("test_objects/test_query.fna", 3) {}

    void shared_tests() {
      auto seqs = host.get_seqs();
      ASSERT_EQ(seq_0, (*seqs)[0]);
      ASSERT_EQ(seq_3, (*seqs)[3]);
      ASSERT_EQ(4, seqs->size());
      ASSERT_EQ(15, host.get_total_letters());
      ASSERT_EQ(7, host.get_total_positions());
    }
};

TEST_F(SequenceFileParserTest, ParseKseqppStreams) {
  host.parse_kseqpp_streams();
  shared_tests();
}

TEST_F(SequenceFileParserTest, ParseKseqppRead) {
  host.parse_kseqpp_read();
  shared_tests();
}

TEST_F(SequenceFileParserTest, ParseKseqppGzStream) {
  host.parse_kseqpp_gz_stream();
  shared_tests();
}

TEST_F(SequenceFileParserTest, ParseKseqppGzRead) {
  host.parse_kseqpp_gz_read();
  shared_tests();
}

}
