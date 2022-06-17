#include <memory>
#include <stdexcept>

#include <gtest/gtest.h>

#include "QueryReader.h"

using std::make_unique;
using std::unique_ptr;

namespace sbwt_search {

const auto seq_0
  = "GACTGCAATGGGCAATATGTCTCTGTGTGGATTAAAAAAAGAGTCTCTGACAGCAGCTTCTGAACTGGT"
    "TACCTGCCGTGAGTAAATTAAAATTTTATTG";
const auto seq_6
  = "GTTTCATGGATGTTGTGTACTCTGTAATTTTTATCTGTCTGTGCGCTATGCCTATATTGGTTAAAGTAT"
    "TTAGTGACCTAAGTCAATAAAATTTTAATTT";

class QueryReaderTest: public ::testing::Test {
protected:
  unique_ptr<QueryReader> host;

  QueryReaderTest() {
    host = make_unique<QueryReader>("test_objects/test_query.fna", 30);
  }

  void shared_tests() {
    ASSERT_EQ(seq_0, host->get_seqs()[0]);
    ASSERT_EQ(seq_6, host->get_seqs()[6]);
    ASSERT_EQ(7, host->get_seqs().size());
    ASSERT_EQ(700, host->get_total_letters());
    ASSERT_EQ(497, host->get_total_positions());
  }
};

TEST_F(QueryReaderTest, ReadKseqppStreams) {
  host->parse_kseqpp_streams();
  shared_tests();
}

TEST_F(QueryReaderTest, ReadKseqppRead) {
  host->parse_kseqpp_read();
  shared_tests();
}

TEST_F(QueryReaderTest, AlreadyParsed) {
  const auto assertion_string = "QueryReader has already parsed a file";
  host->parse_kseqpp_read();
  try {
    host->parse_kseqpp_read();
  } catch (std::logic_error &e) {
    ASSERT_EQ(string(e.what()), assertion_string);
  }
  try {
    host->parse_kseqpp_streams();
  } catch (std::logic_error &e) {
    ASSERT_EQ(string(e.what()), assertion_string);
  }
}

}
