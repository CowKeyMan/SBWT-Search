#include <gtest/gtest.h>

#include "QueryReader.h"

namespace sbwt_search {

const auto read_0
  = "GACTGCAATGGGCAATATGTCTCTGTGTGGATTAAAAAAAGAGTCTCTGACAGCAGCTTCTGAACTGGT"
    "TACCTGCCGTGAGTAAATTAAAATTTTATTG";
const auto read_6
  = "GTTTCATGGATGTTGTGTACTCTGTAATTTTTATCTGTCTGTGCGCTATGCCTATATTGGTTAAAGTAT"
    "TTAGTGACCTAAGTCAATAAAATTTTAATTT";

TEST(QueryReader, ReadFASTA) {
  auto host = QueryReader("test_objects/test_query.fna", 30).read();
  ASSERT_EQ(read_0, host.get_reads()[0]);
  ASSERT_EQ(read_6, host.get_reads()[6]);
  ASSERT_EQ(7, host.get_reads().size());
  ASSERT_EQ(700, host.get_total_letters());
  ASSERT_EQ(497, host.get_total_positions());
}

}
