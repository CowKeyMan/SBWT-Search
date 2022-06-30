#include <cmath>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "IndexFileParser.h"
#include "TypeDefinitionUtils.h"

using std::make_unique;
using std::unique_ptr;

namespace sbwt_search {

#include <iostream>
using std::cout;

class IndexFileParserTest: public ::testing::Test {
protected:
  void assert_c_file_correct(IndexFileParser &host) {
    vector<u64> bits = {1, 65564098, 127380657, 188944656};
    ASSERT_EQ(bits.size(), host.get_bit_vector_size());
    for (size_t i = 0; i < bits.size(); ++i) {
      ASSERT_EQ(bits[i], host.get_bit_vector_pointer()[i])
        << " unequal at index " << i;
    }
  }
  void assert_bwt_file_correct(IndexFileParser &host) {
    ASSERT_EQ(254572979, host.get_bits_total());
    ASSERT_EQ(3977703, host.get_bit_vector_size());
    ASSERT_EQ(18446744073709522218ULL, host.get_bit_vector_pointer()[0]);
  }
};

TEST_F(IndexFileParserTest, TestParseBitVectorsCFile) {
  auto host = IndexFileParser("test_objects/C.bit_vector");
  host.parse_c_bit_vector();
  assert_c_file_correct(host);
}

TEST_F(IndexFileParserTest, TestParseBitVectorsBwtFile) {
  auto host = IndexFileParser("test_objects/BWT_A.bit_vector");
  host.parse_sbwt_bit_vector();
  assert_bwt_file_correct(host);
}


}
