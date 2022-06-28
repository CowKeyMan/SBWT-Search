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


class IndexFileParserTest: public ::testing::Test {
protected:
  void shared_tests(IndexFileParser &host) {
    vector<u64> bits = {1, 65564098, 127380657, 188944656};
    ASSERT_EQ(bits.size(), host.get_bit_vector_size());
    for (size_t i = 0; i < bits.size(); ++i) {
      ASSERT_EQ(bits[i], host.get_bit_vector_pointer()[i]) << " unequal at index " << i;
    }
  }
};

TEST_F(IndexFileParserTest, TestParseBitVectors) {
  auto host = IndexFileParser("test_objects/C.bit_vector");
  host.parse_bit_vectors();
  shared_tests(host);
}

}
