#include <stdexcept>
#include <gtest/gtest.h>

#include "Utils/MemoryUnitsParser.h"

using std::runtime_error;

namespace units_parser {

TEST(MemoryUnitsParserTest, TestAll) {
  EXPECT_EQ(MemoryUnitsParser::convert("1GB"), 8 * 1024 * 1024 * 1024ULL);
  EXPECT_EQ(MemoryUnitsParser::convert("1 GB"), 8 * 1024 * 1024 * 1024ULL);
  EXPECT_EQ(
    MemoryUnitsParser::convert("102 GB"), 102 * 8 * 1024 * 1024 * 1024ULL
  );
  EXPECT_EQ(
    MemoryUnitsParser::convert("102GB"), 102 * 8 * 1024 * 1024 * 1024ULL
  );
  EXPECT_EQ(MemoryUnitsParser::convert("102 B"), 102 * 8ULL);
  EXPECT_EQ(MemoryUnitsParser::convert("102B"), 102 * 8ULL);
  EXPECT_EQ(MemoryUnitsParser::convert("102MB"), 102 * 8 * 1024 * 1024ULL);
  EXPECT_EQ(MemoryUnitsParser::convert("167MB"), 167 * 8 * 1024 * 1024ULL);
  EXPECT_EQ(MemoryUnitsParser::convert("167KB"), 167 * 8 * 1024ULL);
}

TEST(MemoryUnitsParserTest, InvalidInput) {
  try {
  EXPECT_EQ(MemoryUnitsParser::convert("1G"), 8 * 1024 * 1024 * 1024ULL);
  } catch(runtime_error &e) {
    ASSERT_STREQ(e.what(), "Unable to infer bits from 1G");
  }
}

}  // namespace units_parser
