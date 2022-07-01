#include <gtest/gtest.h>

#include "RankIndexBuilder.h"

class RankIndexBuilder_test: public ::testing::Test {
protected:
  RankIndexBuilder host;
  RankIndexBuilder_test():
    host(
      raw_sequences,
      (2 + 0 + 1 + 2 + 30 + 0),
      (4 + 2 + 3 + 4 + 32 + 2),
      kmer_size
    ) {}
  void shared_tests() {}
};

TEST_F(RawSequencesParserTest, ParseSerial) {
  host.build_index();
  shared_tests();
}
