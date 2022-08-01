#include <cmath>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "RawSequencesParser/RawSequencesParser.hpp"
#include "Utils/TypeDefinitions.h"

using std::make_shared;

namespace sbwt_search {

const vector<string> raw_sequences = {
  "ACgT",  // 00011011
  "gA",  // 1000
  "GAT",  // 100011
  "GtCa",  // 10110100
  "AAAAaAAaAAAAAAAaAAAAAAAAAAAAAAAA",  // 32 As = 64 0s
  "GC"  // 1001
};
// 1st 64b: 0001101110001000111011010000000000000000000000000000000000000000
// 2nd 64b: 0000000000000000000000000010010000000000000000000000000000000000
// We apply 0 padding to the right
// Using some online converter, we get the following decimal equivalents:
const vector<u64> bits = { 1984096220112486400, 154618822656 };
const int kmer_size = 3;

#include <iostream>
using namespace std;

class RawSequencesParserTest: public ::testing::Test {
  protected:
    RawSequencesParser<CharToBitsVector> host;
    RawSequencesParserTest():
        host(
          make_unique<vector<string>>(raw_sequences),
          (2 + 0 + 1 + 2 + 30 + 0),
          (4 + 2 + 3 + 4 + 32 + 2),
          kmer_size
        ) {}
    void shared_tests() {
      auto seqs = host.get_bit_seqs();
      auto positions = host.get_positions();
      ASSERT_EQ(bits, *seqs);
      ASSERT_EQ(0, (*positions)[0]);
      ASSERT_EQ(6, (*positions)[2]);
      ASSERT_EQ(42, (*positions)[34]);
    }
};

TEST_F(RawSequencesParserTest, ParseSerial) {
  host.parse_serial();
  shared_tests();
}

TEST(RawSequencesParserTestCharToBits, CharToBitsArray) {
  RawSequencesParser<CharToBitsArray> host(
    make_unique<vector<string>>(raw_sequences),
    (2 + 0 + 1 + 2 + 30 + 0),
    (4 + 2 + 3 + 4 + 32 + 2),
    kmer_size
  );
  host.parse_serial();
  ASSERT_EQ(bits, *host.get_bit_seqs());
}

TEST(RawSequencesParserTestCharToBits, CharToBitsSwitch) {
  RawSequencesParser<CharToBitsSwitch> host(
    make_unique<vector<string>>(raw_sequences),
    (2 + 0 + 1 + 2 + 30 + 0),
    (4 + 2 + 3 + 4 + 32 + 2),
    kmer_size
  );
  host.parse_serial();
  ASSERT_EQ(bits, *host.get_bit_seqs());
}

}
