#include <cmath>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "SeqToBitsConverter/SeqToBitsConverter.hpp"
#include "Utils/TypeDefinitions.h"

using std::make_shared;

namespace sbwt_search {

const vector<string> string_seqs = {
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

class SeqToBitsConverterTest: public ::testing::Test {
  protected:
    shared_ptr<vector<u64>> bit_seqs;
    SeqToBitsConverterTest():
        bit_seqs(make_shared<vector<u64>>(2)) {}
    void shared_tests() {
      ASSERT_EQ(bits, *bit_seqs);
    }
};

template <class Converter>
void convert_strings(Converter &host) {
  u64 cumulative_size = 0;
  for (auto &s: string_seqs) {
    host.convert(s, cumulative_size);
    cumulative_size += s.size();
  }
  host.add_int();
}

TEST_F(SeqToBitsConverterTest, CharToBitsVector) {
  auto host = SeqToBitsConverter<CharToBitsVector>(bit_seqs);
  convert_strings<decltype(host)>(host);
  shared_tests();
}

}
