#include <cmath>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "GlobalDefinitions.h"
#include "RawSequencesParser.h"

using std::make_unique;
using std::unique_ptr;

namespace sbwt_search {

auto bit_string_to_vector(string s) -> vector<u64> {
  s.resize(
    size_t(ceil(s.size() / 64.0)) * 64, '0'
  );  // resize to next multiple of 64 and pad with 0s
  auto bits = vector<uint64_t>();
  for (auto i = 0; i < s.size() / 64; ++i) {
    string sub = s.substr(64 * i, 64 * (i + 1));
    bits.push_back(strtoull(sub.c_str(), nullptr, 2));
  }
  return bits;
}

const auto raw_sequences = vector<string>{
  "ACgT",
  "gA",
  "GAT",
  "GtCa",
  "AAAAaAAaAAAAAAAaAAAAAAAAAAAAAAAA",  // 32 As
  "GC"};
const auto bit_string = string(
  "00011011"
  "1000"
  "100011"
  "10110100"
  "0000000000000000000000000000000000000000000000000000000000000000"  // 32 00s
  "1001"
);

class RawSequencesParserTest: public ::testing::Test {
protected:
  unique_ptr<RawSequencesParser> host;
  const vector<u64> bits = bit_string_to_vector(bit_string);

  RawSequencesParserTest() {
    host = make_unique<RawSequencesParser>(
      raw_sequences, (2 + 0 + 1 + 2 + 30 + 0), (4 + 2 + 3 + 4 + 32 + 2), 3
    );
  }

  void shared_tests() {
    ASSERT_EQ(bits, host->get_bit_seqs());
    ASSERT_EQ(0, host->get_positions()[0]);
    ASSERT_EQ(6, host->get_positions()[2]);
    ASSERT_EQ(42, host->get_positions()[34]);
  }
};

TEST_F(RawSequencesParserTest, ParseSerial) {
  host->parse_serial();
  shared_tests();
}

TEST_F(RawSequencesParserTest, AlreadyParsed) {
  const auto assertion_string = "RawSequencesParser has already parsed a file";
  host->parse_serial();
  try {
    host->parse_serial();
  } catch (std::logic_error &e) {
    ASSERT_EQ(string(e.what()), assertion_string);
  }
  try {
    host->parse_serial();
  } catch (std::logic_error &e) {
    ASSERT_EQ(string(e.what()), assertion_string);
  }
}

}
