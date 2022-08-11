#include <cmath>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "SeqToBitsConverter/ContinuousSeqToBitsConverter.hpp"
#include "TestUtils/GeneralTestUtils.hpp"
#include "Utils/TypeDefinitions.h"

using std::make_shared;
using std::string;
using std::vector;

namespace sbwt_search {

auto string_seqs = vector<string>({
  "ACgT",  // 00011011
  "gA",  // 1000
  "GAT",  // 100011
  "GtCa",  // 10110100
  "AAAAaAAaAAAAAAAaAAAAAAAAAAAAAAAA",  // 32 As = 64 0s
  "GC"  // 1001
});
// 1st 64b: 0001101110001000111011010000000000000000000000000000000000000000
// 2nd 64b: 0000000000000000000000000010010000000000000000000000000000000000
// We apply 0 padding on the right to get decimal equivalent
// Using some online converter, we get the following decimal equivalents:
const vector<u64> bits = { 1984096220112486400, 154618822656 };
vector<u32> batch_indexes = { 0, 0 };
vector<u32> in_batch_indexes = { 0, 1 };
vector<u64> string_indexes = { 0, 4 };
vector<u64> character_indexes = { 0, 32 - 13 };

vector<u64> expected_output = bits;

class DummyParser {
  public:
  private:
    int counter = 0;

  public:
    vector<u32> batch_indexes, in_batch_indexes;
    vector<u64> string_indexes, character_indexes;
    DummyParser(
      vector<u32> batch_indexes,
      vector<u32> in_batch_indexes,
      vector<u64> string_indexes,
      vector<u64> character_indexes
    ):
        batch_indexes(batch_indexes),
        in_batch_indexes(in_batch_indexes),
        string_indexes(string_indexes),
        character_indexes(character_indexes) {}
    auto operator>>(tuple<u32 &, u32 &, u64 &, u64 &> t) -> bool {
      if (counter < batch_indexes.size()) {
        set_output_variables(t);
        ++counter;
        return true;
      }
      return false;
    }

    auto set_output_variables(tuple<u32 &, u32 &, u64 &, u64 &> &t) -> void {
      auto [batch_index, in_batch_index, string_index, character_index] = t;
      batch_index = batch_indexes[counter];
      in_batch_index = in_batch_indexes[counter];
      string_index = string_indexes[counter];
      character_index = character_indexes[counter];
    }
};

TEST(SeqToBitsConverterTest, SingleBatch) {
  auto parser = DummyParser(
    batch_indexes, in_batch_indexes, string_indexes, character_indexes
  );
  auto host = ContinuousSeqToBitsConverter<DummyParser>(parser, 2, 1, 2);
  host.update(make_shared<vector<string>>(string_seqs));
  host.generate();
  vector<u64> output;
  host >> output;
  assert_vectors_equal(expected_output, output);
}

}
