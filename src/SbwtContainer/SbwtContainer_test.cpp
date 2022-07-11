#include <cmath>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <sdsl/bit_vectors.hpp>

#include "IndexFileParser/IndexFileParser.hpp"
#include "IndexWriter/IndexWriter.hpp"
#include "SbwtContainer/SbwtContainer.hpp"
#include "Utils/TestUtils.hpp"
#include "Utils/TypeDefinitionUtils.h"
#include "sdsl/bit_vectors.hpp"

using sdsl::bit_vector;
using std::make_unique;
using std::move;
using std::unique_ptr;

namespace sbwt_search {

const vector<vector<u64>> acgt = {
  // 00000000 00000000 00000000 00000000 00111010 01010101 01011101 10111100
  // 00000000 00000000 00000000 00000000 00110000 11111011 10101000 00000101
  { 978673084ULL, 821798917ULL },
  // 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000010
  // 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000
  { 2ULL, 0ULL },
  // 00000000 00000000 00000000 00000000 00000000 00000000 00000100 00000000
  // 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000
  { static_cast<u64>(pow(2, 10)), 0ULL },
  // 00000000 00000000 00000001 00000000 00000000 00000000 00000000 00000000
  // 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000
  { static_cast<u64>(pow(2, 40)), 0ULL },
};

template <class Container>
auto assert_containers_equal(Container &a, Container &b) {
  ASSERT_EQ(a.get_bits_total(), b.get_bits_total());
  ASSERT_EQ(a.get_bit_vector_size(), b.get_bit_vector_size());
  auto size = a.get_bit_vector_size();
  for (auto i = 0; i < 4; ++i) {
    ACGT letter = static_cast<ACGT>(i);
    assert_arrays_equals<u64>(a.get_acgt(letter), b.get_acgt(letter), size);
  }
}

TEST(SbwtContainerTest, ConstructWriteRead) {
  auto host = BitVectorSbwtContainer(
    move(acgt[0]), move(acgt[1]), move(acgt[2]), move(acgt[3]), 64 + 8 * 4
  );
  auto writer = BitVectorIndexWriter(host);
  writer.write("test_objects/tmp/bitvector");
  auto loaded_container
    = BitVectorIndexFileParser("test_objects/tmp/bitvector").parse(false);
  assert_containers_equal<BitVectorSbwtContainer>(host, loaded_container);
  ASSERT_EQ(loaded_container.get_acgt(ACGT::A)[0], 978673084);
}

TEST(SdslContainerTest, ConstructWriteRead) {
  vector<bit_vector> sdsl_acgt(acgt.size());
  for (auto i = 0; i < acgt.size(); ++i) {
    sdsl_acgt[i].bit_resize(64 + 8 * 4);
    for (auto i2 = 0; i2 < acgt[i].size(); ++i2) {
      sdsl_acgt[i].set_int(i2 * 64, acgt[i][i2], 64);
    }
  }
  auto host = SdslSbwtContainer(
    move(sdsl_acgt[0]),
    move(sdsl_acgt[1]),
    move(sdsl_acgt[2]),
    move(sdsl_acgt[3])
  );
  auto writer = SdslIndexWriter(host);
  writer.write("test_objects/tmp/bitvector");
  auto loaded_container
    = SdslIndexFileParser("test_objects/tmp/bitvector").parse(false);
  assert_containers_equal<SdslSbwtContainer>(host, loaded_container);
  ASSERT_EQ(loaded_container.get_acgt(ACGT::A)[0], 978673084);
}

TEST(SbwtContainerTest, ChangeEndianness) {
  auto host = BitVectorSbwtContainer(
    move(acgt[0]), move(acgt[1]), move(acgt[2]), move(acgt[3]), 64 + 8 * 4
  );
  host.change_acgt_endianness();
  // change endianness of 978673084 is: 13573098559561007104, which is:
  // 10111100 01011101 01010101 00111010 00000000 00000000 00000000 00000000
  // change endianness of 821798917 is: 407851949854031872, which is:
  // 00000101 10101000 11111011 00110000 00000000 00000000 00000000 00000000
  ASSERT_EQ(host.get_acgt(ACGT::A)[0], 13573098559561007104ULL);
  ASSERT_EQ(host.get_acgt(ACGT::A)[1], 407851949854031872ULL);
}

}
