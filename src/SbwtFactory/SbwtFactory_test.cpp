#include <cmath>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <sdsl/bit_vectors.hpp>

#include "SbwtContainer/SbwtContainer.hpp"
#include "SbwtFactory/SbwtFactory.hpp"
#include "SbwtParser/SbwtParser.hpp"
#include "TestUtils/GeneralTestUtils.hpp"
#include "Utils/TypeDefinitions.h"
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

TEST(SbwtFactoryTest, BitVectorConstructWriteRead) {
  auto container = BitVectorSbwtContainer(
    move(acgt[0]), move(acgt[1]), move(acgt[2]), move(acgt[3]), 64 + 8 * 4
  );
  auto factory = BitVectorSbwtFactory();
  auto writer
    = factory.get_sbwt_writer(container, "test_objects/tmp/bitvector");
  writer.write();
  auto loaded_container
    = factory.get_sbwt_parser("test_objects/tmp/bitvector").parse();
  assert_containers_equal<BitVectorSbwtContainer>(container, loaded_container);
  ASSERT_EQ(loaded_container.get_acgt(ACGT::A)[0], 978673084);
}

auto build_sdsl_bit_vectors() -> vector<bit_vector> {
  vector<bit_vector> sdsl_acgt(acgt.size());
  for (auto i = 0; i < acgt.size(); ++i) {
    sdsl_acgt[i].bit_resize(64 + 8 * 4);
    for (auto i2 = 0; i2 < acgt[i].size(); ++i2) {
      sdsl_acgt[i].set_int(i2 * 64, acgt[i][i2], 64);
    }
  }
  return sdsl_acgt;
}

TEST(SbwtFactoryTest, SdslConstructWriteRead) {
  auto sdsl_acgt = build_sdsl_bit_vectors();
  // last 2 bytes of first number in acgt
  vector<u64> expected = { 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0 };
  for (size_t i = 0; i < expected.size(); ++i) {
    // assert reverse are equal
    ASSERT_EQ(expected[expected.size() - i - 1], sdsl_acgt[0][i]);
  }
  auto container = SdslSbwtContainer(
    move(sdsl_acgt[0]),
    move(sdsl_acgt[1]),
    move(sdsl_acgt[2]),
    move(sdsl_acgt[3])
  );
  auto factory = SdslSbwtFactory();
  auto writer
    = factory.get_sbwt_writer(container, "test_objects/tmp/bitvector");
  writer.write();
  auto sbwt_parser = factory.get_sbwt_parser("test_objects/tmp/bitvector");
  auto loaded_container = sbwt_parser.parse();
  assert_containers_equal<SdslSbwtContainer>(container, loaded_container);
  ASSERT_EQ(loaded_container.get_acgt(ACGT::A)[0], 978673084);
}

TEST(SbwtFactoryTest, BitVectorChangeEndianness) {
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

string incorrect_sbwt_filepath = "test_objects/tmp/incorrect.sbwt";

void write_incorrect_sbwt() {
  ThrowingOfstream stream(
    incorrect_sbwt_filepath, std::ios::out | std::ios::binary
  );
  string incorrect_format = "not_plain_matrix";
  u64 string_size = incorrect_format.size();
  stream.write(reinterpret_cast<char *>(&string_size), sizeof(u64));
  stream << incorrect_format;
}

TEST(SbwtFactoryTest, InvalidSdsl) {
  write_incorrect_sbwt();
  try {
    SdslSbwtParser(incorrect_sbwt_filepath).parse();
  } catch (runtime_error &e) {
    ASSERT_EQ(string(e.what()), "Error input is not a plain-matrix SBWT");
  }
}

}
