#include <array>
#include <istream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <sdsl/bit_vectors.hpp>

#include "SbwtContainer/SbwtContainer.hpp"
#include "SbwtParser/SbwtParser.hpp"
#include "Utils/IOUtils.hpp"
#include "Utils/MathUtils.hpp"
#include "Utils/TypeDefinitions.h"

using sdsl::bit_vector;
using std::begin;
using std::end;
using std::istream;
using std::move;
using std::runtime_error;
using std::string;
using std::vector;

namespace sbwt_search {

auto SdslSbwtParser::do_parse() -> SdslSbwtContainer {
  ThrowingIfstream stream(filename, std::ios::in);
  assert_plain_matrix(stream);
  vector<bit_vector> acgt(4);
  for (int i = 0; i < 4; ++i) { acgt[i].load(stream); }
  return SdslSbwtContainer(
    move(acgt[0]), move(acgt[1]), move(acgt[2]), move(acgt[3])
  );
}

// Function credits:
// https://github.com/algbio/SBWT/blob/master/src/globals.cpp
void SdslSbwtParser::assert_plain_matrix(istream &stream) const {
  size_t size;
  stream.read(reinterpret_cast<char *>(&size), sizeof(u64));
  string variant(size, '\0');
  stream.read(reinterpret_cast<char *>(&variant[0]), size);
  if (variant != "plain-matrix") {
    throw runtime_error("Error input is not a plain-matrix SBWT");
  }
}

}
