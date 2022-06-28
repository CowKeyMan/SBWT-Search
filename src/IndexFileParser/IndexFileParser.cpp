#include <vector>
#include <array>

#include "IndexFileParser.h"
#include "IndexFileParser.h"
#include <sdsl/bit_vectors.hpp>
#include "TypeDefinitionUtils.h"
#include "MathUtils.hpp"

using sdsl::bit_vector;
using std::vector;
using std::begin;
using std::end;

namespace sbwt_search {

void IndexFileParser::parse_sdsl() {
  bit_vector sdsl_vector;
  load_from_file(sdsl_vector, filename.c_str());
  bit_vector_size = round_up<u64>(sdsl_vector.size(), 64) / 64;
  bit_vector_pointer = sdsl_vector.data();
}

#include <iostream>
using std::cout;

void IndexFileParser::parse_bit_vectors() {
  std::ifstream file(filename, std::ios::in | std::ios::binary);
  file.read(reinterpret_cast<char*>(&bit_vector_size), sizeof(u64));
  custom_bit_vector.resize(bit_vector_size, 0);
  file.read(reinterpret_cast<char*>(&custom_bit_vector[0]), sizeof(u64) * bit_vector_size);
  bit_vector_pointer = &custom_bit_vector[0];
}

}
