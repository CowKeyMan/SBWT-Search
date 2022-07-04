#include <array>
#include <vector>

#include <sdsl/bit_vectors.hpp>

#include "IndexFileParser/IndexFileParser.h"
#include "Utils/MathUtils.hpp"
#include "Utils/TypeDefinitionUtils.h"

using sdsl::bit_vector;
using std::begin;
using std::end;
using std::vector;

namespace sbwt_search {

auto IndexFileParser::parse_sdsl() -> void {
  bit_vector sdsl_vector;
  load_from_file(sdsl_vector, filename.c_str());
  bit_vector_size = round_up<u64>(sdsl_vector.size(), 64) / 64;
  bit_vector_pointer = sdsl_vector.data();
}

auto IndexFileParser::parse_c_bit_vector() -> void {
  std::ifstream file(filename, std::ios::in | std::ios::binary);
  file.read(reinterpret_cast<char *>(&bit_vector_size), sizeof(u64));
  custom_bit_vector.resize(bit_vector_size, 0);
  file.read(
    reinterpret_cast<char *>(&custom_bit_vector[0]),
    sizeof(u64) * bit_vector_size
  );
  bit_vector_pointer = &custom_bit_vector[0];
}

auto IndexFileParser::parse_sbwt_bit_vector() -> void {
  std::ifstream file(filename, std::ios::in | std::ios::binary);
  file.read(reinterpret_cast<char *>(&bits_total), sizeof(u64));
  bit_vector_size = round_up<u64>(bits_total, 64) / 64;
  u64 total_characters = round_up<u64>(bits_total, 8) / 8;
  custom_bit_vector.resize(bit_vector_size, 0);
  file.read(reinterpret_cast<char *>(&custom_bit_vector[0]), total_characters);
  for (auto i = size_t(0); i < bit_vector_size; ++i) {
    char *char_buffer = (char *)(&custom_bit_vector[i]);
    custom_bit_vector[i]
      = u64(char_buffer[0]) << (8 * 7) | u64(char_buffer[1]) << (8 * 6)
      | u64(char_buffer[2]) << (8 * 5) | u64(char_buffer[3]) << (8 * 4)
      | u64(char_buffer[4]) << (8 * 3) | u64(char_buffer[5]) << (8 * 2)
      | u64(char_buffer[6]) << (8 * 1) | u64(char_buffer[7]) << (8 * 0);
  }
  bit_vector_pointer = &custom_bit_vector[0];
}

}
