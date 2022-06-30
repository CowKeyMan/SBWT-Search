#include <array>
#include <vector>

#include <sdsl/bit_vectors.hpp>

#include "IndexFileParser.h"
#include "MathUtils.hpp"
#include "TypeDefinitionUtils.h"

using sdsl::bit_vector;
using std::begin;
using std::end;
using std::vector;

namespace sbwt_search {

void IndexFileParser::parse_sdsl() {
  bit_vector sdsl_vector;
  load_from_file(sdsl_vector, filename.c_str());
  bit_vector_size = round_up<u64>(sdsl_vector.size(), 64) / 64;
  bit_vector_pointer = sdsl_vector.data();
}

void IndexFileParser::parse_c_bit_vector() {
  std::ifstream file(filename, std::ios::in | std::ios::binary);
  file.read(reinterpret_cast<char *>(&bit_vector_size), sizeof(u64));
  custom_bit_vector.resize(bit_vector_size, 0);
  file.read(
    reinterpret_cast<char *>(&custom_bit_vector[0]),
    sizeof(u64) * bit_vector_size
  );
  bit_vector_pointer = &custom_bit_vector[0];
}

void IndexFileParser::parse_sbwt_bit_vector() {
  std::ifstream file(filename, std::ios::in | std::ios::binary);
  file.read(reinterpret_cast<char *>(&bits_total), sizeof(u64));
	bit_vector_size = round_up<u64>(bits_total, 64) / 64;
	auto char_buffer = vector<char>(round_up<u64>(bits_total, 64) / 8, 0);
  custom_bit_vector.resize(bit_vector_size, 0);
  file.read(reinterpret_cast<char *>(&char_buffer[0]), char_buffer.size());
	for (auto i = size_t(0); i < bit_vector_size; ++i) {
		auto current_first_char = i * 8;
		custom_bit_vector[i] =
			  u64(char_buffer[current_first_char + 0]) << (8 * 7)
			| u64(char_buffer[current_first_char + 1]) << (8 * 6)
			| u64(char_buffer[current_first_char + 2]) << (8 * 5)
			| u64(char_buffer[current_first_char + 3]) << (8 * 4)
			| u64(char_buffer[current_first_char + 4]) << (8 * 3)
			| u64(char_buffer[current_first_char + 5]) << (8 * 2)
			| u64(char_buffer[current_first_char + 6]) << (8 * 1)
			| u64(char_buffer[current_first_char + 7]) << (8 * 0);
	}
	bit_vector_pointer = &custom_bit_vector[0];
}

}
