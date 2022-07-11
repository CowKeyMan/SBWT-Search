#include <iostream>
#include <istream>

#include "IndexFileParser/IndexFileParser.hpp"
#include "Utils/BitVectorUtils.h"
#include "Utils/IOUtils.hpp"
#include "Utils/MathUtils.hpp"

using std::cerr;

namespace sbwt_search {

auto BitVectorIndexFileParser::do_parse(bool has_index)
  -> BitVectorSbwtContainer {
  if (has_index) { cerr << "Set has_index to bit vector format. Ignoring\n"; }
  auto acgt = parse_acgt();
  return BitVectorSbwtContainer(
    move(acgt[0]), move(acgt[1]), move(acgt[2]), move(acgt[3]), bits_total
  );
}

auto BitVectorIndexFileParser::parse_acgt() -> vector<vector<u64>> {
  vector<vector<u64>> result;
  result.reserve(4);
  for (auto &postfix: acgt_postfixes) {
    result.push_back(parse_single_acgt(files_prefix + postfix));
  }
  return result;
}

auto BitVectorIndexFileParser::parse_single_acgt(string filename)
  -> vector<u64> {
  ThrowingIfstream stream(filename, std::ios::in | std::ios::binary);
  stream.read(reinterpret_cast<char *>(&bits_total), sizeof(u64));
  auto bit_vector_size = round_up<u64>(bits_total, 64) / 64;
  auto total_characters = round_up<u64>(bits_total, 8) / 8;
  vector<u64> result(bit_vector_size);
  stream.read(reinterpret_cast<char *>(&result[0]), total_characters);
  return result;
}

auto BitVectorIndexFileParser::parse_c_map() const -> vector<u64> {
  vector<u64> result;
  u64 bit_vector_size;
  ThrowingIfstream stream(
    files_prefix + c_map_postfix, std::ios::in | std::ios::binary
  );
  stream.read(reinterpret_cast<char *>(&bit_vector_size), sizeof(u64));
  result.resize(bit_vector_size, 0);
  stream.read(
    reinterpret_cast<char *>(&result[0]), sizeof(u64) * bit_vector_size
  );
  return result;
}

}