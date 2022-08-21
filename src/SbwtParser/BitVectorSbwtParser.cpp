#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "SbwtContainer/CpuSbwtContainer.hpp"
#include "SbwtParser/SbwtParser.hpp"
#include "Utils/BitVectorUtils.h"
#include "Utils/IOUtils.h"
#include "Utils/MathUtils.hpp"
#include "Utils/TypeDefinitions.h"

using io_utils::ThrowingIfstream;
using math_utils::round_up;
using std::cerr;
using std::make_shared;
using std::shared_ptr;

namespace sbwt_search {

auto BitVectorSbwtParser::do_parse() -> shared_ptr<BitVectorSbwtContainer> {
  auto acgt = parse_acgt();
  return make_shared<BitVectorSbwtContainer>(
    move(acgt[0]), move(acgt[1]), move(acgt[2]), move(acgt[3]), bits_total
  );
}

auto BitVectorSbwtParser::parse_acgt() -> vector<vector<u64>> {
  vector<vector<u64>> result;
  result.reserve(4);
  for (auto &postfix: acgt_postfixes) {
    result.push_back(parse_single_acgt(files_prefix + postfix));
  }
  return result;
}

auto BitVectorSbwtParser::parse_single_acgt(string filename) -> vector<u64> {
  ThrowingIfstream stream(filename, std::ios::in | std::ios::binary);
  stream.read(reinterpret_cast<char *>(&bits_total), sizeof(u64));
  auto bit_vector_size = round_up<u64>(bits_total, 64) / 64;
  auto total_characters = round_up<u64>(bits_total, 8) / 8;
  vector<u64> result(bit_vector_size);
  stream.read(reinterpret_cast<char *>(&result[0]), total_characters);
  return result;
}

}  // namespace sbwt_search
