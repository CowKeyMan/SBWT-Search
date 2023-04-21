#ifndef CHAR_TO_BITS_H
#define CHAR_TO_BITS_H

/**
 * @file CharToBits.h
 * @brief Contains mapping functions which map the characters ACGT to their
 * corresponding bit value
 */

#include <array>
#include <vector>

#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using std::array;
using std::fill;
using std::vector;

const u64 invalid_char_to_bits_value = 99;
const u64 num_ascii_characters = 256;

class CharToBits {
private:
  vector<u64> char_to_bits;

public:
  CharToBits(): char_to_bits(get_char_to_bits()){};
  auto operator()(char c) const -> u64 { return char_to_bits[c]; }

private:
  auto get_char_to_bits() -> vector<u64> {
    vector<u64> char_to_bits(num_ascii_characters, invalid_char_to_bits_value);
    char_to_bits['A'] = char_to_bits['a'] = 0;
    char_to_bits['C'] = char_to_bits['c'] = 1;
    char_to_bits['G'] = char_to_bits['g'] = 2;
    char_to_bits['T'] = char_to_bits['t'] = 3;
    return char_to_bits;
  }
};

}  // namespace sbwt_search

#endif
