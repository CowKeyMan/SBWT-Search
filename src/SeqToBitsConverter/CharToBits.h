#ifndef CHAR_TO_BITS_H
#define CHAR_TO_BITS_H

/**
 * @file CharToBits.h
 * @brief Contains mapping functions which map the characters ACGT to their
 *        corresponding bit value
 * */

#include <array>
#include <vector>

#include "Utils/TypeDefinitions.h"

using std::array;
using std::fill;
using std::vector;

namespace sbwt_search {

const u64 default_char_to_bits_value = 99;

vector<u64> get_char_to_bits() {
  vector<u64> char_to_bits(256, default_char_to_bits_value);
  char_to_bits['A'] = char_to_bits['a'] = 0;
  char_to_bits['C'] = char_to_bits['c'] = 1;
  char_to_bits['G'] = char_to_bits['g'] = 2;
  char_to_bits['T'] = char_to_bits['t'] = 3;
  return char_to_bits;
}

class CharToBitsVector {
  private:
    const vector<u64> char_to_bits;

  public:
    CharToBitsVector(): char_to_bits(get_char_to_bits()){};
    u64 operator()(char c) { return char_to_bits[c]; }
};

;

}

#endif
