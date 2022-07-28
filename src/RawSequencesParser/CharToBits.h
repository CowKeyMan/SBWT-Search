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

class CharToBitsVector {
  private:
    vector<u64> char_to_bits;

  public:
    CharToBitsVector(): char_to_bits(256, default_char_to_bits_value) {
      char_to_bits['A'] = char_to_bits['a'] = 0;
      char_to_bits['C'] = char_to_bits['c'] = 1;
      char_to_bits['G'] = char_to_bits['g'] = 2;
      char_to_bits['T'] = char_to_bits['t'] = 3;
    };
    u64 operator()(char c) { return char_to_bits[c]; }
};

class CharToBitsArray {
  private:
    array<u64, 256> char_to_bits;

  public:
    CharToBitsArray() {
      fill(
        char_to_bits.begin(), char_to_bits.end(), default_char_to_bits_value
      );
      char_to_bits['A'] = char_to_bits['a'] = 0;
      char_to_bits['C'] = char_to_bits['c'] = 1;
      char_to_bits['G'] = char_to_bits['g'] = 2;
      char_to_bits['T'] = char_to_bits['t'] = 3;
    };
    u64 operator()(char c) { return char_to_bits[c]; }
};

class CharToBitsCArray {
  private:
    u64 char_to_bits[256];

  public:
    CharToBitsCArray() {
      fill(char_to_bits, char_to_bits + 256, default_char_to_bits_value);
      char_to_bits['A'] = char_to_bits['a'] = 0;
      char_to_bits['C'] = char_to_bits['c'] = 1;
      char_to_bits['G'] = char_to_bits['g'] = 2;
      char_to_bits['T'] = char_to_bits['t'] = 3;
    };
    u64 operator()(char c) { return char_to_bits[c]; }
};

class CharToBitsSwitch {
  public:
    u64 operator()(char c) {
      switch (c) {
        case 'a':
        case 'A': return 0;
        case 'c':
        case 'C': return 1;
        case 'g':
        case 'G': return 2;
        case 't':
        case 'T': return 3;
        default: return default_char_to_bits_value;
      }
    }
};

}

#endif
