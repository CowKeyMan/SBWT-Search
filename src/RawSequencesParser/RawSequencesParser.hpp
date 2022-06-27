#ifndef RAW_SEQUENCES_PARSER_HPP
#define RAW_SEQUENCES_PARSER_HPP

/**
 * @file RawSequencesParser.hpp
 * @brief Contains functions for parsing raw (ASCII format) sequences
 *        into bit vectors and also generating the positions which
 *        will be used as the starting kmer positions for the parallel
 *        implementation
 * */

#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "TypeDefinitionUtils.h"
#include "MathUtils.hpp"

using std::string;
using std::vector;
using std::array;
using std::fill;
using std::ceil;

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
    fill(char_to_bits.begin(), char_to_bits.end(), default_char_to_bits_value);
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
    switch(c) {
      case 'a':
      case 'A':
        return 0;
      case 'c':
      case 'C':
        return 1;
      case 'g':
      case 'G':
        return 2;
      case 't':
      case 'T':
        return 3;
      default:
        return default_char_to_bits_value;
    }
  }
};

template <class CharToBits = CharToBitsVector>
class RawSequencesParser {
private:
  bool has_parsed = false;
  void check_if_has_parsed(){
    if (has_parsed) {
      throw std::logic_error("RawSequencesParser has already parsed a file");
    }
    has_parsed = true;
  }
  const vector<string> &string_seqs;

  class PositionsBuilder {
  private:
    size_t positions_index = 0;
    size_t global_index = 0;
    u64 kmer_size;
    vector<u64> positions;
  public:
    PositionsBuilder(u64 kmer_size, size_t total_positions):
      kmer_size(kmer_size),
      positions(round_up<u64>(total_positions, 64), 0) {}
    void add_position(const size_t seq_length, const u64 seq_index) {
      if (
        // First check is to make sure we do not get underflow
        seq_length >= kmer_size && seq_index <= seq_length - kmer_size
      ) {
        positions[positions_index] = global_index + seq_index;
        ++positions_index;
      }
    }
    void add_to_global_index(const size_t seq_length) {
      global_index += seq_length;
    }
    auto &get_positions() { return positions; };
  } positions_builder;

  class BitSeqsBuilder {
  private:
    size_t vector_index = 0;
    u64 internal_shift = 62;
    u64 total_letters;
    vector<u64> bit_seqs;
    CharToBits char_to_bits;
  public:
    BitSeqsBuilder(size_t total_letters):
      bit_seqs(round_up<u64>(total_letters, 64) / 64 * 2, 0) {}
    void add_character(char c) {
      auto bits = char_to_bits(c);
      bits <<= internal_shift;
      bit_seqs[vector_index] |= bits;
      internal_shift -= 2;
      if (internal_shift > 64) {
        internal_shift = 62;
        ++vector_index;
      }
    }
    auto &get_bit_seqs() { return bit_seqs; };
  } bit_seqs_builder;
public:
  RawSequencesParser(
    const vector<string> &seqs,
    const size_t total_positions,
    const size_t total_letters,
    const u64 kmer_size
  ):
    string_seqs(seqs),
    positions_builder(kmer_size, total_positions),
    bit_seqs_builder(total_letters) {}
  auto &get_bit_seqs() { return bit_seqs_builder.get_bit_seqs(); };
  auto &get_positions() { return positions_builder.get_positions(); };
  void parse_serial(){
    check_if_has_parsed();
    for (auto &seq: string_seqs) {
      for (size_t seq_index = 0; seq_index < seq.length(); ++seq_index) {
        positions_builder.add_position(seq.length(), seq_index);
        bit_seqs_builder.add_character(seq[seq_index]);
      }
      positions_builder.add_to_global_index(seq.length());
    }
  }
};

}

#endif
