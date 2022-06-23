#ifndef RAW_SEQUENCES_PARSER_H
#define RAW_SEQUENCES_PARSER_H

/**
 * @file RawSequencesParser.h
 * @brief Contains functions for parsing raw (ASCII format) sequences
 *        into bit vectors and also generating the positions which
 *        will be used as the starting kmer positions for the parallel
 *        implementation
 * */

#include <string>
#include <vector>

#include "GlobalDefinitions.h"

using std::string;
using std::vector;

namespace sbwt_search {

const u64 default_char_to_bits_value = 99;

class RawSequencesParser {
private:
  bool has_parsed = false;
  void check_if_has_parsed();
  const vector<string> &string_seqs;

  class PositionsBuilder {
  private:
    size_t positions_index = 0;
    size_t global_index = 0;
    u64 kmer_size;
    vector<u64> positions;

  public:
    PositionsBuilder(u64 kmer_size, size_t total_positions);
    void add_position(const size_t seq_length, const u64 seq_index);
    void add_to_global_index(const size_t seq_length) {
      global_index += seq_length;
    }
    auto &get_positions() { return positions; };
  } positions_builder;

  class BitSeqsBuilder {
  private:
    const vector<u64> char_to_bits;
    size_t vector_index = 0;
    u64 internal_shift = 62;
    u64 total_letters;
    vector<u64> bit_seqs;

  public:
    BitSeqsBuilder(size_t total_letters);
    void add_character(char c);
    auto &get_bit_seqs() { return bit_seqs; };
  } bit_seqs_builder;

public:
  RawSequencesParser(
    const vector<string> &seqs,
    const size_t total_positions,
    const size_t total_letters,
    const u64 kmer_size
  );
  auto &get_bit_seqs() { return bit_seqs_builder.get_bit_seqs(); };
  auto &get_positions() { return positions_builder.get_positions(); };
  void parse_serial();
};

}

#endif
