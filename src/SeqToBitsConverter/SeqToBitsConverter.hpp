#ifndef SEQ_TO_BITS_CONVERTER_HPP
#define SEQ_TO_BITS_CONVERTER_HPP

/**
 * @file SeqToBitsConverter.hpp
 * @brief Contains functions for parsing raw (ASCII format) sequences
 *        into bit vectors and also generating the positions which
 *        will be used as the starting kmer positions for the parallel
 *        implementation
 * */

#include <memory>
#include <string>
#include <vector>

#include "Builder/Builder.h"
#include "SeqToBitsConverter/CharToBits.h"
#include "Utils/MathUtils.hpp"
#include "Utils/TypeDefinitions.h"

using std::make_unique;
using std::shared_ptr;
using std::string;
using std::vector;

namespace sbwt_search {

template <class CharToBits = CharToBitsVector>
class SeqToBitsConverter {
  private:
    CharToBits char_to_bits;
    size_t vector_index;
    u64 bit_index;
    u64 internal_shift;
    u64 working_int = 0;
    shared_ptr<vector<u64>> seqs;

  public:
    SeqToBitsConverter(shared_ptr<vector<u64>> seqs): seqs(seqs) {}

    auto convert(const string &s, u64 starting_character_index) -> void {
      bit_index = starting_character_index * 2;
      reset();
      for (auto c: s) { add_character(c); }
    }

    auto add_int() -> void {
      (*seqs)[vector_index] |= working_int;
      reset();
      working_int = 0;
    }

  private:
    auto add_character(char c) -> void {
      auto bits = char_to_bits(c);
      bits <<= internal_shift;
      working_int |= bits;
      internal_shift -= 2;
      bit_index += 2;
      if (internal_shift > 64) { add_int(); }
    }

    auto reset() {
      vector_index = bit_index / 64;
      internal_shift = 62 - (bit_index % 64);
    }
};

}

#endif
