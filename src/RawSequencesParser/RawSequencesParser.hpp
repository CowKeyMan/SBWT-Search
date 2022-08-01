#ifndef RAW_SEQUENCES_PARSER_HPP
#define RAW_SEQUENCES_PARSER_HPP

/**
 * @file RawSequencesParser.hpp
 * @brief Contains functions for parsing raw (ASCII format) sequences
 *        into bit vectors and also generating the positions which
 *        will be used as the starting kmer positions for the parallel
 *        implementation
 * */

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "Builder/Builder.h"
#include "RawSequencesParser/CharToBits.h"
#include "Utils/MathUtils.hpp"
#include "Utils/TypeDefinitions.h"

using std::array;
using std::ceil;
using std::fill;
using std::make_unique;
using std::unique_ptr;
using std::string;
using std::vector;

namespace sbwt_search {

template <class CharToBits = CharToBitsVector>
class RawSequencesParser: Builder {
  private:
    const unique_ptr<vector<string>> string_seqs;

    class PositionsBuilder {
      private:
        size_t positions_index = 0;
        size_t global_index = 0;
        u64 kmer_size;
        unique_ptr<vector<u64>> positions;

      public:
        PositionsBuilder(u64 kmer_size, size_t total_positions):
            kmer_size(kmer_size),
            positions(make_unique<vector<u64>>(total_positions, 0)) {}

        auto add_position(const size_t seq_length, const u64 seq_index)
          -> void {
          if (seq_length >= kmer_size && seq_index <= seq_length - kmer_size) {
            (*positions.get())[positions_index] = global_index + seq_index;
            ++positions_index;
          }
        }

        auto add_to_global_index(const size_t seq_length) -> void {
          global_index += seq_length;
        }

        auto get_positions() { return move(positions); };
    } positions_builder;

    class BitSeqsBuilder {
      private:
        size_t vector_index = 0;
        u64 internal_shift = 62;
        u64 total_letters;
        unique_ptr<vector<u64>> bit_seqs;
        CharToBits char_to_bits;

      public:
        BitSeqsBuilder(size_t total_letters):
            bit_seqs(make_unique<vector<u64>>(
              round_up<u64>(total_letters, 64) / 64 * 2, 0
            )) {}

        auto add_character(char c) -> void {
          auto bits = char_to_bits(c);
          bits <<= internal_shift;
          (*bit_seqs.get())[vector_index] |= bits;
          internal_shift -= 2;
          if (internal_shift > 64) {
            internal_shift = 62;
            ++vector_index;
          }
        }

        auto get_bit_seqs() { return move(bit_seqs); };
    } bit_seqs_builder;

  public:
    RawSequencesParser(
      unique_ptr<vector<string>> string_seqs,
      const size_t total_positions,
      const size_t total_letters,
      const u64 kmer_size
    ):
        string_seqs(move(string_seqs)),
        positions_builder(kmer_size, total_positions),
        bit_seqs_builder(total_letters) {}

    auto get_bit_seqs() { return move(bit_seqs_builder.get_bit_seqs()); };

    auto get_positions() { return move(positions_builder.get_positions()); };

    auto parse_serial() -> void {
      check_if_has_built();
      for (auto &seq: *string_seqs) {
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
