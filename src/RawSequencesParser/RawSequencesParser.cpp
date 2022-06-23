#include <cmath>
#include <stdexcept>

#include "GlobalDefinitions.h"
#include "RawSequencesParser.h"

namespace sbwt_search {

auto build_char_to_bits() -> vector<u64> {
  auto char_to_bits = vector<u64>(256, default_char_to_bits_value);
  char_to_bits['A'] = char_to_bits['a'] = 0;
  char_to_bits['C'] = char_to_bits['c'] = 1;
  char_to_bits['G'] = char_to_bits['g'] = 2;
  char_to_bits['T'] = char_to_bits['t'] = 3;
  return char_to_bits;
}

auto RawSequencesParser::parse_serial() -> void {
  check_if_has_parsed();
  for (auto &seq: string_seqs) {
    for (auto seq_index = 0; seq_index < seq.length(); ++seq_index) {
      positions_builder.add_position(seq.length(), seq_index);
      bit_seqs_builder.add_character(seq[seq_index]);
    }
    positions_builder.add_to_global_index(seq.length());
  }
}

RawSequencesParser::PositionsBuilder::PositionsBuilder(
  u64 kmer_size, size_t total_positions
):
  kmer_size(kmer_size),
  positions(size_t(ceil(total_positions / 64.0)) * 64, 0) {}

RawSequencesParser::BitSeqsBuilder::BitSeqsBuilder(size_t total_letters):
  bit_seqs(size_t(ceil(total_letters / 64.0)) * 2, 0),
  char_to_bits(build_char_to_bits()) {}

RawSequencesParser::RawSequencesParser(
  const vector<string> &seqs_strings,
  const size_t total_positions,
  const size_t total_letters,
  const u64 kmer_size
):
  string_seqs(seqs_strings),
  positions_builder(kmer_size, total_positions),
  bit_seqs_builder(total_letters) {}

auto RawSequencesParser::PositionsBuilder::add_position(
  const size_t seq_length, const u64 seq_index
) -> void {
  if (
    // First check is to make sure we do not get underflow
    seq_length >= kmer_size && seq_index <= seq_length - kmer_size
  ) {
    positions[positions_index] = global_index + seq_index;
    ++positions_index;
  }
}

auto RawSequencesParser::BitSeqsBuilder::add_character(char c) -> void {
  auto bits = char_to_bits[c];
  bits <<= internal_shift;
  bit_seqs[vector_index] |= bits;
  internal_shift -= 2;
  if (internal_shift > 64) {
    internal_shift = 62;
    ++vector_index;
  }
}

auto RawSequencesParser::check_if_has_parsed() -> void {
  if (has_parsed) {
    throw std::logic_error("RawSequencesParser has already parsed a file");
  }
  has_parsed = true;
}

}
