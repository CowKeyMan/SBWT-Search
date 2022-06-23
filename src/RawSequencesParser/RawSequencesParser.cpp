#include <cmath>
#include <stdexcept>

#include "GlobalDefinitions.h"
#include "RawSequencesParser.h"

namespace sbwt_search {

auto build_bits_hash_map() -> vector<u64> {
  auto bits_hash_map = vector<u64>(256, default_bits_hash_map_value);
  bits_hash_map['A'] = bits_hash_map['a'] = 0;
  bits_hash_map['C'] = bits_hash_map['c'] = 1;
  bits_hash_map['G'] = bits_hash_map['g'] = 2;
  bits_hash_map['T'] = bits_hash_map['t'] = 3;
  return bits_hash_map;
}

RawSequencesParser::PositionsGenerator::PositionsGenerator(
  u64 kmer_size, size_t total_positions
):
  kmer_size(kmer_size),
  positions(size_t(ceil(total_positions / 64.0)) * 64, 0)
{}

RawSequencesParser::RawSequencesParser(
  const vector<string> &seqs_strings,
  const size_t total_positions,
  const size_t total_letters,
  const u64 kmer_size
):
  seqs_strings(seqs_strings),
  bits_hash_map(build_bits_hash_map()),
  positions_generator(kmer_size, total_positions)
{
  seqs_bits.resize(size_t(ceil(total_letters / 64.0)) * 2, 0);
}

auto RawSequencesParser::parse_serial() -> void{
  check_if_has_parsed();
  auto vector_index = size_t(0);
  auto internal_shift = 62;
  for (auto &seq: seqs_strings) {
    for (auto seq_index = 0; seq_index < seq.length(); ++seq_index) {
      positions_generator.add_position(seq.length(), seq_index);
      add_bits(seq, seq_index, internal_shift, vector_index);
    }
    positions_generator.add_to_global_index(seq.length());
  }
}

auto RawSequencesParser::PositionsGenerator::add_position(
  const size_t sequence_length,
  const u64 sequence_index
) -> void {
  if (sequence_index <= sequence_length - kmer_size * 1.0) {
    positions[positions_index] = global_index + sequence_index;
    ++positions_index;
  }
}

auto RawSequencesParser::add_bits(
  const string &sequence,
  const u64 sequence_index,
  int &internal_shift,
  size_t &vector_index
) -> void {
  auto character = sequence[sequence_index];
  auto bits = bits_hash_map[character];
  bits <<= internal_shift;
  seqs_bits[vector_index] |= bits;
  internal_shift -= 2;
  if (internal_shift < 0) {
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
