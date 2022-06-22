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

RawSequencesParser::RawSequencesParser(
  const vector<string> &seqs_strings,
  const u64 total_positions,
  const u64 total_letters,
  const u64 kmer_size
):
  seqs_strings(seqs_strings),
  bits_hash_map(build_bits_hash_map()),
  kmer_size(kmer_size)
{
  seqs_bits.resize(size_t(ceil(total_letters / 64.0)) * 2, 0);
  positions.resize(size_t(ceil(total_positions / 64.0)) * 64, 0);
}

auto RawSequencesParser::parse_serial() -> void{
  check_if_has_parsed();
  auto positions_index = size_t(0);
  auto global_index = size_t(0);
  auto vector_index = size_t(0);
  auto internal_shift = 62;
  for (auto &sequence: seqs_strings) {
    for (
      auto sequence_index = 0;
      sequence_index < sequence.length();
      ++sequence_index
    ) {
      add_position(
        sequence.length(), sequence_index, positions_index, global_index
      );
      add_bits(sequence, sequence_index, internal_shift, vector_index);
    }
    global_index += sequence.length();
  }
}

auto RawSequencesParser::parse_parallel() -> void {}

auto RawSequencesParser::add_position(
  const size_t sequence_length,
  const u64 sequence_index,
  size_t &positions_index,
  const size_t global_index
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
