#include <bit>
#include <cstddef>
#include <fstream>
#include <ios>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <ext/alloc_traits.h>

#include "PoppyBuilder/PoppyBuilder.h"
#include "SbwtBuilder/SbwtBuilder.h"
#include "SbwtContainer/CpuSbwtContainer.h"
#include "Tools/IOUtils.h"
#include "Tools/Logger.h"
#include "Tools/MathUtils.hpp"
#include "Tools/TypeDefinitions.h"
#include "fmt/core.h"
#include "sdsl/int_vector.hpp"
#include "sdsl/rank_support.hpp"

namespace sbwt_search {

using fmt::format;
using io_utils::ThrowingIfstream;
using log_utils::Logger;
using math_utils::round_up;
using std::bit_cast;
using std::ifstream;
using std::ios;
using std::make_unique;
using std::runtime_error;
using std::unique_ptr;

SbwtBuilder::SbwtBuilder(string dbg_filename_, string colors_filename_):
    dbg_filename(std::move(dbg_filename_)),
    colors_filename(std::move(colors_filename_)) {}

auto SbwtBuilder::get_cpu_sbwt() -> unique_ptr<CpuSbwtContainer> {
  Logger::log_timed_event("SBWTReadAndPopppy", Logger::EVENT_STATE::START);
  auto [acgt, poppys, c_map, num_bits, acgt_size, kmer_size]
    = get_dbg_components();
  auto container = make_unique<CpuSbwtContainer>(
    std::move(acgt),
    std::move(poppys),
    std::move(c_map),
    num_bits,
    acgt_size,
    kmer_size,
    get_key_kmer_marks()
  );
  Logger::log_timed_event("SBWTReadAndPopppy", Logger::EVENT_STATE::STOP);
  return container;
}

auto SbwtBuilder::get_dbg_components()
  -> tuple<vector<vector<u64>>, vector<Poppy>, vector<u64>, u64, u64, u64> {
  ThrowingIfstream in_stream(dbg_filename, std::ios::in);
  const string variant = in_stream.read_string_with_size();
  if (variant != "v0.1") {  // may not contain variant string
    if (variant != "plain-matrix") {
      throw runtime_error("Error input is not a plain-matrix SBWT");
    }
    const string version = in_stream.read_string_with_size();
    if (version != "v0.1") { throw runtime_error("Error: wrong SBWT version"); }
  }
  u64 num_bits = in_stream.read_real<u64>();
  const u64 vectors_start_position = in_stream.tellg();
  const u64 bit_vector_bytes = round_up<u64>(num_bits, u64_bits) / sizeof(u64);
  vector<vector<u64>> acgt(4);
  vector<Poppy> poppys(4);
  vector<u64> c_map(cmap_size, 1);
  u64 kmer_size = -1;
  kmer_size = read_k(in_stream, bit_vector_bytes);
  std::tie(acgt, poppys, c_map)
    = get_dbg_bitvectors(bit_vector_bytes, vectors_start_position, num_bits);
  return {acgt, poppys, c_map, num_bits, acgt[0].size(), kmer_size};
}

auto SbwtBuilder::read_k(istream &in_stream, u64 bit_vector_bytes) -> u64 {
  u64 kmer_size = -1;
  in_stream.seekg(
    static_cast<std::ios::off_type>(bit_vector_bytes), ios::cur
  );  // skip first vector
  skip_unecessary_dbg_components(in_stream);
  in_stream.read(bit_cast<char *>(&kmer_size), sizeof(u64));
  Logger::log(
    Logger::LOG_LEVEL::DEBUG, format("Using kmer size: {}", kmer_size)
  );
  return kmer_size;
}

auto SbwtBuilder::skip_unecessary_dbg_components(istream &in_stream) -> void {
  // skip acgt vectors and 4 rank structure vectors
  for (int i = 0; i < 3 + 4; ++i) { skip_bits_vector(in_stream); }
  skip_bits_vector(in_stream);             // skip suffix group starts
  skip_bytes_vector(in_stream);            // skip C map
  skip_bytes_vector(in_stream);            // skip kmer_prefix_calc
  in_stream.seekg(sizeof(u64), ios::cur);  // skip precalc_k
  in_stream.seekg(sizeof(u64), ios::cur);  // skip n_nodes
  in_stream.seekg(sizeof(u64), ios::cur);  // skip n_kmers
}

auto SbwtBuilder::get_dbg_bitvectors(
  u64 bit_vector_bytes, u64 vectors_start_position, u64 num_bits
) -> tuple<vector<vector<u64>>, vector<Poppy>, vector<u64>> {
  vector<vector<u64>> acgt(4);
  vector<Poppy> poppys(4);
  vector<u64> c_map(cmap_size, 1);
#pragma omp parallel for
  for (u64 i = 0; i < 4; ++i) {
    ifstream st(dbg_filename);
    st.seekg(
      static_cast<std::ios::off_type>(
        vectors_start_position + i * (bit_vector_bytes + sizeof(u64))
      ),
      ios::beg
    );
    acgt[i] = vector<u64>(bit_vector_bytes / sizeof(u64));
    st.read(
      bit_cast<char *>(acgt[i].data()),
      static_cast<std::streamsize>(bit_vector_bytes)
    );
    auto builder = PoppyBuilder(acgt[i], num_bits);
    poppys[i] = builder.get_poppy();
    c_map[i + 1] = poppys[i].total_1s;
  }
  for (int i = 0; i < 4; ++i) { c_map[i + 1] += c_map[i]; }
  return {std::move(acgt), std::move(poppys), std::move(c_map)};
}

auto SbwtBuilder::skip_bits_vector(istream &stream) -> void {
  u64 bits = 0;
  stream.read(bit_cast<char *>(&bits), sizeof(u64));
  u64 bytes = round_up<u64>(bits, u64_bits) / sizeof(u64);
  stream.seekg(static_cast<std::ios::off_type>(bytes), ios::cur);
}

auto SbwtBuilder::skip_bytes_vector(istream &stream) -> void {
  u64 bytes = 0;
  stream.read(bit_cast<char *>(&bytes), sizeof(u64));
  stream.seekg(static_cast<std::ios::off_type>(bytes), ios::cur);
}

auto SbwtBuilder::get_key_kmer_marks() -> vector<u64> {
  if (colors_filename.empty()) { return {}; }
  ThrowingIfstream in_stream(colors_filename, ios::in | ios::binary);
  string filetype = in_stream.read_string_with_size();
  if (filetype != "sdsl-hybrid-v4") {
    throw runtime_error(
      "The colors file has an incorrect format. Expected 'sdsl-hybrid-v4'"
    );
  }
  skip_unecessary_colors_components(in_stream);
  u64 num_bits = in_stream.read_real<u64>();
  const u64 bit_vector_bytes = round_up<u64>(num_bits, u64_bits) / sizeof(u64);
  vector<u64> key_kmer_marks(bit_vector_bytes / sizeof(u64));
  in_stream.read(
    bit_cast<char *>(key_kmer_marks.data()),
    static_cast<std::streamsize>(bit_vector_bytes)
  );
  return key_kmer_marks;
}

auto SbwtBuilder::skip_unecessary_colors_components(istream &in_stream)
  -> void {
  sdsl::int_vector<> vector_discard;
  sdsl::rank_support_v5 rank_discard;
  sdsl::bit_vector bit_discard;

  bit_discard.load(in_stream);     // skip dense_arrays
  vector_discard.load(in_stream);  // skip dense_arrays_intervals

  vector_discard.load(in_stream);  // skip sparse_arrays
  vector_discard.load(in_stream);  // skip sparse_arrays_intervals

  bit_discard.load(in_stream);     // skip is_dense_marks
  // skip is_dense_marks rank structure
  rank_discard.load(in_stream, &bit_discard);
}

}  // namespace sbwt_search
