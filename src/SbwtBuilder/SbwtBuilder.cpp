#include <cstddef>
#include <ios>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <ext/alloc_traits.h>

#include "PoppyBuilder/PoppyBuilder.h"
#include "SbwtBuilder/SbwtBuilder.h"
#include "SbwtContainer/SbwtContainer.h"
#include "Utils/IOUtils.h"
#include "Utils/Logger.h"
#include "Utils/MathUtils.hpp"
#include "Utils/TypeDefinitions.h"

using io_utils::ThrowingIfstream;
using log_utils::Logger;
using math_utils::round_up;
using std::ios_base;
using std::make_unique;
using std::move;
using std::runtime_error;
using std::unique_ptr;

namespace sbwt_search {

auto SbwtBuilder::get_cpu_sbwt(bool build_index)
  -> unique_ptr<CpuSbwtContainer> {
  ThrowingIfstream stream(filename, std::ios::in);
  Logger::log_timed_event("SBWTRead", Logger::EVENT_STATE::START);
  string variant = load_string(stream);
  if (variant != "plain-matrix") {
    throw runtime_error("Error input is not a plain-matrix SBWT");
  }
  string version = load_string(stream);
  if (version != "v0.1") { throw runtime_error("Error: wrong SBWT version"); }
  u64 num_bits;
  stream.read(reinterpret_cast<char *>(&num_bits), sizeof(u64));
  size_t vectors_start_position = stream.tellg();
  size_t bit_vector_bytes = round_up<u64>(num_bits, 64) / sizeof(u64);
  stream.seekg(bit_vector_bytes, ios_base::cur); // skip first vector
  for (int i = 0; i < 3 + 4; ++i) {
    // skip the other 3 vectors and 4 rank structure vectors
    skip_bits_vector(stream);
  }
  vector<unique_ptr<vector<u64>>> acgt(4);
  load_bit_vectors(bit_vector_bytes, acgt, vectors_start_position);
  skip_bits_vector(stream);  // skip suffix group starts
  skip_bytes_vector(stream);  // skip C map
  skip_bytes_vector(stream);  // skip kmer_prefix_calc
  u64 kmer_size = 4;
  stream.seekg(
    sizeof(u64) * 3, ios_base::cur
  );  // skip precalc_k, n_nodes, n_kmers
  stream.read(reinterpret_cast<char *>(&kmer_size), sizeof(u64));
  Logger::log_timed_event("SBWTRead", Logger::EVENT_STATE::STOP);
  auto container = make_unique<CpuSbwtContainer>(
    num_bits, acgt[0], acgt[1], acgt[2], acgt[3], kmer_size
  );
  Logger::log_timed_event("Poppy", Logger::EVENT_STATE::START);
  if (build_index) { build_poppy(container.get()); }
  Logger::log_timed_event("Poppy", Logger::EVENT_STATE::STOP);
  return container;
}

auto SbwtBuilder::load_bit_vectors(
  u64 bit_vector_bytes, vector<unique_ptr<vector<u64>>> &acgt, size_t start_position
) -> void {
#pragma omp parallel for
  for (int i = 0; i < 4; ++i) {
    ifstream st(filename);
    st.seekg(
      start_position + i * (bit_vector_bytes + 8), ios_base::beg
    );
    acgt[i] = make_unique<vector<u64>>(bit_vector_bytes / 8);
    st.read(reinterpret_cast<char *>(&(*acgt[i])[0]), bit_vector_bytes);
  }
}

auto SbwtBuilder::build_poppy(CpuSbwtContainer *container) -> void {
  vector<vector<u64>> layer_0(4), layer_1_2(4);
  vector<u64> c_map(5);
  c_map[0] = 1;
#pragma omp parallel for
  for (int i = 0; i < 4; ++i) {
    auto builder = PoppyBuilder(
      container->get_bits_total(), container->get_acgt(static_cast<ACGT>(i))
    );
    builder.build();
    layer_0[i] = builder.get_layer_0();
    layer_1_2[i] = builder.get_layer_1_2();
    c_map[i + 1] = builder.get_total_count();
  }
  for (int i = 0; i < 4; ++i) { c_map[i + 1] += c_map[i]; }
  container->set_index(move(c_map), move(layer_0), move(layer_1_2));
}

auto SbwtBuilder::load_string(istream &stream) const -> string {
  size_t size;
  stream.read(reinterpret_cast<char *>(&size), sizeof(u64));
  string s(size, '\0');
  stream.read(reinterpret_cast<char *>(&s[0]), size);
  return s;
}

auto SbwtBuilder::skip_bits_vector(istream &stream) const -> void {
  size_t bits;
  stream.read(reinterpret_cast<char *>(&bits), sizeof(u64));
  size_t bytes = round_up<size_t>(bits, 64) / sizeof(u64);
  stream.seekg(bytes, ios_base::cur);
}

auto SbwtBuilder::skip_bytes_vector(istream &stream) const -> void {
  size_t bytes;
  stream.read(reinterpret_cast<char *>(&bytes), sizeof(u64));
  stream.seekg(bytes, ios_base::cur);
}

}  // namespace sbwt_search
