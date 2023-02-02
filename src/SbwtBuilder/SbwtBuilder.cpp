#include <bit>
#include <cstddef>
#include <fstream>
#include <ios>
#include <memory>
#include <stdexcept>
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

namespace sbwt_search {

using fmt::format;
using io_utils::ThrowingIfstream;
using log_utils::Logger;
using math_utils::round_up;
using std::bit_cast;
using std::ifstream;
using std::ios_base;
using std::make_unique;
using std::runtime_error;
using std::unique_ptr;

auto SbwtBuilder::get_cpu_sbwt(bool build_index)
  -> unique_ptr<CpuSbwtContainer> {
  ThrowingIfstream stream(filename, std::ios::in);
  Logger::log_timed_event("SBWTRead", Logger::EVENT_STATE::START);
  const string variant = read_string_with_size(stream);
  if (variant != "plain-matrix") {
    throw runtime_error("Error input is not a plain-matrix SBWT");
  }
  const string version = read_string_with_size(stream);
  if (version != "v0.1") { throw runtime_error("Error: wrong SBWT version"); }
  u64 num_bits = 0;
  stream.read(bit_cast<char *>(&num_bits), sizeof(u64));
  const size_t vectors_start_position = stream.tellg();
  const size_t bit_vector_bytes
    = round_up<u64>(num_bits, u64_bits) / sizeof(u64);
  stream.seekg(
    static_cast<std::ios::off_type>(bit_vector_bytes), ios_base::cur
  );  // skip first vector
#pragma unroll
  // skip the other 3 vectors and 4 rank structure vectors
  for (int i = 0; i < 3 + 4; ++i) { skip_bits_vector(stream); }
  vector<unique_ptr<vector<u64>>> acgt(4);
  load_bit_vectors(bit_vector_bytes, acgt, vectors_start_position);
  skip_bits_vector(stream);  // skip suffix group starts
  skip_bytes_vector(stream);  // skip C map
  skip_bytes_vector(stream);  // skip kmer_prefix_calc
  u64 kmer_size = -1;
  stream.seekg(sizeof(u64), ios_base::cur);  // skip precalc_k
  stream.seekg(sizeof(u64), ios_base::cur);  // skip n_nodes
  stream.seekg(sizeof(u64), ios_base::cur);  // skip n_kmers
  stream.read(bit_cast<char *>(&kmer_size), sizeof(u64));
  Logger::log(
    Logger::LOG_LEVEL::DEBUG, format("Using kmer size: {}", kmer_size)
  );
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
  u64 bit_vector_bytes,
  vector<unique_ptr<vector<u64>>> &acgt,
  size_t start_position
) -> void {
#pragma omp parallel for
  for (size_t i = 0; i < 4; ++i) {
    ifstream st(filename);
    st.seekg(
      static_cast<std::ios::off_type>(
        start_position + i * (bit_vector_bytes + sizeof(u64))
      ),
      ios_base::beg
    );
    acgt[i] = make_unique<vector<u64>>(bit_vector_bytes / sizeof(u64));
    st.read(
      bit_cast<char *>(acgt[i]->data()),
      static_cast<std::streamsize>(bit_vector_bytes)
    );
  }
}

auto SbwtBuilder::build_poppy(CpuSbwtContainer *container) -> void {
  vector<vector<u64>> layer_0(4);
  vector<vector<u64>> layer_1_2(4);
  const uint cmap_size = 5;
  vector<u64> c_map(cmap_size);
  c_map[0] = 1;
#pragma omp parallel for
  for (size_t i = 0; i < 4; ++i) {
    auto builder = PoppyBuilder(
      container->get_num_bits(), container->get_acgt(static_cast<ACGT>(i))
    );
    builder.build();
    layer_0[i] = builder.get_layer_0();
    layer_1_2[i] = builder.get_layer_1_2();
    c_map[i + 1] = builder.get_total_count();
  }
  for (int i = 0; i < 4; ++i) { c_map[i + 1] += c_map[i]; }
  container->set_index(
    std::move(c_map), std::move(layer_0), std::move(layer_1_2)
  );
}

auto SbwtBuilder::skip_bits_vector(istream &stream) -> void {
  size_t bits = 0;
  stream.read(bit_cast<char *>(&bits), sizeof(u64));
  size_t bytes = round_up<size_t>(bits, u64_bits) / sizeof(u64);
  stream.seekg(static_cast<std::ios::off_type>(bytes), ios_base::cur);
}

auto SbwtBuilder::skip_bytes_vector(istream &stream) -> void {
  size_t bytes = 0;
  stream.read(bit_cast<char *>(&bytes), sizeof(u64));
  stream.seekg(static_cast<std::ios::off_type>(bytes), ios_base::cur);
}

}  // namespace sbwt_search
