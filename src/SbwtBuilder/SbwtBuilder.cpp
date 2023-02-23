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
using std::ios;
using std::make_unique;
using std::runtime_error;
using std::unique_ptr;

auto SbwtBuilder::get_cpu_sbwt() -> unique_ptr<CpuSbwtContainer> {
  ThrowingIfstream in_stream(filename, std::ios::in);
  Logger::log_timed_event("SBWTReadAndPopppy", Logger::EVENT_STATE::START);
  const string variant = in_stream.read_string_with_size();
  if (variant != "plain-matrix") {
    throw runtime_error("Error input is not a plain-matrix SBWT");
  }
  const string version = in_stream.read_string_with_size();
  if (version != "v0.1") { throw runtime_error("Error: wrong SBWT version"); }
  u64 num_bits = 0;
  in_stream.read(bit_cast<char *>(&num_bits), sizeof(u64));
  const u64 vectors_start_position = in_stream.tellg();
  const u64 bit_vector_bytes = round_up<u64>(num_bits, u64_bits) / sizeof(u64);
  in_stream.seekg(
    static_cast<std::ios::off_type>(bit_vector_bytes), ios::cur
  );  // skip first vector
#pragma unroll
  // skip the other 3 vectors and 4 rank structure vectors
  for (int i = 0; i < 3 + 4; ++i) { skip_bits_vector(in_stream); }
  skip_bits_vector(in_stream);             // skip suffix group starts
  skip_bytes_vector(in_stream);            // skip C map
  skip_bytes_vector(in_stream);            // skip kmer_prefix_calc
  u64 kmer_size = -1;
  in_stream.seekg(sizeof(u64), ios::cur);  // skip precalc_k
  in_stream.seekg(sizeof(u64), ios::cur);  // skip n_nodes
  in_stream.seekg(sizeof(u64), ios::cur);  // skip n_kmers
  in_stream.read(bit_cast<char *>(&kmer_size), sizeof(u64));
  Logger::log(
    Logger::LOG_LEVEL::DEBUG, format("Using kmer size: {}", kmer_size)
  );
  auto [acgt, poppys, c_map] = get_container_components(
    num_bits, bit_vector_bytes, vectors_start_position
  );
  u64 acgt_size = acgt[0].size();
  auto container = make_unique<CpuSbwtContainer>(
    std::move(acgt),
    std::move(poppys),
    std::move(c_map),
    num_bits,
    acgt_size,
    kmer_size
  );
  Logger::log_timed_event("SBWTReadAndPopppy", Logger::EVENT_STATE::STOP);
  return container;
}

auto SbwtBuilder::get_container_components(
  u64 num_bits, u64 bit_vector_bytes, u64 start_position
) -> tuple<vector<vector<u64>>, vector<Poppy>, vector<u64>> {
  vector<vector<u64>> acgt(4);
  vector<Poppy> poppys(4);
  vector<u64> c_map(cmap_size, 1);
#pragma omp parallel for
  for (u64 i = 0; i < 4; ++i) {
    ifstream st(filename);
    st.seekg(
      static_cast<std::ios::off_type>(
        start_position + i * (bit_vector_bytes + sizeof(u64))
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
  return {acgt, poppys, c_map};
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

}  // namespace sbwt_search
