#ifndef SBWT_BUILDER_H
#define SBWT_BUILDER_H

/**
 * @file SbwtBuilder.h
 * @brief Loads SBWT from disk and can also build the index using other
 * components. CPU only
 */

#include <istream>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "SbwtContainer/CpuSbwtContainer.h"
#include "Tools/TypeDefinitions.h"
#include "sdsl/int_vector.hpp"

namespace sbwt_search {

using std::istream;
using std::string;
using std::tuple;
using std::unique_ptr;
using std::vector;

class SbwtBuilder {
private:
  string dbg_filename;
  string colors_filename;

public:
  explicit SbwtBuilder(string sbwt_filename, string colors_filename = "");
  auto get_cpu_sbwt() -> unique_ptr<CpuSbwtContainer>;

private:
  auto get_dbg_components()
    -> tuple<vector<vector<u64>>, vector<Poppy>, vector<u64>, u64, u64, u64>;
  auto skip_unecessary_dbg_components(istream &in_stream) -> void;
  auto read_k(istream &in_stream, u64 bit_vector_bytes) -> u64;
  auto get_dbg_bitvectors(
    u64 bit_vector_bytes, u64 vectors_start_position, u64 num_bits
  ) -> tuple<vector<vector<u64>>, vector<Poppy>, vector<u64>>;
  auto get_colors_components() -> vector<u64>;
  auto skip_bits_vector(istream &stream) -> void;
  auto skip_bytes_vector(istream &stream) -> void;
  auto get_key_kmer_marks() -> sdsl::int_vector<>;
  auto skip_unecessary_colors_components(istream &in_stream) -> void;
};

}  // namespace sbwt_search
#endif
