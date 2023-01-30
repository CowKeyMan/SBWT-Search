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
#include <vector>

#include "SbwtContainer/CpuSbwtContainer.h"
#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using std::istream;
using std::string;
using std::unique_ptr;
using std::vector;

class SbwtBuilder {
private:
  string filename;

public:
  explicit SbwtBuilder(string filename): filename(std::move(filename)) {}
  auto get_cpu_sbwt(bool build_index = true) -> unique_ptr<CpuSbwtContainer>;

private:
  auto load_string(istream &in) -> string;
  auto build_poppy(CpuSbwtContainer *container) -> void;
  auto load_bit_vectors(
    u64 bit_vector_bytes,
    vector<unique_ptr<vector<u64>>> &acgt,
    size_t start_position
  ) -> void;
  auto skip_bits_vector(istream &stream) -> void;
  auto skip_bytes_vector(istream &stream) -> void;
};

}  // namespace sbwt_search
#endif
