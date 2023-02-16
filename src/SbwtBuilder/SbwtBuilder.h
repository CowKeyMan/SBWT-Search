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

namespace sbwt_search {

using std::istream;
using std::string;
using std::tuple;
using std::unique_ptr;
using std::vector;

class SbwtBuilder {
private:
  string filename;

public:
  explicit SbwtBuilder(string filename): filename(std::move(filename)) {}
  auto get_cpu_sbwt() -> unique_ptr<CpuSbwtContainer>;

private:
  auto get_container_components(
    u64 num_bits, u64 bit_vector_bytes, u64 start_position
  ) -> tuple<vector<vector<u64>>, vector<Poppy>, vector<u64>>;
  auto skip_bits_vector(istream &stream) -> void;
  auto skip_bytes_vector(istream &stream) -> void;
};

}  // namespace sbwt_search
#endif
