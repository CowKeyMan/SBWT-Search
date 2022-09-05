#ifndef SBWT_BUILDER_H
#define SBWT_BUILDER_H

/**
 * @file SbwtBuilder.h
 * @brief Loads SBWT from disk and can also build the index using other
 * components. CPU only
 * */

#include <istream>
#include <memory>
#include <string>
#include <vector>

#include "Utils/TypeDefinitions.h"

namespace sbwt_search {
class CpuSbwtContainer;
}  // namespace sbwt_search

using std::istream;
using std::string;
using std::unique_ptr;
using std::vector;

namespace sbwt_search {

class SbwtBuilder {
  private:
    string filename;

  public:
    SbwtBuilder(string &filename): filename(filename) {}
    auto get_cpu_sbwt(bool build_index = true) -> unique_ptr<CpuSbwtContainer>;

  private:
    auto assert_plain_matrix(istream &in) const -> void;
    auto build_poppy(CpuSbwtContainer *container) -> void;
    auto load_bit_vectors(u64 bits_total, vector<unique_ptr<vector<u64>>> &acgt)
      -> void;
};

}  // namespace sbwt_search
#endif
