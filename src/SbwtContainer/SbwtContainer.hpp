#ifndef SBWT_CONTAINER_HPP
#define SBWT_CONTAINER_HPP

/**
 * @file SbwtContainer.hpp
 * @brief Contains data class for SBWT index
 * */

#include <cstddef>
#include <vector>

#include "Utils/TypeDefinitions.h"

using std::size_t;
using std::vector;

namespace sbwt_search {

class GpuSbwtContainer;

enum class ACGT { A = 0, C = 1, G = 2, T = 3 };

template <class Implementation>
class SbwtContainer {
  private:
    Implementation *const host;

  protected:
    const size_t bit_vector_size;
    const size_t bits_total;

    SbwtContainer(const size_t bits_total, const size_t bit_vector_size):
        bits_total(bits_total),
        bit_vector_size(bit_vector_size),
        host(static_cast<Implementation *>(this)) {}

  public:
    const u64 *get_acgt(ACGT letter) const { return host->do_get_acgt(letter); }

    u64 get_bit_vector_size() const { return bit_vector_size; }
    u64 get_bits_total() const { return bits_total; }
};

}

#endif
