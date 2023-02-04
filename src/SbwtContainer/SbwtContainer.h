#ifndef SBWT_CONTAINER_H
#define SBWT_CONTAINER_H

/**
 * @file SbwtContainer.h
 * @brief Contains data class for SBWT index
 */

#include <cstddef>

#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

enum class ACGT { A = 0, C = 1, G = 2, T = 3 };
const uint cmap_size = 5;

class SbwtContainer {
private:
  size_t bit_vector_size;
  size_t num_bits;
  uint kmer_size;

protected:
  SbwtContainer(size_t num_bits, size_t bit_vector_size, u64 kmer_size):
      num_bits(num_bits),
      bit_vector_size(bit_vector_size),
      kmer_size(kmer_size) {}

public:
  [[nodiscard]] auto get_bit_vector_size() const -> u64;
  [[nodiscard]] auto get_num_bits() const -> u64;
  [[nodiscard]] auto get_kmer_size() const -> uint;
};

}  // namespace sbwt_search

#endif
