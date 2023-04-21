#ifndef SBWT_CONTAINER_H
#define SBWT_CONTAINER_H

/**
 * @file SbwtContainer.h
 * @brief Contains data class for SBWT index. Contains the more generic items
 * from the SBWT items such as the bit vector sizes, and the kmer size. Cpu and
 * Gpu specific items are in the subclasses.
 */

#include <cstddef>

#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

enum class ACGT { A = 0, C = 1, G = 2, T = 3 };
const u64 cmap_size = 5;

class SbwtContainer {
private:
  u64 num_bits;
  u64 bit_vector_size;
  u64 kmer_size;

protected:
  SbwtContainer(u64 num_bits_, u64 bit_vector_size_, u64 kmer_size_);

public:
  [[nodiscard]] auto get_bit_vector_size() const -> u64;
  [[nodiscard]] auto get_num_bits() const -> u64;
  [[nodiscard]] auto get_kmer_size() const -> u64;
};

}  // namespace sbwt_search

#endif
