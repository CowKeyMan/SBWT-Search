#include "SbwtContainer/SbwtContainer.h"

namespace sbwt_search {

SbwtContainer::SbwtContainer(
  size_t num_bits_, size_t bit_vector_size_, u64 kmer_size_
):
    num_bits(num_bits_),
    bit_vector_size(bit_vector_size_),
    kmer_size(kmer_size_) {}

auto SbwtContainer::get_bit_vector_size() const -> u64 {
  return bit_vector_size;
}
auto SbwtContainer::get_num_bits() const -> u64 { return num_bits; }
auto SbwtContainer::get_kmer_size() const -> u64 { return kmer_size; }

}  // namespace sbwt_search
