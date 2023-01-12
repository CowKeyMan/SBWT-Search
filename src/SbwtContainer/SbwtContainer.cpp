#include "SbwtContainer/SbwtContainer.h"

namespace sbwt_search {

auto SbwtContainer::get_bit_vector_size() const -> u64 {
  return bit_vector_size;
}
auto SbwtContainer::get_num_bits() const -> u64 { return num_bits; }
auto SbwtContainer::get_kmer_size() const -> uint { return kmer_size; }

}  // namespace sbwt_search
