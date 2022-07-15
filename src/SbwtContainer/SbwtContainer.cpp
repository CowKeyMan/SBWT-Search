#include <byteswap.h>

#include <sdsl/int_vector.hpp>

#include "SbwtContainer/CpuSbwtContainer.hpp"
#include "SbwtContainer/GpuSbwtContainer.hpp"
#include "SbwtContainer/SbwtContainer.hpp"
#include "sdsl/bit_vectors.hpp"

using sdsl::bit_vector;

namespace sbwt_search {

auto SdslSbwtContainer::do_get_acgt(ACGT letter) -> const u64 * {
  return acgt[static_cast<int>(letter)].data();
}

auto SdslSbwtContainer::get_acgt_sdsl(ACGT letter) const -> bit_vector {
  return acgt[static_cast<int>(letter)];
}

auto BitVectorSbwtContainer::do_get_acgt(ACGT letter) const -> const u64 * {
  return &acgt[static_cast<int>(letter)][0];
}

auto BitVectorSbwtContainer::change_acgt_endianness() -> void {
  for (auto &bit_vector: acgt) {
    for (auto &value: bit_vector) { value = bswap_64(value); }
  }
}

}
